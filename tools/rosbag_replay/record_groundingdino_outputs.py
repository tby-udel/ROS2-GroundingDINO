#!/usr/bin/env python3

import argparse
import json
import signal
import time
from pathlib import Path

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from vision_msgs.msg import Detection2DArray


def _stamp_to_dict(stamp):
    return {
        "sec": int(stamp.sec),
        "nanosec": int(stamp.nanosec),
        "nanoseconds": int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec),
    }


def _bbox_center_xy(bbox):
    center = bbox.center
    if hasattr(center, "position"):
        return float(center.position.x), float(center.position.y)
    return float(center.x), float(center.y)


def _parse_labels(text):
    if "," in text:
        parts = text.split(",")
    else:
        parts = text.replace("\n", ".").replace(";", ".").split(".")
    return [part.strip().lower() for part in parts if part.strip()]


class GroundingDINOOutputRecorder(Node):
    def __init__(self, args):
        super().__init__("groundingdino_output_recorder")
        self.args = args
        self.bridge = CvBridge()
        self.output_dir = Path(args.output_dir).expanduser().resolve()
        self.frames_dir = self.output_dir / "sample_frames"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)

        self.labels = [label.strip() for label in args.labels.split(",") if label.strip()]
        self.current_labels = list(self.labels)
        self.query_events = []
        self.started_at = time.monotonic()
        self.last_message_at = None
        self.done = False

        self.video_path = self.output_dir / f"{args.name}_annotated.mp4"
        self.legacy_path = self.output_dir / f"{args.name}_legacy_detections.jsonl"
        self.structured_path = self.output_dir / f"{args.name}_structured_detections.jsonl"
        self.query_events_path = self.output_dir / f"{args.name}_query_events.jsonl"
        self.summary_path = self.output_dir / f"{args.name}_summary.json"

        self.video_writer = None
        self.video_frames = 0
        self.legacy_messages = 0
        self.structured_messages = 0
        self.first_image_stamp = None
        self.last_image_stamp = None

        self.legacy_file = self.legacy_path.open("w", encoding="utf-8")
        self.structured_file = self.structured_path.open("w", encoding="utf-8")
        self.query_events_file = self.query_events_path.open("w", encoding="utf-8")

        self.create_subscription(Image, args.image_topic, self.image_callback, 10)
        self.create_subscription(String, args.legacy_detection_topic, self.legacy_detection_callback, 10)
        self.create_subscription(Detection2DArray, args.structured_detection_topic, self.structured_detection_callback, 10)
        self.create_subscription(String, args.query_topic, self.query_callback, 10)
        self.create_timer(0.25, self.idle_check)

        self.get_logger().info(f"Recording annotated images from {args.image_topic} to {self.video_path}")
        self.get_logger().info(f"Recording legacy detections from {args.legacy_detection_topic} to {self.legacy_path}")
        self.get_logger().info(f"Recording structured detections from {args.structured_detection_topic} to {self.structured_path}")

    def _mark_seen(self):
        self.last_message_at = time.monotonic()

    def query_callback(self, msg):
        labels = _parse_labels(msg.data)
        if labels:
            self.current_labels = labels
        record = {
            "received_unix_time": time.time(),
            "query": msg.data,
            "labels": list(self.current_labels),
        }
        self.query_events.append(record)
        self.query_events_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.query_events_file.flush()

    def _open_video(self, width, height):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(str(self.video_path), fourcc, float(self.args.fps), (width, height))
        if not self.video_writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {self.video_path}")

    def image_callback(self, msg):
        self._mark_seen()
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        height, width = frame.shape[:2]
        if self.video_writer is None:
            self._open_video(width, height)
            self.first_image_stamp = _stamp_to_dict(msg.header.stamp)

        self.video_writer.write(frame)
        self.video_frames += 1
        self.last_image_stamp = _stamp_to_dict(msg.header.stamp)

        if self.video_frames == 1 or (
            self.args.sample_every > 0 and self.video_frames % self.args.sample_every == 0
        ):
            sample_path = self.frames_dir / f"{self.args.name}_frame_{self.video_frames:06d}.jpg"
            cv2.imwrite(str(sample_path), frame)

    def legacy_detection_callback(self, msg):
        self._mark_seen()
        self.legacy_messages += 1
        record = {
            "message_index": self.legacy_messages,
            "received_unix_time": time.time(),
            "data": msg.data,
        }
        self.legacy_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.legacy_file.flush()

    def structured_detection_callback(self, msg):
        self._mark_seen()
        self.structured_messages += 1
        detections = []
        for detection in msg.detections:
            cx, cy = _bbox_center_xy(detection.bbox)
            results = []
            for result in detection.results:
                class_id = str(result.hypothesis.class_id)
                label = class_id
                if class_id.isdigit():
                    idx = int(class_id)
                    if 0 <= idx < len(self.current_labels):
                        label = self.current_labels[idx]
                results.append(
                    {
                        "class_id": class_id,
                        "label": label,
                        "score": float(result.hypothesis.score),
                    }
                )
            detections.append(
                {
                    "center_x": cx,
                    "center_y": cy,
                    "size_x": float(detection.bbox.size_x),
                    "size_y": float(detection.bbox.size_y),
                    "results": results,
                }
            )

        record = {
            "message_index": self.structured_messages,
            "stamp": _stamp_to_dict(msg.header.stamp),
            "frame_id": msg.header.frame_id,
            "query_labels": list(self.current_labels),
            "detections": detections,
        }
        self.structured_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.structured_file.flush()

    def idle_check(self):
        now = time.monotonic()
        if self.last_message_at is None:
            if now - self.started_at > self.args.startup_timeout:
                self.get_logger().warning("No output messages received before startup timeout.")
                self.finish()
            return

        if now - self.last_message_at > self.args.idle_timeout:
            self.finish()

    def finish(self):
        if self.done:
            return
        self.done = True
        if self.video_writer is not None:
            self.video_writer.release()
        self.legacy_file.close()
        self.structured_file.close()
        self.query_events_file.close()

        summary = {
            "name": self.args.name,
            "labels": self.labels,
            "final_query_labels": self.current_labels,
            "query_events_path": str(self.query_events_path),
            "query_events": self.query_events,
            "image_topic": self.args.image_topic,
            "legacy_detection_topic": self.args.legacy_detection_topic,
            "structured_detection_topic": self.args.structured_detection_topic,
            "video_path": str(self.video_path),
            "legacy_detections_path": str(self.legacy_path),
            "structured_detections_path": str(self.structured_path),
            "sample_frames_dir": str(self.frames_dir),
            "video_frames": self.video_frames,
            "legacy_messages": self.legacy_messages,
            "structured_messages": self.structured_messages,
            "first_image_stamp": self.first_image_stamp,
            "last_image_stamp": self.last_image_stamp,
        }
        self.summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        self.get_logger().info(f"Finished recording {self.args.name}: {self.video_frames} video frames.")


def parse_args():
    parser = argparse.ArgumentParser(description="Record GroundingDINO ROS2 output topics to inspection artifacts.")
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--image-topic", default="/yolo/inference_image")
    parser.add_argument("--query-topic", default="/input_query")
    parser.add_argument("--legacy-detection-topic", default="/yolo/detections")
    parser.add_argument("--structured-detection-topic", default="/output_detections")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--sample-every", type=int, default=45)
    parser.add_argument("--idle-timeout", type=float, default=6.0)
    parser.add_argument("--startup-timeout", type=float, default=120.0)
    return parser.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = GroundingDINOOutputRecorder(args)

    def handle_signal(signum, frame):
        node.finish()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.2)
    finally:
        node.finish()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
