#!/usr/bin/env python3

import argparse
import json
import math
import shutil
import subprocess
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray


def _stamp_ns(stamp):
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _percentile(values, percentile):
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * percentile / 100.0
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return values[int(rank)]
    return values[low] * (high - rank) + values[high] * (rank - low)


class GroundingDINOProfiler(Node):
    def __init__(self, args):
        super().__init__("groundingdino_runtime_profiler")
        self.args = args
        self.output_dir = Path(args.output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.samples_path = self.output_dir / f"{args.name}_profile_samples.jsonl"
        self.summary_path = self.output_dir / f"{args.name}_profile_summary.json"

        self.samples_file = self.samples_path.open("w", encoding="utf-8")
        self.started_wall = time.time()
        self.started_mono = time.monotonic()
        self.last_message_mono = None
        self.done = False

        self.input_times = {}
        self.input_first_mono = None
        self.input_last_mono = None
        self.output_first_mono = None
        self.output_last_mono = None
        self.input_count = 0
        self.output_count = 0
        self.matched_outputs = 0
        self.unmatched_outputs = 0
        self.total_detections = 0
        self.latency_ms = []
        self.gpu_samples = []
        self.nvidia_smi = shutil.which("nvidia-smi")

        self.create_subscription(Image, args.input_image_topic, self.input_callback, 50)
        self.create_subscription(Detection2DArray, args.output_detection_topic, self.output_callback, 50)
        self.create_timer(args.gpu_sample_period, self.sample_gpu)
        self.create_timer(0.25, self.idle_check)

        self.get_logger().info(f"Profiling input {args.input_image_topic} and output {args.output_detection_topic}")

    def _mark_seen(self):
        self.last_message_mono = time.monotonic()

    def input_callback(self, msg):
        now = time.monotonic()
        self._mark_seen()
        self.input_count += 1
        if self.input_first_mono is None:
            self.input_first_mono = now
        self.input_last_mono = now
        self.input_times[_stamp_ns(msg.header.stamp)] = now

        if len(self.input_times) > self.args.max_unmatched_inputs:
            oldest_keys = sorted(self.input_times)[: len(self.input_times) - self.args.max_unmatched_inputs]
            for key in oldest_keys:
                self.input_times.pop(key, None)

    def output_callback(self, msg):
        now = time.monotonic()
        self._mark_seen()
        self.output_count += 1
        self.total_detections += len(msg.detections)
        if self.output_first_mono is None:
            self.output_first_mono = now
        self.output_last_mono = now

        stamp = _stamp_ns(msg.header.stamp)
        input_time = self.input_times.pop(stamp, None)
        latency = None
        if input_time is None:
            self.unmatched_outputs += 1
        else:
            latency = (now - input_time) * 1000.0
            self.latency_ms.append(latency)
            self.matched_outputs += 1

        record = {
            "received_unix_time": time.time(),
            "stamp_ns": stamp,
            "output_index": self.output_count,
            "detections": len(msg.detections),
            "matched_input": input_time is not None,
            "latency_ms": latency,
        }
        self.samples_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.samples_file.flush()

    def sample_gpu(self):
        if not self.nvidia_smi:
            return
        try:
            completed = subprocess.run(
                [
                    self.nvidia_smi,
                    "--query-gpu=memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=1.0,
            )
        except (subprocess.SubprocessError, OSError):
            return

        line = completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else ""
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            return
        sample = {
            "unix_time": time.time(),
            "memory_used_mb": float(parts[0]),
            "memory_total_mb": float(parts[1]),
            "gpu_utilization_percent": float(parts[2]),
        }
        self.gpu_samples.append(sample)

    def idle_check(self):
        now = time.monotonic()
        if self.last_message_mono is None:
            if now - self.started_mono > self.args.startup_timeout:
                self.get_logger().warning("No profiled messages received before startup timeout.")
                self.finish()
            return
        if now - self.last_message_mono > self.args.idle_timeout:
            self.finish()

    def finish(self):
        if self.done:
            return
        self.done = True
        self.samples_file.close()

        input_duration = None
        if self.input_first_mono is not None and self.input_last_mono is not None:
            input_duration = max(0.0, self.input_last_mono - self.input_first_mono)
        output_duration = None
        if self.output_first_mono is not None and self.output_last_mono is not None:
            output_duration = max(0.0, self.output_last_mono - self.output_first_mono)

        latencies = self.latency_ms
        gpu_mem_values = [sample["memory_used_mb"] for sample in self.gpu_samples]
        gpu_util_values = [sample["gpu_utilization_percent"] for sample in self.gpu_samples]

        summary = {
            "name": self.args.name,
            "started_unix_time": self.started_wall,
            "finished_unix_time": time.time(),
            "input_image_topic": self.args.input_image_topic,
            "output_detection_topic": self.args.output_detection_topic,
            "input_count": self.input_count,
            "output_count": self.output_count,
            "matched_outputs": self.matched_outputs,
            "unmatched_outputs": self.unmatched_outputs,
            "unprocessed_input_estimate": max(0, self.input_count - self.matched_outputs),
            "input_duration_seconds": input_duration,
            "output_duration_seconds": output_duration,
            "input_hz": self.input_count / input_duration if input_duration and input_duration > 0 else None,
            "output_hz": self.output_count / output_duration if output_duration and output_duration > 0 else None,
            "output_per_input_ratio": self.output_count / self.input_count if self.input_count else None,
            "total_detections": self.total_detections,
            "detections_per_output": self.total_detections / self.output_count if self.output_count else None,
            "latency_ms": {
                "count": len(latencies),
                "mean": sum(latencies) / len(latencies) if latencies else None,
                "min": min(latencies) if latencies else None,
                "p50": _percentile(latencies, 50),
                "p90": _percentile(latencies, 90),
                "p95": _percentile(latencies, 95),
                "p99": _percentile(latencies, 99),
                "max": max(latencies) if latencies else None,
            },
            "gpu": {
                "sample_count": len(self.gpu_samples),
                "memory_used_mb_max": max(gpu_mem_values) if gpu_mem_values else None,
                "memory_used_mb_mean": sum(gpu_mem_values) / len(gpu_mem_values) if gpu_mem_values else None,
                "utilization_percent_max": max(gpu_util_values) if gpu_util_values else None,
                "utilization_percent_mean": sum(gpu_util_values) / len(gpu_util_values) if gpu_util_values else None,
            },
            "samples_path": str(self.samples_path),
        }
        self.summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        self.get_logger().info(f"Finished profiling {self.args.name}: {self.output_count} outputs.")


def parse_args():
    parser = argparse.ArgumentParser(description="Profile GroundingDINO ROS2 runtime latency and throughput.")
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--input-image-topic", default="/camera/camera/color/image_raw")
    parser.add_argument("--output-detection-topic", default="/output_detections")
    parser.add_argument("--idle-timeout", type=float, default=8.0)
    parser.add_argument("--startup-timeout", type=float, default=120.0)
    parser.add_argument("--gpu-sample-period", type=float, default=1.0)
    parser.add_argument("--max-unmatched-inputs", type=int, default=5000)
    return parser.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = GroundingDINOProfiler(args)
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
