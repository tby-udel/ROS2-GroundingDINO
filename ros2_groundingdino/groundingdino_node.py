import os
import re
import sys
import time
import gc
from pathlib import Path

import cv2
import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose


def _discover_groundingdino_root():
    env_root = os.environ.get("GROUNDINGDINO_DIR")
    if env_root:
        return str(Path(env_root).expanduser().resolve())

    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py").exists():
            return str(parent)

    return "/home/boyang/safeai/GroundingDINO"


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(value)


def _as_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_precision(value):
    precision = str(value or "fp32").strip().lower()
    if precision in ("float16", "half", "16"):
        return "fp16"
    if precision in ("float32", "full", "32"):
        return "fp32"
    if precision not in ("fp16", "fp32"):
        return "fp32"
    return precision


def _normalize_text(text):
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .")


def parse_query(query):
    prompt = query.strip().strip("][()")
    if "," in prompt:
        parts = prompt.split(",")
    else:
        parts = re.split(r"[.\n;]+", prompt)

    labels = [_normalize_text(part) for part in parts if _normalize_text(part)]
    if not labels:
        labels = ["a person"]

    caption = " . ".join(labels) + " ."
    return caption, labels


def set_bbox_center(bbox, x, y):
    if hasattr(bbox.center, "position"):
        bbox.center.position.x = float(x)
        bbox.center.position.y = float(y)
        if hasattr(bbox.center, "orientation"):
            bbox.center.orientation.w = 1.0
        return

    bbox.center.x = float(x)
    bbox.center.y = float(y)
    if hasattr(bbox.center, "theta"):
        bbox.center.theta = 0.0


class GroundingDINOPredictor:
    def __init__(self, groundingdino_dir, config_path, checkpoint_path, device, image_size, max_size, precision="fp32"):
        groundingdino_root = Path(groundingdino_dir).expanduser().resolve()
        if str(groundingdino_root) not in sys.path:
            sys.path.insert(0, str(groundingdino_root))

        from groundingdino.datasets import transforms as T
        from groundingdino.models import build_model
        from groundingdino.util.inference import preprocess_caption
        from groundingdino.util.misc import clean_state_dict
        from groundingdino.util.slconfig import SLConfig
        from groundingdino.util.utils import get_phrases_from_posmap

        self.device = device
        self.precision = _normalize_precision(precision)
        self.tensor_dtype = torch.float16 if self.precision == "fp16" and str(device).startswith("cuda") else torch.float32
        self.preprocess_caption = preprocess_caption
        self.get_phrases_from_posmap = get_phrases_from_posmap

        args = SLConfig.fromfile(str(config_path))
        # Build on CPU first for FP16 CUDA so parameters can be narrowed before
        # the large device transfer on memory-constrained Jetsons.
        args.device = "cpu" if self.tensor_dtype == torch.float16 else device
        self.model = build_model(args)

        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        self.load_message = self.model.load_state_dict(clean_state_dict(state_dict), strict=False)
        del checkpoint
        del state_dict
        gc.collect()
        if self.tensor_dtype == torch.float16:
            self.model.half()
            torch.cuda.empty_cache()
        self.model.to(device)
        self.model.eval()

        self.transform = T.Compose(
            [
                T.RandomResize([image_size], max_size=max_size),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def preprocess_image(self, image_bgr):
        from PIL import Image as PILImage

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_pil = PILImage.fromarray(image_rgb)
        image_tensor, _ = self.transform(image_pil, None)
        return image_tensor

    def predict(self, image_bgr, caption, box_threshold, text_threshold):
        image_tensor = self.preprocess_image(image_bgr).to(device=self.device, dtype=self.tensor_dtype)
        caption = self.preprocess_caption(caption)

        with torch.no_grad():
            outputs = self.model(image_tensor[None], captions=[caption])

        logits = outputs["pred_logits"].sigmoid()[0].cpu()
        boxes = outputs["pred_boxes"][0].cpu()
        scores = logits.max(dim=1).values
        keep = scores > box_threshold
        logits = logits[keep]
        boxes = boxes[keep]
        scores = scores[keep]

        tokenized = self.model.tokenizer(caption)
        phrases = []
        for logit in logits:
            phrase = self.get_phrases_from_posmap(
                logit > text_threshold,
                tokenized,
                self.model.tokenizer,
            ).replace(".", "")
            phrases.append(phrase.strip() or "object")

        return boxes, scores, phrases


class GroundingDINOSubscriber(Node):
    def __init__(self):
        super().__init__("groundingdino_subscriber")

        default_root = _discover_groundingdino_root()
        self.declare_parameter("groundingdino_dir", default_root)
        self.declare_parameter("config", str(Path(default_root) / "groundingdino/config/GroundingDINO_SwinT_OGC.py"))
        self.declare_parameter("checkpoint", str(Path(default_root) / "weights/groundingdino_swint_ogc.pth"))
        self.declare_parameter("model", "GroundingDINO-T Swin-T")
        self.declare_parameter("device", "cuda")
        self.declare_parameter("image_encoder_engine", "")
        self.declare_parameter("thresholds", 0.3)
        self.declare_parameter("box_threshold", 0.3)
        self.declare_parameter("text_threshold", 0.25)
        self.declare_parameter("image_size", 800)
        self.declare_parameter("max_size", 1333)
        self.declare_parameter("precision", "fp32")
        self.declare_parameter("initial_query", "a person, a box")
        self.declare_parameter("publish_output_image", False)
        self.declare_parameter("publish_legacy_outputs", False)
        self.declare_parameter("legacy_detection_topic", "/yolo/detections")
        self.declare_parameter("legacy_image_topic", "/yolo/inference_image")
        self.declare_parameter("drop_frames_when_busy", True)

        self.cv_br = CvBridge()
        self.processing_image = False

        self.output_publisher = self.create_publisher(Detection2DArray, "output_detections", 10)
        self.output_image_publisher = self.create_publisher(Image, "output_image", 10)

        self.query_subscription = self.create_subscription(
            String,
            "input_query",
            self.query_listener_callback,
            10,
        )
        self.image_subscription = self.create_subscription(
            Image,
            "input_image",
            self.listener_callback,
            1,
        )

        groundingdino_dir = self.get_parameter("groundingdino_dir").value
        config_path = Path(self.get_parameter("config").value).expanduser()
        checkpoint_path = Path(self.get_parameter("checkpoint").value).expanduser()
        device = str(self.get_parameter("device").value)
        if device.startswith("cuda") and not torch.cuda.is_available():
            self.get_logger().warning("CUDA is not available; falling back to CPU inference.")
            device = "cpu"
        self.device = device

        self.thresholds = _as_float(self.get_parameter("thresholds").value, 0.3)
        self.box_threshold = _as_float(self.get_parameter("box_threshold").value, self.thresholds)
        self.text_threshold = _as_float(self.get_parameter("text_threshold").value, 0.25)
        self.image_size = _as_int(self.get_parameter("image_size").value, 800)
        self.max_size = _as_int(self.get_parameter("max_size").value, 1333)
        self.precision = _normalize_precision(self.get_parameter("precision").value)
        if self.precision == "fp16" and not str(device).startswith("cuda"):
            self.get_logger().warning("FP16 is only enabled for CUDA inference; using FP32 tensors on CPU.")
            self.precision = "fp32"
        self.publish_output_image = _as_bool(self.get_parameter("publish_output_image").value)
        self.publish_legacy_outputs = _as_bool(self.get_parameter("publish_legacy_outputs").value)
        self.legacy_detection_topic = str(self.get_parameter("legacy_detection_topic").value)
        self.legacy_image_topic = str(self.get_parameter("legacy_image_topic").value)
        self.drop_frames_when_busy = _as_bool(self.get_parameter("drop_frames_when_busy").value)

        self.legacy_detection_publisher = self.create_publisher(String, self.legacy_detection_topic, 10)
        self.legacy_image_publisher = self.create_publisher(Image, self.legacy_image_topic, 10)

        initial_query = str(self.get_parameter("initial_query").value)
        self._update_query_cache(initial_query)

        self.get_logger().info(f"Loading GroundingDINO from {checkpoint_path}")
        self.get_logger().info(
            f"GroundingDINO runtime: device={self.device}, precision={self.precision}, "
            f"image_size={self.image_size}, max_size={self.max_size}"
        )
        self.predictor = GroundingDINOPredictor(
            groundingdino_dir=groundingdino_dir,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            device=self.device,
            image_size=self.image_size,
            max_size=self.max_size,
            precision=self.precision,
        )
        self.get_logger().info("GroundingDINO ready.")

    def _update_query_cache(self, query):
        self.query = query
        self.caption, self.query_text = parse_query(query)
        self.get_logger().info("Updated query: %s" % self.query_text)

    def query_listener_callback(self, msg):
        if msg.data != self.query:
            self._update_query_cache(msg.data)

    def _class_id_for_phrase(self, phrase):
        phrase_norm = _normalize_text(phrase)
        for index, label in enumerate(self.query_text):
            if phrase_norm == label or phrase_norm.endswith(label) or label in phrase_norm:
                return str(index), label
        return phrase_norm or "object", phrase_norm or "object"

    def _detections_to_msg(self, header, boxes, scores, phrases, image_shape):
        from torchvision.ops import box_convert

        detections_arr = Detection2DArray()
        detections_arr.header = header

        height, width = image_shape[:2]
        if len(boxes) == 0:
            return detections_arr, []

        scale = torch.tensor([width, height, width, height], dtype=boxes.dtype)
        boxes_xyxy = box_convert(boxes * scale, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, width - 1)
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, height - 1)

        legacy_detections = []
        for box, score, phrase in zip(boxes_xyxy, scores.numpy(), phrases):
            x1, y1, x2, y2 = [float(value) for value in box]
            class_id, class_name = self._class_id_for_phrase(phrase)

            obj = Detection2D()
            obj.header = header
            obj.bbox.size_x = abs(x2 - x1)
            obj.bbox.size_y = abs(y2 - y1)
            set_bbox_center(obj.bbox, (x1 + x2) / 2.0, (y1 + y2) / 2.0)

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = class_id
            hyp.hypothesis.score = float(score)
            obj.results.append(hyp)
            detections_arr.detections.append(obj)

            legacy_detections.append(
                f"{class_name} [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}] ({float(score):.2f})"
            )

        return detections_arr, legacy_detections

    def _draw_detections(self, image_bgr, boxes, scores, phrases):
        from torchvision.ops import box_convert

        annotated = image_bgr.copy()
        height, width = annotated.shape[:2]
        if len(boxes) == 0:
            return annotated

        scale = torch.tensor([width, height, width, height], dtype=boxes.dtype)
        boxes_xyxy = box_convert(boxes * scale, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, width - 1)
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, height - 1)

        for box, score, phrase in zip(boxes_xyxy, scores.numpy(), phrases):
            x1, y1, x2, y2 = [int(round(value)) for value in box]
            _, class_name = self._class_id_for_phrase(phrase)
            label = f"{class_name} {float(score):.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (34, 197, 94), 2)
            text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_w, text_h = text_size
            y_text = max(y1, text_h + baseline + 4)
            cv2.rectangle(
                annotated,
                (x1, y_text - text_h - baseline - 4),
                (x1 + text_w + 6, y_text + baseline),
                (34, 197, 94),
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (x1 + 3, y_text - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        return annotated

    def listener_callback(self, data):
        if self.processing_image and self.drop_frames_when_busy:
            return

        self.processing_image = True
        start = time.time()

        try:
            cv_img = self.cv_br.imgmsg_to_cv2(data, desired_encoding="bgr8")
            boxes, scores, phrases = self.predictor.predict(
                cv_img,
                self.caption,
                self.box_threshold,
                self.text_threshold,
            )

            detections_arr, legacy_detections = self._detections_to_msg(
                data.header,
                boxes,
                scores,
                phrases,
                cv_img.shape,
            )
            self.output_publisher.publish(detections_arr)

            if self.publish_legacy_outputs:
                legacy_message = "; ".join(legacy_detections) if legacy_detections else "no detections"
                self.legacy_detection_publisher.publish(String(data=legacy_message))

            if self.publish_output_image or self.publish_legacy_outputs:
                image = self._draw_detections(cv_img, boxes, scores, phrases)
                image_msg = self.cv_br.cv2_to_imgmsg(image, "bgr8")
                image_msg.header = data.header
                if self.publish_output_image:
                    self.output_image_publisher.publish(image_msg)
                if self.publish_legacy_outputs:
                    self.legacy_image_publisher.publish(image_msg)

            elapsed_ms = (time.time() - start) * 1000.0
            self.get_logger().debug(f"GroundingDINO frame processed in {elapsed_ms:.1f} ms")
        except Exception as exc:
            self.get_logger().error(f"GroundingDINO inference failed: {exc}")
        finally:
            self.processing_image = False


def main(args=None):
    rclpy.init(args=args)
    node = GroundingDINOSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
