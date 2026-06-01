#!/usr/bin/env python3
import argparse
import gc
import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt
import torch
from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _repo_root():
    return Path(__file__).resolve().parents[2]


def _default_groundingdino_dir():
    return _repo_root().parent / "GroundingDINO"


def _normalize_label(label):
    label = label.strip().lower()
    label = re.sub(r"\s+", " ", label)
    return label.strip(" .")


def parse_query(query):
    prompt = query.strip().strip("][()")
    if "," in prompt:
        parts = prompt.split(",")
    else:
        parts = re.split(r"[.\n;]+", prompt)
    labels = [_normalize_label(part) for part in parts if _normalize_label(part)]
    if not labels:
        labels = ["box"]
    caption = " . ".join(labels) + " ."
    return caption, labels


def preprocess_caption(caption):
    caption = caption.lower().strip()
    return caption if caption.endswith(".") else caption + "."


class GroundingDINOTextEncoder(torch.nn.Module):
    def __init__(self, model, max_text_len):
        super().__init__()
        self.tokenizer = model.tokenizer
        self.specical_tokens = model.specical_tokens
        self.sub_sentence_present = model.sub_sentence_present
        self.bert = model.bert
        self.feat_map = model.feat_map
        self.max_text_len = max_text_len

    @torch.no_grad()
    def encode(self, caption):
        from groundingdino.models.GroundingDINO.bertwarper import (
            generate_masks_with_special_tokens_and_transfer_map,
        )

        caption = preprocess_caption(caption)
        tokenized = self.tokenizer([caption], padding="longest", return_tensors="pt")
        text_self_attention_masks, position_ids, cate_to_token_mask_list = generate_masks_with_special_tokens_and_transfer_map(
            tokenized,
            self.specical_tokens,
            self.tokenizer,
        )
        category_masks = cate_to_token_mask_list[0]

        if text_self_attention_masks.shape[1] > self.max_text_len:
            text_self_attention_masks = text_self_attention_masks[
                :, : self.max_text_len, : self.max_text_len
            ]
            position_ids = position_ids[:, : self.max_text_len]
            tokenized["input_ids"] = tokenized["input_ids"][:, : self.max_text_len]
            tokenized["attention_mask"] = tokenized["attention_mask"][:, : self.max_text_len]
            if "token_type_ids" in tokenized:
                tokenized["token_type_ids"] = tokenized["token_type_ids"][:, : self.max_text_len]
            category_masks = category_masks[:, : self.max_text_len]

        if self.sub_sentence_present:
            tokenized_for_encoder = {k: v for k, v in tokenized.items() if k != "attention_mask"}
            tokenized_for_encoder["attention_mask"] = text_self_attention_masks
            tokenized_for_encoder["position_ids"] = position_ids
        else:
            tokenized_for_encoder = tokenized

        bert_output = self.bert(**tokenized_for_encoder)
        encoded_text = self.feat_map(bert_output["last_hidden_state"])
        text_token_mask = tokenized.attention_mask.bool()

        if encoded_text.shape[1] > self.max_text_len:
            encoded_text = encoded_text[:, : self.max_text_len, :]
            text_token_mask = text_token_mask[:, : self.max_text_len]
            position_ids = position_ids[:, : self.max_text_len]
            text_self_attention_masks = text_self_attention_masks[
                :, : self.max_text_len, : self.max_text_len
            ]

        seq_len = encoded_text.shape[1]
        if seq_len < self.max_text_len:
            pad_len = self.max_text_len - seq_len
            encoded_text = torch.nn.functional.pad(encoded_text, (0, 0, 0, pad_len), value=0.0)
            text_token_mask = torch.nn.functional.pad(text_token_mask, (0, pad_len), value=False)
            position_ids = torch.nn.functional.pad(position_ids, (0, pad_len), value=0)
            padded_attention = torch.zeros(
                (1, self.max_text_len, self.max_text_len),
                dtype=text_self_attention_masks.dtype,
            )
            padded_attention[:, :seq_len, :seq_len] = text_self_attention_masks
            text_self_attention_masks = padded_attention
            padded_category_masks = torch.zeros(
                (category_masks.shape[0], self.max_text_len),
                dtype=category_masks.dtype,
            )
            padded_category_masks[:, : category_masks.shape[1]] = category_masks
            category_masks = padded_category_masks

        return {
            "encoded_text": encoded_text.contiguous(),
            "text_token_mask": text_token_mask.contiguous(),
            "position_ids": position_ids.contiguous(),
            "text_self_attention_masks": text_self_attention_masks.contiguous(),
            "category_masks": category_masks.contiguous(),
            "tokenized": self.tokenizer(caption),
            "caption": caption,
            "seq_len": seq_len,
        }


def load_text_encoder(groundingdino_dir, config_path, checkpoint_path, max_text_len):
    groundingdino_dir = Path(groundingdino_dir).expanduser().resolve()
    if str(groundingdino_dir) not in sys.path:
        sys.path.insert(0, str(groundingdino_dir))

    from groundingdino.models import build_model
    from groundingdino.util.misc import clean_state_dict
    from groundingdino.util.slconfig import SLConfig

    args = SLConfig.fromfile(str(config_path))
    args.device = "cpu"
    model = build_model(args)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(clean_state_dict(state_dict), strict=False)
    del checkpoint
    del state_dict
    gc.collect()
    model.eval()

    encoder = GroundingDINOTextEncoder(model, max_text_len).eval()
    del model
    gc.collect()
    return encoder


def trt_dtype_to_torch(dtype):
    if dtype == trt.float32:
        return torch.float32
    if dtype == trt.float16:
        return torch.float16
    if dtype == trt.int32:
        return torch.int32
    if dtype == trt.int64:
        return torch.int64
    if dtype == trt.bool:
        return torch.bool
    raise TypeError(f"Unsupported TensorRT dtype: {dtype}")


class TorchTensorRTRunner:
    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as engine_file:
            self.engine = self.runtime.deserialize_cuda_engine(engine_file.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context")

        self.input_names = []
        self.output_names = []
        self.tensor_shapes = {}
        self.tensor_dtypes = {}
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = trt_dtype_to_torch(self.engine.get_tensor_dtype(name))
            self.tensor_shapes[name] = shape
            self.tensor_dtypes[name] = dtype
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        self.outputs = {
            name: torch.empty(self.tensor_shapes[name], dtype=self.tensor_dtypes[name], device="cuda")
            for name in self.output_names
        }

    def infer(self, inputs):
        for name in self.input_names:
            tensor = inputs[name]
            expected_shape = self.tensor_shapes[name]
            expected_dtype = self.tensor_dtypes[name]
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(f"{name} has shape {tuple(tensor.shape)}, expected {expected_shape}")
            if tensor.dtype != expected_dtype:
                raise TypeError(f"{name} has dtype {tensor.dtype}, expected {expected_dtype}")
            self.context.set_tensor_address(name, tensor.data_ptr())

        for name, tensor in self.outputs.items():
            self.context.set_tensor_address(name, tensor.data_ptr())

        stream = torch.cuda.current_stream()
        ok = self.context.execute_async_v3(stream.cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT execution failed")
        stream.synchronize()
        return {name: tensor.detach().cpu() for name, tensor in self.outputs.items()}


def preprocess_image(frame_bgr, height, width):
    image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    image = resized.astype(np.float32) / 255.0
    image = (image - MEAN) / STD
    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(image).unsqueeze(0).to(device="cuda", dtype=torch.float16).contiguous()


def open_bag_reader(bag_path):
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=str(bag_path), storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader.open(storage_options, converter_options)
    topics = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}
    return reader, topics


def color_for_label(label):
    palette = [
        (65, 148, 255),
        (80, 210, 120),
        (230, 115, 80),
        (215, 175, 70),
        (190, 105, 220),
        (70, 205, 210),
        (110, 130, 255),
    ]
    return palette[abs(hash(label)) % len(palette)]


def draw_label(frame, text, x1, y1, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    label_y1 = max(0, y1 - text_h - baseline - 5)
    label_y2 = label_y1 + text_h + baseline + 5
    label_x2 = min(frame.shape[1] - 1, x1 + text_w + 8)
    cv2.rectangle(frame, (x1, label_y1), (label_x2, label_y2), color, -1)
    cv2.putText(frame, text, (x1 + 4, label_y2 - baseline - 3), font, scale, (20, 20, 20), thickness, cv2.LINE_AA)


def annotate_frame(frame, detections, frame_index, query, inference_ms):
    annotated = frame.copy()
    height, width = annotated.shape[:2]
    for detection in detections:
        x1, y1, x2, y2 = detection["xyxy"]
        label = detection["label"]
        score = detection["score"]
        color = color_for_label(label)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        draw_label(annotated, f"{label} {score:.2f}", x1, y1, color)

    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (width, 52), (25, 25, 25), -1)
    annotated = cv2.addWeighted(overlay, 0.72, annotated, 0.28, 0.0)
    cv2.putText(
        annotated,
        f"Query: {query}",
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        f"frame {frame_index} | detections {len(detections)} | TRT {inference_ms:.1f} ms",
        (8, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    return annotated


def postprocess(
    outputs,
    tokenized,
    tokenizer,
    image_shape,
    box_threshold,
    text_threshold,
    max_detections,
    labels=None,
    category_masks=None,
):
    from groundingdino.util.utils import get_phrases_from_posmap

    frame_h, frame_w = image_shape[:2]
    logits = outputs["pred_logits"].float().sigmoid()[0]
    boxes = outputs["pred_boxes"].float()[0]

    label_indices = None
    if labels and category_masks is not None and category_masks.numel() > 0:
        category_masks = category_masks.to(dtype=torch.bool)
        category_scores = []
        for mask in category_masks:
            if mask.any():
                category_scores.append(logits[:, mask].max(dim=1).values)
            else:
                category_scores.append(torch.zeros(logits.shape[0], dtype=logits.dtype))
        category_scores = torch.stack(category_scores, dim=1)
        scores, label_indices = category_scores.max(dim=1)
    else:
        scores = logits.max(dim=1).values

    keep = torch.nonzero(scores > box_threshold, as_tuple=False).flatten()
    if keep.numel() == 0:
        return []

    keep_scores = scores[keep]
    order = torch.argsort(keep_scores, descending=True)
    keep = keep[order[:max_detections]]

    detections = []
    for index in keep.tolist():
        score = float(scores[index].item())
        box = boxes[index]
        cx, cy, bw, bh = [float(value) for value in box.tolist()]
        x1 = int(round((cx - bw / 2.0) * frame_w))
        y1 = int(round((cy - bh / 2.0) * frame_h))
        x2 = int(round((cx + bw / 2.0) * frame_w))
        y2 = int(round((cy + bh / 2.0) * frame_h))
        x1 = max(0, min(frame_w - 1, x1))
        y1 = max(0, min(frame_h - 1, y1))
        x2 = max(0, min(frame_w - 1, x2))
        y2 = max(0, min(frame_h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        if label_indices is not None:
            label = labels[int(label_indices[index].item())]
        else:
            posmap = logits[index] > text_threshold
            phrase = get_phrases_from_posmap(posmap.clone(), tokenized, tokenizer).replace(".", "")
            label = phrase.strip() or "object"
        detections.append(
            {
                "label": label,
                "score": score,
                "xyxy": [x1, y1, x2, y2],
            }
        )
    return detections


def score_count_summary(outputs, thresholds, category_masks=None):
    logits = outputs["pred_logits"].float().sigmoid()[0]
    if category_masks is not None and category_masks.numel() > 0:
        category_masks = category_masks.to(dtype=torch.bool)
        category_scores = []
        for mask in category_masks:
            if mask.any():
                category_scores.append(logits[:, mask].max(dim=1).values)
            else:
                category_scores.append(torch.zeros(logits.shape[0], dtype=logits.dtype))
        scores = torch.stack(category_scores, dim=1).max(dim=1).values
    else:
        scores = logits.max(dim=1).values
    return {str(threshold): int((scores > threshold).sum().item()) for threshold in thresholds}


def parse_args():
    repo = _repo_root()
    groundingdino_dir = _default_groundingdino_dir()
    parser = argparse.ArgumentParser(description="Render compressed GroundingDINO TensorRT detections from a ROS 2 bag.")
    parser.add_argument(
        "--bag",
        default=str(repo.parent.parent / "testing_rosbags/indoor_640x480_15hz"),
        help="Path to the rosbag2 directory.",
    )
    parser.add_argument("--topic", default="/camera/camera/color/image_raw")
    parser.add_argument(
        "--engine",
        default=str(repo / "artifacts/tensorrt/groundingdino_swint_open_vocab_128x192_text32_q100_e2d2_fp16.engine"),
    )
    parser.add_argument("--groundingdino-dir", default=str(groundingdino_dir))
    parser.add_argument(
        "--config",
        default=str(groundingdino_dir / "groundingdino/config/GroundingDINO_SwinT_OGC.py"),
    )
    parser.add_argument(
        "--checkpoint",
        default=str(groundingdino_dir / "weights/groundingdino_swint_ogc.pth"),
    )
    parser.add_argument("--query", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--box-threshold", type=float, default=0.1)
    parser.add_argument("--text-threshold", type=float, default=0.1)
    parser.add_argument("--max-detections", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=30)
    parser.add_argument("--log-score-stats", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the TensorRT engine")

    bag_path = Path(args.bag).expanduser().resolve()
    engine_path = Path(args.engine).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve() if args.output else (
        _repo_root() / "artifacts/rosbag_tests/indoor_640x480_15hz_groundingdino_trt_e2d2_fp16.mp4"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = output_path.with_suffix(".json")

    caption, labels = parse_query(args.query)
    print(f"query labels: {labels}")
    print(f"groundingdino caption: {caption}")

    print("loading text encoder...")
    text_encoder = load_text_encoder(
        args.groundingdino_dir,
        Path(args.config).expanduser().resolve(),
        Path(args.checkpoint).expanduser().resolve(),
        max_text_len=32,
    )
    text = text_encoder.encode(caption)
    print(f"text tokens used: {text['seq_len']} / 32")
    text_inputs = {
        "encoded_text": text["encoded_text"].to(device="cuda", dtype=torch.float16).contiguous(),
        "text_token_mask": text["text_token_mask"].to(device="cuda").contiguous(),
        "position_ids": text["position_ids"].to(device="cuda").contiguous(),
        "text_self_attention_masks": text["text_self_attention_masks"].to(device="cuda").contiguous(),
    }

    print(f"loading TensorRT engine: {engine_path}")
    runner = TorchTensorRTRunner(engine_path)
    image_shape = runner.tensor_shapes["image"]
    _, _, engine_h, engine_w = image_shape
    print(f"engine image shape: {engine_h}x{engine_w}")

    reader, topics = open_bag_reader(bag_path)
    if args.topic not in topics:
        raise RuntimeError(f"Topic {args.topic} not found. Available topics: {sorted(topics)}")
    msg_type = get_message(topics[args.topic])
    bridge = CvBridge()

    writer = None
    frames = 0
    total_detections = 0
    inference_ms_values = []
    score_count_samples = []
    started = time.perf_counter()
    first_stamp = None
    last_stamp = None

    try:
        while reader.has_next():
            topic, serialized, timestamp = reader.read_next()
            if topic != args.topic:
                continue
            msg = deserialize_message(serialized, msg_type)
            frame = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            if writer is None:
                height, width = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(output_path), fourcc, args.fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError(f"Failed to open video writer: {output_path}")
                first_stamp = int(timestamp)

            image_tensor = preprocess_image(frame, engine_h, engine_w)
            inputs = {"image": image_tensor, **text_inputs}

            infer_start = time.perf_counter()
            outputs = runner.infer(inputs)
            inference_ms = (time.perf_counter() - infer_start) * 1000.0
            inference_ms_values.append(inference_ms)

            detections = postprocess(
                outputs,
                text["tokenized"],
                text_encoder.tokenizer,
                frame.shape,
                args.box_threshold,
                args.text_threshold,
                args.max_detections,
                labels=labels,
                category_masks=text["category_masks"],
            )
            total_detections += len(detections)
            score_counts = score_count_summary(outputs, (0.1, 0.3, 0.5, 0.7, 0.9), text["category_masks"])
            score_count_samples.append(score_counts)
            frames += 1
            last_stamp = int(timestamp)
            writer.write(annotate_frame(frame, detections, frames, ", ".join(labels), inference_ms))

            if args.log_every > 0 and (frames == 1 or frames % args.log_every == 0):
                score_text = f", score_counts={score_counts}" if args.log_score_stats else ""
                print(
                    f"processed {frames} frames, last detections={len(detections)}, "
                    f"TRT={inference_ms:.1f} ms{score_text}"
                )

            if args.max_frames > 0 and frames >= args.max_frames:
                break
    finally:
        if writer is not None:
            writer.release()

    elapsed = time.perf_counter() - started
    summary = {
        "bag": str(bag_path),
        "topic": args.topic,
        "engine": str(engine_path),
        "output_video": str(output_path),
        "query": args.query,
        "labels": labels,
        "caption": caption,
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "frames": frames,
        "fps": args.fps,
        "elapsed_seconds": elapsed,
        "processed_fps": frames / elapsed if elapsed > 0 else None,
        "total_detections": total_detections,
        "detections_per_frame": total_detections / frames if frames else None,
        "trt_inference_ms": {
            "mean": float(np.mean(inference_ms_values)) if inference_ms_values else None,
            "min": float(np.min(inference_ms_values)) if inference_ms_values else None,
            "max": float(np.max(inference_ms_values)) if inference_ms_values else None,
        },
        "score_count_samples": score_count_samples[:20],
        "first_bag_timestamp_ns": first_stamp,
        "last_bag_timestamp_ns": last_stamp,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote video: {output_path}")
    print(f"wrote summary: {summary_path}")
    print(f"frames={frames} total_detections={total_detections} mean_trt_ms={summary['trt_inference_ms']['mean']}")


if __name__ == "__main__":
    main()
