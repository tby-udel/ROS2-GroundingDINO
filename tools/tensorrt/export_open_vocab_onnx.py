#!/usr/bin/env python3
import argparse
import gc
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


def _groundingdino_root(path):
    root = Path(path).expanduser().resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _resize_embedding(embedding, rows):
    if embedding is None or embedding.num_embeddings == rows:
        return embedding
    if rows > embedding.num_embeddings:
        raise ValueError(f"Cannot expand embedding from {embedding.num_embeddings} to {rows} rows")
    resized = nn.Embedding(rows, embedding.embedding_dim)
    resized.weight.data.copy_(embedding.weight.data[:rows])
    return resized


def _truncate_module_list(modules, length):
    if modules is None or length is None or len(modules) == length:
        return modules
    if length < 1 or length > len(modules):
        raise ValueError(f"Cannot truncate module list of length {len(modules)} to {length}")
    return nn.ModuleList(list(modules)[:length])


def _load_detector(config_path, checkpoint_path, num_queries=None, encoder_layers=None, decoder_layers=None):
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

    # The TensorRT engine takes encoded text as input, so BERT/tokenization is
    # intentionally kept outside the exported graph for runtime open-vocab use.
    model.bert = nn.Identity()
    model.feat_map = nn.Identity()
    for module in model.modules():
        if hasattr(module, "use_checkpoint"):
            module.use_checkpoint = False
        if hasattr(module, "use_transformer_ckpt"):
            module.use_transformer_ckpt = False
    if num_queries is not None:
        model.num_queries = num_queries
        model.transformer.num_queries = num_queries
        if hasattr(model.transformer, "tgt_embed"):
            model.transformer.tgt_embed = _resize_embedding(model.transformer.tgt_embed, num_queries)
        if hasattr(model.transformer, "refpoint_embed"):
            model.transformer.refpoint_embed = _resize_embedding(
                model.transformer.refpoint_embed,
                num_queries,
            )
        if hasattr(model, "refpoint_embed"):
            model.refpoint_embed = _resize_embedding(model.refpoint_embed, num_queries)
    if encoder_layers is not None:
        encoder = model.transformer.encoder
        encoder.layers = _truncate_module_list(encoder.layers, encoder_layers)
        encoder.text_layers = _truncate_module_list(encoder.text_layers, encoder_layers)
        encoder.fusion_layers = _truncate_module_list(encoder.fusion_layers, encoder_layers)
        encoder.num_layers = encoder_layers
        model.transformer.num_encoder_layers = encoder_layers
    if decoder_layers is not None:
        decoder = model.transformer.decoder
        decoder.layers = _truncate_module_list(decoder.layers, decoder_layers)
        decoder.num_layers = decoder_layers
        model.bbox_embed = _truncate_module_list(model.bbox_embed, decoder_layers)
        model.class_embed = _truncate_module_list(model.class_embed, decoder_layers)
        decoder.bbox_embed = model.bbox_embed
        decoder.class_embed = model.class_embed
        model.transformer.num_decoder_layers = decoder_layers
    return model


class OpenVocabGroundingDINO(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        image,
        encoded_text,
        text_token_mask,
        position_ids,
        text_self_attention_masks,
    ):
        from groundingdino.util.misc import inverse_sigmoid

        batch, _, height, width = image.shape
        image_mask = torch.zeros((batch, height, width), dtype=torch.bool, device=image.device)
        samples = ExportNestedTensor(image, image_mask)

        features, poss = self.model.backbone(samples)
        srcs = []
        masks = []
        for level, feat in enumerate(features):
            src, src_mask = feat.decompose()
            srcs.append(self.model.input_proj[level](src))
            masks.append(src_mask)

        if self.model.num_feature_levels > len(srcs):
            len_srcs = len(srcs)
            for level in range(len_srcs, self.model.num_feature_levels):
                if level == len_srcs:
                    src = self.model.input_proj[level](features[-1].tensors)
                else:
                    src = self.model.input_proj[level](srcs[-1])
                src_mask = F.interpolate(samples.mask[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.model.backbone[1](ExportNestedTensor(src, src_mask)).to(src.dtype)
                srcs.append(src)
                masks.append(src_mask)
                poss.append(pos_l)

        text_dict = {
            "encoded_text": encoded_text,
            "text_token_mask": text_token_mask,
            "position_ids": position_ids,
            "text_self_attention_masks": text_self_attention_masks,
        }
        hs, reference, _, _, _ = self.model.transformer(
            srcs,
            masks,
            None,
            poss,
            None,
            None,
            text_dict,
        )

        coord_outputs = []
        for layer_ref_sig, layer_bbox_embed, layer_hs in zip(reference[:-1], self.model.bbox_embed, hs):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            coord_outputs.append(layer_outputs_unsig.sigmoid())
        boxes = torch.stack(coord_outputs)[-1]

        class_outputs = []
        for layer_hs in hs:
            layer_logits = layer_hs @ text_dict["encoded_text"].transpose(-1, -2)
            layer_logits = layer_logits.masked_fill(~text_dict["text_token_mask"][:, None, :], float("-inf"))
            class_outputs.append(layer_logits)
        logits = torch.stack(class_outputs)[-1]
        return logits, boxes


class ExportNestedTensor:
    def __init__(self, tensors, mask):
        self.tensors = tensors
        self.mask = mask

    @property
    def device(self):
        return self.tensors.device

    def decompose(self):
        return self.tensors, self.mask


PRESETS = {
    "jetson-tiny-e2d2": {
        "height": 128,
        "width": 192,
        "max_text_len": 32,
        "num_queries": 100,
        "encoder_layers": 2,
        "decoder_layers": 2,
    },
    "jetson-ultra": {
        "height": 224,
        "width": 320,
        "max_text_len": 64,
        "num_queries": None,
        "encoder_layers": None,
        "decoder_layers": None,
    },
    "jetson-balanced": {
        "height": 320,
        "width": 480,
        "max_text_len": 96,
        "num_queries": None,
        "encoder_layers": None,
        "decoder_layers": None,
    },
    "desktop-640": {
        "height": 480,
        "width": 640,
        "max_text_len": 256,
        "num_queries": None,
        "encoder_layers": None,
        "decoder_layers": None,
    },
}


def _resolve_preset(args):
    preset = PRESETS[args.preset]
    args.height = args.height if args.height is not None else preset["height"]
    args.width = args.width if args.width is not None else preset["width"]
    args.max_text_len = args.max_text_len if args.max_text_len is not None else preset["max_text_len"]
    args.num_queries = args.num_queries if args.num_queries is not None else preset["num_queries"]
    args.encoder_layers = args.encoder_layers if args.encoder_layers is not None else preset["encoder_layers"]
    args.decoder_layers = args.decoder_layers if args.decoder_layers is not None else preset["decoder_layers"]
    return args


def export(args):
    args = _resolve_preset(args)
    _groundingdino_root(args.groundingdino_dir)
    detector = _load_detector(
        Path(args.config).expanduser().resolve(),
        Path(args.checkpoint).expanduser().resolve(),
        args.num_queries,
        args.encoder_layers,
        args.decoder_layers,
    )
    wrapper = OpenVocabGroundingDINO(detector).eval()

    image = torch.randn(1, 3, args.height, args.width, dtype=torch.float32)
    encoded_text = torch.randn(1, args.max_text_len, detector.hidden_dim, dtype=torch.float32)
    text_token_mask = torch.ones(1, args.max_text_len, dtype=torch.bool)
    position_ids = torch.arange(args.max_text_len, dtype=torch.long).unsqueeze(0)
    text_self_attention_masks = torch.ones(1, args.max_text_len, args.max_text_len, dtype=torch.bool)

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        logits, boxes = wrapper(
            image,
            encoded_text,
            text_token_mask,
            position_ids,
            text_self_attention_masks,
        )
    print(f"dry_run logits={tuple(logits.shape)} boxes={tuple(boxes.shape)}")

    torch.onnx.export(
        wrapper,
        (image, encoded_text, text_token_mask, position_ids, text_self_attention_masks),
        str(output_path),
        input_names=[
            "image",
            "encoded_text",
            "text_token_mask",
            "position_ids",
            "text_self_attention_masks",
        ],
        output_names=["pred_logits", "pred_boxes"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes=None,
        dynamo=args.dynamo,
    )
    print(f"exported {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="jetson-ultra",
        help="Static export shape preset. jetson-ultra is the strongest compression preset.",
    )
    parser.add_argument("--groundingdino-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--max-text-len", type=int, default=None)
    parser.add_argument("--num-queries", type=int, default=None)
    parser.add_argument("--encoder-layers", type=int, default=None)
    parser.add_argument("--decoder-layers", type=int, default=None)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--dynamo", action="store_true", help="Use the newer torch.export-based ONNX exporter")
    parser.add_argument("--output", required=True)
    export(parser.parse_args())


if __name__ == "__main__":
    main()
