#!/usr/bin/env python3
import argparse
from pathlib import Path

import onnx
from onnxconverter_common import float16


def convert(args):
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = onnx.load(str(input_path))
    model_fp16 = float16.convert_float_to_float16(
        model,
        keep_io_types=args.keep_io_types,
        disable_shape_infer=args.disable_shape_infer,
    )
    onnx.checker.check_model(model_fp16)
    onnx.save(model_fp16, str(output_path))
    print(f"wrote {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--keep-io-types",
        action="store_true",
        help="Keep float model inputs/outputs as FP32 while converting internal tensors to FP16.",
    )
    parser.add_argument(
        "--disable-shape-infer",
        action="store_true",
        help="Skip ONNX shape inference during conversion to reduce conversion memory use.",
    )
    convert(parser.parse_args())


if __name__ == "__main__":
    main()
