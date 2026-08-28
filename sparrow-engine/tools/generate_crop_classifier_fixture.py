#!/usr/bin/env python3
# /// script
# dependencies = ["onnx>=1.17"]
# ///
"""Generate tiny dynamic/static image-classifier fixtures for crop batching."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def build_model(batch: str | int) -> onnx.ModelProto:
    input_info = helper.make_tensor_value_info(
        "pixel_values", TensorProto.FLOAT, [batch, 3, 2, 2]
    )
    output_info = helper.make_tensor_value_info(
        "logits", TensorProto.FLOAT, [batch, 2]
    )
    axes_reduce = helper.make_tensor(
        "axes_reduce", TensorProto.INT64, [3], [1, 2, 3]
    )
    axes_unsqueeze = helper.make_tensor(
        "axes_unsqueeze", TensorProto.INT64, [1], [1]
    )
    nodes = [
        helper.make_node(
            "ReduceMean",
            ["pixel_values", "axes_reduce"],
            ["mean"],
            keepdims=0,
        ),
        helper.make_node("Unsqueeze", ["mean", "axes_unsqueeze"], ["positive"]),
        helper.make_node("Neg", ["positive"], ["negative"]),
        helper.make_node(
            "Concat", ["positive", "negative"], ["logits"], axis=1
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "synthetic_crop_classifier",
        [input_info],
        [output_info],
        [axes_reduce, axes_unsqueeze],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18)],
        producer_name="sparrow-engine-fixture",
    )
    onnx.checker.check_model(model)
    return model


def write_fixture(root: Path, name: str, batch: str | int) -> None:
    destination = root / name
    destination.mkdir(parents=True, exist_ok=True)
    onnx.save(build_model(batch), destination / "model.onnx")
    (destination / "labels.txt").write_text(
        "positive\nnegative\n", encoding="utf-8"
    )
    (destination / "manifest.toml").write_text(
        f"""[model]
id = "{name}"
format = "onnx"
file = "model.onnx"

[preprocessing]
method = "resize"
input_size = [2, 2]
layout = "nchw"
normalization = "unit"
interpolation = "nearest"

[inference]
strategy = "single"

[postprocessing]
method = "softmax"

[labels]
file = "labels.txt"
format = "one_per_line"
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    write_fixture(args.output_root, "synthetic-classifier-dynamic", "batch")
    write_fixture(args.output_root, "synthetic-classifier-static", 1)


if __name__ == "__main__":
    main()
