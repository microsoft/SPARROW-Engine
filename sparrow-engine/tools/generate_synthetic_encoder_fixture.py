#!/usr/bin/env python3
"""Generate the tiny deterministic image-encoder ONNX fixture."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

MODEL_ID = "synthetic-image-encoder"
EMBEDDING_VERSION = "synthetic-encoder-v1"
EMBEDDING_DIM = 8
INPUT_SIZE = 16


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "sparrow-engine-core" / "tests" / "fixtures" / "image" / MODEL_ID
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.onnx"
    manifest_path = out_dir / "manifest.toml"

    input_tensor = helper.make_tensor_value_info(
        "image", TensorProto.FLOAT, [1, 3, INPUT_SIZE, INPUT_SIZE]
    )
    output_tensor = helper.make_tensor_value_info(
        "embedding", TensorProto.FLOAT, [1, EMBEDDING_DIM]
    )

    weights = np.array(
        [
            [0.25, -0.10, 0.05, 0.30, -0.20, 0.15, 0.40, -0.05],
            [-0.30, 0.20, 0.10, -0.15, 0.35, -0.25, 0.05, 0.45],
            [0.12, 0.18, -0.22, 0.08, 0.16, 0.28, -0.32, 0.06],
        ],
        dtype=np.float32,
    )
    bias = np.array([0.11, -0.07, 0.19, 0.03, -0.13, 0.17, 0.23, -0.09], dtype=np.float32)

    graph = helper.make_graph(
        [
            helper.make_node("ReduceMean", ["image"], ["channel_mean"], axes=[2, 3], keepdims=0),
            helper.make_node("Gemm", ["channel_mean", "weights", "bias"], ["embedding"]),
        ],
        "synthetic_image_encoder",
        [input_tensor],
        [output_tensor],
        initializer=[numpy_helper.from_array(weights, "weights"), numpy_helper.from_array(bias, "bias")],
    )
    model = helper.make_model(
        graph,
        producer_name="sparrow-engine synthetic encoder fixture",
        opset_imports=[helper.make_opsetid("", 11)],
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    onnx.save(model, model_path)

    size = model_path.stat().st_size
    if size >= 1_000_000:
        raise RuntimeError(f"fixture too large: {size} bytes")
    sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()

    manifest_path.write_text(
        f'''[model]\nid = "{MODEL_ID}"\nformat = "onnx"\nfile = "model.onnx"\nversion = "test-fixture-1"\ndescription = "Tiny deterministic image encoder test fixture"\nonnx_sha256 = "{sha256}"\n\n[preprocessing]\nmethod = "resize"\ninput_size = [{INPUT_SIZE}, {INPUT_SIZE}]\nlayout = "nchw"\nnormalization = "unit"\nchannel_order = "rgb"\n\n[inference]\nstrategy = "single"\n\n[postprocessing]\nmethod = "embedding"\nnormalize = true\n\n[embedding]\nversion = "{EMBEDDING_VERSION}"\ndim = {EMBEDDING_DIM}\nmetric = "cosine"\n'''
    )
    print(f"wrote {model_path} ({size} bytes)")
    print(f"wrote {manifest_path} (onnx_sha256={sha256})")


if __name__ == "__main__":
    main()
