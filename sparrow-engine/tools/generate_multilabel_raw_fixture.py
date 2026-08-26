#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#   "onnx==1.17.0",
#   "torch==2.7.1",
# ]
# ///
"""Generate the tiny raw-audio multi-label ONNX fixture."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class TinyRawMultiLabel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "base_probabilities",
            torch.tensor(
                [
                    [0.80, 0.40, 0.90],
                    [0.70, 0.30, 0.85],
                    [0.20, 0.95, 0.40],
                    [0.10, 0.10, 0.10],
                ],
                dtype=torch.float32,
            ),
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        frame_signal = audio.reshape(audio.shape[0], 4, 1200).mean(dim=2)
        return (self.base_probabilities.unsqueeze(0) + frame_signal.unsqueeze(2) * 0.01).clamp(
            0.0, 1.0
        )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output_dir = (
        root
        / "sparrow-engine-core"
        / "tests"
        / "fixtures"
        / "audio"
        / "multilabel_raw_tiny"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model.onnx"

    model = TinyRawMultiLabel().eval()
    example = torch.zeros(2, 4800, dtype=torch.float32)
    torch.onnx.export(
        model,
        example,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
        input_names=["audio"],
        output_names=["probabilities"],
        dynamic_axes={
            "audio": {0: "batch"},
            "probabilities": {0: "batch"},
        },
    )

    import onnx

    graph = onnx.load(output_path)
    onnx.checker.check_model(graph)
    size = output_path.stat().st_size
    if size >= 1_000_000:
        raise RuntimeError(f"fixture too large: {size} bytes")
    print(f"wrote {output_path} ({size} bytes)")


if __name__ == "__main__":
    main()
