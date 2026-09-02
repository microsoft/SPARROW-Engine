#!/usr/bin/env python3
"""Generate the tiny recording-level audio ensemble integration fixture."""

from __future__ import annotations

import hashlib
from pathlib import Path
import wave

import numpy as np
import torch
import torch.nn as nn


ROOT = (
    Path(__file__).resolve().parents[1]
    / "sparrow-engine-core/tests/fixtures/audio/frame_ensemble_tiny"
)


class FixedFrameModel(nn.Module):
    def __init__(self, values: list[list[float]]) -> None:
        super().__init__()
        self.register_buffer("values", torch.tensor(values, dtype=torch.float32))

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        batch = spectrogram.shape[0]
        dependency = spectrogram.sum(dim=(1, 2, 3)) * 0.0
        return self.values.unsqueeze(0).expand(batch, -1, -1) + dependency[:, None, None]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_f32(name: str, values: np.ndarray) -> dict:
    path = ROOT / name
    contiguous = np.ascontiguousarray(values, dtype="<f4")
    path.write_bytes(contiguous.tobytes())
    return {"file": name, "sha256": sha256(path)}


def export_model(name: str, values: list[list[float]], rows: int, columns: int) -> dict:
    path = ROOT / name
    model = FixedFrameModel(values).eval()
    example = torch.zeros((1, 1, rows, columns), dtype=torch.float32)
    batch = torch.export.Dim("batch", min=1, max=32)
    torch.onnx.export(
        model,
        (example,),
        str(path),
        input_names=["spectrogram"],
        output_names=["frame_probabilities"],
        opset_version=18,
        dynamo=True,
        external_data=False,
        dynamic_shapes={"spectrogram": {0: batch}},
    )
    return {
        "file": name,
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def write_wav() -> None:
    samples = (
        0.45
        * np.sin(
            2.0
            * np.pi
            * 3.0
            * np.arange(64, dtype=np.float32)
            / 32.0
        )
    )
    pcm = np.clip(samples * 32767.0, -32768, 32767).astype("<i2")
    with wave.open(str(ROOT / "input.wav"), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(32)
        handle.writeframes(pcm.tobytes())


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    main_window = np.hanning(9).astype(np.float32)[:-1]
    main_filterbank = np.zeros((2, 9), dtype=np.float32)
    main_filterbank[0, 1] = 1.0
    main_filterbank[1, 2] = 1.0
    low_window = np.hanning(5).astype(np.float32)[:-1]
    low_filterbank = np.zeros((2, 5), dtype=np.float32)
    low_filterbank[0, 1] = 1.0
    low_filterbank[1, 2] = 1.0

    main_window_meta = write_f32("main-window.f32", main_window)
    main_filter_meta = write_f32("main-filterbank.f32", main_filterbank)
    low_window_meta = write_f32("low-window.f32", low_window)
    low_filter_meta = write_f32("low-filterbank.f32", low_filterbank)

    common = [
        [0.90, 0.10, 0.20],
        [0.92, 0.10, 0.20],
        [0.88, 0.10, 0.20],
        [0.91, 0.10, 0.20],
    ]
    members = [
        export_model("member-1.onnx", common, 2, 8),
        export_model("member-2.onnx", common, 2, 8),
        export_model("member-3.onnx", common, 2, 8),
    ]
    auxiliary = export_model(
        "auxiliary.onnx",
        [
            [0.85, 0.10],
            [0.20, 0.10],
            [0.86, 0.10],
            [0.90, 0.10],
        ],
        2,
        8,
    )

    (ROOT / "labels.txt").write_text("primary\naux-merged\nunused\n")
    (ROOT / "aux-labels.txt").write_text("source\nrejection\n")
    write_wav()

    lines = [
        "[ensemble]",
        'id = "frame-ensemble-tiny"',
        'kind = "audio_frame_ensemble"',
        'version = "1"',
        'description = "Deterministic tiny recording-level audio frame ensemble"',
        "default = false",
        "frame_rate_hz = 4.0",
        "frames_per_window = 4",
        "class_count = 3",
        "confidence_threshold = 0.7",
        "max_classes = 3",
        "inference_batch_size = 2",
        'combine = "mean"',
        'labels_file = "labels.txt"',
        f'labels_sha256 = "{sha256(ROOT / "labels.txt")}"',
        "",
        "[frontend]",
        "sample_rate = 32",
        "n_fft = 16",
        "win_length = 8",
        "hop_length = 4",
        "filter_rows = 2",
        "filter_columns = 9",
        "chunk_samples = 96",
        "chunk_columns = 24",
        "window_duration_s = 1.0",
        "window_columns = 8",
        "min_coverage_columns = 2",
        "audio_power = 0.7",
        'channel_selection = "average"',
        "channel_check_seconds = 1.0",
        'short_window = "stop"',
        f'window_file = "{main_window_meta["file"]}"',
        f'window_sha256 = "{main_window_meta["sha256"]}"',
        f'filterbank_file = "{main_filter_meta["file"]}"',
        f'filterbank_sha256 = "{main_filter_meta["sha256"]}"',
        "",
    ]
    for index, (member, offset, lead) in enumerate(
        zip(members, [0.0, 0.25, 0.5], [False, True, True]), start=1
    ):
        lines.extend(
            [
                "[[member]]",
                f'id = "member-{index}"',
                f'file = "{member["file"]}"',
                f'sha256 = "{member["sha256"]}"',
                f'size_bytes = {member["size_bytes"]}',
                'input_name = "spectrogram"',
                'output_name = "frame_probabilities"',
                f"offset_s = {offset}",
                f"lead_window = {str(lead).lower()}",
                "",
            ]
        )
    lines.extend(
        [
            "[auxiliary]",
            'id = "auxiliary"',
            f'file = "{auxiliary["file"]}"',
            f'sha256 = "{auxiliary["sha256"]}"',
            f'size_bytes = {auxiliary["size_bytes"]}',
            'input_name = "spectrogram"',
            'output_name = "frame_probabilities"',
            'labels_file = "aux-labels.txt"',
            f'labels_sha256 = "{sha256(ROOT / "aux-labels.txt")}"',
            "class_count = 2",
            "frames_per_window = 4",
            "frame_rate_hz = 4.0",
            "offset_s = 0.0",
            "lead_window = false",
            'operation = "max"',
            "",
            "[auxiliary.frontend]",
            "sample_rate = 16",
            "n_fft = 8",
            "win_length = 4",
            "hop_length = 2",
            "filter_rows = 2",
            "filter_columns = 5",
            "chunk_samples = 48",
            "chunk_columns = 24",
            "window_duration_s = 1.0",
            "window_columns = 8",
            "min_coverage_columns = 2",
            "audio_power = 1.2",
            'channel_selection = "average"',
            "channel_check_seconds = 1.0",
            'short_window = "stop"',
            f'window_file = "{low_window_meta["file"]}"',
            f'window_sha256 = "{low_window_meta["sha256"]}"',
            f'filterbank_file = "{low_filter_meta["file"]}"',
            f'filterbank_sha256 = "{low_filter_meta["sha256"]}"',
            "",
            "[[auxiliary.merge]]",
            'from = "source"',
            'to = "aux-merged"',
            "",
        ]
    )
    (ROOT / "ensemble.toml").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
