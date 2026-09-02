#!/usr/bin/env python3
"""Generate a small librosa/SoXR reference for Rust resampler regression tests."""

from __future__ import annotations

from pathlib import Path
import wave

import librosa
import numpy as np


ROOT = (
    Path(__file__).resolve().parents[1]
    / "sparrow-engine-core/tests/fixtures/audio/resample_soxr_tiny"
)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    source_rate = 44_100
    target_rate = 28_000
    samples = np.arange(source_rate // 2, dtype=np.float32)
    time = samples / source_rate
    signal = (
        0.35 * np.sin(2 * np.pi * 440.0 * time)
        + 0.2 * np.sin(2 * np.pi * 4_300.0 * time)
        + 0.1 * np.sin(2 * np.pi * (500.0 + 4_000.0 * time) * time)
    )
    signal[2_000] += 0.4
    pcm = np.clip(signal * 32767.0, -32768, 32767).astype("<i2")
    source = ROOT / "source_44100.wav"
    with wave.open(str(source), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(source_rate)
        handle.writeframes(pcm.tobytes())

    decoded = pcm.astype(np.float32) / 32768.0
    expected = librosa.resample(
        decoded,
        orig_sr=source_rate,
        target_sr=target_rate,
        res_type="soxr_hq",
        fix=True,
        scale=False,
    ).astype("<f4")
    expected.tofile(ROOT / "expected_28000.f32")


if __name__ == "__main__":
    main()
