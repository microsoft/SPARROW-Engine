"""Tests for pw_tools.convert._load_model pickle safety (CWE-502)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pw_tools.convert import _load_model  # noqa: E402


class _Payload:
    def __init__(self, sentinel: str) -> None:
        self.sentinel = sentinel

    def __reduce__(self):
        return (os.mkdir, (self.sentinel,))


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(2, 2)

    def forward(self, x):
        return self.fc(x)


def test_default_refuses_code_carrying_pickle(tmp_path: Path) -> None:
    sentinel = tmp_path / "pwned"
    src = tmp_path / "evil.pt"
    torch.save({"model": _Payload(str(sentinel))}, src)

    with pytest.raises(RuntimeError, match="--trust-pickle"):
        _load_model(src)
    assert not sentinel.exists()


def test_default_refuses_pickled_module(tmp_path: Path) -> None:
    src = tmp_path / "full.pt"
    torch.save(_Tiny(), src)

    with pytest.raises(RuntimeError, match="--trust-pickle"):
        _load_model(src)


def test_trust_pickle_loads_pickled_module(tmp_path: Path) -> None:
    src = tmp_path / "full.pt"
    torch.save(_Tiny(), src)

    model, is_ts = _load_model(src, trust_pickle=True)
    assert isinstance(model, _Tiny)
    assert not is_ts
    assert not model.training


def test_torchscript_loads_without_trust_pickle(tmp_path: Path) -> None:
    src = tmp_path / "scripted.pt"
    torch.jit.script(_Tiny()).save(str(src))

    model, is_ts = _load_model(src)
    assert is_ts
