"""Helpers shared by the per-module torch registration layers (``*/_ops.py``)."""

from __future__ import annotations

import torch
from torch import Tensor

from quantem.cuda import _core


def _launch_args(volume: Tensor) -> tuple[Tensor, int, int, int, int, int]:
    vol = volume.contiguous()
    b, d, h, w = vol.shape
    stream = torch.cuda.current_stream(vol.device).cuda_stream
    return vol, b, d, h, w, stream


def _as_batched(volume: Tensor, fn: str) -> Tensor:
    if volume.ndim < 3:
        raise ValueError(
            f"{fn} expects [..., D, H, W] with ndim >= 3, got shape {tuple(volume.shape)}"
        )
    if volume.dtype != torch.float32:
        raise TypeError(f"{fn} is fp32-only (got {volume.dtype}).")
    if not volume.is_cuda:
        raise ValueError(f"{fn} requires a CUDA tensor (got device {volume.device}).")
    d, h, w = volume.shape[-3:]
    return volume.reshape(-1, d, h, w)


def cudart_version() -> int:
    """CUDA runtime version the extension was compiled against (e.g. 13000)."""
    return int(_core.__cudart_version__)
