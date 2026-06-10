"""Shared CUDA kernels usable across quantem modules (regularizers, volume ops)."""

from quantem.cuda.core._ops import (
    tv_loss_iso_3d as tv_loss_iso_3d,
    tv_loss_sq_3d as tv_loss_sq_3d,
)
