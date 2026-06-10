"""Shared CUDA kernels usable across quantem modules (regularizers, volume ops).

``quantem.cuda.core.ml`` holds the kernels mirroring ``quantem.core.ml``
(K-Planes / tensor-decomposition models).
"""

from quantem.cuda.core import ml as ml
from quantem.cuda.core._ops import (
    tv_loss_iso_3d as tv_loss_iso_3d,
    tv_loss_sq_3d as tv_loss_sq_3d,
)
