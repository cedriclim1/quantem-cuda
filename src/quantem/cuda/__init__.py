"""CUDA-accelerated kernels for the quantem electron-microscopy toolkit.

This package installs into the ``quantem`` namespace alongside the core
``quantem`` distribution (mirroring ``quantem.widget``) but is fully usable
standalone — it depends only on torch and numpy at runtime.

All public functions take and return ``torch.Tensor`` and are registered as
torch custom ops, so they compose with autograd and ``torch.compile``.
"""

from importlib.metadata import version

from quantem.cuda._ops import (
    cudart_version as cudart_version,
    kplanes_tilted_fuse as kplanes_tilted_fuse,
    tv_loss_iso_3d as tv_loss_iso_3d,
    tv_loss_sq_3d as tv_loss_sq_3d,
)

__version__ = version("quantem-cuda")
