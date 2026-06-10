"""CUDA-accelerated kernels for the quantem electron-microscopy toolkit.

This package installs into the ``quantem`` namespace alongside the core
``quantem`` distribution (mirroring ``quantem.widget``) but is fully usable
standalone — it depends only on torch and numpy at runtime.

Kernels are grouped into submodules mirroring the quantem module they
accelerate (``quantem.cuda.core``, ``quantem.cuda.tomography``, ...); all
public functions take and return ``torch.Tensor`` and are registered as
torch custom ops, so they compose with autograd and ``torch.compile``.
"""

from importlib.metadata import version

from quantem.cuda import core as core
from quantem.cuda._common import cudart_version as cudart_version

__version__ = version("quantem-cuda")
