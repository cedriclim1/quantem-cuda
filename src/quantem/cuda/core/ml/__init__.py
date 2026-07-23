"""CUDA kernels accelerating quantem.core.ml (K-Planes / tensor-decomposition models)."""

from quantem.cuda.core.ml._ops import (
    density_tail as density_tail,
    kplanes_tilted_fuse as kplanes_tilted_fuse,
    kplanes_tilted_fuse_ms as kplanes_tilted_fuse_ms,
    kplanes_tilted_fuse_ms_tv as kplanes_tilted_fuse_ms_tv,
    kplanes_tilted_tv_fuse as kplanes_tilted_tv_fuse,
    plane_tv_loss as plane_tv_loss,
)

# Consumers use this identity to respect instrumentation or overrides of the
# single-level entry point by falling back to that overridden implementation.
_kplanes_tilted_fuse_builtin = kplanes_tilted_fuse
