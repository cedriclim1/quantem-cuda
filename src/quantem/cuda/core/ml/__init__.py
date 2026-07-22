"""CUDA kernels accelerating quantem.core.ml (K-Planes / tensor-decomposition models)."""

from quantem.cuda.core.ml._ops import (
    kplanes_tilted_fuse as kplanes_tilted_fuse,
    kplanes_tilted_fuse_ms as kplanes_tilted_fuse_ms,
    kplanes_tilted_tv_fuse as kplanes_tilted_tv_fuse,
)

# Consumers use this identity to respect instrumentation or overrides of the
# single-level entry point by falling back to that overridden implementation.
_kplanes_tilted_fuse_builtin = kplanes_tilted_fuse
