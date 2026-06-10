"""torch-native registration of the shared (cross-module) kernels.

Each kernel pair is exposed as a ``torch.library.custom_op`` (forward) plus
a companion backward op, wired together with ``register_autograd`` and given
fake-tensor implementations so the ops compose with autograd and
``torch.compile`` without graph breaks.

The compiled module (``quantem.cuda._core``) is torch-free: it receives raw
device pointers, shapes, and the current CUDA stream. Everything
torch-facing — validation, contiguity, device guards, normalization —
happens here.

Conventions shared by all ops:

* registered ops take a contiguous-izable fp32 CUDA tensor shaped
  ``[B, D, H, W]``; the public wrappers below flatten any leading
  batch/channel dims into ``B``;
* forward ops return a 0-dim fp32 scalar on the input's device;
* backward ops fully overwrite their output, so gradients are exact in a
  single kernel launch (no accumulation passes).
"""

from __future__ import annotations

import torch
from torch import Tensor

from quantem.cuda import _core
from quantem.cuda._common import _as_batched, _launch_args

# ── isotropic 3-D TV ──────────────────────────────────────────────────────


def _iso_n_corners(shape: tuple[int, ...]) -> int:
    b, d, h, w = shape
    return max(1, b * (d - 1) * (h - 1) * (w - 1))


@torch.library.custom_op("quantem_cuda::tv_loss_iso_3d", mutates_args=())
def _tv_loss_iso_3d(volume: Tensor, eps: float) -> Tensor:
    vol, b, d, h, w, stream = _launch_args(volume)
    acc = torch.zeros(1, dtype=torch.float32, device=vol.device)
    with torch.cuda.device(vol.device):
        _core.tv_loss_iso_3d_cuda(vol.data_ptr(), acc.data_ptr(), b, d, h, w, float(eps), stream)
    return acc.squeeze(0) / float(_iso_n_corners(vol.shape))


@_tv_loss_iso_3d.register_fake
def _(volume: Tensor, eps: float) -> Tensor:
    return volume.new_empty(())


@torch.library.custom_op("quantem_cuda::tv_loss_iso_3d_bwd", mutates_args=())
def _tv_loss_iso_3d_bwd(volume: Tensor, grad_out: Tensor, eps: float) -> Tensor:
    vol, b, d, h, w, stream = _launch_args(volume)
    g_scaled = (
        grad_out.detach().to(dtype=torch.float32, device=vol.device).reshape(1)
        / float(_iso_n_corners(vol.shape))
    ).contiguous()
    grad_vol = torch.zeros_like(vol)
    with torch.cuda.device(vol.device):
        _core.tv_loss_iso_3d_grad_cuda(
            vol.data_ptr(),
            g_scaled.data_ptr(),
            grad_vol.data_ptr(),
            b,
            d,
            h,
            w,
            float(eps),
            stream,
        )
    return grad_vol


@_tv_loss_iso_3d_bwd.register_fake
def _(volume: Tensor, grad_out: Tensor, eps: float) -> Tensor:
    return torch.empty_like(volume)


def _iso_setup_context(ctx, inputs, output) -> None:
    volume, eps = inputs
    ctx.save_for_backward(volume)
    ctx.eps = eps


def _iso_backward(ctx, grad_out):
    (volume,) = ctx.saved_tensors
    return _tv_loss_iso_3d_bwd(volume, grad_out, ctx.eps), None


_tv_loss_iso_3d.register_autograd(_iso_backward, setup_context=_iso_setup_context)


# ── squared-anisotropic 3-D TV (quantem tv_vol parity) ────────────────────


@torch.library.custom_op("quantem_cuda::tv_loss_sq_3d", mutates_args=())
def _tv_loss_sq_3d(volume: Tensor) -> Tensor:
    vol, b, d, h, w, stream = _launch_args(volume)
    acc = torch.zeros(1, dtype=torch.float32, device=vol.device)
    with torch.cuda.device(vol.device):
        _core.tv_loss_sq_3d_cuda(vol.data_ptr(), acc.data_ptr(), b, d, h, w, stream)
    return acc.squeeze(0)


@_tv_loss_sq_3d.register_fake
def _(volume: Tensor) -> Tensor:
    return volume.new_empty(())


@torch.library.custom_op("quantem_cuda::tv_loss_sq_3d_bwd", mutates_args=())
def _tv_loss_sq_3d_bwd(volume: Tensor, grad_out: Tensor) -> Tensor:
    vol, b, d, h, w, stream = _launch_args(volume)
    g = grad_out.detach().to(dtype=torch.float32, device=vol.device).reshape(1).contiguous()
    grad_vol = torch.empty_like(vol)
    with torch.cuda.device(vol.device):
        _core.tv_loss_sq_3d_grad_cuda(
            vol.data_ptr(), g.data_ptr(), grad_vol.data_ptr(), b, d, h, w, stream
        )
    return grad_vol


@_tv_loss_sq_3d_bwd.register_fake
def _(volume: Tensor, grad_out: Tensor) -> Tensor:
    return torch.empty_like(volume)


def _sq_setup_context(ctx, inputs, output) -> None:
    (volume,) = inputs
    ctx.save_for_backward(volume)


def _sq_backward(ctx, grad_out):
    (volume,) = ctx.saved_tensors
    return _tv_loss_sq_3d_bwd(volume, grad_out)


_tv_loss_sq_3d.register_autograd(_sq_backward, setup_context=_sq_setup_context)


# ── public wrappers ───────────────────────────────────────────────────────


def tv_loss_iso_3d(volume: Tensor, eps: float = 1e-8) -> Tensor:
    """Isotropic 3-D total-variation loss (fused CUDA forward + backward).

        loss = mean over corners of sqrt( dd² + dh² + dw² + eps )

    where dd/dh/dw are forward differences along the three trailing dims
    and the corner set is ``[0, D−2] × [0, H−2] × [0, W−2]`` per leading
    slice. Leading dims (channels and/or batch) are flattened and included
    in the mean.

    Args:
        volume: fp32 CUDA tensor, shape ``[D, H, W]`` or ``[..., D, H, W]``.
        eps:    smoothing constant inside the sqrt (default 1e-8).

    Returns:
        0-dim fp32 tensor on the same device as ``volume``; differentiable.
    """
    return _tv_loss_iso_3d(_as_batched(volume, "tv_loss_iso_3d"), eps)


def tv_loss_sq_3d(volume: Tensor) -> Tensor:
    """Squared-anisotropic 3-D total-variation sum (fused CUDA kernels).

        loss = Σ (Δd)² + Σ (Δh)² + Σ (Δw)²

    over forward differences along the three trailing dims, each summed
    over its full complementary index range — exactly quantem's ``tv_vol``
    regularizer before its ``weight / numel`` scaling. Leading dims
    (channels and/or batch) are flattened and included in the sum.

    Args:
        volume: fp32 CUDA tensor, shape ``[D, H, W]`` or ``[..., D, H, W]``.

    Returns:
        0-dim fp32 tensor on the same device as ``volume``; differentiable.
    """
    return _tv_loss_sq_3d(_as_batched(volume, "tv_loss_sq_3d"))
