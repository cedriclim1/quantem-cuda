"""torch-native registration of the compiled kernels.

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


def _launch_args(volume: Tensor) -> tuple[Tensor, int, int, int, int, int]:
    vol = volume.contiguous()
    b, d, h, w = vol.shape
    stream = torch.cuda.current_stream(vol.device).cuda_stream
    return vol, b, d, h, w, stream


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


# ── fused TILTED K-Planes interpolation ───────────────────────────────────


def _channels_last(plane: Tensor) -> Tensor:
    """(3T, C, H, W) → contiguous (3T, H, W, C) for the kernels. The grid is
    tiny next to the per-point traffic, so this pass is noise — and it buys
    contiguous per-cell channel reads inside the kernels."""
    return plane.permute(0, 2, 3, 1).contiguous()


@torch.library.custom_op("quantem_cuda::kplanes_tilted_fuse", mutates_args=())
def _kplanes_tilted_fuse(pts: Tensor, rotations: Tensor, plane: Tensor) -> Tensor:
    p = pts.contiguous()
    r = rotations.contiguous()
    c, h, w = plane.shape[1:]
    g = _channels_last(plane)
    b = p.shape[0]
    t = r.shape[0]
    out = torch.empty((b, t * c), dtype=torch.float32, device=p.device)
    stream = torch.cuda.current_stream(p.device).cuda_stream
    with torch.cuda.device(p.device):
        _core.kplanes_tilted_fuse_cuda(
            p.data_ptr(), r.data_ptr(), g.data_ptr(), out.data_ptr(), b, t, c, h, w, stream
        )
    return out


@_kplanes_tilted_fuse.register_fake
def _(pts: Tensor, rotations: Tensor, plane: Tensor) -> Tensor:
    return pts.new_empty((pts.shape[0], rotations.shape[0] * plane.shape[1]))


@torch.library.custom_op("quantem_cuda::kplanes_tilted_fuse_bwd", mutates_args=())
def _kplanes_tilted_fuse_bwd(
    pts: Tensor, rotations: Tensor, plane: Tensor, grad_out: Tensor
) -> list[Tensor]:
    p = pts.contiguous()
    r = rotations.contiguous()
    c, h, w = plane.shape[1:]
    g = _channels_last(plane)
    b = p.shape[0]
    t = r.shape[0]
    gout = grad_out.detach().to(dtype=torch.float32, device=p.device).contiguous()
    grad_pts = torch.zeros_like(p)  # accumulated across the T threads per point
    grad_r = torch.zeros_like(r)
    grad_plane_cl = torch.zeros_like(g)
    stream = torch.cuda.current_stream(p.device).cuda_stream
    with torch.cuda.device(p.device):
        _core.kplanes_tilted_fuse_grad_cuda(
            p.data_ptr(),
            r.data_ptr(),
            g.data_ptr(),
            gout.data_ptr(),
            grad_plane_cl.data_ptr(),
            grad_r.data_ptr(),
            grad_pts.data_ptr(),
            b,
            t,
            c,
            h,
            w,
            stream,
        )
    # (3T, H, W, C) → (3T, C, H, W), matching the parameter layout
    return [grad_pts, grad_r, grad_plane_cl.permute(0, 3, 1, 2).contiguous()]


@_kplanes_tilted_fuse_bwd.register_fake
def _(pts: Tensor, rotations: Tensor, plane: Tensor, grad_out: Tensor) -> list[Tensor]:
    return [torch.empty_like(pts), torch.empty_like(rotations), torch.empty_like(plane)]


def _kpt_setup_context(ctx, inputs, output) -> None:
    pts, rotations, plane = inputs
    ctx.save_for_backward(pts, rotations, plane)


def _kpt_backward(ctx, grad_out):
    pts, rotations, plane = ctx.saved_tensors
    grad_pts, grad_r, grad_plane = _kplanes_tilted_fuse_bwd(pts, rotations, plane, grad_out)
    return grad_pts, grad_r, grad_plane


_kplanes_tilted_fuse.register_autograd(_kpt_backward, setup_context=_kpt_setup_context)


# ── public wrappers ───────────────────────────────────────────────────────


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


def kplanes_tilted_fuse(pts: Tensor, rotations: Tensor, plane: Tensor) -> Tensor:
    """Fused TILTED K-Planes feature interpolation for one multiscale level.

    For each point ``p`` and each rotation ``R_t``, bilinearly samples the
    three planes of rotation ``t`` at the (XY, ZX, YZ) pairs of ``R_t @ p``
    and Hadamard-multiplies them — equivalent to (but much faster than)::

        rotated = torch.einsum("tij,bj->tbi", rotations, pts)
        coords = rotated[..., [[0, 1], [2, 0], [1, 2]]]      # (T, B, 3, 2)
        sampled = F.grid_sample(plane, coords.permute(0, 2, 1, 3).reshape(3*T, B, 1, 2),
                                align_corners=True, mode="bilinear",
                                padding_mode="border")
        out = sampled.squeeze(-1).view(T, 3, C, B).prod(1).permute(2, 0, 1).reshape(B, T*C)

    Differentiable w.r.t. all three inputs (analytic CUDA backward, with
    torch's border-clip zero-gradient convention for coordinates).

    Args:
        pts:       fp32 CUDA tensor ``[B, 3]``, coordinates in ``[-1, 1]``.
        rotations: fp32 CUDA tensor ``[T, 3, 3]``.
        plane:     fp32 CUDA tensor ``[3*T, C, H, W]`` (plane ``t*3 + p``).

    Returns:
        fp32 tensor ``[B, T*C]`` (``out[b, t*C + c]``), differentiable.
    """
    if pts.ndim != 2 or pts.shape[-1] != 3:
        raise ValueError(f"kplanes_tilted_fuse expects pts [B, 3], got {tuple(pts.shape)}")
    if rotations.ndim != 3 or rotations.shape[-2:] != (3, 3):
        raise ValueError(
            f"kplanes_tilted_fuse expects rotations [T, 3, 3], got {tuple(rotations.shape)}"
        )
    if plane.ndim != 4 or plane.shape[0] != 3 * rotations.shape[0]:
        raise ValueError(
            "kplanes_tilted_fuse expects plane [3*T, C, H, W] with T = "
            f"rotations.shape[0]; got plane {tuple(plane.shape)} for T={rotations.shape[0]}"
        )
    for name, t in (("pts", pts), ("rotations", rotations), ("plane", plane)):
        if t.dtype != torch.float32:
            raise TypeError(f"kplanes_tilted_fuse is fp32-only ({name} is {t.dtype}).")
        if not t.is_cuda:
            raise ValueError(f"kplanes_tilted_fuse requires CUDA tensors ({name} on {t.device}).")
    if plane.shape[1] * plane.shape[2] * plane.shape[3] >= 2**31:
        raise ValueError(
            "kplanes_tilted_fuse uses 32-bit per-plane offsets; "
            f"C*H*W must be < 2^31, got plane {tuple(plane.shape)}"
        )
    return _kplanes_tilted_fuse(pts, rotations, plane)


def cudart_version() -> int:
    """CUDA runtime version the extension was compiled against (e.g. 13000)."""
    return int(_core.__cudart_version__)
