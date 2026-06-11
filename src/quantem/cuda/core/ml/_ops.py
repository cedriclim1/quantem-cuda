"""torch-native registration of the core.ml kernels (K-Planes models).

Same conventions as the package's other ``_ops.py`` layers: each kernel
pair is a ``torch.library.custom_op`` (forward) plus a companion backward
op, wired together with ``register_autograd`` and given fake-tensor
implementations so the ops compose with autograd and ``torch.compile``
without graph breaks. The compiled module (``quantem.cuda._core``) is
torch-free; everything torch-facing happens here.
"""

from __future__ import annotations

import torch
from torch import Tensor

from quantem.cuda import _core

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


# ── TV-specialized fused TILTED K-Planes interpolation ───────────────────
#
# kplanes_tilted_tv_fuse evaluates kplanes_tilted_fuse at 4 tap locations:
#   tap 0: R·x              (identical in layout to kplanes_tilted_fuse output)
#   tap 1: R·x + h·R[:,0]  (world x finite-difference)
#   tap 2: R·x + h·R[:,1]  (world y finite-difference)
#   tap 3: R·x + h·R[:,2]  (world z finite-difference)
# Output: [4, B, T*C] — tap dimension outermost.
# h is a fixed hyperparameter; no gradient is tracked for it.


@torch.library.custom_op("quantem_cuda::kplanes_tilted_tv_fuse", mutates_args=())
def _kplanes_tilted_tv_fuse(pts: Tensor, rotations: Tensor, plane: Tensor, h: float) -> Tensor:
    p = pts.contiguous()
    r = rotations.contiguous()
    c, hw, w = plane.shape[1], plane.shape[2], plane.shape[3]
    h_dim = hw  # H dimension
    g = _channels_last(plane)
    b = p.shape[0]
    t = r.shape[0]
    out = torch.empty((4, b, t * c), dtype=torch.float32, device=p.device)
    stream = torch.cuda.current_stream(p.device).cuda_stream
    with torch.cuda.device(p.device):
        _core.kplanes_tilted_tv_fuse_cuda(
            p.data_ptr(),
            r.data_ptr(),
            g.data_ptr(),
            out.data_ptr(),
            b,
            t,
            c,
            h_dim,
            w,
            float(h),
            stream,
        )
    return out


@_kplanes_tilted_tv_fuse.register_fake
def _(pts: Tensor, rotations: Tensor, plane: Tensor, h: float) -> Tensor:
    return pts.new_empty((4, pts.shape[0], rotations.shape[0] * plane.shape[1]))


@torch.library.custom_op("quantem_cuda::kplanes_tilted_tv_fuse_bwd", mutates_args=())
def _kplanes_tilted_tv_fuse_bwd(
    pts: Tensor, rotations: Tensor, plane: Tensor, grad_out: Tensor, h: float
) -> list[Tensor]:
    p = pts.contiguous()
    r = rotations.contiguous()
    c, h_dim, w = plane.shape[1], plane.shape[2], plane.shape[3]
    g = _channels_last(plane)
    b = p.shape[0]
    t = r.shape[0]
    gout = grad_out.detach().to(dtype=torch.float32, device=p.device).contiguous()
    grad_pts = torch.zeros_like(p)
    grad_r = torch.zeros_like(r)
    grad_plane_cl = torch.zeros_like(g)
    stream = torch.cuda.current_stream(p.device).cuda_stream
    with torch.cuda.device(p.device):
        _core.kplanes_tilted_tv_fuse_grad_cuda(
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
            h_dim,
            w,
            float(h),
            stream,
        )
    # (3T, H, W, C) → (3T, C, H, W), matching the parameter layout
    return [grad_pts, grad_r, grad_plane_cl.permute(0, 3, 1, 2).contiguous()]


@_kplanes_tilted_tv_fuse_bwd.register_fake
def _(pts: Tensor, rotations: Tensor, plane: Tensor, grad_out: Tensor, h: float) -> list[Tensor]:
    return [torch.empty_like(pts), torch.empty_like(rotations), torch.empty_like(plane)]


def _kpt_tv_setup_context(ctx, inputs, output) -> None:
    pts, rotations, plane, h = inputs
    ctx.save_for_backward(pts, rotations, plane)
    ctx.h = h


def _kpt_tv_backward(ctx, grad_out):
    pts, rotations, plane = ctx.saved_tensors
    grad_pts, grad_r, grad_plane = _kplanes_tilted_tv_fuse_bwd(
        pts, rotations, plane, grad_out, ctx.h
    )
    # h has no gradient
    return grad_pts, grad_r, grad_plane, None


_kplanes_tilted_tv_fuse.register_autograd(_kpt_tv_backward, setup_context=_kpt_tv_setup_context)


# ── fused NON-TILTED K-Planes interpolation ───────────────────────────────


@torch.library.custom_op("quantem_cuda::kplanes_fuse", mutates_args=())
def _kplanes_fuse(pts: Tensor, plane: Tensor) -> Tensor:
    p = pts.contiguous()
    c, h, w = plane.shape[1:]
    g = _channels_last(plane)
    b = p.shape[0]
    out = torch.empty((b, c), dtype=torch.float32, device=p.device)
    stream = torch.cuda.current_stream(p.device).cuda_stream
    with torch.cuda.device(p.device):
        _core.kplanes_fuse_cuda(p.data_ptr(), g.data_ptr(), out.data_ptr(), b, c, h, w, stream)
    return out


@_kplanes_fuse.register_fake
def _(pts: Tensor, plane: Tensor) -> Tensor:
    return pts.new_empty((pts.shape[0], plane.shape[1]))


@torch.library.custom_op("quantem_cuda::kplanes_fuse_bwd", mutates_args=())
def _kplanes_fuse_bwd(pts: Tensor, plane: Tensor, grad_out: Tensor) -> list[Tensor]:
    p = pts.contiguous()
    c, h, w = plane.shape[1:]
    g = _channels_last(plane)
    b = p.shape[0]
    gout = grad_out.detach().to(dtype=torch.float32, device=p.device).contiguous()
    grad_pts = torch.zeros_like(p)  # accumulated across a point's channel chunks
    grad_plane_cl = torch.zeros_like(g)
    stream = torch.cuda.current_stream(p.device).cuda_stream
    with torch.cuda.device(p.device):
        _core.kplanes_fuse_grad_cuda(
            p.data_ptr(),
            g.data_ptr(),
            gout.data_ptr(),
            grad_plane_cl.data_ptr(),
            grad_pts.data_ptr(),
            b,
            c,
            h,
            w,
            stream,
        )
    # (3, H, W, C) → (3, C, H, W), matching the parameter layout
    return [grad_pts, grad_plane_cl.permute(0, 3, 1, 2).contiguous()]


@_kplanes_fuse_bwd.register_fake
def _(pts: Tensor, plane: Tensor, grad_out: Tensor) -> list[Tensor]:
    return [torch.empty_like(pts), torch.empty_like(plane)]


def _kp_setup_context(ctx, inputs, output) -> None:
    pts, plane = inputs
    ctx.save_for_backward(pts, plane)


def _kp_backward(ctx, grad_out):
    pts, plane = ctx.saved_tensors
    grad_pts, grad_plane = _kplanes_fuse_bwd(pts, plane, grad_out)
    return grad_pts, grad_plane


_kplanes_fuse.register_autograd(_kp_backward, setup_context=_kp_setup_context)


# ── public wrappers ───────────────────────────────────────────────────────


def kplanes_fuse(pts: Tensor, plane: Tensor) -> Tensor:
    """Fused NON-TILTED K-Planes feature interpolation for one multiscale level.

    For each point ``p``, bilinearly samples the three planes at quantem's
    ``interpolate_ms_features`` coordinate pairs and Hadamard-multiplies
    them — equivalent to (but much faster than)::

        coords = torch.stack([pts[:, [0, 1]], pts[:, [0, 2]], pts[:, [1, 2]]]).view(3, -1, 1, 2)
        feats = F.grid_sample(plane, coords, align_corners=True,
                              mode="bilinear", padding_mode="border").reshape(3, C, -1)
        out = (feats[0] * feats[1] * feats[2]).T

    Differentiable w.r.t. both inputs (analytic CUDA backward, with torch's
    border-clip zero-gradient convention for coordinates).

    Args:
        pts:   fp32 CUDA tensor ``[B, 3]``, coordinates in ``[-1, 1]``.
        plane: fp32 CUDA tensor ``[3, C, H, W]``.

    Returns:
        fp32 tensor ``[B, C]``, differentiable.
    """
    if pts.ndim != 2 or pts.shape[-1] != 3:
        raise ValueError(f"kplanes_fuse expects pts [B, 3], got {tuple(pts.shape)}")
    if plane.ndim != 4 or plane.shape[0] != 3:
        raise ValueError(f"kplanes_fuse expects plane [3, C, H, W], got {tuple(plane.shape)}")
    for name, t in (("pts", pts), ("plane", plane)):
        if t.dtype != torch.float32:
            raise TypeError(f"kplanes_fuse is fp32-only ({name} is {t.dtype}).")
        if not t.is_cuda:
            raise ValueError(f"kplanes_fuse requires CUDA tensors ({name} on {t.device}).")
    if plane.shape[1] * plane.shape[2] * plane.shape[3] >= 2**31:
        raise ValueError(
            "kplanes_fuse uses 32-bit per-plane offsets; "
            f"C*H*W must be < 2^31, got plane {tuple(plane.shape)}"
        )
    return _kplanes_fuse(pts, plane)


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


def kplanes_tilted_tv_fuse(pts: Tensor, rotations: Tensor, plane: Tensor, h: float) -> Tensor:
    """TV-specialized fused TILTED K-Planes feature interpolation for one multiscale level.

    Evaluates ``kplanes_tilted_fuse`` at 4 tap locations per point to support
    total-variation regularization via finite differences. For each point ``p``
    and rotation ``R_t``:

        tap 0: feature(R_t · x)              — base, identical layout to
                                               kplanes_tilted_fuse output
        tap 1: feature(R_t · x + h·R_t[:,0]) — world-x finite difference
        tap 2: feature(R_t · x + h·R_t[:,1]) — world-y finite difference
        tap 3: feature(R_t · x + h·R_t[:,2]) — world-z finite difference

    Key identities used by the kernel:
        R·(x + h·e_i) = R·x + h·R[:,i]
    so the base rotation is computed once per (point, rotation), and the three
    column offsets ``h·R[:,i]`` are per-rotation constants shared across all
    points. This saves 3 matrix-vector multiplications per point.

    Output layout: ``[4, B, T*C]`` — tap dimension outermost. ``out[0]`` is
    bit-for-bit compatible in layout with ``kplanes_tilted_fuse(pts, ...)``,
    enabling ``out.unbind(0)`` to yield four ``[B, T*C]`` tensors.

    ``h`` is treated as a fixed hyperparameter: no gradient is returned for it.
    Differentiable w.r.t. ``pts``, ``rotations``, and ``plane`` (analytic CUDA
    backward, with torch's border-clip zero-gradient convention).

    Args:
        pts:       fp32 CUDA tensor ``[B, 3]``, coordinates in ``[-1, 1]``.
        rotations: fp32 CUDA tensor ``[T, 3, 3]``.
        plane:     fp32 CUDA tensor ``[3*T, C, H, W]`` (plane ``t*3 + p``).
        h:         finite-difference step in world ``[-1, 1]`` units.

    Returns:
        fp32 tensor ``[4, B, T*C]``, differentiable w.r.t. pts/rotations/plane.
    """
    if pts.ndim != 2 or pts.shape[-1] != 3:
        raise ValueError(f"kplanes_tilted_tv_fuse expects pts [B, 3], got {tuple(pts.shape)}")
    if rotations.ndim != 3 or rotations.shape[-2:] != (3, 3):
        raise ValueError(
            f"kplanes_tilted_tv_fuse expects rotations [T, 3, 3], got {tuple(rotations.shape)}"
        )
    if plane.ndim != 4 or plane.shape[0] != 3 * rotations.shape[0]:
        raise ValueError(
            "kplanes_tilted_tv_fuse expects plane [3*T, C, H, W] with T = "
            f"rotations.shape[0]; got plane {tuple(plane.shape)} for "
            f"T={rotations.shape[0]}"
        )
    for name, t in (("pts", pts), ("rotations", rotations), ("plane", plane)):
        if t.dtype != torch.float32:
            raise TypeError(f"kplanes_tilted_tv_fuse is fp32-only ({name} is {t.dtype}).")
        if not t.is_cuda:
            raise ValueError(
                f"kplanes_tilted_tv_fuse requires CUDA tensors ({name} on {t.device})."
            )
    if plane.shape[1] * plane.shape[2] * plane.shape[3] >= 2**31:
        raise ValueError(
            "kplanes_tilted_tv_fuse uses 32-bit per-plane offsets; "
            f"C*H*W must be < 2^31, got plane {tuple(plane.shape)}"
        )
    return _kplanes_tilted_tv_fuse(pts, rotations, plane, float(h))
