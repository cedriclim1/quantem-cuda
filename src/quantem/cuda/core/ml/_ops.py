"""torch-native registration of the core.ml kernels (K-Planes models).

Same conventions as the package's other ``_ops.py`` layers: each kernel
pair is a ``torch.library.custom_op`` (forward) plus a companion backward
op, wired together with ``register_autograd`` and given fake-tensor
implementations so the ops compose with autograd and ``torch.compile``
without graph breaks. The compiled module (``quantem.cuda._core``) is
torch-free; everything torch-facing happens here.
"""

from __future__ import annotations

import os

import torch
from torch import Tensor

from quantem.cuda import _core

# ── fused TILTED K-Planes interpolation ───────────────────────────────────


def _channels_last(plane: Tensor) -> Tensor:
    """Return a contiguous ``(3T, H, W, C)`` kernel view.

    K-Planes parameters use logical NCHW shape with channels-last physical
    storage, so their NHWC permutation is already contiguous. Arbitrary input
    layouts retain the materializing fallback for correctness.
    """
    plane_cl = plane.permute(0, 2, 3, 1)
    if plane_cl.is_contiguous():
        return plane_cl
    return plane_cl.contiguous()


def _restore_plane_layout(grad_plane_cl: Tensor, plane: Tensor) -> Tensor:
    """Return an NCHW gradient with the input plane's contiguous layout."""
    grad_plane = grad_plane_cl.permute(0, 3, 1, 2)
    if plane.permute(0, 2, 3, 1).is_contiguous():
        return grad_plane
    return grad_plane.contiguous()


# ── three-level plane-wise 2-D squared TV ───────────────────────────────


def _plane_tv_num_blocks(grids: tuple[Tensor, Tensor, Tensor]) -> int:
    """Bound the deterministic partial buffer while retaining grid-stride coverage."""
    max_elements = max(grid.numel() for grid in grids)
    return min(1024, max(1, (max_elements + 255) // 256))


@torch.library.custom_op("quantem_cuda::plane_tv_loss", mutates_args=())
def _plane_tv_loss(plane0: Tensor, plane1: Tensor, plane2: Tensor, rotations: int) -> Tensor:
    planes = (plane0, plane1, plane2)
    grids = tuple(_channels_last(plane) for plane in planes)
    dims = tuple(dim for plane in planes for dim in plane.shape)
    num_blocks = _plane_tv_num_blocks(grids)
    partials = torch.empty((3 * num_blocks,), dtype=torch.float32, device=plane0.device)
    output = torch.empty((), dtype=torch.float32, device=plane0.device)
    stream = torch.cuda.current_stream(plane0.device).cuda_stream
    with torch.cuda.device(plane0.device):
        _core.plane_tv_loss_cuda(
            *(grid.data_ptr() for grid in grids),
            partials.data_ptr(),
            output.data_ptr(),
            *dims,
            rotations,
            num_blocks,
            stream,
        )
    return output


@_plane_tv_loss.register_fake
def _(plane0: Tensor, plane1: Tensor, plane2: Tensor, rotations: int) -> Tensor:
    del plane1, plane2, rotations
    return plane0.new_empty(())


@torch.library.custom_op("quantem_cuda::plane_tv_loss_bwd", mutates_args=())
def _plane_tv_loss_bwd(
    plane0: Tensor,
    plane1: Tensor,
    plane2: Tensor,
    grad_out: Tensor,
    rotations: int,
) -> list[Tensor]:
    planes = (plane0, plane1, plane2)
    grids = tuple(_channels_last(plane) for plane in planes)
    dims = tuple(dim for plane in planes for dim in plane.shape)
    num_blocks = _plane_tv_num_blocks(grids)
    gout = grad_out.detach().to(dtype=torch.float32, device=plane0.device).reshape(1).contiguous()
    grad_grids = tuple(torch.empty_like(grid) for grid in grids)
    stream = torch.cuda.current_stream(plane0.device).cuda_stream
    with torch.cuda.device(plane0.device):
        _core.plane_tv_loss_grad_cuda(
            *(grid.data_ptr() for grid in grids),
            gout.data_ptr(),
            *(grad_grid.data_ptr() for grad_grid in grad_grids),
            *dims,
            rotations,
            num_blocks,
            stream,
        )
    return [
        _restore_plane_layout(grad_grid, plane) for grad_grid, plane in zip(grad_grids, planes)
    ]


@_plane_tv_loss_bwd.register_fake
def _(
    plane0: Tensor,
    plane1: Tensor,
    plane2: Tensor,
    grad_out: Tensor,
    rotations: int,
) -> list[Tensor]:
    del grad_out, rotations
    return [torch.empty_like(plane0), torch.empty_like(plane1), torch.empty_like(plane2)]


def _plane_tv_setup_context(ctx, inputs, output) -> None:
    del output
    plane0, plane1, plane2, rotations = inputs
    ctx.save_for_backward(plane0, plane1, plane2)
    ctx.rotations = rotations


def _plane_tv_backward(ctx, grad_out):
    plane0, plane1, plane2 = ctx.saved_tensors
    return (*_plane_tv_loss_bwd(plane0, plane1, plane2, grad_out, ctx.rotations), None)


_plane_tv_loss.register_autograd(_plane_tv_backward, setup_context=_plane_tv_setup_context)


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
            p.data_ptr(),
            r.data_ptr(),
            g.data_ptr(),
            out.data_ptr(),
            b,
            t,
            c,
            h,
            w,
            g.dtype == torch.bfloat16,
            stream,
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
    # V4 reads a compact bf16 plane but accumulates its gradient in a
    # separate fp32 buffer; the scatter precision is never reduced.
    grad_plane_cl = torch.zeros_like(g, dtype=torch.float32)
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
            g.dtype == torch.bfloat16,
            stream,
        )
    # (3T, H, W, C) → (3T, C, H, W), matching the parameter layout
    return [grad_pts, grad_r, _restore_plane_layout(grad_plane_cl, plane)]


@_kplanes_tilted_fuse_bwd.register_fake
def _(pts: Tensor, rotations: Tensor, plane: Tensor, grad_out: Tensor) -> list[Tensor]:
    return [
        torch.empty_like(pts),
        torch.empty_like(rotations),
        torch.empty_like(plane, dtype=torch.float32),
    ]


def _kpt_setup_context(ctx, inputs, output) -> None:
    pts, rotations, plane = inputs
    ctx.save_for_backward(pts, rotations, plane)


def _kpt_backward(ctx, grad_out):
    pts, rotations, plane = ctx.saved_tensors
    grad_pts, grad_r, grad_plane = _kplanes_tilted_fuse_bwd(pts, rotations, plane, grad_out)
    return grad_pts, grad_r, grad_plane


_kplanes_tilted_fuse.register_autograd(_kpt_backward, setup_context=_kpt_setup_context)


# ── three-level multiscale fused TILTED K-Planes interpolation ───────────


@torch.library.custom_op("quantem_cuda::kplanes_tilted_fuse_ms", mutates_args=())
def _kplanes_tilted_fuse_ms(
    pts: Tensor,
    rotations: Tensor,
    plane0: Tensor,
    plane1: Tensor,
    plane2: Tensor,
    scale0: float,
    scale1: float,
    scale2: float,
) -> Tensor:
    p = pts.contiguous()
    r = rotations.contiguous()
    planes = (plane0, plane1, plane2)
    grids = tuple(_channels_last(plane) for plane in planes)
    dims = tuple(dim for plane in planes for dim in plane.shape[1:])
    b = p.shape[0]
    t = r.shape[0]
    out_width = t * sum(plane.shape[1] for plane in planes)
    out = torch.empty((b, out_width), dtype=torch.float32, device=p.device)
    stream = torch.cuda.current_stream(p.device).cuda_stream
    with torch.cuda.device(p.device):
        _core.kplanes_tilted_fuse_ms_cuda(
            p.data_ptr(),
            r.data_ptr(),
            *(grid.data_ptr() for grid in grids),
            out.data_ptr(),
            b,
            t,
            *dims,
            scale0,
            scale1,
            scale2,
            grids[0].dtype == torch.bfloat16,
            stream,
        )
    return out


@_kplanes_tilted_fuse_ms.register_fake
def _(
    pts: Tensor,
    rotations: Tensor,
    plane0: Tensor,
    plane1: Tensor,
    plane2: Tensor,
    scale0: float,
    scale1: float,
    scale2: float,
) -> Tensor:
    del scale0, scale1, scale2
    width = rotations.shape[0] * (plane0.shape[1] + plane1.shape[1] + plane2.shape[1])
    return pts.new_empty((pts.shape[0], width))


@torch.library.custom_op("quantem_cuda::kplanes_tilted_fuse_ms_bwd", mutates_args=())
def _kplanes_tilted_fuse_ms_bwd(
    pts: Tensor,
    rotations: Tensor,
    plane0: Tensor,
    plane1: Tensor,
    plane2: Tensor,
    grad_out: Tensor,
    scale0: float,
    scale1: float,
    scale2: float,
) -> list[Tensor]:
    p = pts.contiguous()
    r = rotations.contiguous()
    planes = (plane0, plane1, plane2)
    grids = tuple(_channels_last(plane) for plane in planes)
    dims = tuple(dim for plane in planes for dim in plane.shape[1:])
    b = p.shape[0]
    t = r.shape[0]
    gout = grad_out.detach().to(dtype=torch.float32, device=p.device)
    # Slices produced by a concatenated consumer have stride
    # (total_feature_width, 1). Preserve that view and pass its row stride;
    # only exotic non-unit inner strides need a materializing fallback.
    if gout.stride(1) != 1:
        gout = gout.contiguous()
    grad_pts = torch.zeros_like(p)
    grad_r = torch.zeros_like(r)
    grad_grids = tuple(torch.zeros_like(grid, dtype=torch.float32) for grid in grids)
    stream = torch.cuda.current_stream(p.device).cuda_stream
    with torch.cuda.device(p.device):
        _core.kplanes_tilted_fuse_ms_grad_cuda(
            p.data_ptr(),
            r.data_ptr(),
            *(grid.data_ptr() for grid in grids),
            gout.data_ptr(),
            *(grad_grid.data_ptr() for grad_grid in grad_grids),
            grad_r.data_ptr(),
            grad_pts.data_ptr(),
            b,
            t,
            *dims,
            gout.stride(0),
            scale0,
            scale1,
            scale2,
            grids[0].dtype == torch.bfloat16,
            stream,
        )
    restored = tuple(
        _restore_plane_layout(grad_grid, plane) for grad_grid, plane in zip(grad_grids, planes)
    )
    return [grad_pts, grad_r, *restored]


@_kplanes_tilted_fuse_ms_bwd.register_fake
def _(
    pts: Tensor,
    rotations: Tensor,
    plane0: Tensor,
    plane1: Tensor,
    plane2: Tensor,
    grad_out: Tensor,
    scale0: float,
    scale1: float,
    scale2: float,
) -> list[Tensor]:
    del grad_out, scale0, scale1, scale2
    return [
        torch.empty_like(pts),
        torch.empty_like(rotations),
        torch.empty_like(plane0, dtype=torch.float32),
        torch.empty_like(plane1, dtype=torch.float32),
        torch.empty_like(plane2, dtype=torch.float32),
    ]


def _kpt_ms_setup_context(ctx, inputs, output) -> None:
    del output
    pts, rotations, plane0, plane1, plane2, scale0, scale1, scale2 = inputs
    ctx.save_for_backward(pts, rotations, plane0, plane1, plane2)
    ctx.scales = (scale0, scale1, scale2)


def _kpt_ms_backward(ctx, grad_out):
    pts, rotations, plane0, plane1, plane2 = ctx.saved_tensors
    grads = _kplanes_tilted_fuse_ms_bwd(
        pts, rotations, plane0, plane1, plane2, grad_out, *ctx.scales
    )
    return *grads, None, None, None


_kplanes_tilted_fuse_ms.register_autograd(_kpt_ms_backward, setup_context=_kpt_ms_setup_context)


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
    return [grad_pts, grad_r, _restore_plane_layout(grad_plane_cl, plane)]


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


# ── public wrappers ───────────────────────────────────────────────────────


def plane_tv_loss(plane0: Tensor, plane1: Tensor, plane2: Tensor) -> Tensor:
    """Three-level plane-wise 2-D squared-TV loss with analytic backward.

    Every input is a logical NCHW grid ``[3*T, C, H, W]``. For each grid,
    this computes the mean squared H difference plus the mean squared W
    difference independently for every plane, sums the three planes per
    rotation, averages over ``T``, then sums the three grid levels. The H
    and W means retain their distinct ``C*(H-1)*W`` and ``C*H*(W-1)``
    denominators.

    Production K-Planes parameters use channels-last physical storage, which
    reaches the kernel without a copy. Other NCHW layouts are accepted via a
    contiguous NHWC staging view and receive gradients in their original
    contiguous layout.
    """
    name = "plane_tv_loss"
    planes = (plane0, plane1, plane2)
    rotations = None
    device = plane0.device
    for level, plane in enumerate(planes):
        if plane.ndim != 4:
            raise ValueError(
                f"{name} expects plane{level} [3*T, C, H, W], got {tuple(plane.shape)}"
            )
        if plane.shape[0] == 0 or plane.shape[0] % 3 != 0:
            raise ValueError(
                f"{name} expects plane{level}.shape[0] to be 3*T, got {plane.shape[0]}"
            )
        level_rotations = plane.shape[0] // 3
        if rotations is None:
            rotations = level_rotations
        elif level_rotations != rotations:
            raise ValueError(f"{name} requires the same T for all three levels")
        if plane.dtype != torch.float32:
            raise TypeError(f"{name} is fp32-only (plane{level} is {plane.dtype}).")
        if not plane.is_cuda:
            raise ValueError(f"{name} requires CUDA tensors (plane{level} on {plane.device}).")
        if plane.device != device:
            raise ValueError(
                f"{name} requires all planes on {device} (plane{level} on {plane.device})."
            )
        if any(dim == 0 for dim in plane.shape[1:]):
            raise ValueError(f"{name} requires non-empty C/H/W dimensions")
        if any(dim >= 2**31 for dim in plane.shape):
            raise ValueError(f"{name} requires every grid dimension to fit in int32")
    assert rotations is not None
    return _plane_tv_loss(plane0, plane1, plane2, rotations)


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
        plane:     fp32 CUDA tensor ``[3*T, C, H, W]`` (plane ``t*3 + p``), or
                   bf16 when ``QUANTEM_KPLANES_BWD_VARIANT`` is 4 or 5.

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
    for name, t in (("pts", pts), ("rotations", rotations)):
        if t.dtype != torch.float32:
            raise TypeError(f"kplanes_tilted_fuse is fp32-only ({name} is {t.dtype}).")
        if not t.is_cuda:
            raise ValueError(f"kplanes_tilted_fuse requires CUDA tensors ({name} on {t.device}).")
    bf16_experiment = plane.dtype == torch.bfloat16 and os.environ.get(
        "QUANTEM_KPLANES_BWD_VARIANT"
    ) in {"4", "5"}
    if plane.dtype != torch.float32 and not bf16_experiment:
        raise TypeError(
            "kplanes_tilted_fuse is fp32-only unless "
            "QUANTEM_KPLANES_BWD_VARIANT=4 or 5 selects a bf16 plane "
            f"(plane is {plane.dtype})."
        )
    if not plane.is_cuda:
        raise ValueError(f"kplanes_tilted_fuse requires CUDA tensors (plane on {plane.device}).")
    if plane.shape[1] * plane.shape[2] * plane.shape[3] >= 2**31:
        raise ValueError(
            "kplanes_tilted_fuse uses 32-bit per-plane offsets; "
            f"C*H*W must be < 2^31, got plane {tuple(plane.shape)}"
        )
    return _kplanes_tilted_fuse(pts, rotations, plane)


def kplanes_tilted_fuse_ms(
    pts: Tensor,
    rotations: Tensor,
    plane0: Tensor,
    plane1: Tensor,
    plane2: Tensor,
    scale0: float = 1.0,
    scale1: float = 1.0,
    scale2: float = 1.0,
) -> Tensor:
    """Fuse three K-Planes scales into one output and one launch per direction.

    Each plane is logically ``[3*T, C_l, H_l, W_l]``. The output is
    ``[B, T*(C_0+C_1+C_2)]`` with scale-major slices, matching
    ``torch.cat([kplanes_tilted_fuse(..., plane_l)], dim=-1)``. The three
    scalar gates are applied in the CUDA forward epilogue and at the backward
    upstream-gradient load.
    """
    name = "kplanes_tilted_fuse_ms"
    if pts.ndim != 2 or pts.shape[-1] != 3:
        raise ValueError(f"{name} expects pts [B, 3], got {tuple(pts.shape)}")
    if rotations.ndim != 3 or rotations.shape[-2:] != (3, 3):
        raise ValueError(f"{name} expects rotations [T, 3, 3], got {tuple(rotations.shape)}")
    planes = (plane0, plane1, plane2)
    for level, plane in enumerate(planes):
        if plane.ndim != 4 or plane.shape[0] != 3 * rotations.shape[0]:
            raise ValueError(
                f"{name} expects plane{level} [3*T, C, H, W]; got "
                f"{tuple(plane.shape)} for T={rotations.shape[0]}"
            )
        if plane.shape[1] * plane.shape[2] * plane.shape[3] >= 2**31:
            raise ValueError(
                f"{name} uses 32-bit per-plane offsets; C*H*W must be < 2^31, "
                f"got plane{level} {tuple(plane.shape)}"
            )
    for tensor_name, tensor in (("pts", pts), ("rotations", rotations)):
        if tensor.dtype != torch.float32:
            raise TypeError(f"{name} is fp32-only ({tensor_name} is {tensor.dtype}).")
        if not tensor.is_cuda:
            raise ValueError(f"{name} requires CUDA tensors ({tensor_name} on {tensor.device}).")
    grid_dtype = plane0.dtype
    if any(plane.dtype != grid_dtype for plane in planes[1:]):
        raise TypeError(f"{name} requires all three planes to have the same dtype.")
    bf16_experiment = grid_dtype == torch.bfloat16 and os.environ.get(
        "QUANTEM_KPLANES_BWD_VARIANT"
    ) in {"4", "5"}
    if grid_dtype != torch.float32 and not bf16_experiment:
        raise TypeError(
            f"{name} is fp32-only unless QUANTEM_KPLANES_BWD_VARIANT=4 or 5 "
            f"selects bf16 planes (planes are {grid_dtype})."
        )
    for level, plane in enumerate(planes):
        if not plane.is_cuda or plane.device != pts.device:
            raise ValueError(f"{name} requires plane{level} on {pts.device} (got {plane.device}).")
    if rotations.device != pts.device:
        raise ValueError(f"{name} requires rotations on {pts.device} (got {rotations.device}).")
    return _kplanes_tilted_fuse_ms(
        pts,
        rotations,
        plane0,
        plane1,
        plane2,
        float(scale0),
        float(scale1),
        float(scale2),
    )


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
