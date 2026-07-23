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
import threading
import warnings

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

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


def _kplanes_ms_bf16_output_enabled() -> bool:
    """Select direct bf16 feature stores for CUDA bf16 autocast."""
    return (
        os.environ.get("QUANTEM_KPLANES_MS_BF16_OUT", "1") != "0"
        and torch.is_autocast_enabled("cuda")
        and torch.get_autocast_dtype("cuda") == torch.bfloat16
    )


# ── cuBLASLt fused sigma-head MLP ──────────────────────────────────────


def _mlp_aux_ld(width: int) -> int:
    """Return the cuBLASLt ReLU-mask leading dimension, measured in bits."""
    return ((width + 127) // 128) * 128


def _mlp_workspace(device: torch.device) -> Tensor:
    """Allocate workspace through PyTorch's stream-aware caching allocator."""
    try:
        workspace_mb = int(os.environ.get("QUANTEM_FUSED_MLP_WORKSPACE_MB", "32"))
    except ValueError:
        workspace_mb = 32
    workspace_mb = min(1024, max(0, workspace_mb))
    return torch.empty(workspace_mb * 1024 * 1024, dtype=torch.uint8, device=device)


_MlpShapeKey = tuple[int | None, int, int, int, int, int]
_unsupported_mlp_shapes: dict[_MlpShapeKey, str] = {}
_warned_mlp_fallback_reasons: set[str] = set()
_mlp_fallback_lock = threading.Lock()


def _mlp_shape_key(x: Tensor, w1: Tensor, w2: Tensor, w3: Tensor) -> _MlpShapeKey:
    return (
        x.device.index,
        x.shape[0],
        x.shape[1],
        w1.shape[0],
        w2.shape[0],
        w3.shape[0],
    )


def _unsupported_mlp_reason(key: _MlpShapeKey) -> str | None:
    with _mlp_fallback_lock:
        return _unsupported_mlp_shapes.get(key)


def _memoize_unsupported_mlp_shape(key: _MlpShapeKey, reason: str) -> None:
    with _mlp_fallback_lock:
        _unsupported_mlp_shapes[key] = reason
        should_warn = reason not in _warned_mlp_fallback_reasons
        _warned_mlp_fallback_reasons.add(reason)
    if should_warn:
        warnings.warn(
            "Fused cuBLASLt MLP backward is unsupported for this device/shape; "
            f"future calls will use the eager path. Reason: {reason}",
            RuntimeWarning,
            stacklevel=3,
        )


@torch.library.custom_op("quantem_cuda::cublaslt_mlp", mutates_args=())
def _cublaslt_mlp(
    x: Tensor,
    w1: Tensor,
    b1: Tensor,
    w2: Tensor,
    b2: Tensor,
    w3: Tensor,
    b3: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    m, k = x.shape
    h1_width, h2_width, out_width = w1.shape[0], w2.shape[0], w3.shape[0]
    h1 = torch.empty((m, h1_width), dtype=torch.bfloat16, device=x.device)
    h2 = torch.empty((m, h2_width), dtype=torch.bfloat16, device=x.device)
    out = torch.empty((m, out_width), dtype=torch.bfloat16, device=x.device)
    aux1_ld = _mlp_aux_ld(h1_width)
    aux2_ld = _mlp_aux_ld(h2_width)
    aux1 = torch.empty((m, aux1_ld // 8), dtype=torch.uint8, device=x.device)
    aux2 = torch.empty((m, aux2_ld // 8), dtype=torch.uint8, device=x.device)
    workspace = _mlp_workspace(x.device)
    stream = torch.cuda.current_stream(x.device).cuda_stream
    with torch.cuda.device(x.device):
        _core.cublaslt_mlp_forward_cuda(
            x.data_ptr(),
            w1.data_ptr(),
            b1.data_ptr(),
            w2.data_ptr(),
            b2.data_ptr(),
            w3.data_ptr(),
            b3.data_ptr(),
            h1.data_ptr(),
            h2.data_ptr(),
            out.data_ptr(),
            aux1.data_ptr(),
            aux2.data_ptr(),
            m,
            k,
            h1_width,
            h2_width,
            out_width,
            aux1_ld,
            aux2_ld,
            workspace.data_ptr(),
            workspace.numel(),
            stream,
        )
    return out, h1, h2, aux1, aux2


@_cublaslt_mlp.register_fake
def _(
    x: Tensor,
    w1: Tensor,
    b1: Tensor,
    w2: Tensor,
    b2: Tensor,
    w3: Tensor,
    b3: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    del b1, b2, b3
    m = x.shape[0]
    aux1_ld = _mlp_aux_ld(w1.shape[0])
    aux2_ld = _mlp_aux_ld(w2.shape[0])
    return (
        x.new_empty((m, w3.shape[0])),
        x.new_empty((m, w1.shape[0])),
        x.new_empty((m, w2.shape[0])),
        torch.empty((m, aux1_ld // 8), dtype=torch.uint8, device=x.device),
        torch.empty((m, aux2_ld // 8), dtype=torch.uint8, device=x.device),
    )


@torch.library.custom_op("quantem_cuda::cublaslt_mlp_bwd", mutates_args=())
def _cublaslt_mlp_bwd(
    x: Tensor,
    w1: Tensor,
    w2: Tensor,
    w3: Tensor,
    h1: Tensor,
    h2: Tensor,
    aux1: Tensor,
    aux2: Tensor,
    grad_out: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    grad_out = grad_out.detach().to(device=x.device, dtype=torch.bfloat16).contiguous()
    grad_x = torch.empty_like(x, memory_format=torch.contiguous_format)
    grad_w1 = torch.empty_like(w1, dtype=torch.float32)
    grad_w2 = torch.empty_like(w2, dtype=torch.float32)
    grad_w3 = torch.empty_like(w3, dtype=torch.float32)
    # DRELU_BGRAD requires bias-grad storage to match its bf16 D matrix.
    grad_b1 = torch.empty(w1.shape[0], dtype=torch.bfloat16, device=x.device)
    grad_b2 = torch.empty(w2.shape[0], dtype=torch.bfloat16, device=x.device)
    dz1 = torch.empty((x.shape[0], w1.shape[0]), dtype=torch.bfloat16, device=x.device)
    dz2 = torch.empty((x.shape[0], w2.shape[0]), dtype=torch.bfloat16, device=x.device)
    workspace = _mlp_workspace(x.device)
    stream = torch.cuda.current_stream(x.device).cuda_stream
    with torch.cuda.device(x.device):
        _core.cublaslt_mlp_backward_cuda(
            x.data_ptr(),
            w1.data_ptr(),
            w2.data_ptr(),
            w3.data_ptr(),
            h1.data_ptr(),
            h2.data_ptr(),
            aux1.data_ptr(),
            aux2.data_ptr(),
            grad_out.data_ptr(),
            grad_x.data_ptr(),
            grad_w1.data_ptr(),
            grad_w2.data_ptr(),
            grad_w3.data_ptr(),
            grad_b1.data_ptr(),
            grad_b2.data_ptr(),
            dz1.data_ptr(),
            dz2.data_ptr(),
            x.shape[0],
            x.shape[1],
            w1.shape[0],
            w2.shape[0],
            w3.shape[0],
            _mlp_aux_ld(w1.shape[0]),
            _mlp_aux_ld(w2.shape[0]),
            workspace.data_ptr(),
            workspace.numel(),
            stream,
        )
    return grad_x, grad_w1, grad_b1, grad_w2, grad_b2, grad_w3


@_cublaslt_mlp_bwd.register_fake
def _(
    x: Tensor,
    w1: Tensor,
    w2: Tensor,
    w3: Tensor,
    h1: Tensor,
    h2: Tensor,
    aux1: Tensor,
    aux2: Tensor,
    grad_out: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    del h1, h2, aux1, aux2, grad_out
    return (
        torch.empty_like(x),
        torch.empty_like(w1, dtype=torch.float32),
        torch.empty(w1.shape[0], dtype=torch.bfloat16, device=x.device),
        torch.empty_like(w2, dtype=torch.float32),
        torch.empty(w2.shape[0], dtype=torch.bfloat16, device=x.device),
        torch.empty_like(w3, dtype=torch.float32),
    )


class _FusedHiddenMLPFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, b1, w2, b2, w3, b3):
        # Explicit copies mirror autocast's bf16 GEMM operands while keeping
        # fp32 master parameters as the Function inputs and gradient targets.
        tensors = tuple(
            t.detach().to(dtype=torch.bfloat16).contiguous() for t in (x, w1, b1, w2, b2, w3, b3)
        )
        x_bf16, w1_bf16, b1_bf16, w2_bf16, b2_bf16, w3_bf16, b3_bf16 = tensors
        out, h1, h2, aux1, aux2 = _cublaslt_mlp(
            x_bf16, w1_bf16, b1_bf16, w2_bf16, b2_bf16, w3_bf16, b3_bf16
        )
        ctx.save_for_backward(
            x_bf16,
            w1_bf16,
            w2_bf16,
            w3_bf16,
            h1,
            h2,
            aux1,
            aux2,
            w1,
            b1,
            w2,
            b2,
            w3,
            b3,
        )
        ctx.input_dtype = x.dtype
        ctx.mlp_shape_key = _mlp_shape_key(x, w1, w2, w3)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        (
            x,
            w1_bf16,
            w2_bf16,
            w3_bf16,
            h1,
            h2,
            aux1,
            aux2,
            w1,
            b1,
            w2,
            b2,
            w3,
            b3,
        ) = ctx.saved_tensors
        try:
            grad_x, grad_w1, grad_b1, grad_w2, grad_b2, grad_w3 = _cublaslt_mlp_bwd(
                x, w1_bf16, w2_bf16, w3_bf16, h1, h2, aux1, aux2, grad_out
            )
        except (RuntimeError, TypeError, ValueError) as error:
            # A device may expose the forward AUX epilogue but no compatible
            # DRELU_BGRAD heuristic. Recompute the unchanged eager graph so an
            # unsupported backward never breaks training after fused forward.
            reason = f"{type(error).__name__}: {error}"
            _memoize_unsupported_mlp_shape(ctx.mlp_shape_key, reason)
            with torch.enable_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                reference_inputs = tuple(
                    tensor.detach().requires_grad_(True) for tensor in (x, w1, b1, w2, b2, w3, b3)
                )
                x_ref, w1_ref, b1_ref, w2_ref, b2_ref, w3_ref, b3_ref = reference_inputs
                h1_ref = torch.relu(torch.nn.functional.linear(x_ref, w1_ref, b1_ref))
                h2_ref = torch.relu(torch.nn.functional.linear(h1_ref, w2_ref, b2_ref))
                out_ref = torch.nn.functional.linear(h2_ref, w3_ref, b3_ref)
                return torch.autograd.grad(out_ref, reference_inputs, grad_out)
        grad_b3 = grad_out.detach().float().sum(dim=0)
        return (
            grad_x.to(ctx.input_dtype),
            grad_w1,
            grad_b1.float(),
            grad_w2,
            grad_b2.float(),
            grad_w3,
            grad_b3,
        )


def fused_hidden_mlp(
    x: Tensor,
    w1: Tensor,
    b1: Tensor,
    w2: Tensor,
    b2: Tensor,
    w3: Tensor,
    b3: Tensor,
) -> Tensor:
    """Run the full two-hidden-layer bf16 sigma head with cuBLASLt epilogues.

    This is intentionally strict.  Model-side dispatch handles unsupported
    dtype, layout, device, shape, and compilation cases by using its unchanged
    ``nn.Sequential`` implementation. This custom path is not governed by
    ``torch.use_deterministic_algorithms``; disable it when strict PyTorch
    deterministic-mode behavior is required.
    """
    tensors = (x, w1, b1, w2, b2, w3, b3)
    if x.ndim != 2 or any(t.device != x.device or not t.is_cuda for t in tensors):
        raise ValueError("fused_hidden_mlp requires 2-D tensors on one CUDA device")
    if x.dtype != torch.bfloat16:
        raise TypeError("fused_hidden_mlp requires a bf16 input")
    if any(t.dtype != torch.float32 for t in tensors[1:]):
        raise TypeError("fused_hidden_mlp requires fp32 master parameters")
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("fused_hidden_mlp requires contiguous tensors")
    if w1.shape[1] != x.shape[1] or w2.shape[1] != w1.shape[0] or w3.shape[1] != w2.shape[0]:
        raise ValueError("fused_hidden_mlp layer shapes do not compose")
    if b1.shape != (w1.shape[0],) or b2.shape != (w2.shape[0],) or b3.shape != (w3.shape[0],):
        raise ValueError("fused_hidden_mlp bias shapes do not match their weights")
    shape_key = _mlp_shape_key(x, w1, w2, w3)
    unsupported_reason = _unsupported_mlp_reason(shape_key)
    if unsupported_reason is not None:
        raise RuntimeError(
            "fused_hidden_mlp is disabled for a previously unsupported device/shape: "
            f"{unsupported_reason}"
        )
    return _FusedHiddenMLPFunction.apply(x, w1, b1, w2, b2, w3, b3)


# ── fused trunc-exp density tail ────────────────────────────────────────


def _density_tail_shape_strides(tensor: Tensor) -> tuple[int, int, int, int]:
    if tensor.ndim == 1:
        return tensor.shape[0], 1, tensor.stride(0), 0
    return tensor.shape[0], tensor.shape[1], tensor.stride(0), tensor.stride(1)


@torch.library.custom_op("quantem_cuda::density_tail", mutates_args=())
def _density_tail(values: Tensor, offset: float) -> Tensor:
    output = torch.empty_like(values, memory_format=torch.contiguous_format)
    rows, cols, value_stride0, value_stride1 = _density_tail_shape_strides(values)
    _, _, output_stride0, output_stride1 = _density_tail_shape_strides(output)
    stream = torch.cuda.current_stream(values.device).cuda_stream
    with torch.cuda.device(values.device):
        _core.density_tail_cuda(
            values.data_ptr(),
            output.data_ptr(),
            rows,
            cols,
            value_stride0,
            value_stride1,
            output_stride0,
            output_stride1,
            float(offset),
            values.dtype == torch.bfloat16,
            stream,
        )
    return output


@_density_tail.register_fake
def _(values: Tensor, offset: float) -> Tensor:
    del offset
    return torch.empty_like(values, memory_format=torch.contiguous_format)


@torch.library.custom_op("quantem_cuda::density_tail_bwd", mutates_args=())
def _density_tail_bwd(values: Tensor, grad_out: Tensor, offset: float) -> Tensor:
    grad_out = grad_out.detach().to(device=values.device, dtype=values.dtype)
    grad_values = torch.empty_like(values, memory_format=torch.contiguous_format)
    rows, cols, value_stride0, value_stride1 = _density_tail_shape_strides(values)
    _, _, gout_stride0, gout_stride1 = _density_tail_shape_strides(grad_out)
    _, _, grad_stride0, grad_stride1 = _density_tail_shape_strides(grad_values)
    stream = torch.cuda.current_stream(values.device).cuda_stream
    with torch.cuda.device(values.device):
        _core.density_tail_grad_cuda(
            values.data_ptr(),
            grad_out.data_ptr(),
            grad_values.data_ptr(),
            rows,
            cols,
            value_stride0,
            value_stride1,
            gout_stride0,
            gout_stride1,
            grad_stride0,
            grad_stride1,
            float(offset),
            values.dtype == torch.bfloat16,
            stream,
        )
    return grad_values


@_density_tail_bwd.register_fake
def _(values: Tensor, grad_out: Tensor, offset: float) -> Tensor:
    del grad_out, offset
    return torch.empty_like(values, memory_format=torch.contiguous_format)


def _density_tail_setup_context(ctx, inputs, output) -> None:
    del output
    values, offset = inputs
    ctx.save_for_backward(values)
    ctx.offset = offset


def _density_tail_backward(ctx, grad_out):
    (values,) = ctx.saved_tensors
    return _density_tail_bwd(values, grad_out, ctx.offset), None


_density_tail.register_autograd(_density_tail_backward, setup_context=_density_tail_setup_context)


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
            False,
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
    output_is_bf16: bool,
) -> Tensor:
    p = pts.contiguous()
    r = rotations.contiguous()
    planes = (plane0, plane1, plane2)
    grids = tuple(_channels_last(plane) for plane in planes)
    dims = tuple(dim for plane in planes for dim in plane.shape[1:])
    b = p.shape[0]
    t = r.shape[0]
    out_width = t * sum(plane.shape[1] for plane in planes)
    out_dtype = torch.bfloat16 if output_is_bf16 else torch.float32
    out = torch.empty((b, out_width), dtype=out_dtype, device=p.device)
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
            output_is_bf16,
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
    output_is_bf16: bool,
) -> Tensor:
    del scale0, scale1, scale2
    width = rotations.shape[0] * (plane0.shape[1] + plane1.shape[1] + plane2.shape[1])
    dtype = torch.bfloat16 if output_is_bf16 else torch.float32
    return pts.new_empty((pts.shape[0], width), dtype=dtype)


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
    gout = grad_out.detach().to(device=p.device)
    if gout.dtype not in (torch.float32, torch.bfloat16):
        gout = gout.float()
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
            gout.dtype == torch.bfloat16,
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
    pts, rotations, plane0, plane1, plane2, scale0, scale1, scale2, output_is_bf16 = inputs
    del output_is_bf16
    ctx.save_for_backward(pts, rotations, plane0, plane1, plane2)
    ctx.scales = (scale0, scale1, scale2)


def _kpt_ms_backward(ctx, grad_out):
    pts, rotations, plane0, plane1, plane2 = ctx.saved_tensors
    grads = _kplanes_tilted_fuse_ms_bwd(
        pts, rotations, plane0, plane1, plane2, grad_out, *ctx.scales
    )
    return *grads, None, None, None, None


_kplanes_tilted_fuse_ms.register_autograd(_kpt_ms_backward, setup_context=_kpt_ms_setup_context)


# ── optional multiscale interpolation + plane-TV autograd path ──────────


@torch.library.custom_op("quantem_cuda::kplanes_tilted_fuse_ms_tv", mutates_args=())
def _kplanes_tilted_fuse_ms_tv(
    pts: Tensor,
    rotations: Tensor,
    plane0: Tensor,
    plane1: Tensor,
    plane2: Tensor,
    scale0: float,
    scale1: float,
    scale2: float,
    output_is_bf16: bool,
) -> tuple[Tensor, Tensor]:
    p = pts.contiguous()
    r = rotations.contiguous()
    planes = (plane0, plane1, plane2)
    grids = tuple(_channels_last(plane) for plane in planes)
    dims = tuple(dim for plane in planes for dim in plane.shape[1:])
    tv_dims = tuple(dim for plane in planes for dim in plane.shape)
    b = p.shape[0]
    t = r.shape[0]
    out_width = t * sum(plane.shape[1] for plane in planes)
    out_dtype = torch.bfloat16 if output_is_bf16 else torch.float32
    out = torch.empty((b, out_width), dtype=out_dtype, device=p.device)
    num_blocks = _plane_tv_num_blocks(grids)
    partials = torch.empty((3 * num_blocks,), dtype=torch.float32, device=p.device)
    tv = torch.empty((), dtype=torch.float32, device=p.device)
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
            False,
            output_is_bf16,
            stream,
        )
        _core.plane_tv_loss_cuda(
            *(grid.data_ptr() for grid in grids),
            partials.data_ptr(),
            tv.data_ptr(),
            *tv_dims,
            t,
            num_blocks,
            stream,
        )
    return out, tv


@_kplanes_tilted_fuse_ms_tv.register_fake
def _(
    pts: Tensor,
    rotations: Tensor,
    plane0: Tensor,
    plane1: Tensor,
    plane2: Tensor,
    scale0: float,
    scale1: float,
    scale2: float,
    output_is_bf16: bool,
) -> tuple[Tensor, Tensor]:
    del scale0, scale1, scale2
    width = rotations.shape[0] * (plane0.shape[1] + plane1.shape[1] + plane2.shape[1])
    dtype = torch.bfloat16 if output_is_bf16 else torch.float32
    return pts.new_empty((pts.shape[0], width), dtype=dtype), plane0.new_empty(())


@torch.library.custom_op("quantem_cuda::kplanes_tilted_fuse_ms_tv_bwd", mutates_args=())
def _kplanes_tilted_fuse_ms_tv_bwd(
    pts: Tensor,
    rotations: Tensor,
    plane0: Tensor,
    plane1: Tensor,
    plane2: Tensor,
    grad_out: Tensor,
    grad_tv: Tensor,
    scale0: float,
    scale1: float,
    scale2: float,
) -> list[Tensor]:
    p = pts.contiguous()
    r = rotations.contiguous()
    planes = (plane0, plane1, plane2)
    grids = tuple(_channels_last(plane) for plane in planes)
    dims = tuple(dim for plane in planes for dim in plane.shape[1:])
    tv_dims = tuple(dim for plane in planes for dim in plane.shape)
    b = p.shape[0]
    t = r.shape[0]
    gout = grad_out.detach().to(device=p.device)
    if gout.dtype not in (torch.float32, torch.bfloat16):
        gout = gout.float()
    if gout.stride(1) != 1:
        gout = gout.contiguous()
    tv_gout = grad_tv.detach().to(dtype=torch.float32, device=p.device).reshape(1).contiguous()
    grad_pts = torch.zeros_like(p)
    grad_r = torch.zeros_like(r)
    grad_grids = tuple(torch.zeros_like(grid, dtype=torch.float32) for grid in grids)
    num_blocks = _plane_tv_num_blocks(grids)
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
            False,
            gout.dtype == torch.bfloat16,
            stream,
        )
        _core.plane_tv_loss_grad_cuda(
            *(grid.data_ptr() for grid in grids),
            tv_gout.data_ptr(),
            *(grad_grid.data_ptr() for grad_grid in grad_grids),
            *tv_dims,
            t,
            num_blocks,
            True,
            stream,
        )
    restored = tuple(
        _restore_plane_layout(grad_grid, plane) for grad_grid, plane in zip(grad_grids, planes)
    )
    return [grad_pts, grad_r, *restored]


@_kplanes_tilted_fuse_ms_tv_bwd.register_fake
def _(
    pts: Tensor,
    rotations: Tensor,
    plane0: Tensor,
    plane1: Tensor,
    plane2: Tensor,
    grad_out: Tensor,
    grad_tv: Tensor,
    scale0: float,
    scale1: float,
    scale2: float,
) -> list[Tensor]:
    del grad_out, grad_tv, scale0, scale1, scale2
    return [
        torch.empty_like(pts),
        torch.empty_like(rotations),
        torch.empty_like(plane0, dtype=torch.float32),
        torch.empty_like(plane1, dtype=torch.float32),
        torch.empty_like(plane2, dtype=torch.float32),
    ]


def _kpt_ms_tv_setup_context(ctx, inputs, output) -> None:
    del output
    pts, rotations, plane0, plane1, plane2, scale0, scale1, scale2, output_is_bf16 = inputs
    del output_is_bf16
    ctx.save_for_backward(pts, rotations, plane0, plane1, plane2)
    ctx.scales = (scale0, scale1, scale2)


def _kpt_ms_tv_backward(ctx, grad_out, grad_tv):
    pts, rotations, plane0, plane1, plane2 = ctx.saved_tensors
    grads = _kplanes_tilted_fuse_ms_tv_bwd(
        pts,
        rotations,
        plane0,
        plane1,
        plane2,
        grad_out,
        grad_tv,
        *ctx.scales,
    )
    return *grads, None, None, None, None


_kplanes_tilted_fuse_ms_tv.register_autograd(
    _kpt_ms_tv_backward, setup_context=_kpt_ms_tv_setup_context
)


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


def density_tail(values: Tensor, offset: float = 0.0) -> Tensor:
    """Fused ``exp(values - offset)`` with a trunc-exp backward.

    Supports non-empty fp32/bf16 CUDA vectors and matrices. Inputs may be
    strided; the result is dense and has the same shape and dtype.
    """
    name = "density_tail"
    if values.ndim not in (1, 2):
        raise ValueError(f"{name} expects a 1-D or 2-D tensor, got {tuple(values.shape)}")
    if values.numel() == 0:
        raise ValueError(f"{name} requires a non-empty tensor")
    if values.dtype not in (torch.float32, torch.bfloat16):
        raise TypeError(f"{name} supports fp32 and bf16 (values are {values.dtype}).")
    if not values.is_cuda:
        raise ValueError(f"{name} requires a CUDA tensor (values on {values.device}).")
    if any(stride < 0 for stride in values.stride()):
        raise ValueError(f"{name} does not support negative strides")
    return _density_tail(values, float(offset))


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
    upstream-gradient load. Under CUDA bf16 autocast, the kernel stores bf16
    features directly unless ``QUANTEM_KPLANES_MS_BF16_OUT=0``.
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
        _kplanes_ms_bf16_output_enabled(),
    )


def kplanes_tilted_fuse_ms_tv(
    pts: Tensor,
    rotations: Tensor,
    plane0: Tensor,
    plane1: Tensor,
    plane2: Tensor,
    scale0: float = 1.0,
    scale1: float = 1.0,
    scale2: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Return multiscale features and raw plane-TV through one autograd node.

    The forward reuses the existing multiscale and plane-TV launchers. During
    backward, plane-TV accumulates directly into the multiscale op's three
    dense fp32 grid-gradient buffers, eliminating separate TV gradients and
    the three autograd add passes. This experimental path is fp32-grid only.
    """
    name = "kplanes_tilted_fuse_ms_tv"
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
        if any(dim == 0 for dim in plane.shape[1:]):
            raise ValueError(f"{name} requires non-empty C/H/W dimensions")
    for tensor_name, tensor in (
        ("pts", pts),
        ("rotations", rotations),
        *[(f"plane{i}", p) for i, p in enumerate(planes)],
    ):
        if tensor.dtype != torch.float32:
            raise TypeError(f"{name} is fp32-grid only ({tensor_name} is {tensor.dtype}).")
        if not tensor.is_cuda or tensor.device != pts.device:
            raise ValueError(
                f"{name} requires {tensor_name} on {pts.device} (got {tensor.device})."
            )
    return _kplanes_tilted_fuse_ms_tv(
        pts,
        rotations,
        plane0,
        plane1,
        plane2,
        float(scale0),
        float(scale1),
        float(scale2),
        _kplanes_ms_bf16_output_enabled(),
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
