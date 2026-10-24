"""kopt-convp runtime (class A, bitwise): p-major PHYSICAL layout for the deduplicated GDN conv-window buffer.

apply_convp.py copies this file to <sglang>/srt/mem_cache/kopt_convp.py and adds one hook at the top of
MambaPool._allocate_deduplicated_conv_window (memory_pool.py). With SGL_CONVP=1 the hook calls allocate_pmajor();
a (phys, view) result replaces the stock allocation, None means "use the stock code below" (fallback).

Stock (sglang @1cf2b8c54, GDN conv_shape = (C, W-1), window axis last):
    phys [L, R, C, NW]          NW = T + W - 2, p (window position) contiguous
    view [L, R, T, C, W-1]      strides (sL, sR, 1, NW, 1)       step t, channel c, tap j -> phys[.., c, t + j]
kopt-convp:
    phys [L, R, NW, C]          C contiguous
    view [L, R, T, C, W-1]      strides (sL, sR, C, 1, C)        step t, channel c, tap j -> phys[.., t + j, c]
Same logical shape, same aliasing (element (t, c, j) and (t', c, j') share memory iff c == c' and t + j == t' + j'),
so every consumer that addresses the window through the view's strides reads and writes exactly the same logical
values; only addresses change. The a2 conv kernel then stores the window along C (16 B vectors) like its out store.
k1b_conv H200: B64 1.847 -> 1.548 ms/step for 48 layers, bitwise out / conv_state / window, eager and CUDA graph.

Never raises: any unsupported case or error logs ONE line "[kopt-convp] fallback: <reason> -> stock layout" (once
per reason) and returns None, so the stock allocation runs.
"""
import os
import sys

STATE = {"on": [], "fallback": []}
_SEEN = set()


def _log(msg):
    if msg in _SEEN:
        return
    _SEEN.add(msg)
    if os.environ.get("RANK", "0") in ("0", ""):
        print(f"[kopt-convp] {msg}", file=sys.stderr, flush=True)


def _fallback(reason):
    STATE["fallback"].append(reason)
    _log(f"fallback: {reason} -> stock layout")
    return None


def pmajor_geometry(conv_shape, num_layers, rows, num_draft_tokens):
    """Pure-python geometry (shared with the CPU tests): (phys_shape, view_shape, view_strides) in elements."""
    C, W1 = int(conv_shape[0]), int(conv_shape[1])
    T = int(num_draft_tokens)
    NW = T + W1 - 1
    s1 = NW * C
    s0 = rows * s1
    return (num_layers, rows, NW, C), (num_layers, rows, T, C, W1), (s0, s1, C, 1, C)


def stock_geometry(conv_shape, num_layers, rows, num_draft_tokens):
    """The stock dedup geometry for a (C, W-1) conv shape (window axis last), same convention."""
    C, W1 = int(conv_shape[0]), int(conv_shape[1])
    T = int(num_draft_tokens)
    NW = T + W1 - 1
    s1 = C * NW
    s0 = rows * s1
    return (num_layers, rows, C, NW), (num_layers, rows, T, C, W1), (s0, s1, 1, NW, 1)


def _a1_active():
    if os.environ.get("SGLANG_A1_SSM_PTR", "") == "0":
        return False
    import importlib.util

    try:
        return importlib.util.find_spec("sglang.srt.mem_cache.gdnk_a1") is not None
    except (ImportError, ValueError):
        return False


def unsupported_reason(pool, conv_shape, conv_dtype, num_draft_tokens):
    """None when the p-major layout is covered by the convp check, else a short reason (-> stock layout)."""
    import torch

    if not torch.cuda.is_available() or getattr(torch.version, "hip", None):
        return "not a CUDA (non-ROCm) device"
    if len(conv_shape) != 2:
        return f"conv shape {tuple(conv_shape)} is not 2-D"
    axis = getattr(pool, "conv_window_axis", -1) % 2
    if axis != 1:
        return f"window axis {axis} is not the last axis of conv shape {tuple(conv_shape)} (stock is already p-major)"
    if conv_dtype != torch.bfloat16:
        return f"conv dtype {conv_dtype} != bfloat16 (only bf16 is verified)"
    if int(num_draft_tokens) < 1:
        return f"num_draft_tokens {num_draft_tokens} < 1"
    if os.environ.get("SGL_A2_FUSED") == "1":
        return "SGL_A2_FUSED=1 (the fused conv+verify consumer is not covered by the convp check)"
    if _a1_active():
        return "A1 (gdnk_a1) is installed and active (its commit is not covered by the convp check)"
    return None


def allocate_pmajor(pool, *, conv_shape, num_mamba_layers, spec_state_size, speculative_num_draft_tokens,
                    conv_dtype):
    """Drop-in for MambaPool._allocate_deduplicated_conv_window: (phys, view) in the p-major layout, or None."""
    try:
        import torch

        why = unsupported_reason(pool, conv_shape, conv_dtype, speculative_num_draft_tokens)
        if why:
            return _fallback(why)
        rows = spec_state_size + 1
        pshape, vshape, vstrides = pmajor_geometry(conv_shape, num_mamba_layers, rows, speculative_num_draft_tokens)
        phys = torch.zeros(pshape, dtype=conv_dtype, device="cuda")
        if tuple(phys.stride()) != (vstrides[0], vstrides[1], vstrides[2], 1):
            return _fallback(f"unexpected phys strides {tuple(phys.stride())}")
        view = phys.as_strided(vshape, vstrides)
        # invariants every stride-driven consumer relies on (same as the stock dedup view):
        C, W1, T, NW = vshape[3], vshape[4], vshape[2], pshape[2]
        if not (view.stride(2) == view.stride(4) and view.stride(3) == 1 and view.stride(2) == C
                and (T - 1) * view.stride(2) + (C - 1) + (W1 - 1) * view.stride(4) == NW * C - 1
                and tuple(view.shape) == vshape and view.data_ptr() == phys.data_ptr()):
            return _fallback(f"view self-check failed (strides {view.stride()})")
        STATE["on"].append((pshape, vstrides))
        _log(f"ON: dedup conv window phys {list(pshape)} (C contiguous), view {list(vshape)} strides {list(vstrides)}")
        return phys, view
    except Exception as e:  # noqa: BLE001  never raise: the stock allocation runs instead
        return _fallback(f"{type(e).__name__}: {str(e)[:200]}")
