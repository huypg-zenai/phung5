"""kopt-convp GPU check (run by `apply_convp.py --check`, or directly with the patched sglang on PYTHONPATH).

Stock vs convp dedup conv-window buffers, built by the PATCHED SGLang allocation function
(MambaPool._allocate_deduplicated_conv_window with SGL_CONVP=0 / 1) and, when it constructs, by a real MambaPool.
On identical inputs, for several shapes (production dims C 10240 B 64 T 8 included), every combination of
  conv  : the installed gdn_backend.causal_conv1d_update (= the a2 production wrapper -> gdnk_a2.a2_conv_verify_kernel)
          and the stock Triton causal_conv1d_update (what the wrapper falls back to)
  commit: the installed hybrid_linear_attn_backend.scatter_mamba_states_after_mtp_verify (stock, or the gdnk-b
          wrapper -> stock when no B verify ran), fused_conv_window_scatter_with_mask (the gdnk-b / ReplaySSM conv
          commit) and fused_conv_window_scatter_multi; accept rows AND track rows (mamba extra_buffer)
runs several verify+commit rounds eager, then captures one round per arm in a CUDA graph and replays it with new
inputs, comparing bitwise: conv outputs, conv_state pool, temporal pool, the window buffer (every physical element,
in a canonical order) and the logical window view. Real pool: also copy_from / clear_slots.
Negative controls (must be CAUGHT): a commit that reads the p-major buffer with stock strides, a view with a wrong
step stride, a 1-ulp flip in the p-major buffer. Gate controls: SGL_CONVP=0 -> stock phys shape; fallback cases.
Last line: 'convp check: PASS ...' (exit 0) or 'convp check: FAIL ...' (exit 1).
Options: --allow-no-a2 (do not FAIL when the a2 production conv is not installed), --quick (fewer shapes / rounds).
"""
import argparse
import os
import sys
import traceback
import types

import torch

P = lambda *a: print(*a, flush=True)  # noqa: E731
FAILS, NOTES = [], []


def eq(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    a, b = a.contiguous(), b.contiguous()
    if a.dtype in (torch.bfloat16, torch.float16):
        return torch.equal(a.view(torch.int16), b.view(torch.int16))
    if a.dtype == torch.float32:
        return torch.equal(a.view(torch.int32), b.view(torch.int32))
    return torch.equal(a, b)


def capture(fn):
    fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return g


ap = argparse.ArgumentParser()
ap.add_argument("--allow-no-a2", action="store_true")
ap.add_argument("--quick", action="store_true")
ARGS = ap.parse_args()

dev, bf = torch.device("cuda"), torch.bfloat16
import sglang.srt.mem_cache.memory_pool as mp  # noqa: E402
from sglang.srt.mem_cache import kopt_convp as kc  # noqa: E402

P(f"convp check: torch {torch.__version__} gpu {torch.cuda.get_device_name()}; memory_pool {mp.__file__}")
import inspect  # noqa: E402

if "# kopt convp:alloc" not in inspect.getsource(mp.MambaPool._allocate_deduplicated_conv_window):
    P("convp check: FAIL - the imported MambaPool._allocate_deduplicated_conv_window has no kopt convp hook")
    sys.exit(1)

# ------------------------------------------------------------------ consumers as the server resolves them
import sglang.srt.layers.attention.linear.gdn_backend as gb  # noqa: E402  (runs the a2 / braw installs if applied)
import sglang.srt.layers.attention.hybrid_linear_attn_backend as hlab  # noqa: E402
from sglang.kernels.ops.mamba import mamba_state_scatter_triton as mst  # noqa: E402
from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update as stock_conv_fn  # noqa: E402

conv_installed = gb.causal_conv1d_update
A2 = bool(getattr(conv_installed, "_gdnk_a2", False))
K2 = None
if A2:
    from sglang.kernels.ops.attention.fla import gdnk_a2 as K2  # noqa: E402
conv_stock = getattr(conv_installed, "_gdnk_a2_orig", stock_conv_fn)
scatter_installed = hlab.scatter_mamba_states_after_mtp_verify
BRAW = bool(getattr(hlab, "_gdnk_b", False))
P(f"consumers: conv installed = {'a2 wrapper (gdnk_a2)' if A2 else 'stock (a2 NOT installed)'}; commit installed = "
  f"{'gdnk-b wrapper (stock path until a B verify ran; its conv commit = fused_conv_window_scatter_with_mask)' if BRAW else 'stock'}"
  f"; SGL_A2_FUSED={os.environ.get('SGL_A2_FUSED', '')!r}")
if not A2:
    msg = "a2 production conv wrapper not installed in this sglang (run /opt/kern/a2/apply_a2.py first)"
    (NOTES if ARGS.allow_no_a2 else FAILS).append(msg)
CONVS = [("a2", conv_installed)] if A2 else []
CONVS.append(("stock", conv_stock))


def alloc(conv_shape, L, R, T, convp, axis=1, dtype=bf):
    old = os.environ.get("SGL_CONVP")
    os.environ["SGL_CONVP"] = "1" if convp else "0"
    try:
        ns = types.SimpleNamespace(conv_window_axis=axis)
        return mp.MambaPool._allocate_deduplicated_conv_window(
            ns, conv_shape=conv_shape, num_mamba_layers=L, spec_state_size=R - 1, speculative_num_draft_tokens=T,
            conv_dtype=dtype)
    finally:
        if old is None:
            os.environ.pop("SGL_CONVP", None)
        else:
            os.environ["SGL_CONVP"] = old


# ------------------------------------------------------------------ gate controls
def gate_controls():
    C, L, R, T = 640, 2, 4, 8
    ps, vs = alloc((C, 3), L, R, T, False)
    pp, vp = alloc((C, 3), L, R, T, True)
    ok = (tuple(ps.shape) == (L, R, C, T + 2) and tuple(pp.shape) == (L, R, T + 2, C) and vs.shape == vp.shape
          and tuple(vp.stride()) == kc.pmajor_geometry((C, 3), L, R, T)[2]
          and tuple(vs.stride()) == kc.stock_geometry((C, 3), L, R, T)[2])
    P(f"  gate: SGL_CONVP=0 phys {list(ps.shape)} strides {list(vs.stride())} | SGL_CONVP=1 phys {list(pp.shape)} "
      f"strides {list(vp.stride())} -> {'OK' if ok else 'WRONG'}")
    if not ok:
        FAILS.append("gate: SGL_CONVP 0/1 did not give the stock / p-major layouts; last fallback reasons: "
                     + "; ".join(kc.STATE["fallback"][-3:]))
    cases = [("window axis first (3, C)", dict(conv_shape=(3, C), axis=0)),
             ("fp32 conv dtype", dict(conv_shape=(C, 3), dtype=torch.float32)),
             ("SGL_A2_FUSED=1", dict(conv_shape=(C, 3), env=("SGL_A2_FUSED", "1")))]
    for name, c in cases:
        env = c.pop("env", None)
        old = os.environ.get(env[0]) if env else None
        if env:
            os.environ[env[0]] = env[1]
        n0 = len(kc.STATE["fallback"])
        try:
            ph, _ = alloc(c["conv_shape"], L, R, T, True, axis=c.get("axis", 1), dtype=c.get("dtype", bf))
            st, _ = alloc(c["conv_shape"], L, R, T, False, axis=c.get("axis", 1), dtype=c.get("dtype", bf))
        finally:
            if env:
                if old is None:
                    os.environ.pop(env[0], None)
                else:
                    os.environ[env[0]] = old
        fb = len(kc.STATE["fallback"]) > n0
        good = fb and tuple(ph.shape) == tuple(st.shape) and ph.stride() == st.stride()
        P(f"  gate fallback '{name}': stock layout {tuple(ph.shape) == tuple(st.shape)}, logged {fb} -> "
          f"{'OK' if good else 'WRONG'}")
        if not good:
            FAILS.append(f"gate fallback '{name}' did not give the stock layout")


# ------------------------------------------------------------------ the verify + commit simulation
class Arm:
    def __init__(self, name, phys, view, conv, temporal, issm, B, T, C, L, commit_view=None):
        self.name, self.phys, self.view, self.conv, self.temporal, self.issm = name, phys, view, conv, temporal, issm
        self.commit_view = view if commit_view is None else commit_view
        self.out = torch.zeros(L, B, C, T, device=dev, dtype=bf)
        self.layer_cache = None   # real pool: (pool) -> per-layer views via mamba2_layer_cache

    def snap(self):
        return dict(out=self.out.clone(), conv=self.conv.clone(), temporal=self.temporal.clone(),
                    window=(self.phys.transpose(-1, -2) if self.pmajor() else self.phys).clone(),
                    logical=self.view.contiguous().clone())

    def pmajor(self):
        return self.phys.stride(-1) == 1 and self.view.stride(3) == 1


class Sim:
    """Shared inputs + one commit/conv combo; arms are stepped with identical inputs."""

    def __init__(self, C, B, T, L, R, S, seed):
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.C, self.B, self.T, self.L, self.R, self.S = C, B, T, L, R, S
        self.x = torch.zeros(L, B * T, C, device=dev, dtype=bf)
        self.w = [(torch.randn(C, 4, generator=g) * 0.35).to(bf).to(dev) for _ in range(L)]
        perm = torch.randperm(S - 1, generator=g)[: 2 * B] + 1
        self.slots = perm[:B].to(torch.int32).to(dev)
        self.track = torch.zeros(B, dtype=torch.int32, device=dev)
        self.last = torch.zeros(B, dtype=torch.int32, device=dev)
        self.tsteps = torch.zeros(B, dtype=torch.int32, device=dev)
        self.track_pool = perm[B:2 * B].to(torch.int32)
        self.rows = torch.arange(B, dtype=torch.int32, device=dev)
        self.g = g
        HV, V, K = 2, 32, 32
        self.init = dict(conv=torch.randn(L, S, C, 3, generator=g).to(bf).to(dev),
                         temporal=torch.randn(L, S, HV, V, K, generator=g).to(bf).to(dev),
                         issm=torch.randn(L, R, T, HV, V, K, generator=g).to(bf).to(dev))

    def new_inputs(self):
        g, B, T = self.g, self.B, self.T
        self.x.copy_((torch.randn(self.L, B * T, self.C, generator=g)).to(bf))
        self.last.copy_(torch.randint(-1, T, (B,), generator=g).to(torch.int32))
        tr = self.track_pool.clone()
        tr[torch.rand(B, generator=g) < 0.3] = -1
        self.track.copy_(tr)
        self.tsteps.copy_(torch.randint(-1, T, (B,), generator=g).to(torch.int32))

    def arm(self, name, phys, view, commit_view=None):
        a = Arm(name, phys, view, self.init["conv"].clone(), self.init["temporal"].clone(),
                self.init["issm"].clone(), self.B, self.T, self.C, self.L, commit_view)
        return a

    def reset(self, a):
        a.conv.copy_(self.init["conv"])
        a.temporal.copy_(self.init["temporal"])
        a.issm.copy_(self.init["issm"])
        a.phys.zero_()
        a.out.zero_()

    def step(self, a, conv_fn, commit):
        B, T, C = self.B, self.T, self.C
        for l in range(self.L):
            if a.layer_cache is not None:        # real pool: exactly what gdn_backend.forward_extend reads
                lc = a.layer_cache(l)
                cs, win = lc.conv[0], lc.intermediate_conv_window[0]
            else:
                cs, win = a.conv[l], a.view[l]
            o = conv_fn(self.x[l].view(B, T, C).transpose(1, 2), cs, self.w[l], None, "silu",
                        conv_state_indices=self.slots, intermediate_conv_window=win,
                        intermediate_state_indices=self.rows)
            a.out[l].copy_(o)
        cv = a.commit_view
        if commit == "installed":
            caches = a.caches if getattr(a, "caches", None) is not None else types.SimpleNamespace(
                temporal=a.temporal, intermediate_ssm=a.issm, conv=[a.conv], intermediate_conv_window=[cv])
            scatter_installed(caches, self.slots, self.last, self.track, self.tsteps)
        elif commit == "with_mask":
            mst.fused_conv_window_scatter_with_mask(a.conv, cv, self.slots, self.last)
            mst.fused_conv_window_scatter_with_mask(a.conv, cv, self.track, self.tsteps)
        elif commit == "multi":
            mst.fused_conv_window_scatter_multi([(a.conv, cv)], self.slots, self.last, self.track, self.tsteps)
        else:
            raise ValueError(commit)


def compare(ra, rb):
    return {k: eq(ra[k], rb[k]) for k in ("out", "conv", "temporal", "window", "logical")}


def fmt(ok):
    return " ".join(f"{k}={'Y' if v else 'N'}" for k, v in ok.items())


KEEP = []   # keeps the fused_conv_window_scatter_multi meta tensors alive for captured graphs


def run_combo(tag, sim, arm_s, arm_p, conv_name, conv_fn, commit, n_eager, n_graph):
    good = True
    for a in (arm_s, arm_p):
        sim.reset(a)
    if K2 is not None:
        K2.LAST_COMPILED.pop("conv", None)
    for r in range(n_eager):
        sim.new_inputs()
        sim.step(arm_s, conv_fn, commit)
        sim.step(arm_p, conv_fn, commit)
        torch.cuda.synchronize()
        ok = compare(arm_s.snap(), arm_p.snap())
        good &= all(ok.values())
        if not all(ok.values()) or r == n_eager - 1:
            P(f"    [{tag} conv={conv_name} commit={commit} eager r{r}] {fmt(ok)}")
    if conv_name == "a2" and K2 is not None and "conv" not in K2.LAST_COMPILED:
        FAILS.append(f"{tag}: the a2 wrapper did not take the a2 kernel path (fell back to stock)")
        good = False
    if n_graph:
        graphs = []
        for a in (arm_s, arm_p):
            sim.reset(a)
            sim.new_inputs()
            graphs.append(capture(lambda a=a: sim.step(a, conv_fn, commit)))
            KEEP.append(list(mst._conv_multi_meta_cache.values()))
        for a in (arm_s, arm_p):
            sim.reset(a)
        for r in range(n_graph):
            sim.new_inputs()
            graphs[0].replay()
            graphs[1].replay()
            torch.cuda.synchronize()
            ok = compare(arm_s.snap(), arm_p.snap())
            good &= all(ok.values())
            if not all(ok.values()) or r == n_graph - 1:
                P(f"    [{tag} conv={conv_name} commit={commit} graph r{r}] {fmt(ok)}")
        del graphs
    if not good:
        FAILS.append(f"{tag} conv={conv_name} commit={commit}: stock vs convp NOT bitwise identical")
    return good


def shape_section(tag, C, B, T, L, n_eager, n_graph, negatives):
    R, S = B + 3, 2 * B + 6
    sim = Sim(C, B, T, L, R, S, seed=C * 7 + B)
    ps, vs = alloc((C, 3), L, R, T, False)
    pp, vp = alloc((C, 3), L, R, T, True)
    if not (tuple(ps.shape) == (L, R, C, T + 2) and tuple(pp.shape) == (L, R, T + 2, C)):
        FAILS.append(f"{tag}: unexpected phys shapes stock {list(ps.shape)} convp {list(pp.shape)}; last fallback "
                     "reasons: " + "; ".join(kc.STATE["fallback"][-3:]))
        return
    if A2 and not (K2.window_mode(vs[0]) == 1 and K2.window_mode(vp[0]) == 1):
        FAILS.append(f"{tag}: a2 window_mode is not 1 (dedup) for both layouts")
    arm_s, arm_p = sim.arm("stock", ps, vs), sim.arm("convp", pp, vp)
    P(f"  == {tag}: C {C} B {B} T {T} L {L} rows {R} slots {S}")
    for conv_name, conv_fn in CONVS:
        for commit in ("installed", "with_mask", "multi"):
            try:
                run_combo(tag, sim, arm_s, arm_p, conv_name, conv_fn, commit, n_eager, n_graph)
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                FAILS.append(f"{tag} conv={conv_name} commit={commit}: crashed {type(e).__name__}: {str(e)[:200]}")
    if not negatives:
        return
    # ---- negative controls: each must be CAUGHT (some compared tensor differs from the stock arm)
    conv_name, conv_fn = CONVS[0]
    neg = []
    sv = kc.stock_geometry((C, 3), L, R, T)
    bad_reader = pp.as_strided(sv[1], sv[2])                         # a consumer assuming the stock strides
    neg.append(("commit reads p-major phys with stock strides", sim.arm("bad1", pp, vp, commit_view=bad_reader), None))
    pg = kc.pmajor_geometry((C, 3), L, R, T)
    bad_step = pp.as_strided(pg[1], (pg[2][0], pg[2][1], 1, 1, C))   # wrong step stride (1 instead of C)
    neg.append(("view with wrong step stride", sim.arm("bad2", pp, bad_step), None))
    neg.append(("1-ulp flip in the p-major buffer", sim.arm("bad3", pp, vp), "flip"))
    for name, bad, post in neg:
        for a in (arm_s, bad):
            sim.reset(a)
        sim.new_inputs()
        sim.step(arm_s, conv_fn, "installed")
        sim.step(bad, conv_fn, "installed")
        if post == "flip":
            t = int(sim.rows[0])
            idx = (0, t, 1, 5) if pp.shape[2] > 1 else (0, t, 0, 5)
            pp.view(torch.int16)[idx].add_(1)
        torch.cuda.synchronize()
        ok = compare(arm_s.snap(), bad.snap())
        caught = not all(ok.values())
        P(f"    [negative '{name}'] {fmt(ok)} -> {'CAUGHT' if caught else 'NOT CAUGHT'}")
        if not caught:
            FAILS.append(f"{tag}: negative control '{name}' was NOT caught")


def real_pool_section(n_eager):
    """Real MambaPool (stock vs SGL_CONVP=1) with Qwen-like GDN dims (scaled down)."""
    try:
        from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateDType, Mamba2StateShape
        H, HV, K, V, L, T, B = 2, 4, 128, 128, 3, 8, 6
        pools = []
        for convp in (False, True):
            os.environ["SGL_CONVP"] = "1" if convp else "0"
            shape = Mamba2StateShape.create(tp_world_size=1, intermediate_size=HV * V, n_groups=H, num_heads=HV,
                                            head_dim=V, state_size=K, conv_kernel=4,
                                            conv_shard_groups=[H * K, H * K, HV * V])
            cp = Mamba2CacheParams(shape=shape, layers=list(range(L)),
                                   dtype=Mamba2StateDType(conv=bf, temporal=bf))
            pools.append(mp.MambaPool(size=2 * B + 5, spec_state_size=B + 2, cache_params=cp,
                                      mamba_layer_ids=list(range(L)), device="cuda", speculative_num_draft_tokens=T,
                                      speculative_eagle_topk=None))
        os.environ.pop("SGL_CONVP", None)
    except Exception as e:  # noqa: BLE001
        os.environ.pop("SGL_CONVP", None)
        traceback.print_exc()
        NOTES.append(f"real-pool test SKIPPED (MambaPool construction failed: {type(e).__name__}: {str(e)[:160]})")
        P(f"  real MambaPool: SKIPPED - construction failed: {type(e).__name__}: {e}")
        return
    ps_, pp_ = pools
    phs, php = ps_._intermediate_conv_window_phys[0], pp_._intermediate_conv_window_phys[0]
    vs, vp = ps_.mamba_cache.intermediate_conv_window[0], pp_.mamba_cache.intermediate_conv_window[0]
    C = ps_.mamba_cache.conv[0].shape[2]
    lay = (tuple(phs.shape) == (L, B + 3, C, T + 2) and tuple(php.shape) == (L, B + 3, T + 2, C)
           and vs.shape == vp.shape and abs(ps_.mem_usage - pp_.mem_usage) < 1e-12)
    P(f"  == real MambaPool (C {C}, L {L}, T {T}): stock phys {list(phs.shape)} convp phys {list(php.shape)}; "
      f"view {list(vp.shape)} strides {list(vp.stride())}; mem_usage {ps_.mem_usage:.6f} / {pp_.mem_usage:.6f} GB "
      f"-> {'OK' if lay else 'WRONG'}")
    if not lay:
        FAILS.append("real pool: layouts / mem_usage not as expected")
        return
    S, R = ps_.mamba_cache.conv[0].shape[1], phs.shape[1]
    sim = Sim(C, B, T, L, R, S, seed=11)
    HVt = ps_.mamba_cache.temporal.shape[2:]
    sim.init["temporal"] = torch.randn(L, S, *HVt, generator=sim.g).to(bf).to(dev)
    sim.init["issm"] = torch.randn(L, R, T, *HVt, generator=sim.g).to(bf).to(dev)
    arms = []
    for pool, ph, v in ((ps_, phs, vs), (pp_, php, vp)):
        mc = pool.mamba_cache
        a = Arm("pool", ph, v, mc.conv[0], mc.temporal, mc.intermediate_ssm, B, T, C, L)
        a.layer_cache = pool.mamba2_layer_cache
        a.caches = pool.get_speculative_mamba2_params_all_layers()
        arms.append(a)
    for conv_name, conv_fn in CONVS:
        run_combo("real-pool", sim, arms[0], arms[1], conv_name, conv_fn, "installed", n_eager, 2)
    # copy / clear paths of the pool (radix COW / fork / free): conv + temporal only, compared bitwise
    src = sim.slots[:3].to(torch.int64)
    dst = sim.track_pool[:3].to(torch.int64).to(dev)
    for pool in (ps_, pp_):
        pool.copy_from(src, dst)
    torch.cuda.synchronize()
    ok_c = eq(ps_.mamba_cache.conv[0], pp_.mamba_cache.conv[0]) and eq(ps_.mamba_cache.temporal,
                                                                       pp_.mamba_cache.temporal)
    for pool in (ps_, pp_):
        pool.clear_slots(src)
    torch.cuda.synchronize()
    ok_z = eq(ps_.mamba_cache.conv[0], pp_.mamba_cache.conv[0]) and eq(ps_.mamba_cache.temporal,
                                                                       pp_.mamba_cache.temporal)
    P(f"    [real-pool copy_from] conv+temporal identical {ok_c} | [clear_slots] identical {ok_z}")
    if not (ok_c and ok_z):
        FAILS.append("real pool: copy_from / clear_slots results differ")


try:
    P("-- gate controls")
    gate_controls()
    P("-- stock vs convp (allocation function), every conv x commit path, eager + CUDA graph")
    n_e, n_g = (2, 2) if ARGS.quick else (3, 3)
    shapes = [("prod-dims", 10240, 64, 8, 3, True), ("small-B", 10240, 5, 8, 2, False),
              ("odd-C", 1000, 7, 8, 2, True), ("T3", 640, 3, 3, 2, False)]
    if ARGS.quick:
        shapes = shapes[:1] + shapes[2:3]
    for tag, C, B, T, L, negs in shapes:
        shape_section(tag, C, B, T, L, n_e, n_g, negs)
    P("-- real MambaPool")
    real_pool_section(n_e)
except Exception as e:  # noqa: BLE001
    traceback.print_exc()
    FAILS.append(f"check crashed: {type(e).__name__}: {str(e)[:300]}")

for f in FAILS:
    P(f"  !! {f}")
for n in NOTES:
    P(f"  note: {n}")
if FAILS:
    P(f"convp check: FAIL ({len(FAILS)} problem(s))")
    sys.exit(1)
P(f"convp check: PASS (stock vs SGL_CONVP=1 bitwise identical: conv {[c for c, _ in CONVS]} x commit "
  f"[installed, with_mask, multi], eager + CUDA graph; negative controls caught; gate + fallbacks OK"
  + (f"; {'; '.join(NOTES)}" if NOTES else "") + ")")
