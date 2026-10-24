"""apply_foldsplit.py - kopt-foldx (class A): the braw GDN fold without the per-token track store in every CTA.

Production (kern_shim a2b, extra_buffer scheduling) passes mamba_track_indices/steps on every commit -> HAS_TRACK=True
-> every CTA's token loop carries the conditional bf16 track store -> fold ~3.85 ms (B64, accept 5.4, 4 track rows)
vs 2.94 ms for the same kernel built without it. With SGL_FOLD_X=1 the wrapper runs gdn_b_fold_x_kernel twice:
MODE 1 (grid V/BV x MAXTR x L*HV, only the tracked rows, picked on device: argsort + sum, no host sync) writes the
track slots from the OLD state, then MODE 0 (every row, no store in the loop) writes the final state. The arithmetic
line is the production text `S = S * tl.exp(gg) + vn[:, None] * kk[None, :]` (k4d: bitwise = production incl. track
slots; an explicit fma(vn,kk,S*e) is NOT). A CTA-level split inside ONE kernel was not bitwise (k4c) - do not merge.

Patches the braw kernel SOURCE (default /opt/kern/braw/kernels.py; apply_b.py copies it into sglang at every launch):
  python apply_foldsplit.py [--file F]            apply (idempotent; backup F.kopt_foldx.orig)
  python apply_foldsplit.py [--file F] --check    edit present + compiles; with a GPU: bitwise self-test vs the
                                                  backup (track rows 0/4/64, eager + graph) and timing
  python apply_foldsplit.py [--file F] --revert   restore the backup
Runtime: SGL_FOLD_X=1 (read at each fold launch; unset/0 = the production kernel); SGL_FOLD_X_MAXTR (8).
"""
import hashlib
import importlib.util
import os
import shutil
import sys

MARK = "# kopt foldx"
ANCHOR_HOST = "# ============================================================================================== host wrappers\n"
NEW_KERNEL = """@triton.jit
def gdn_b_fold_x_kernel(  """ + MARK + """: fold without the in-loop track store + a separate track pass (same arithmetic)
    h, stash, state_idx, last_step, track_idx, track_step, order, ntrack,
    stride_h_layer, stride_h_slot, stride_s_layer, stride_s_row,
    TS: tl.constexpr, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BV: tl.constexpr,
    NULL_ID: tl.constexpr, MODE: tl.constexpr, MAXTR: tl.constexpr,
):
    # MODE 0: grid (V/BV, B, L*HV), every row, final state only (= gdn_b_fold_kernel with HAS_TRACK False).
    # MODE 1: grid (V/BV, MAXTR, L*HV), rows order[j], order[j+MAXTR], ... < ntrack (tracked rows first in order):
    #         state after track_step+1 tokens -> slot track_idx. Must run BEFORE mode 0 (reads the old state).
    i_v = tl.program_id(0)
    i_j = tl.program_id(1)
    i_lh = tl.program_id(2)
    i_l = (i_lh // HV).to(tl.int64)
    i_hv = i_lh % HV
    i_h = i_hv // (HV // H)
    offs_k = tl.arange(0, K)
    offs_v = i_v * BV + tl.arange(0, BV)
    mask_v = offs_v < V
    m_h = mask_v[:, None] & (offs_k[None, :] < K)
    hb = h + i_l * stride_h_layer + i_hv * V * K + offs_v[:, None] * K + offs_k[None, :]
    if MODE == 0:
        i_n = i_j
        idx = tl.load(state_idx + i_n).to(tl.int64)
        if idx <= NULL_ID:
            return
        n = tl.load(last_step + i_n).to(tl.int32) + 1
        if n <= 0:
            return
        S = tl.load(hb + idx * stride_h_slot, mask=m_h, other=0.0).to(tl.float32)
        sb = stash + i_l * stride_s_layer + i_n.to(tl.int64) * stride_s_row
        p_vn = sb + i_hv * TS * V + offs_v
        p_k = sb + HV * TS * V + i_h * TS * K + offs_k
        p_g = sb + HV * TS * V + H * TS * K + i_hv * TS
        for t in tl.static_range(TS):
            if t < n:
                vn = tl.load(p_vn + t * V, mask=mask_v, other=0.0)
                kk = tl.load(p_k + t * K)
                gg = tl.load(p_g + t)
                S = S * tl.exp(gg) + vn[:, None] * kk[None, :]
        tl.store(hb + idx * stride_h_slot, S.to(h.dtype.element_ty), mask=m_h)
    else:
        nt = tl.load(ntrack)
        for r in range(i_j, nt, MAXTR):
            i_n = tl.load(order + r)
            idx = tl.load(state_idx + i_n).to(tl.int64)
            n = tl.load(last_step + i_n).to(tl.int32) + 1
            tr_idx = tl.load(track_idx + i_n).to(tl.int64)
            tr_step = tl.load(track_step + i_n).to(tl.int32)
            if (idx > NULL_ID) & (n > 0) & (tr_idx > NULL_ID):
                S = tl.load(hb + idx * stride_h_slot, mask=m_h, other=0.0).to(tl.float32)
                sb = stash + i_l * stride_s_layer + i_n.to(tl.int64) * stride_s_row
                p_vn = sb + i_hv * TS * V + offs_v
                p_k = sb + HV * TS * V + i_h * TS * K + offs_k
                p_g = sb + HV * TS * V + H * TS * K + i_hv * TS
                for t in tl.static_range(TS):
                    if (t < n) & (t <= tr_step):   # tokens after the track step are not needed here
                        vn = tl.load(p_vn + t * V, mask=mask_v, other=0.0)
                        kk = tl.load(p_k + t * K)
                        gg = tl.load(p_g + t)
                        S = S * tl.exp(gg) + vn[:, None] * kk[None, :]
                        if t == tr_step:
                            tl.store(hb + tr_idx * stride_h_slot, S.to(h.dtype.element_ty), mask=m_h)


"""
ANCHOR_LAUNCH = "    gdn_b_fold_kernel[(triton.cdiv(V, BV), B, L * HV)](\n"
NEW_LAUNCH = """    if has_track and stash_rows is None and os.environ.get("SGL_FOLD_X", "0") == "1":  """ + MARK + """
        if not globals().get("_kopt_foldx_logged"):
            globals()["_kopt_foldx_logged"] = True
            print("[kopt-foldx] ON (SGL_FOLD_X=1): fold = track pass (tracked rows only) + main pass without the "
                  "in-loop track store", flush=True)
        grid0 = (triton.cdiv(V, BV), B, L * HV)
        common = (temporal.stride(0), temporal.stride(1), st.stride(0), st.stride(1))
        # a row stores a track slot only if (slot valid) & (state valid) & (0 <= track_step <= last_step); production
        # extra_buffer gives EVERY request a track slot and marks "no crossing this step" with step -1 (fxe 21/10)
        tracked = ((track_idx[:B] > null_id) & (state_idx[:B] > null_id) & (track_step[:B] >= 0)
                   & (track_step[:B] <= last_step[:B]))
        order = torch.argsort((~tracked).to(torch.int8), stable=True).to(torch.int32)   # tracked rows first, no sync
        ntrack = tracked.sum().to(torch.int32).reshape(1)
        maxtr = int(os.environ.get("SGL_FOLD_X_MAXTR", "8"))
        gdn_b_fold_x_kernel[(triton.cdiv(V, BV), maxtr, L * HV)](
            temporal, st, state_idx, last_step, track_idx, track_step, order, ntrack, *common,
            TS=TS, H=num_k_heads, HV=HV, K=K, V=V, BV=BV, NULL_ID=null_id, MODE=1, MAXTR=maxtr,
            num_warps=num_warps, num_stages=1)
        gdn_b_fold_x_kernel[grid0](
            temporal, st, state_idx, last_step, track_idx, track_step, order, ntrack, *common,
            TS=TS, H=num_k_heads, HV=HV, K=K, V=V, BV=BV, NULL_ID=null_id, MODE=0, MAXTR=maxtr,
            num_warps=num_warps, num_stages=1)
        return
"""
EDITS = [(ANCHOR_HOST, NEW_KERNEL + ANCHOR_HOST), (ANCHOR_LAUNCH, NEW_LAUNCH + ANCHOR_LAUNCH)]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


F = arg("--file", "/opt/kern/braw/kernels.py")
BAK = F + ".kopt_foldx.orig"


def apply():
    src = open(F).read()
    if MARK in src:
        print(f"foldsplit: already applied to {F}")
        return 0
    for old, _ in EDITS:
        if src.count(old) != 1:
            print(f"foldsplit !! anchor not found exactly once in {F} (nothing written): {old.splitlines()[0][:80]!r}")
            return 1
    new = src
    for old, rep in EDITS:
        new = new.replace(old, rep)
    compile(new, F, "exec")
    shutil.copy2(F, BAK)
    tmp = F + ".kopt_foldx_tmp"
    open(tmp, "w").write(new)
    os.replace(tmp, F)
    print(f"foldsplit: applied to {F} (backup {BAK}, sha256 before {sha(BAK)[:12]})")
    return 0


def revert():
    if not os.path.exists(BAK):
        if MARK in open(F).read():
            print(f"foldsplit !! {F} patched but no backup")
            return 8
        print("foldsplit: not applied")
        return 0
    if MARK not in open(F).read():
        print(f"foldsplit !! {F} changed by someone else after apply (marker gone); backup kept")
        return 7
    os.replace(BAK, F)
    print(f"foldsplit: reverted {F} (sha256 {sha(F)[:12]})")
    return 0


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def check():
    src = open(F).read()
    if src.count(MARK) != 2 or not os.path.exists(BAK):
        print(f"foldsplit check: FAIL - not applied / incomplete ({src.count(MARK)}/2 markers, backup "
              f"{os.path.exists(BAK)})")
        return 1
    compile(src, F, "exec")
    try:
        import torch
        if not torch.cuda.is_available():
            raise ImportError
    except ImportError:
        print("foldsplit check: PASS (compile only, no GPU)")
        return 0
    import triton
    import tempfile
    d = tempfile.mkdtemp()
    shutil.copy2(BAK, os.path.join(d, "fs_orig.py"))
    shutil.copy2(F, os.path.join(d, "fs_new.py"))
    KO, KN = load("fs_orig", os.path.join(d, "fs_orig.py")), load("fs_new", os.path.join(d, "fs_new.py"))
    dev, bf = torch.device("cuda"), torch.bfloat16
    L, B, HV, H, K, V, T = int(os.environ.get("FOLDX_CHECK_L", "4")), 64, 48, 16, 128, 128, 8
    NF = KO.stash_floats(H, HV, K, V, T)
    NSLOT = 3 * B + 8
    torch.manual_seed(0)
    pool0 = (torch.randn(L, NSLOT, HV, V, K, device=dev) * 0.05).to(bf)
    inter = torch.zeros(L, B + 1, T, HV, V, K, device=dev, dtype=bf)
    st = inter.view(torch.float32).reshape(L, B + 1, -1)
    st[:, :, :HV * T * V] = torch.randn(L, B + 1, HV * T * V, device=dev) * 0.3
    kk = torch.randn(L, B + 1, H * T, K, device=dev)
    st[:, :, HV * T * V:HV * T * V + H * T * K] = (kk / kk.norm(dim=-1, keepdim=True)).reshape(L, B + 1, -1)
    st[:, :, HV * T * V + H * T * K:NF] = -torch.rand(L, B + 1, HV * T, device=dev) * 0.5
    g = torch.Generator().manual_seed(1)
    perm = torch.randperm(NSLOT, generator=g).to(torch.int32)
    slots = perm[:B].clone().to(dev)
    nacc = 1 + torch.binomial(torch.full((B,), 7.0), torch.full((B,), 0.65), generator=g).long()
    last = (nacc - 1).to(torch.int32).to(dev)
    last[3] = -1
    slots[5] = -1
    pool = pool0.clone()
    ok, rows = True, []
    for ntr, allslot in ((0, False), (4, False), (B, False), (2, True), (0, True)):
        # allslot = production extra_buffer shape: every row has a track slot, step -1 = no crossing; plus rows whose
        # step is >= accepted (never reached -> no store)
        tri = torch.full((B,), -1, dtype=torch.int32)
        trs = torch.full((B,), -1, dtype=torch.int32)
        if allslot:
            tri[:] = perm[B:2 * B]
            for rr in (7, 9):
                trs[rr] = int(nacc[rr]) + (rr - 7) // 2
        for j, rr in enumerate(torch.randperm(B, generator=g)[:ntr].tolist()):
            tri[rr] = int(perm[B + j]) if not allslot else tri[rr]
            trs[rr] = int(torch.randint(0, int(nacc[rr]), (1,), generator=g))
        tri, trs = tri.to(dev), trs.to(dev)

        def run(M):
            M.gdn_b_fold_all_layers(pool, inter, slots, last, tri, trs, num_k_heads=H)
        os.environ["SGL_FOLD_X"] = "0"
        pool.copy_(pool0); run(KO); torch.cuda.synchronize(); ref = pool.clone()
        res = {}
        for sp in ("0", "1"):
            os.environ["SGL_FOLD_X"] = sp
            pool.copy_(pool0); run(KN); torch.cuda.synchronize()
            e = torch.equal(pool.view(torch.int16), ref.view(torch.int16))
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                run(KN)
            torch.cuda.current_stream().wait_stream(s)
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                run(KN)
            pool.copy_(pool0); gr.replay(); torch.cuda.synchronize()
            gq = torch.equal(pool.view(torch.int16), ref.view(torch.int16))
            del gr
            ms = triton.testing.do_bench(lambda: run(KN), warmup=5, rep=50, return_mode="median")
            res[sp] = (e, gq, ms)
            ok = ok and e and gq
        keep = st[:, :, 0].clone()
        st[:, :, 0] += 1e-6
        os.environ["SGL_FOLD_X"] = "1"
        pool.copy_(pool0); run(KN); torch.cuda.synchronize()
        caught = not torch.equal(pool.view(torch.int16), ref.view(torch.int16))
        st[:, :, 0] = keep
        ok = ok and caught
        rows.append(f"track rows {ntr:3d}{' (all rows have slots, prod shape)' if allslot else ''}: X=0 bitwise eager {res['0'][0]} graph {res['0'][1]} {res['0'][2]:.3f} ms | "
                    f"X=1 bitwise eager {res['1'][0]} graph {res['1'][1]} {res['1'][2]:.3f} ms | perturbed caught {caught}")
    os.environ.pop("SGL_FOLD_X", None)
    for r in rows:
        print("  " + r)
    print(f"foldsplit check: {'PASS' if ok else 'FAIL'} (L {L} B {B}, accept mean {float(nacc.float().mean()):.2f}, "
          f"GPU {torch.cuda.get_device_name()})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(check() if "--check" in sys.argv else revert() if "--revert" in sys.argv else apply())
