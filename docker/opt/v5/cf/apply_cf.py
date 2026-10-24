"""apply_cf.py - kopt convp + foldx (kernel-optim -> pipeline-speed handovers 21/10, CLASS A = bitwise identical), one item
of /opt/v5/kopt_shim.sh behind ONE switch: SGL_CONVP_FOLDX=1 (image default 0 = off, nothing applied).
  python apply_cf.py            apply both (idempotent):
                                  foldx: text patch of the braw kernel SOURCE /opt/kern/braw/kernels.py (apply_foldsplit.py,
                                         backup kernels.py.kopt_foldx.orig), then /opt/kern/braw/apply_b.py re-copies it into
                                         sglang (kern_shim ran before v5, so the sglang copy is refreshed here)
                                  convp: hook in sglang srt/mem_cache/memory_pool.py + kopt_convp.py (apply_convp.py)
  python apply_cf.py --check    foldx GPU self-test (bitwise vs the unpatched kernel, eager + graph) + convp GPU bitwise
                                check (needs SGL_CONVP=1 SGL_GDN_B=1 A2_CONFIG); last line 'apply_cf check: PASS' or FAIL
  python apply_cf.py --revert   foldx revert + apply_b.py re-copy (sglang back to the production kernels.py) + convp revert
Runtime switches the shim exports when the check passed: SGL_CONVP=1 SGL_FOLD_X=1 (both read by the patched code; 0/unset =
production paths byte for byte). Logs at server start: '[kopt-convp] ON: ...' and '[kopt-foldx] ON (SGL_FOLD_X=1) ...'.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
KB = os.environ.get("CF_BRAW_DIR", "/opt/kern/braw")
FOLD = [PY, os.path.join(HERE, "apply_foldsplit.py"), "--file", os.path.join(KB, "kernels.py")]
APPLY_B = [PY, os.path.join(KB, "apply_b.py")]
CONVP = [PY, os.path.join(HERE, "apply_convp.py")]


def run(cmd, env=None):
    r = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, **(env or {})))
    out = (r.stdout + r.stderr).strip()
    print(out[-3000:], flush=True)
    return r.returncode, out


def apply():
    for c in (FOLD, APPLY_B, CONVP):
        rc, _ = run(c)
        if rc:
            print(f"apply_cf: !! {' '.join(c[1:2])} failed rc {rc}")
            return rc
    print("apply_cf: applied (foldx in kernels.py + sglang copy refreshed, convp hook)")
    return 0


def check():
    rc, out = run(FOLD + ["--check"])
    fold_ok = rc == 0 and "foldsplit check: PASS (L" in out   # the GPU self-test line, not the compile-only one
    rc2, out2 = run(CONVP + ["--check"], {"SGL_CONVP": "1", "SGL_GDN_B": os.environ.get("SGL_GDN_B", "1"),
                                           "A2_CONFIG": os.environ.get("A2_CONFIG", "/opt/kern/a2/a2_v.json")})
    convp_ok = rc2 == 0 and "check: PASS (hook + module current, modules import, GPU bitwise check PASS)" in out2
    ok = fold_ok and convp_ok
    print(f"apply_cf check: {'PASS' if ok else 'FAIL'} (foldx GPU self-test {'PASS' if fold_ok else 'FAIL'}, "
          f"convp GPU bitwise {'PASS' if convp_ok else 'FAIL'})")
    return 0 if ok else 1


def revert():
    rcs = [run(FOLD + ["--revert"])[0], run(APPLY_B)[0], run(CONVP + ["--revert"])[0]]
    print(f"apply_cf: revert rc foldx {rcs[0]} apply_b {rcs[1]} convp {rcs[2]}")
    return max(rcs)


if __name__ == "__main__":
    sys.exit(check() if "--check" in sys.argv else revert() if "--revert" in sys.argv else apply())
