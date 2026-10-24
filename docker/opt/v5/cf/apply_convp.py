"""Apply the kopt-convp patch (p-major physical layout of the deduplicated GDN conv-window buffer; see kopt_convp.py
and README.md) to the sglang that `import sglang` resolves to, in place, idempotent, atomic, sha256-recorded.

  python apply_convp.py                         patch (exit 0; an already-present identical hook is kept)
  python apply_convp.py --check                 hook + module present and current, patched files compile, the patched
                                                modules import, then the GPU bitwise check (convp_gpucheck.py, needs a
                                                CUDA GPU and the server env); last line 'apply_convp check: PASS ...'
                                                (exit 0) or '... FAIL ...' (exit != 0)
  python apply_convp.py --check --compile-only  hook + module present and compile (no sglang deps, no GPU)
  python apply_convp.py --revert                remove the hook + module; the file must come back to the sha256
                                                recorded before the first apply (else exit 7)
  python apply_convp.py --root DIR              operate on the package at DIR (the directory that contains srt/)
  --allow-no-a2 / --quick (check)               passed to convp_gpucheck.py

Runtime switch: SGL_CONVP=1, read once per allocation (= at MambaPool construction, i.e. server start). Unset/0 = the
hook does nothing and the stock allocation runs byte for byte. Unsupported cases fall back to the stock layout and log
one line '[kopt-convp] fallback: <reason> -> stock layout'; the ON case logs '[kopt-convp] ON: ...'.

Edits (marked '# kopt convp:alloc'):
  srt/mem_cache/memory_pool.py   MambaPool._allocate_deduplicated_conv_window: hook at the top of the body
  srt/mem_cache/kopt_convp.py    NEW: copy of kopt_convp.py
Record: srt/mem_cache/.kopt_convp_record.json (sha256 before/after) + memory_pool.py.kopt_convp.orig (backup).
Coexists with the kern shim patches (a2 / braw only touch gdn_backend.py and fla/gdnk_*.py), topk20 / topkbf /
normcfg / seed (other files) and slim (memory_pool.py, a different anchor: intermediate_ssm allocation).
"""
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TAG = "apply_convp"
MODULE_SRC = os.path.join(HERE, "kopt_convp.py")
CHECK_SRC = os.path.join(HERE, "convp_gpucheck.py")
MODULE_DST = "srt/mem_cache/kopt_convp.py"
RECORD = "srt/mem_cache/.kopt_convp_record.json"
BACKUP = ".kopt_convp.orig"
MP = "srt/mem_cache/memory_pool.py"
STOCK_SHA = "41e87128dbdf9178dad65f695e7a98428ecfeebc1e3407fee57df5067f4205c3"   # memory_pool.py @1cf2b8c54
MARK = "# kopt convp:alloc"

ANCHOR = ("    ) -> Tuple[torch.Tensor, torch.Tensor]:\n"
          "        window_axis = self.conv_window_axis % len(conv_shape)\n")
HOOK = (
    "    ) -> Tuple[torch.Tensor, torch.Tensor]:\n"
    f"        {MARK}  SGL_CONVP=1 -> window axis physically before C (kernel-optim/convp); else stock below\n"
    "        if os.environ.get(\"SGL_CONVP\", \"0\") == \"1\":\n"
    "            try:\n"
    "                from sglang.srt.mem_cache.kopt_convp import allocate_pmajor as _kopt_convp_alloc\n"
    "            except Exception as _kopt_convp_e:  # never raise: stock layout\n"
    "                print(f\"[kopt-convp] fallback: import failed ({_kopt_convp_e!r}) -> stock layout\", flush=True)\n"
    "                _kopt_convp_alloc = None\n"
    "            if _kopt_convp_alloc is not None:\n"
    "                _kopt_convp_r = _kopt_convp_alloc(\n"
    "                    self,\n"
    "                    conv_shape=conv_shape,\n"
    "                    num_mamba_layers=num_mamba_layers,\n"
    "                    spec_state_size=spec_state_size,\n"
    "                    speculative_num_draft_tokens=speculative_num_draft_tokens,\n"
    "                    conv_dtype=conv_dtype,\n"
    "                )\n"
    "                if _kopt_convp_r is not None:\n"
    "                    return _kopt_convp_r\n"
    "        window_axis = self.conv_window_axis % len(conv_shape)\n"
)
NEEDS = ["\nimport os\n", "    def _allocate_deduplicated_conv_window(\n"]   # must exist in the original file


def sha(s):
    return hashlib.sha256(s.encode() if isinstance(s, str) else s).hexdigest()


def find_root():
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        sys.exit(f"{TAG}: `import sglang` does not resolve (set PYTHONPATH or use --root)")
    return list(spec.submodule_search_locations)[0]


def read(p):
    with open(p) as fh:
        return fh.read()


def write_atomic(p, s):
    tmp = p + ".kopt_convp_tmp"
    with open(tmp, "w") as fh:
        fh.write(s)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, p)
    if p.endswith(".py"):
        pyc = os.path.join(os.path.dirname(p), "__pycache__")
        stem = os.path.basename(p)[:-3] + "."
        if os.path.isdir(pyc):
            for c in os.listdir(pyc):
                if c.startswith(stem):
                    try:
                        os.remove(os.path.join(pyc, c))
                    except OSError:
                        pass


def compile_err(text, name):
    try:
        compile(text, name, "exec")
        return None
    except SyntaxError as e:
        return f"{name}: {e}"


def die(msg, rc=1):
    print(f"{TAG}: !! {msg}")
    raise SystemExit(rc)


def load_record(root):
    p = os.path.join(root, RECORD)
    if os.path.exists(p):
        try:
            return json.loads(read(p))
        except ValueError:
            pass
    return {}


def hook_state(src):
    """'absent' | 'current' (exactly one verbatim hook, no other marker) | 'stale' (marker present, block differs)."""
    if MARK not in src:
        return "absent"
    if src.count(MARK) == 1 and src.count(HOOK) == 1:
        return "current"
    return "stale"


def unhook(src):
    return src.replace(HOOK, ANCHOR, 1)


def backup_ok(root, rec):
    want = rec.get("sha256_before")
    p = os.path.join(root, MP) + BACKUP
    if want and os.path.exists(p):
        b = read(p)
        if sha(b) == want:
            return b
    return None


def do_apply(root):
    """Stage everything (nothing written on any problem), then: module -> backup -> record -> memory_pool.py."""
    p = os.path.join(root, MP)
    if not os.path.exists(p):
        die(f"target missing: {p} - nothing written")
    body = read(MODULE_SRC)
    err = compile_err(body, MODULE_SRC)
    if err:
        die(f"kopt_convp.py does not compile - nothing written: {err}")
    src = read(p)
    rec = load_record(root)
    st = hook_state(src)
    if st == "current":
        base = unhook(src)
        if rec.get("sha256_before") and sha(base) != rec["sha256_before"]:
            # hook present, but the rest of the file changed since our apply (another patch): keep it, record it
            print(f"  {MP}: hook already present (note: file changed by another patch since our apply; revert will "
                  f"report it)")
        else:
            print(f"  {MP}: hook already present")
        new = src
    elif st == "stale":
        base = backup_ok(root, rec)
        if base is None or sha(src) != rec.get("sha256_after"):
            die(f"stale hook in {MP}: '{MARK}' marker present but the block differs from this kit and the file is "
                f"not exactly what the last apply wrote (or no verified backup) - nothing written; restore the "
                f"original file by hand")
        print(f"  {MP}: stale hook -> re-applied from the verified original")
        new = None
    else:
        base, new = src, None
        rec = {}   # no hook in the file: this is a fresh first apply (an old record / backup is obsolete)
    if new is None:
        for need in NEEDS:
            if need not in base:
                die(f"{MP}: required text {need.strip()!r} not found (SGLang differs from @1cf2b8c54?) - nothing "
                    f"written")
        n = base.count(ANCHOR)
        if n != 1:
            die(f"{MP}: anchor found {n} times (need 1; SGLang differs from @1cf2b8c54?) - nothing written")
        new = base.replace(ANCHOR, HOOK, 1)
        err = compile_err(new, p)
        if err:
            die(f"patched text does not compile - nothing written: {err}")
    if sha(base) != STOCK_SHA:
        print(f"  note: original {MP} sha256 {sha(base)[:16]} != sglang@1cf2b8c54 {STOCK_SHA[:16]} (anchor OK)")
    try:
        mdst = os.path.join(root, MODULE_DST)
        if not (os.path.exists(mdst) and read(mdst) == body):
            write_atomic(mdst, body)                                           # 1. module (unused until hook)
        if new != src or not rec:
            b = p + BACKUP
            if not (os.path.exists(b) and read(b) == base):
                write_atomic(b, base)                                          # 2. backup of the original
            rec = {"sha256_before": rec.get("sha256_before") or sha(base), "sha256_after": sha(new),
                   "module_sha256": sha(body), "file": MP}
            write_atomic(os.path.join(root, RECORD), json.dumps(rec, indent=1))   # 3. record
        if new != src:
            write_atomic(p, new)                                               # 4. the file
            print(f"  {MP}: hook applied; sha256 before {sha(base)[:12]} after {sha(new)[:12]}")
    except OSError as e:
        die(f"write failed: {e}; state is recorded in {RECORD} (+ {BACKUP} backup): re-run apply or --revert")
    print(f"  {MODULE_DST}: installed")
    print(f"{TAG}: done (active only with SGL_CONVP=1)")
    return 0


def do_revert(root):
    """Exit 0 = memory_pool.py back to the recorded original sha256 (or the @1cf2b8c54 sha when no record) and the
    module/record/backup removed; 7 = hook removed but the file differs from the original; 8 = refused (nothing
    written)."""
    p = os.path.join(root, MP)
    rec = load_record(root)
    want = rec.get("sha256_before") or STOCK_SHA
    if not os.path.exists(p):
        die(f"{p} missing - nothing written", rc=8)
    src = read(p)
    st = hook_state(src)
    if st == "current":
        new, how = unhook(src), "hook removed"
    elif st == "stale":
        base = backup_ok(root, rec)
        if base is None or sha(src) != rec.get("sha256_after"):
            die(f"stale hook in {MP}: marker present but the block differs from this kit and no verified backup "
                f"matches - nothing written; restore the file by hand", rc=8)
        new, how = base, "stale hook: original restored from the verified backup"
    else:
        if not rec.get("sha256_before"):
            q = os.path.join(root, MODULE_DST)
            if os.path.exists(q):
                os.remove(q)
                print(f"  removed {MODULE_DST}")
            print(f"  {MP}: no hook and no record (not applied; file sha256 {sha(src)[:12]})")
            print(f"{TAG}: nothing to revert")
            return 0
        new, how = src, "no hook (nothing to remove)"
    if new != src:
        write_atomic(p, new)
    ok = sha(new) == want
    print(f"  {MP}: {how}; sha256 {sha(new)[:12]} {'==' if ok else '!='} original {want[:12]}"
          f"{'' if rec.get('sha256_before') else ' (no record: compared to sglang@1cf2b8c54)'}")
    for rel in (MODULE_DST, RECORD, MP + BACKUP):
        q = os.path.join(root, rel)
        if os.path.exists(q) and (ok or rel == MODULE_DST):
            os.remove(q)
            print(f"  removed {rel}")
    if not ok:
        print(f"{TAG}: !! hook removed but {MP} differs from the recorded original (another patch changed it after "
              f"kopt-convp was applied); record + backup kept")
        return 7
    print(f"{TAG}: reverted")
    return 0


def status(root):
    miss = []
    p = os.path.join(root, MP)
    st = hook_state(read(p)) if os.path.exists(p) else "absent"
    if st != "current":
        miss.append(f"{MP}: hook {st}")
    d = os.path.join(root, MODULE_DST)
    if not os.path.exists(d) or read(d) != read(MODULE_SRC):
        miss.append(f"{MODULE_DST} missing or differs from kopt_convp.py")
    return miss


def do_check(root, compile_only, extra):
    miss = status(root)
    if miss:
        print(f"{TAG} check: FAIL - not applied / incomplete: " + "; ".join(miss))
        return 1
    for f in (MP, MODULE_DST):
        q = os.path.join(root, f)
        err = compile_err(read(q), q)
        if err:
            print(f"{TAG} check: FAIL - does not compile: {err}")
            return 1
    if compile_only:
        print(f"{TAG} check: PASS (compile only: hook + module present, current, and compile)")
        return 0
    env = dict(os.environ, PYTHONPATH=os.path.dirname(root) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    env.pop("SGL_CONVP", None)
    probe = ("import inspect, os\n"
             "import sglang.srt.mem_cache.memory_pool as mp\n"
             "import sglang.srt.mem_cache.kopt_convp as kc\n"
             f"root = {root!r}\n"
             "for m in (mp, kc):\n"
             "    assert os.path.realpath(m.__file__).startswith(os.path.realpath(root)), m.__file__\n"
             f"assert {MARK!r} in inspect.getsource(mp.MambaPool._allocate_deduplicated_conv_window)\n"
             "print('imported', mp.__file__)\n")
    r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, cwd="/", env=env)
    if r.returncode != 0:
        print(f"{TAG} check: FAIL - patched modules do not import from {root}: {r.stderr.strip()[-600:]}")
        return 2
    print(f"  {r.stdout.strip()}")
    try:
        import torch
        gpu = torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        gpu = False
    if not gpu:
        print(f"{TAG} check: FAIL - no CUDA GPU for the bitwise check (use --compile-only for a static check)")
        return 4
    r = subprocess.run([sys.executable, CHECK_SRC] + extra, text=True, cwd="/", env=env)
    if r.returncode != 0:
        print(f"{TAG} check: FAIL - GPU bitwise check failed (exit {r.returncode}, see above)")
        return 5
    print(f"{TAG} check: PASS (hook + module current, modules import, GPU bitwise check PASS)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--allow-no-a2", action="store_true")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    root = os.path.abspath(a.root) if a.root else find_root()
    print(f"{TAG}: sglang package at {root}")
    if a.check:
        extra = (["--allow-no-a2"] if a.allow_no_a2 else []) + (["--quick"] if a.quick else [])
        return do_check(root, a.compile_only, extra)
    if a.revert:
        return do_revert(root)
    return do_apply(root)


if __name__ == "__main__":
    sys.exit(main())
