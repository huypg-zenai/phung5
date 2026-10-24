"""pgh_gsplit.py - [gsplit] GPU split: the coder on 3 GPUs instead of 2 (kernel-optim analysis/gpusplit/RESULT.md).

PGH_GPU_SPLIT = "1" changes, in the parsed configuration.yaml dict before validation (coder side, kernel-optim):
  llm_clients.coder-instance.vllm.gpu_ids            "0,1" -> "0,1,2"
  llm_clients.coder-instance.vllm.data_parallel_size  2    -> 3
  actors.coder.workers                                128  -> 192   (keeps 64 running per engine, as today)
sgl_shim.sh then starts one SGLang per coder GPU (:30001..30003) and one sleeping burst GLM per coder GPU (:8031..8033).
The JUDGE side (free GPU 2: disable / move judge-critic-instance, clients, BURST_GLM_URLS +:8033, balancer, K coefficients)
belongs to pipeline-speed and is applied by judge_side() below, same flag.
Safety: if after both sides any OTHER enabled vLLM client still uses a coder GPU, ONE '!!' line and the dict is returned
unchanged (never double-books a GPU, never fails the start). "" / unset / "0" / "off" = unchanged (= today's 2+2 split).
Pure function of (data, env); config/settings.py and llm/spawn.py call it only when the env is set.
"""
from __future__ import annotations

import copy
import os

OFF = ("", "0", "off", "false", "no")
CODER = "coder-instance"
CODER_GPUS = "0,1,2"
CODER_DP = 3
CODER_WORKERS = 192


# ---------------------------------------------------------------- judge side (pipeline-speed), same flag
# Config (judge_side, below): judge-critic-instance (judge.client + critic.client, :8002, vllm_enc.sh) moves GPU 2 -> the
# GPU of its extra replica judge-critic-instance-b (3); -b is disabled AND removed from actors.judge.extra_clients
# (settings.py refuses a disabled extra client); actors.judge.workers (duel slots in the coder phase) 48 -> 24 = today's
# per-GLM duel load (48 duels / 2 GLMs), so the one judge's vLLM queue (2-8 calls per duel, request timeout 300 s, a
# timed-out call is re-sampled) stays where it is today. Runtime (only when the split was ACCEPTED in this process, on()):
#   burst_urls()  BURST_GLM_URLS extended to one burst GLM per coder GPU (+:8033 = last port + 1, sgl_shim's 8031+i)
#   burst_slots() BURST_SLOTS unset -> per-GLM slots x (judges + burst GLMs) - judge slots = 24 x 4 - 24 = 72 (total 96 =
#                 today's burst total); an explicit BURST_SLOTS wins
#   scale_adaptive_k() / scale_kcpu()  the 2-coder-GPU coefficients: coder_s_per_draft / coder_s_per_k divided by
#                 PGH_GPU_SPLIT_CODER_X (default 1.4), overhead_s + PGH_GPU_SPLIT_TAIL_S (default 50 s: one judge GPU's
#                 backlog adds tail, gpusplit sim b0-b3 mean +52 s). CODER_X=1 and TAIL_S=0 = unscaled.
#   gpu_tag()     batch_stats header token gpus=c012/j3/b0123 (+ vb/vb2/vb3 vLLM tags for the burst GLMs)
JUDGE = "judge-critic-instance"
BURST_PORT0 = 8031
CODER_X, TAIL_S = 1.4, 50.0
APPLIED: bool | None = None    # outcome of apply_cfg in THIS process (None = not called / flag off)
SPLIT: dict = {}               # what judge_side resolved (valid when APPLIED)


def judge_side(out: dict, log=print) -> tuple:
    """Free the coder GPUs from the judges: the main judge client moves onto its extra replica's GPU, the extra replica is
    disabled and leaves extra_clients; judge duel slots scale to the remaining judge count. Returns (out, ok).
    Unexpected layout -> one '!!' line, (out unchanged, False): apply_cfg then keeps 2+2 (APPLIED=False) even when no
    GPU clashes (e.g. the judge already off the coder GPUs), since the runtime hooks need what this side resolves."""
    SPLIT.clear()
    clients, actors = out.get("llm_clients") or {}, out.get("actors") or {}
    judge = actors.get("judge") if isinstance(actors.get("judge"), dict) else {}
    main, extras = judge.get("client"), list(judge.get("extra_clients") or [])
    coder = _gpus(clients.get(CODER))
    names = [main, *extras]
    ok = main in clients and all(isinstance(clients.get(n), dict) and isinstance(clients[n].get("vllm"), dict)
                                 for n in names)
    keep = [n for n in extras if ok and clients[n].get("enabled", True) and not (_gpus(clients[n]) & coder)]
    if not ok or not (_gpus(clients[main]) & coder) or len(extras) != 1 or len(keep) != 1 or \
            any(isinstance(a, dict) and a.get("client") == keep[0] for a in actors.values()):
        log(f"[GSPLIT] !! judge side: unexpected judge layout (judge.client={main!r} extra_clients={extras}; want the "
            f"client on a coder GPU + ONE extra replica off them) -> judge side not applied")
        return out, False
    free = ",".join(sorted(_gpus(clients[keep[0]])))
    n_old, n_new = 1 + len(extras), 1
    w_old = int(judge.get("workers", 48) or 48)
    per_glm = max(1, w_old // n_old)
    raw = str(os.environ.get("PGH_GPU_SPLIT_JUDGE_SLOTS", "") or "").strip()
    slots = per_glm * n_new
    if raw:
        try:
            slots = int(raw)
        except ValueError:
            slots = 0
        if not 1 <= slots <= 256:
            slots = per_glm * n_new
            log(f"[GSPLIT] !! bad PGH_GPU_SPLIT_JUDGE_SLOTS {raw!r} (want 1..256) -> {slots}")
    old_gpu = clients[main]["vllm"].get("gpu_ids")
    clients[main]["vllm"]["gpu_ids"] = free
    clients[keep[0]]["enabled"] = False
    judge["extra_clients"] = []
    judge["workers"] = slots
    SPLIT.update(judge_gpu=free, n_judge=n_new, per_glm=per_glm, judge_slots=slots, moved=main, disabled=keep[0])
    log(f"[GSPLIT] judge side: {main} GPU {old_gpu} -> {free}, {keep[0]} disabled + out of extra_clients, judge duel "
        f"slots {w_old} -> {slots} ({per_glm}/GLM as today)")
    return out, True


def _flag(env=None) -> bool:
    env = os.environ if env is None else env
    return str(env.get("PGH_GPU_SPLIT", "") or "").strip().lower() not in OFF


def on(env=None) -> bool:
    """The split is in force in this process: flag set AND apply_cfg accepted it (both sides)."""
    return _flag(env) and APPLIED is True


def n_coder() -> int:
    return len(SPLIT.get("coder_gpus") or []) or 2


def burst_urls(urls: list, env=None, log=print) -> list:
    """One burst GLM per coder GPU: extend the list by last port + 1 (sgl_shim: 8031 + i). Unchanged when off, when the
    list is empty (burst disabled) or already long enough, or under PGH_MIX_CODERS (its ports are shifted per coder)."""
    from urllib.parse import urlsplit, urlunsplit
    env = os.environ if env is None else env
    if not on(env) or not urls or len(urls) >= n_coder() or \
            str(env.get("PGH_MIX_CODERS", "") or "").strip().lower() not in OFF:
        return list(urls)
    out = list(urls)
    try:
        last = urlsplit(out[-1])
        port = last.port or BURST_PORT0 + len(out) - 1
        while len(out) < n_coder():
            port += 1
            out.append(urlunsplit(last._replace(netloc=f"{last.hostname}:{port}")))
    except ValueError as exc:
        log(f"[GSPLIT] !! burst urls: cannot derive a port from {urls[-1]!r} ({exc}) -> {urls}")
        return list(urls)
    return out


def burst_slots(default: int, env=None) -> int:
    """Extra duel slots while the burst is up. Explicit BURST_SLOTS wins; else keep per-GLM slots constant."""
    env = os.environ if env is None else env
    if not on(env) or str(env.get("BURST_SLOTS", "") or "").strip() or \
            any(k not in SPLIT for k in ("per_glm", "n_judge", "judge_slots")):
        return default
    return SPLIT["per_glm"] * (SPLIT["n_judge"] + n_coder()) - SPLIT["judge_slots"]


def _num_env(env, key, default, lo, hi):
    raw = str(env.get(key, "") or "").strip()
    if not raw:
        return default, ""
    try:
        x = float(raw)
        if not lo <= x <= hi:
            raise ValueError
        return x, ""
    except ValueError:
        return default, f"!! bad {key} {raw!r} (want {lo:g}..{hi:g}) -> {default:g}"


def coder_x(env=None) -> tuple:
    env = os.environ if env is None else env
    x, n1 = _num_env(env, "PGH_GPU_SPLIT_CODER_X", CODER_X, 1.0, 4.0)
    t, n2 = _num_env(env, "PGH_GPU_SPLIT_TAIL_S", TAIL_S, 0.0, 600.0)
    return x, t, "; ".join(n for n in (n1, n2) if n)


def scale_adaptive_k(co: dict, env=None) -> tuple:
    """adaptive_k / time-guard coefficients fitted on 2 coder GPUs -> 3: C / x, overhead + tail. Off -> co unchanged."""
    if not on(env):
        return co, ""
    x, t, bad = coder_x(env)
    out = dict(co, coder_s_per_draft=co["coder_s_per_draft"] / x, overhead_s=co["overhead_s"] + t)
    return out, (f"gsplit: coder_s_per_draft {co['coder_s_per_draft']:g}/{x:g}, overhead_s {co['overhead_s']:g}+{t:g}"
                 + (f"; {bad}" if bad else ""))


def scale_kcpu(p: dict, env=None) -> tuple:
    """pgh_kcpu prior: coder_s_per_k / x (coder_start_s kept: first-token latency does not scale with GPUs)."""
    if not on(env):
        return p, ""
    x, _t, bad = coder_x(env)
    return dict(p, coder_s_per_k=p["coder_s_per_k"] / x), (f"gsplit: coder_s_per_k {p['coder_s_per_k']:g}/{x:g}"
                                                          + (f"; {bad}" if bad else ""))


def gpu_tag() -> str:
    """batch_stats header value: coder / judge / burst GPUs, e.g. c012/j3/b0123 ('' when the split is not in force)."""
    if APPLIED is not True:
        return ""
    c = "".join(SPLIT.get("coder_gpus") or [])
    j = SPLIT.get("judge_gpu", "").replace(",", "")
    return f"c{c}/j{j}/b{''.join(sorted(c + j))}"


def _gpus(client: dict) -> set:
    raw = str(((client or {}).get("vllm") or {}).get("gpu_ids", "") or "")
    return {g.strip() for g in raw.split(",") if g.strip()}


def apply_cfg(data: dict, env=None, log=print) -> dict:
    global APPLIED
    env = os.environ if env is None else env
    if str(env.get("PGH_GPU_SPLIT", "") or "").strip().lower() in OFF:
        return data
    APPLIED = False
    clients = data.get("llm_clients") or {}
    coder = clients.get(CODER)
    if not isinstance(coder, dict) or not isinstance(coder.get("vllm"), dict) or \
            not isinstance(((data.get("actors") or {}).get("coder")), dict):
        log(f"[GSPLIT] !! PGH_GPU_SPLIT ignored (no {CODER}.vllm / actors.coder in configuration.yaml): 2+2 split kept")
        return data
    out = copy.deepcopy(data)
    v, a = out["llm_clients"][CODER]["vllm"], out["actors"]["coder"]
    old = (v.get("gpu_ids"), v.get("data_parallel_size"), a.get("workers"))
    v["gpu_ids"], v["data_parallel_size"], a["workers"] = CODER_GPUS, CODER_DP, CODER_WORKERS
    out, ok = judge_side(out, log=log)
    if not ok:
        SPLIT.clear()
        log("[GSPLIT] !! PGH_GPU_SPLIT ignored: judge side not applied: 2+2 split kept")
        return data
    mine = _gpus(out["llm_clients"][CODER])
    clash = sorted(n for n, c in (out.get("llm_clients") or {}).items()
                   if n != CODER and isinstance(c, dict) and c.get("enabled", True) and _gpus(c) & mine)
    if clash:
        log(f"[GSPLIT] !! PGH_GPU_SPLIT ignored: {', '.join(clash)} still on coder GPUs {CODER_GPUS} "
            f"after the judge side: 2+2 split kept")
        return data
    APPLIED = True
    SPLIT["coder_gpus"] = sorted(mine)
    log(f"[GSPLIT] coder gpu_ids/dp/workers {old[0]}/{old[1]}/{old[2]} -> {CODER_GPUS}/{CODER_DP}/{CODER_WORKERS} (PGH_GPU_SPLIT)")
    return out
