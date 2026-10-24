"""pgh_timeguard.py - reproducible ROUND-BUDGET guard checked at BATCH BOUNDARIES (pgh1v3, 19/10).
Design: the only guard of our time-guard study that keeps the draft pool reproducible (the
within-batch predictive cap, the O9 render deadline and a start-up render benchmark were all REJECTED there because they
make the draft pool depend on speed). Installed by make_pgh1_v3.py as pipeline_service/pgh_timeguard.py.

Switch PGH_TIMEGUARD (read at import = container start; Dockerfile ENV default "live", owner: guard ON):
  live   (or 1 / on)   decide at every batch start AND apply the cut (output-changing when it fires)
  shadow               decide and log exactly what live would do, change nothing (output = no guard)
  off    (or 0 / "")   hooks skipped entirely (= pgh1v2)
  pace                 [cycle1] budget PACER (opt_r6/kschedule.md "REC", from pacer/cand): batch 0 of a CLEAN round runs at
                       K0 = max(k_floor, floor(b0_frac x min(start_k, cap_k))) (b0_frac 0.90), a non-clean round at
                       min(start_k, cap_k). Before batches 1-3: E = clock now - this round's batch-0 start (inter-batch gaps
                       count), allowed = (round_s - margin_s - E) / batches left, unit as live; K_fit = the largest integer
                       in [lowest step, cap_k] with pred(K) <= allowed, pred(K) = max(unit x per_batch(K), b x K) where
                       b = mean(wall / K) of this round's batches (wall ~ K, intercept 0: kschedule.md section 1; the
                       model's intercept under-predicts a RAISE, its larger value at a smaller K keeps CUTS safe);
                       prop_raise false -> unit x per_batch(K) only (= kschedule REC). A cut applies at once; a raise only
                       in a clean round (began at container start-up or after an idle gap > idle_reset_s) and by at most
                       up_max (12) per batch; K may go above the start-up K, never above cap_k (80). margin_s 600.
                       last_unit true (opt-in, default false): from batch 2 on, unit and b are scaled up so the plan is
                       never faster than the last batch's own unit (guards a fast batch 0, e.g. the 16:1:16 render fix).
  anything else        -> one '!!' log line, treated as off.
PGH_TIMEGUARD_PARAMS   JSON object overriding any of DEFAULT_PARAMS (bad -> all defaults + a note in the log).

What fires when (one round = batches_per_round batches; rounds are counted from start-up):
  batch 0 of a round: never (only the pipeline's own per-batch budget acts there); K = the start-up K.
  before batch b = 1..3: E = sum of this round's finished batch walls; allowed = (round_s - margin_s - E) / (4 - b);
    obs = this round's mean batch wall, normalised to the current K with the batch model (= the plain mean while no cut
    happened). obs <= allowed -> nothing. Else K_new = the largest step of {85, 70, 55, 40, 25% of the start-up K, k_floor}
    below the current K with obs * per_batch(K_new) / per_batch(K_cur) <= allowed (none fits -> k_floor). Never raises K
    within a round.
  The cut applies to the WHOLE batch (every prompt has the same K), drafts k = 0..K_new-1 are kept, i.e. the highest
  numbered drafts are dropped first -> a cut batch's pool is a prefix of the uncut pool (seed = task.seed + offset + k).
  A new round (K back to the start-up K) starts after batches_per_round batches, or when the gap since the last batch
  end exceeds idle_reset_s, or when the start-up K changes.
per_batch(K) = overhead_s + K * per_draft(Q) from adaptive_k.py (same coefficients, PGH_ADAPTIVE_K_COEFFS), at the cores
  adaptive_k detected at start-up (read once; detected here once if adaptive K is off). Detection failure -> the model
  at 26 cores with kappa 1.0 (the k_fallback assumption).
Replayable OFFLINE from the batch walls alone: decide() and replay() are pure functions; every decision is logged as one
  '[TIMEGUARD] ...' line with round, batch, E, obs, allowed and K (test: remote8/kern/v3/test_v3_timeguard.py).
Hooks (all switch-guarded, added by make_pgh1_v3.py): GenerationPipeline.run_batch -> GUARD.batch_begin / batch_end;
  stages.multigen_first_iter -> K = min(K, k_cap(K)) right after `K = ensemble_size`.
"""
from __future__ import annotations

import json
import math
import os
import time

DEFAULT_PARAMS: dict = {"round_s": 7200.0, "margin_s": 250.0, "batches_per_round": 4,
                        "steps_pct": [85, 70, 55, 40, 25], "k_floor": 2, "idle_reset_s": 900.0}
FAIL_CORES, FAIL_KAPPA = 26.0, 1.0
# [R82 kbudget, A1 idea 6b] PGH_TIMEGUARD_FINE=1 -> finer cut steps (a 2% overrun no longer costs 15% of the drafts), unless
# PGH_TIMEGUARD_PARAMS sets steps_pct itself. Unset/0 = DEFAULT_PARAMS steps (pgh16 identical).
FINE_STEPS_PCT = [95, 90, 85, 78, 70, 55, 40, 25]
# [cycle1] pace mode only: extra keys and its own default margin (live/shadow keep DEFAULT_PARAMS exactly, pace keys refused)
PACE_PARAMS: dict = {"margin_s": 600.0, "cap_k": 80, "up_max": 12, "b0_frac": 0.90, "prop_raise": True,
                     "last_unit": False}


def parse_mode(raw) -> tuple[str, str]:
    """(mode, note). live | shadow | off."""
    v = str(raw if raw is not None else "").strip().lower()
    if v in ("live", "1", "on"):
        return "live", ""
    if v == "shadow":
        return "shadow", ""
    if v == "pace":
        return "pace", ""
    if v in ("off", "0", ""):
        return "off", ""
    return "off", f"!! PGH_TIMEGUARD={raw!r} is not live/shadow/off -> off"


def normalize_params(raw, mode: str = "live") -> dict:
    base = dict(DEFAULT_PARAMS, **PACE_PARAMS) if mode == "pace" else DEFAULT_PARAMS
    p = dict(base)
    if isinstance(raw, str):
        raw = json.loads(raw) if raw.strip() else {}
    if not isinstance(raw, dict):
        raise ValueError(f"PGH_TIMEGUARD_PARAMS must be a JSON object, got {type(raw).__name__}")
    unknown = sorted(set(raw) - set(base))
    if unknown:
        raise ValueError(f"unknown param(s) {unknown}; known: {sorted(base)}")
    p.update(raw)
    for key in ("round_s", "margin_s", "idle_reset_s"):
        if isinstance(p[key], bool) or not isinstance(p[key], (int, float)) or not math.isfinite(p[key]) or p[key] < 0:
            raise ValueError(f"{key}={p[key]!r} must be a finite number >= 0")
    for key in ("batches_per_round", "k_floor"):
        if isinstance(p[key], bool) or not isinstance(p[key], int) or p[key] < (2 if key == "batches_per_round" else 1):
            raise ValueError(f"{key}={p[key]!r} must be an integer >= {2 if key == 'batches_per_round' else 1}")
    st = p["steps_pct"]
    if not isinstance(st, list) or not st or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not 0 < x < 100
                                                 for x in st):
        raise ValueError(f"steps_pct={st!r} must be a non-empty list of numbers in (0, 100)")
    for key, lo in (("cap_k", 1), ("up_max", 0)):   # [cycle1] pace keys
        if key in p and (isinstance(p[key], bool) or not isinstance(p[key], int) or p[key] < lo):
            raise ValueError(f"{key}={p[key]!r} must be an integer >= {lo}")
    for key in ("prop_raise", "last_unit"):
        if key in p and not isinstance(p[key], bool):
            raise ValueError(f"{key}={p[key]!r} must be true or false")
    if "b0_frac" in p:
        f = p["b0_frac"]
        if isinstance(f, bool) or not isinstance(f, (int, float)) or not math.isfinite(f) or not 0 < f <= 1:
            raise ValueError(f"b0_frac={f!r} must be a number in (0, 1]")
    return p


def steps_for(start_k: int, p: dict) -> list[int]:
    """Fixed per start-up K: floor(start_k * pct / 100) for each pct, plus k_floor; descending, unique."""
    return sorted({max(p["k_floor"], int(math.floor(start_k * x / 100.0 + 1e-9))) for x in p["steps_pct"]} | {p["k_floor"]},
                  reverse=True)


def decide(walls: list[float], ks: list[int], k_cur: int, start_k: int, per_batch, p: dict) -> tuple[int, dict]:
    """Pure batch-boundary decision before batch b = len(walls) (b >= 1) of a round. walls/ks = this round's finished
    batch walls and the K each ran with. Returns (K for batch b, info)."""
    b = len(walls)
    left = p["batches_per_round"] - b
    info = dict(b=b, E=sum(walls), obs=None, allowed=None, k_cur=k_cur, k_new=k_cur, fired=False)
    if b < 1 or left < 1:
        return k_cur, info
    allowed = (p["round_s"] - p["margin_s"] - info["E"]) / left
    unit = sum(w / per_batch(k) for w, k in zip(walls, ks)) / b       # observed seconds per model second
    obs = unit * per_batch(k_cur)
    info.update(obs=obs, allowed=allowed)
    if obs <= allowed:
        return k_cur, info
    k_new = next((s for s in steps_for(start_k, p) if s < k_cur and unit * per_batch(s) <= allowed),
                 min(k_cur, p["k_floor"]))
    info.update(k_new=k_new, fired=True)
    return k_new, info


def decide_pace(walls: list[float], ks: list[int], k_cur: int, start_k: int, per_batch, p: dict, clean: bool,
                E: float | None = None) -> tuple[int, dict]:
    """[cycle1] Pure PACER decision before batch b = len(walls) (b >= 1) of a round. E = seconds since this round's
    batch 0 started (the Guard passes the clock; None -> sum(walls)). unit/obs as decide(). k_fit = the largest integer
    K in [floor, cap_k] with pred(K) <= allowed (pred: see the header) (none -> floor), floor = the lowest step (k_floor). Cut to
    k_fit at once; raise only if `clean`, to at most k_cur + up_max. Returns (K for batch b, info incl. 'what')."""
    b = len(walls)
    left = p["batches_per_round"] - b
    cap, floor = int(p["cap_k"]), min(steps_for(start_k, p))
    E = float(sum(walls)) if E is None else float(E)
    info = dict(b=b, E=E, E_walls=sum(walls), obs=None, allowed=None, k_cur=k_cur, k_new=k_cur, k_fit=None, fired=False,
                what=f"ok K {k_cur}")
    if b < 1 or left < 1:
        return k_cur, info
    allowed = (p["round_s"] - p["margin_s"] - E) / left
    unit = sum(w / per_batch(k) for w, k in zip(walls, ks)) / b       # observed seconds per model second (= live)
    b_prop = sum(w / k for w, k in zip(walls, ks)) / b if p.get("prop_raise", False) else 0.0   # s per draft, wall ~ K
    if p.get("last_unit", False) and b >= 2:                          # opt-in: never plan faster than the last batch
        f = max(1.0, (walls[-1] / per_batch(ks[-1])) / unit)
        unit, b_prop = unit * f, b_prop * f
    obs = unit * per_batch(k_cur)
    pred = lambda k: max(unit * per_batch(k), b_prop * k)
    k_fit = next((k for k in range(cap, floor - 1, -1) if pred(k) <= allowed), floor)
    if k_fit < k_cur:
        k_new, what = k_fit, f"CUT K {k_cur} -> {k_fit}"
    elif k_fit > k_cur and not clean:
        k_new, what = k_cur, f"HOLD K {k_cur} (fits {k_fit}; no raise: round not clean)"
    elif k_fit > k_cur:
        k_new = min(k_fit, k_cur + p["up_max"])
        what = f"RAISE K {k_cur} -> {k_new} (fits {k_fit}, max +{p['up_max']})"
    else:
        k_new, what = k_cur, f"ok K {k_cur}"
    k_new = min(cap, max(floor, k_new))
    info.update(obs=obs, allowed=allowed, k_fit=k_fit, k_new=k_new, fired=k_new != k_cur, what=what)
    return k_new, info


def pace_k0(start_k: int, p: dict, clean: bool) -> int:
    """[cycle1] K of batch 0 in pace mode: clean round -> max(k_floor, floor(b0_frac * min(start_k, cap_k))), else
    min(start_k, cap_k) (a non-clean round is cut-only, so it must not start low)."""
    k = min(start_k, int(p["cap_k"]))
    return max(int(p["k_floor"]), int(math.floor(p["b0_frac"] * k + 1e-9))) if clean else k


def replay(walls_at_k: list[float], start_k: int, per_batch, p: dict, k_rec: int | None = None) -> list[dict]:
    """Offline replay of one round from recorded batch walls (each measured at K = k_rec, default start_k): a cut batch's
    wall is scaled by per_batch(K) / per_batch(k_rec). Returns one dict per batch: K, wall, decision info."""
    k_rec = start_k if k_rec is None else k_rec
    walls, ks, k, out = [], [], start_k, []
    for w in walls_at_k:
        k, info = decide(walls, ks, k, start_k, per_batch, p)
        wall = w * per_batch(k) / per_batch(k_rec)
        walls.append(wall); ks.append(k)
        out.append(dict(info, K=k, wall=wall))
    return out


class Guard:
    def __init__(self, mode: str = "off", params: dict | None = None, per_batch=None, clock=time.monotonic):
        self.mode, self.p, self.per_batch, self.clock = mode, dict(params or DEFAULT_PARAMS), per_batch, clock
        self.model_note, self.env_note, self._noted = "", "", False
        self.round_no, self.walls, self.ks = 0, [], []
        self.start_k = self.k_guard = self.cur = None
        self._t0 = self._last_end = None
        self.clean, self._round_t0 = False, None                # [cycle1] pace: clean round flag, batch-0 start time

    def _model(self):
        if self.per_batch is None:
            self.per_batch, self.model_note = model_from_adaptive_k()
        return self.per_batch

    def batch_begin(self, start_k: int, n_tasks: int | None = None, log=print) -> int:
        """Called once at the start of every batch, before any coder starts. Returns the K this batch runs with."""
        if self.env_note and not self._noted:
            log(f"[TIMEGUARD] {self.env_note}"); self._noted = True
        if self.mode == "off":
            self.cur = None
            return start_k
        now = self.clock()
        pb = self._model()
        new_round = (self.start_k != start_k or len(self.walls) >= self.p["batches_per_round"] or self._last_end is None
                     or now - self._last_end > self.p["idle_reset_s"])
        if new_round and self.mode == "pace":
            self.clean = self._last_end is None or now - self._last_end > self.p["idle_reset_s"]
            self.round_no += 1
            k0 = pace_k0(start_k, self.p, self.clean)
            self.walls, self.ks, self.start_k, self.k_guard, self._round_t0 = [], [], start_k, k0, now
            log(f"[TIMEGUARD] mode=pace round={self.round_no} batch=0 K={k0} (new round; clean="
                f"{'yes' if self.clean else 'no'}; start K {start_k}; model per_batch({k0})={pb(k0):.0f}s"
                f"{', ' + self.model_note if self.model_note else ''}; params={json.dumps(self.p, separators=(',', ':'))})")
        elif new_round:
            self.round_no += 1
            self.walls, self.ks, self.start_k, self.k_guard = [], [], start_k, start_k
            log(f"[TIMEGUARD] mode={self.mode} round={self.round_no} batch=0 K={start_k} (new round; model per_batch("
                f"{start_k})={pb(start_k):.0f}s{', ' + self.model_note if self.model_note else ''}; params="
                f"{json.dumps(self.p, separators=(',', ':'))})")
        elif self.mode == "pace":
            k_new, info = decide_pace(self.walls, self.ks, self.k_guard, self.start_k, pb, self.p, self.clean,
                                      E=now - self._round_t0)
            log(f"[TIMEGUARD] mode=pace round={self.round_no} batch={info['b']} E={info['E']:.0f}s (walls "
                f"{info['E_walls']:.0f}s) obs={info['obs']:.0f}s allowed={info['allowed']:.0f}s walls="
                f"{[round(w) for w in self.walls]} Ks={self.ks} clean={'yes' if self.clean else 'no'} -> {info['what']}")
            self.k_guard = k_new
        else:
            k_new, info = decide(self.walls, self.ks, self.k_guard, self.start_k, pb, self.p)
            what = (f"FIRED K {info['k_cur']} -> {k_new}" if info["fired"] else f"ok K {k_new}")
            if info["fired"] and self.mode == "shadow":
                what += " (shadow: NOT applied, batch runs K %d)" % self.start_k
            log(f"[TIMEGUARD] mode={self.mode} round={self.round_no} batch={info['b']} E={info['E']:.0f}s "
                f"obs={info['obs']:.0f}s allowed={info['allowed']:.0f}s walls={[round(w) for w in self.walls]} "
                f"Ks={self.ks} -> {what}")
            self.k_guard = k_new
        self.cur = self.k_guard if self.mode in ("live", "pace") else start_k
        self._t0 = now
        return self.cur

    def batch_end(self, log=print) -> None:
        if self._t0 is None:
            return
        now = self.clock()
        self.walls.append(now - self._t0); self.ks.append(self.cur)
        log(f"[TIMEGUARD] mode={self.mode} round={self.round_no} batch={len(self.walls) - 1} done: wall="
            f"{now - self._t0:.0f}s K={self.cur}")
        self._t0, self._last_end, self.cur = None, now, None

    def k_cap(self, k_count: int, **_ctx) -> int:
        """Drafts to issue for one prompt of the running batch: live -> min(K, this batch's K); pace -> this batch's K
        (may exceed the start-up K, up to cap_k; stages.py does not clamp it in pace mode); else K."""
        if self.mode == "live" and self.cur is not None:
            return min(k_count, self.cur)
        if self.mode == "pace" and self.cur is not None:
            return self.cur
        return k_count


def model_from_adaptive_k(env=None):
    """per_batch(K) from adaptive_k.py at the cores it detected at start-up (detected once here if it did not run)."""
    try:
        from modules.metrics import adaptive_k as ak
    except ImportError:
        import adaptive_k as ak  # tests: module on sys.path
    env = os.environ if env is None else env
    note = []
    try:
        co = ak.normalize_coeffs(str(env.get("PGH_ADAPTIVE_K_COEFFS", "") or ""))
    except (ValueError, TypeError) as exc:
        co = dict(ak.DEFAULT_COEFFS); note.append(f"coeffs-error ({exc}) -> defaults")
    _gs = __import__("sys").modules.get("pgh_gsplit")   # [gsplit] same rescale as adaptive_k.decide
    if _gs is not None and _gs.on(env):
        co, _gn = _gs.scale_adaptive_k(co, env)
        note.append(_gn)
    det = (getattr(ak, "LAST_DECISION", None) or {}).get("det") or ak.detect()
    cores = det.get("cores") if det.get("ok") else None
    if cores is None or ak.per_draft_s(cores, co) is None:
        co = dict(co, kappa=FAIL_KAPPA); cores = FAIL_CORES
        note.append(f"cores unknown -> model at {FAIL_CORES:g} cores, kappa {FAIL_KAPPA:g}")
    pd = ak.per_draft_s(cores, co)
    note.insert(0, f"cores={cores:g} per_draft={pd:.2f}s overhead={co['overhead_s']:g}s")
    return (lambda k, _o=co["overhead_s"], _pd=pd: _o + k * _pd), "; ".join(note)


def _from_env():
    mode, note = parse_mode(os.environ.get("PGH_TIMEGUARD", "0"))
    try:
        params = normalize_params(os.environ.get("PGH_TIMEGUARD_PARAMS", "") or "", mode)
    except (ValueError, TypeError) as exc:
        params = normalize_params("", mode); note = (note + "; " if note else "") + f"!! bad PGH_TIMEGUARD_PARAMS ({exc}) -> defaults"
    if str(os.environ.get("PGH_TIMEGUARD_FINE", "0") or "0").strip() == "1" and '"steps_pct"' not in (
            os.environ.get("PGH_TIMEGUARD_PARAMS", "") or ""):   # [R82 kbudget] finer steps knob
        params = dict(params, steps_pct=list(FINE_STEPS_PCT))
        note = (note + "; " if note else "") + f"PGH_TIMEGUARD_FINE=1 -> steps_pct {FINE_STEPS_PCT}"
    g = Guard(mode, params)
    g.env_note = note
    return g


GUARD = _from_env()
IMPLEMENTED = True


def enabled() -> bool:
    return GUARD.mode in ("live", "shadow", "pace")


def k_cap(k_count: int, **ctx) -> int:
    return GUARD.k_cap(k_count, **ctx)
