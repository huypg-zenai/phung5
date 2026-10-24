"""[mcache] kernel-optim: coder-instance --max-mamba-cache-size from the env (default off = configuration.yaml's 240).
    PGH_MAMBA_CACHE=<n>   e.g. 192: replace the value after --max-mamba-cache-size in llm_clients.coder-instance.vllm.extra_args
Only admission changes (how many requests share a step): fewer reserved mamba slots leave more KV tokens, so fewer requests
wait or get retracted on heavy batches. S1 heavy-b2 bench (1xH200, 64 running): 240 -> 4115 client tok/s, 192 -> 4518
(+9.8 %, retractions 142 -> 17), 160 -> 4321 (mamba slots become the limit). Invalid value / flag missing -> '!!' + unchanged."""
import os

CODER = "coder-instance"
FLAG = "--max-mamba-cache-size"


def apply_cfg(data: dict, log=print) -> dict:
    raw = str(os.environ.get("PGH_MAMBA_CACHE", "") or "").strip()
    if raw.lower() in ("", "0", "off", "false", "no"):
        return data
    try:
        n = int(raw)
        assert 64 <= n <= 1024
    except (ValueError, AssertionError):
        log(f"[mcache] !! PGH_MAMBA_CACHE={raw!r} is not an int in 64..1024 -> unchanged")
        return data
    args = (((data.get("llm_clients") or {}).get(CODER) or {}).get("vllm") or {}).get("extra_args")
    if not isinstance(args, list) or FLAG not in args or args.index(FLAG) + 1 >= len(args):
        log(f"[mcache] !! {CODER} has no {FLAG} in vllm.extra_args -> unchanged")
        return data
    i = args.index(FLAG) + 1
    old, args[i] = args[i], str(n)
    log(f"[mcache] {CODER} {FLAG} {old} -> {n}")
    return data
