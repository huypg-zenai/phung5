# cf = convp + foldx (kopt item, switch SGL_CONVP_FOLDX, default 0 = OFF)

Two kernel-optim handovers (21/10) as ONE kopt_shim item:
- convp: GDN conv window allocated p-major (memory_pool.py `_allocate_deduplicated_conv_window` hook + kopt_convp.py);
  runtime switch SGL_CONVP.
- foldx: braw fold without the in-loop track store (text patch of /opt/kern/braw/kernels.py, then /opt/kern/braw/apply_b.py
  re-copies it into sglang); runtime switch SGL_FOLD_X.

Both are greedy-bitwise identical to the stock path by design (each --check runs a GPU bitwise comparison).

## Enable
`docker run -e SGL_CONVP_FOLDX=1 ...` (or set it in the Dockerfile ENV). At coder start kopt_shim.sh then runs
`cf/apply_cf.py` (apply) and `cf/apply_cf.py --check`, and requires the lines `foldsplit check: PASS (L` and
`apply_cf check: PASS`. On success it exports SGL_CONVP=1 SGL_FOLD_X=1; on any failure or timeout it runs
`cf/apply_cf.py --revert`, writes the marker v5_cf_refused, and exports both switches as 0 (stock path, fail-closed).
GPU must be "NVIDIA H200" like every v5 item. With SGL_CONVP_FOLDX=0 nothing is applied and both switches are 0.

## Order
apply = apply_foldsplit.py --file /opt/kern/braw/kernels.py -> /opt/kern/braw/apply_b.py -> apply_convp.py.
--revert undoes all three. apply_convp's revert compares against the sha it recorded, so do not apply other patches
to memory_pool.py (e.g. SLIM) after cf in the same process tree.

## Gate
Numbers (greedy identity vs base, step ms / tok/s at 64 running, coder CPU) come from
our coder A/B runs (arm cf64 vs base64/base64b).
