# DRY RUN: discard these numbers

Run on the follow-up pod (a loaded shared host, load average 33 to 69) with `run_rerun.sh --dry-run`, 12 requests per run, 2 blocks, the earlier no-flash-attn binaries, and the load check downgraded to a warning. It tests the script's logic only. Nothing here is a benchmark result, and its analysis correctly printed "NOT EVALUATED" (needs 6 blocks) and flagged all 48 runs for load above the limit. The raw per-run JSON files were discarded.

Refusal paths, tested separately against the same script (each exits before starting any server):
- wrong hostname: `REFUSED: hostname is <the pod>, expected not-this-machine`
- GPU in use (a server was running): `REFUSED: GPU is not idle (1 compute processes, 12740 MiB used)`
- load above the limit, no `--dry-run`: `REFUSED: 1-minute load average 33.23 is above 2`

The analysis logic was also checked on synthetic data, where an injected 30% effect at 8 adapters is called ESTABLISHED and a null effect NOT ESTABLISHED.
