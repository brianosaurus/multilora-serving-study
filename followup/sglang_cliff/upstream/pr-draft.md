# DRAFT (not posted)

**Title:** [Perf] Keep decode inside CUDA graphs: default max_running_requests to the decode graph range, and raise the 24 GB tier's range to 256

**Branch:** `brianosaurus:decode-graph-range` (based on `385262cabb`; not pushed)

## Motivation

Fixes <issue link> (see also #33483). When `--max-running-requests` is not set it is derived from KV capacity (4096 on a 4090 with a 1.5B model), while the decode CUDA graph range on 24 GB GPUs is 48. A decode batch above the range runs eagerly; with LoRA an eager step costs about 31 ms against 2 ms for a replay, so requests drain slower than they arrive, the batch stays above the range, and waiting requests are served only after the load stops.

vLLM avoids this by bounding the two together (`max_num_seqs` 256, graphs captured to 512). This PR does the same.

## Modifications

- `mem_cache/kv_cache_configurator.py`: when `--max-running-requests` is not set, default it to the largest decode CUDA graph batch size (none when decode graphs are disabled or on a prefill-only disaggregation worker). Logged at INFO with the old value and a hint to raise `--cuda-graph-max-bs-decode`. An explicit `--max-running-requests` is unchanged.
- `arg_groups/memory_hook.py`: the 24 GB tier (A10, 4090, 5090) default decode range for tp < 4 goes from 48 to 256, so the default running limit does not drop to 48.

## Accuracy

sgl-eval 0.1.3 GSM8K, 1319 problems, greedy, 128 concurrent, Qwen2.5-1.5B-Instruct on a LoRA server: main 67.5%, this PR 66.3% (`--cuda-graph-max-bs-decode 256` on main: 66.3%, 66.5%). No degenerate generations.

## Memory

RTX 4090, `--mem-fraction-static 0.85`: decode graphs to 256 take 0.54 GB (0.15 GB to 48). The KV pool is 600,201 tokens with and without this PR.

## Benchmark

RTX 4090, Qwen2.5-1.5B-Instruct, 32 rank-16 adapters, `--max-loras-per-batch 16`, Zipf adapter choice, 256 in / 128 out, open-loop Poisson, 500 requests after 50 warm-up, 2 starts each. Shared cloud host, so treat these as screening numbers. "Busy loops": 14 CPU busy-loop processes in the same container.

40 rps (p95 TTFT; median TPOT; output tok/s):

| | no contention | 14 busy loops |
|---|---|---|
| main | 13.5 s, 13.4 s; 43 to 44 ms; 2981, 3040 | 14.5 s, 18.9 s; 53 to 76 ms; 2725, 2155 |
| this PR | 2.25 s, 2.08 s; 12.2 ms; 4665, 4673 | 3.35 s, 7.93 s; 14 to 16 ms; 4598, 4014 |
| vLLM 0.34.0 (reference) | 1.43 s, 1.43 s; 10 ms; 4657, 4658 | 1.44 s, 1.46 s; 10 ms; 4671, 4446 |

20 rps p95 TTFT: main 83 ms and 78 ms without contention; this PR 78 ms and 78 ms.

At 40 rps on main, every request that waited more than 1 s for its first token (110 of 500 in each start) got it only after the last request was sent; with this PR, 19 of 100.

## Not covered

- Larger models on 24 GB, where a range of 256 may take memory the KV pool would otherwise use. If maintainers prefer, the tier can pick the range from free memory after weights, as in #46132.
- The remaining gap to vLLM at 40 rps (traces point at LoRA adapter-load stalls); separate PR.

## Checklist

- [ ] Format with pre-commit
- [ ] Unit test: default running limit follows the graph range; explicit value unchanged; graphs disabled leaves it unchanged
- [x] sgl-eval accuracy run
- [x] Benchmark (shared host)
- [ ] Larger model (7 to 8B)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
