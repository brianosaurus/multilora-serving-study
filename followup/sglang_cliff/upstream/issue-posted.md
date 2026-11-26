# POSTED: https://github.com/sgl-project/sglang/issues/53431 (2026-11-26; body shortened, title trimmed)

**Title:** [Bug][Perf] Decode batches above the largest captured CUDA graph run eagerly and the server stays in a slow state

## Summary

On 24 GB GPUs (tp < 4) the decode CUDA graph range is 48, but `max_running_requests` defaults to a KV-derived value (4096 on a 4090). Decode batches above 48 run eagerly at about 15x the cost of a replay (31 ms vs 2 ms with LoRA), requests drain slower than they arrive, and the batch stays above 48. **Under sustained load the server does not recover:** at 40 rps every request that waited over 1 s (110 of 500) got its first token only after arrivals stopped. CPU contention alone can trigger it at 20 rps.

Same mechanism as #33483 (closed without a fix); still on main `385262cabb`.

## Reproduction

RTX 4090, Qwen2.5-1.5B-Instruct, 32 rank-16 adapters, `--max-loras-per-batch 16 --mem-fraction-static 0.85 --context-length 4096`; open-loop Poisson, 256 in / 128 out, Zipf adapter choice, 500 requests. Shared host, 2 starts each. Scripts and raw data: https://github.com/brianosaurus/multilora-serving-study/tree/0a65e20/followup/sglang_cliff

40 rps, p95 TTFT / output tok/s:

| | no contention | 14 CPU busy loops |
|---|---|---|
| main | 13.5 s / ~3000 | 14.5 to 18.9 s / ~2400 |
| proposed fix | 2.1 to 2.3 s / ~4670 | 3.4 to 7.9 s / ~4300 |
| vLLM 0.34.0 | 1.4 s / ~4660 | 1.4 s / ~4560 |

GSM8K (sgl-eval, greedy): main 67.5%, fix 66.3%.

## Root cause and proposed fix

The running limit and the graph range are set independently and never reconciled. vLLM bounds them together (`max_num_seqs` 256, graphs to 512). Proposed (PR to follow):

1. When `--max-running-requests` is unset, default it to the largest decode graph batch size.
2. Raise the 24 GB tier's default range from 48 to 256 (as #37898 and #46132 did for larger tiers). On the 4090 this costs 0.54 GB of graph memory and leaves the KV pool unchanged.

Not measured: larger models on 24 GB.
