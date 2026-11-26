# DRAFT (not posted)

**Title:** [Perf] Replay CUDA graphs over chunks of decode batches above the graph range

**Branch:** `brianosaurus:decode-graph-split-main` (based on `385262cabb`; not pushed)

## Motivation

Fixes <issue link> (see also #33483). A decode batch larger than the largest captured decode graph runs eagerly. On launch-bound setups (small models, LoRA) an eager step costs about 15 times a replay (31 ms against 2 ms on a 4090 with LoRA), so the running batch grows instead of draining and stays above the graph range.

## Modifications

- `model_executor/runner/decode_graph_split.py` (new): `maybe_execute_split_decode` cuts a plain decode batch above the largest captured size into the fewest chunks that fit, sized within one of each other (`chunk_bounds`), using `TboForwardBatchPreparer.filter_batch`. Each chunk refills the LoRA batch info and replays a graph; outputs are copied out of the static buffers before the next replay and concatenated. Sampling still runs once on the whole batch.
- Chunk size follows the captured range (48, 128, 160, 256, ...), so it composes with `--cuda-graph-max-bs-decode`.
- Falls back to eager when any chunk cannot replay a graph, and for speculative decoding, DP-attention gathers, dLLM, mrope, PP, hisparse and the UOCR greedy graph, and when a replay produces outputs it cannot merge.
- `model_runner.py`: one delegation after the existing graph check.
- `SGLANG_DISABLE_DECODE_GRAPH_SPLIT` (default off) restores the eager path.
- CPU unit tests: chunk bounds (fewest chunks, each within the range, sizes within one) and the kill-switch contract.

## Accuracy

Greedy, 96 concurrent requests, 64 tokens, adapters round-robin. Identical completions against a server with `--cuda-graph-max-bs-decode 256` (every step in a graph): this PR 65 to 67 of 96, stock 61 of 96, the 256 server against itself 79 of 96 (batch makeup changes numerics). No sgl-eval accuracy run yet: <to add before review>.

## Benchmark

RTX 4090, Qwen2.5-1.5B-Instruct, 32 rank-16 adapters, 16 slots, Zipf adapter choice, 256 in / 128 out, open loop, 500 requests after 50 warm-up; shared host, so starts are repeated. "Contention": 14 busy-loop processes in the same container.

| | stock | this PR |
|---|---|---|
| decode step, batch above 48 | 31 to 36 ms (eager) | 3.8 to 4.1 ms (2 to 3 replays) |
| 40 rps p95 TTFT, no contention | 13.5 s, 13.5 s | 9.26 s, 9.37 s |
| 40 rps p95 TTFT, contention | 16.9 s, 16.7 s | 9.34 s, 9.41 s |
| 40 rps TPOT (median) | 43 to 71 ms | 20.2 to 21.0 ms |
| 40 rps output tok/s | 2256 to 3014 | 4036 to 4074 |
| 20 rps p95 TTFT, contention | 10.6 s, 0.82 s | 0.086 s, 2.5 s |

What it does not do: at 40 rps it does not reach `--cuda-graph-max-bs-decode 256` (2.2 to 6.2 s p95) or vLLM 0.34.0 (1.5 s). The remaining overhead is mostly LoRA batch preparation, about 1 ms per chunk plus 1 ms for a whole-batch preparation the split discards; a follow-up PR will address it.

## Checklist

- [ ] Format with pre-commit
- [x] Unit tests (CPU)
- [ ] sgl-eval accuracy run
- [ ] Benchmark on a dedicated machine and a larger model

🤖 Generated with [Claude Code](https://claude.com/claude-code)
