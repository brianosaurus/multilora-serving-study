# DRAFT (not posted) - waiting on step 10 results (fix accuracy and 40 rps)

**Title:** [Bug][Perf] Decode batches above the largest captured CUDA graph run eagerly and the server stays in a slow state (multi-LoRA reproduction of #33483, still on main)

## Summary

On an RTX 4090 the decode CUDA graph range defaults to 48 (`memory_hook.py`, tp < 4), while `max_running_requests` is derived from KV capacity (4096 here). Nothing keeps the running batch within the graph range. Once a decode batch passes 48 it runs eagerly; on a launch-bound setup an eager step costs about 15 times a replay, requests finish more slowly, more of them accumulate, and the batch stays above 48.

**Once eager decoding starts, the queue grows and the server does not recover while the load continues.** At 40 rps, eager steps cap throughput at about 3,000 output tok/s, below the offered load, so the waiting queue keeps growing for the rest of the run (p95 TTFT 13.5 s and rising). At 20 rps the offered load fits even at eager speed, yet a server pushed above 48 stayed slow (p95 TTFT 10.6 s against 82 ms) because the larger batch keeps every step eager. Only a drop in load lets it drain. With this fix every decode step replays a graph and the same load runs at <step 10 result>.

This is the mechanism reported in #33483 (L40, 0.5B model), which was closed for inactivity without a fix. It still reproduces on main, and two things make it worse than that report suggests:

- **LoRA.** With adapters enabled, an eager decode step costs about 31 ms against 1.7 to 2.0 ms for a replay (no LoRA: 18.7 ms against 1.3 ms).
- **CPU contention alone triggers it.** With 14 busy-loop processes in the same container (17.85-CPU quota, not throttled), a server that handles 20 rps at 82 ms p95 TTFT went to 10.6 s in one of two starts. It also happened in one of several quiet starts.

## Setup

- RTX 4090 24 GB, 1 GPU, SGLang `0e9cff98f5`; on main `385262cabb` the 24 GB tier's range (48) and the KV-derived running limit are unchanged.
- Qwen2.5-1.5B-Instruct, 32 synthetic rank-16 adapters on all 7 projection modules, `--max-loras-per-batch 16`, csgmv backend, `--mem-fraction-static 0.85 --context-length 4096`, otherwise defaults.
- Open-loop Poisson load, 256 input / 128 output tokens, adapters chosen by Zipf over 32, 500 requests after 50 warm-up, seed 1.
- Shared cloud host: timings are screening numbers, repeated across starts.

```bash
python -m sglang.launch_server --model-path Qwen/Qwen2.5-1.5B-Instruct --enable-lora \
  --max-lora-rank 16 --max-loras-per-batch 16 \
  --lora-target-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --lora-paths adapter_0000=... adapter_0031=... --mem-fraction-static 0.85 --context-length 4096
```

Harness, raw traces and scripts: <study repo link>.

## Evidence

Median wall time per decode step by batch size (closed loop):

| server | batch 1 to 48 | batch 49 to 96 |
|---|---|---|
| LoRA, default range (48) | 1.7 to 1.9 ms | 31.4 ms (eager) |
| LoRA, `--cuda-graph-max-bs-decode 256` | 1.7 to 1.9 ms | 2.0 ms |
| no LoRA, default | 1.3 ms | 18.7 ms (eager) |
| no LoRA, 256 | 1.2 ms | 1.3 ms |

The eager cost is flat across batch 49 to 96, so it is per-step CPU work (Python and launches), not GPU work. That is also why CPU contention makes it worse: the scheduler thread is CPU-bound (about 85% busy at 20 rps) and its steps get longer.

40 rps, 2 starts each (p95 TTFT; output tok/s):

| config | no contention | 14 busy loops |
|---|---|---|
| default | 13.5 s, 13.5 s; ~3000 | 16.9 s, 16.7 s; ~2300 |
| `--cuda-graph-max-bs-decode 256` | 2.2 s, 6.2 s; 4265 to 4664 | 7.7 s, 9.2 s; 3913 to 4407 |
| `--max-running-requests 48` | 1.8 s, 1.9 s; ~4500 | 7.9 s, 13.3 s; 2384 to 3162 |
| vLLM 0.34.0, same model, adapters, load | 1.5 s, 1.5 s; ~4650 | 2.7 s, 3.7 s; 3976 to 4364 |
| PR below | <step 10> | <step 10> |

GSM8K (sgl-eval 0.1.3, greedy, 128 concurrent): default 67.5%, range 256 66.3% and 66.5%, PR <step 10>.

## Root cause

`max_running_requests` and the decode graph range are set by unrelated code (KV capacity, and a GPU-memory table), and no path reconciles them. vLLM bounds both together: its running limit (`max_num_seqs`, 256 here) sets the capture range (to 512 here), so decode never leaves the graph.

## Proposed fix (PR <link>)

Do the same in SGLang:
1. When `--max-running-requests` is not set, default it to the largest decode graph batch size (logged, with a hint to raise `--cuda-graph-max-bs-decode`). An explicit value is unchanged.
2. Raise the 24 GB tier's default range for tp < 4 from 48 to 256, like #37898 and #46132 for larger tiers. Measured on the 4090: graphs to 256 take 0.54 GB (0.15 GB to 48); the KV pool is unchanged at `--mem-fraction-static 0.85`.

Alternatives considered: only raising the range (the batch can still pass 256, since the running limit stays 4096); only capping at 48 (limits concurrency); splitting an oversized batch into several graph replays inside one step (prototyped; it races with the overlap scheduler and attention planning on shared buffers, and correct versions still cost 11 to 20 ms per step).

## Limits

One model (1.5B), one GPU, shared host. On larger models the range costs a larger share of KV <AB result>.
