# Working-set analysis of the main grid (2026-11-24, local machine, no GPU)

Question: does the data support, contradict or leave undecided "size slots for the working set, not the instantaneous count"?

Data: the study's request logs, 32 adapters, 16 slots, Zipf and uniform popularity, every rate, 3 repeats. vLLM from `results/raw/main/vllm`, SGLang (valid session) from `results/raw/main_rerun1/sglang`. Script: `followup/working_set/analyze.py`; per-repeat numbers in `working_set.json`.

Definitions. Instantaneous: distinct adapters with a request in flight (sent, not finished, so including requests waiting in the engine's queue) at time t, sampled every 10 ms. Window w (1, 5, 10 s): distinct adapters with a request in flight at any time in (t-w, t]. Statistics use the measured span only; the 50 warm-up requests are in the history. Each cell is the median over its 3 repeats of: median / p95 / max / share of time above 16. TTFT p95 is the study's own metric, given to show where the cliff is; no request errors in any cell.

| engine | traffic | rps | reps | TTFT p95 ms (median of reps) | errors | inst med/p95/max/>16 | 1 s med/p95/max/>16 | 5 s | 10 s |
|---|---|---|---|---|---|---|---|---|---|
| sglang | uni32 | 5 | 3 | 66 | 0 | 4 / 8 / 11 / 0% | 8 / 13 / 17 / 0% | 20 / 25 / 27 / 90% | 26 / 30 / 32 / 100% |
| sglang | uni32 | 10 | 3 | 81 | 0 | 9 / 14 / 19 / 2% | 15 / 20 / 25 / 33% | 27 / 30 / 32 / 100% | 31 / 32 / 32 / 100% |
| sglang | uni32 | 20 | 3 | 755 | 0 | 19 / 23 / 27 / 80% | 25 / 28 / 30 / 100% | 31 / 32 / 32 / 100% | 32 / 32 / 32 / 100% |
| sglang | uni32 | 40 | 3 | 13058 | 0 | 31 / 32 / 32 / 79% | 32 / 32 / 32 / 84% | 32 / 32 / 32 / 100% | 32 / 32 / 32 / 100% |
| sglang | zipf32 | 5 | 3 | 51 | 0 | 4 / 7 / 10 / 0% | 7 / 10 / 13 / 0% | 15 / 19 / 20 / 28% | 21 / 24 / 26 / 93% |
| sglang | zipf32 | 10 | 3 | 56 | 0 | 7 / 11 / 13 / 0% | 12 / 15 / 18 / 3% | 21 / 25 / 26 / 99% | 26 / 30 / 31 / 100% |
| sglang | zipf32 | 20 | 3 | 84 | 0 | 13 / 17 / 19 / 9% | 19 / 22 / 23 / 69% | 27 / 31 / 31 / 100% | 31 / 32 / 32 / 100% |
| sglang | zipf32 | 40 | 3 | 12142 | 0 | 29 / 32 / 32 / 80% | 29 / 32 / 32 / 85% | 32 / 32 / 32 / 100% | 32 / 32 / 32 / 100% |
| sglang | zipf32 | 80 | 3 | 9560 | 0 | 31 / 32 / 32 / 73% | 32 / 32 / 32 / 80% | 32 / 32 / 32 / 100% | 32 / 32 / 32 / 100% |
| vllm | uni32 | 5 | 3 | 51 | 0 | 4 / 8 / 11 / 0% | 9 / 13 / 17 / 0% | 20 / 25 / 27 / 91% | 26 / 30 / 32 / 100% |
| vllm | uni32 | 10 | 3 | 51 | 0 | 9 / 14 / 18 / 1% | 15 / 20 / 24 / 32% | 27 / 30 / 32 / 100% | 31 / 32 / 32 / 100% |
| vllm | uni32 | 20 | 3 | 684 | 0 | 18 / 23 / 26 / 72% | 25 / 28 / 29 / 100% | 31 / 32 / 32 / 100% | 32 / 32 / 32 / 100% |
| vllm | uni32 | 40 | 3 | 3075 | 0 | 27 / 31 / 32 / 93% | 31 / 32 / 32 / 100% | 32 / 32 / 32 / 100% | 32 / 32 / 32 / 100% |
| vllm | zipf32 | 5 | 3 | 47 | 0 | 4 / 7 / 10 / 0% | 7 / 10 / 13 / 0% | 15 / 19 / 20 / 28% | 21 / 24 / 26 / 94% |
| vllm | zipf32 | 10 | 3 | 44 | 0 | 7 / 11 / 13 / 0% | 12 / 15 / 18 / 3% | 21 / 25 / 26 / 99% | 26 / 30 / 31 / 100% |
| vllm | zipf32 | 20 | 3 | 50 | 0 | 12 / 16 / 19 / 5% | 18 / 22 / 23 / 67% | 27 / 31 / 31 / 100% | 31 / 32 / 32 / 100% |
| vllm | zipf32 | 40 | 3 | 1456 | 0 | 22 / 26 / 27 / 93% | 26 / 30 / 31 / 100% | 31 / 32 / 32 / 100% | 32 / 32 / 32 / 100% |
| vllm | zipf32 | 80 | 3 | 4663 | 0 | 28 / 32 / 32 / 88% | 30 / 32 / 32 / 99% | 32 / 32 / 32 / 100% | 32 / 32 / 32 / 100% |

## What the table shows

- Healthy cells (TTFT p95 under 100 ms): Zipf 5, 10, 20 rps on both engines and uniform 5, 10. Degraded cells: Zipf 40 and 80 on both engines (vLLM 1.5 s and 4.7 s, SGLang 12.1 s and 9.6 s), uniform 20 (vLLM 0.68 s, SGLang 0.76 s) and uniform 40 (3.1 s and 13.1 s).
- Instantaneous count: above 16 for at most 9% of the time in every healthy cell (Zipf 20: 5% vLLM, 9% SGLang; uniform 10: 1 to 2%), and for 72% or more of the time in every degraded cell. Median at the transition: 12 to 13 in the last healthy Zipf cell, 18 to 19 in the first degraded uniform cell.
- 1 s window: above 16 for 67 to 69% of the time at Zipf 20 and 32 to 33% at uniform 10, both healthy. A threshold of 16 on this window would predict a cliff that does not occur.
- 5 s and 10 s windows: median 21 to 27 and 26 to 31 in healthy cells (Zipf 10, uniform 10), above 16 for 99 to 100% of the time. They are at or near 32 in almost every cell, so they do not separate healthy from degraded cells.

## Verdict

The data contradict the simple form of the statement: in these runs, a count of distinct adapters over a 1, 5 or 10 s window exceeds the 16 slots in cells with no degradation, while the instantaneous count separates healthy from degraded cells better. They do not test other windows (for example 100 ms to 500 ms), other thresholds (a fraction of the slots), or a working set weighted by traffic share, so the statement in those forms is undecided.

## Limits

- Correlation, not mechanism. The in-flight count includes requests waiting for a slot, so in degraded cells the count is partly a result of the degradation (queued requests from many adapters), not only a cause. A cell near the cliff (Zipf 20, uniform 20) is where this matters least and where the separation is already visible.
- 16 slots only, one model, one GPU, one adapter count (32). Rates are the study's grid; no cell lies between uniform 10 (healthy) and uniform 20 (degraded).
- The 100 ms and 10 ms sampling does not resolve the engine's own scheduling steps. Request timestamps are client-side.
