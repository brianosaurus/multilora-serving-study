# DRAFT. Not for publication until the author has read it and every number is checked against the raw files.

Tags: [MEASURED] in the raw data and repeated. [EXPLORATORY] run after the plan was fixed, logged in the changelog. [HYPOTHESIS] not tested.

# What happens when your LoRA adapters outnumber your slots

Short answer: it falls over, politely at first and then not politely at all. Longer answer follows, with numbers, because "it got slow" is not a finding.

## What I did, and what it can't tell you

One RTX 4090. One 1.5B model (Qwen2.5-1.5B-Instruct). Three engines: vLLM 0.34.0, SGLang 0.5.23, mistral.rs v0.9.5. Synthetic adapters (rank 16, random weights), so this measures serving cost, not whether your fine-tune is any good. Open-loop Poisson load, 3 repeats per step, and a rate "passes" if the median p95 time-to-first-token is 500 ms or less. One day, one GPU, one model. If that's not enough for you, good, run your own.

I also could not certify adapter correctness for any engine. Both checks I fixed in advance turned out to be unable to tell a correct engine from a wrong one. I'm reporting that as a failure of my checks, not as a pass.

## Turning LoRA on is cheap [MEASURED]

vLLM with LoRA off, LoRA on but unused, and one adapter all passed at 80 requests per second. At 160 rps, one adapter gave about 20% less throughput than LoRA off (10.3k vs 12.9k tok/s). SGLang passed to 320 rps with LoRA off and on, and to 160 with one adapter. The feature is not the problem.

## The cliff [MEASURED]

Put 32 adapters on 16 slots and things change. Last passing rate:

| 32 adapters, 16 slots | vLLM | SGLang |
|---|---|---|
| Zipf popularity | 20 rps | 20 rps |
| uniform popularity | 10 rps | 10 rps |

Past the cliff, the engines part ways. At 40 rps on Zipf traffic, vLLM's p95 TTFT was about 1.0 to 1.6 s. SGLang's was 11.8 to 13.5 s. Same slots, same traffic, a very different attitude toward failure.

## It's the slots [MEASURED; the SGLang part is EXPLORATORY]

Give the engine more slots and the cliff moves. vLLM, 32 adapters, p95 TTFT range over 3 repeats:

| | 8 slots | 16 slots | 32 slots |
|---|---|---|---|
| uniform, 20 rps | 2471–2937 ms | 496–937 ms | 93–99 ms |
| Zipf, 40 rps | 4669–5794 ms | 957–1564 ms | 136–146 ms |

SGLang with 32 adapters on Zipf traffic: 16 slots gave 11.8 to 13.8 s at 40 rps; 32 slots gave 94 to 108 ms. At 80 rps, 9.7 to 9.8 s against 126 to 131 ms.

So the advice is dull: size the slots to the number of adapters with requests in flight at the same time. This is a configuration cliff, not a bug. I looked for a bug. I'd have been happier with a bug.

Working-set analysis [EXPLORATORY, no new runs]: I checked that advice against the request logs of the main grid (32 adapters, 16 slots, Zipf and uniform traffic, every rate, vLLM and SGLang, 3 repeats). A cell is healthy if its p95 TTFT was under 100 ms. The count of adapters with a request in flight (sent, not finished) was above 16 for at most 9% of the time in every healthy cell and for 72% or more of the time in every degraded cell. Distinct adapters over longer windows do not separate them: over 1 s windows the count was above 16 for 67 to 69% of the time at Zipf 20 rps and 32 to 33% at uniform 10 rps, both healthy, and over 5 and 10 s windows it was above 16 for 99 to 100% of the time in cells such as Zipf 10 rps and uniform 10 rps, also healthy. Caveat: the in-flight count includes requests waiting in the engine's queue, so in degraded cells it is partly a result of the slowdown, not just a cause. This is correlation, not mechanism. It covers 16 slots, one model and one GPU. Details: `followup/working_set/RESULTS.md`.

Why SGLang falls so much harder past the cliff, I don't know. Reading its source suggested three suspects: no anti-starvation by default, synchronous adapter swaps, and eviction policy. I tried the drain threshold (1 s, my pick), overlap loading and FIFO eviction at 16 slots. Drain lowered 40-rps p95 by about 16%, to about 10 s. The others did nothing useful. Suspects mostly acquitted, culprit at large.

A relevant change landed after the version I tested. SGLang PR #46500 (a LoRADrainer revamp, merged 2026-10-29) describes this situation. It is on main but not in v0.5.23: the drainer file at the v0.5.23 tag is identical to the file before that PR. A comparison against main is planned. [PENDING]

### 128 adapters, 32 slots [EXPLORATORY]

Four times more adapters than slots, and the cliff comes back, earlier. p95 TTFT, median (min–max), ms:

| Engine | Popularity | 20 rps | 40 rps |
|---|---|---|---|
| vLLM | Zipf | 113 (98–114) | 3,011 (2,424–3,057) |
| vLLM | uniform | 815 (564–1,520) | 5,440 (4,817–5,480) |
| SGLang | Zipf | 82 (72–5,806) | 11,661 (11,199–13,399) |
| SGLang | uniform | 351 (250–1,032) | 17,408 (16,593–17,741) |

Look at those 20 rps ranges: several straddle the limit, so I'm not ranking anyone there. Past it, SGLang is again much worse than vLLM. I didn't test 64 slots and I didn't measure what each slot costs in memory, so don't read "just buy more slots" into this.

## Did csgmv beat triton by 20 to 80%? No. [MEASURED]

SGLang's docs claim 20 to 80% lower latency with the csgmv backend. My pre-registered bar was at least 20% lower median p95 at concurrency 32 and up, with non-overlapping ranges. At concurrency 128, csgmv was 3.6% lower on end-to-end p95 and 4.7% lower on TTFT p95 (ranges separate), with about 4.5% more throughput. At 32, the TTFT gain was 11% but the ranges overlapped. Not held, on this model and this GPU. The docs may be right elsewhere. The plan didn't say which latency, so I reported three. None reached 20%.

For reference, the benchmark behind the documented 20 to 80% (SGLang PR #10286) next to mine:

| Condition | PR #10286 | This study |
|---|---|---|
| Model | Llama-3.1-8B-Instruct | Qwen2.5-1.5B-Instruct |
| Adapters | 1 | 32 synthetic, Zipf popularity, 16 slots |
| Dataset | ShareGPT | synthetic, 256 input and 128 output tokens |
| Load | open loop, request rates 2, 8, 16 and unlimited | closed loop, concurrency 8, 32 and 128 |
| Radix cache | off | on |
| Metric | median end-to-end latency, 9%, 40%, 70% and 81% lower | p95 TTFT, end-to-end and TPOT; end-to-end p95 3.6% to 4.5% lower at concurrency 32 and 128 |

The comparison is not like for like, so "not held on this setup" says nothing about the docs' own conditions. A commenter on SGLang discussion #52614 suggested some of these differences as the reason. I varied none of them one at a time. A one-condition-at-a-time A/B is planned. [PENDING]

Routing with prefix caching on [FLAGGED]: the host was not quiet during these passes (external CPU use 14 to 17 cores against the plan's limit of 2). Routing correctness does not depend on speed, but the plan's rule applies, so they stay flagged until the SGLang radix-cache passes are repeated in a quiet window (planned, [PENDING]). The prompts share prefixes across adapters; noise 0.0204, margin threshold 0.0611, all four power checks passed on them.

- SGLang 0.5.23, radix cache on, 16 in flight: 64 of 64 requests routed correctly in the cold pass (fresh server) and the warm pass straight after (smallest margins 0.111 and 0.113). The cache served 95.6% (cold) and 100% (warm) of the shared-prefix tokens. A control on fresh servers showed same-adapter reuse of a 579-token prefix (579 cached) and none across adapters (0 cached on the opening call of the other adapter).
- vLLM 0.34.0, prefix caching on: prompt-logprob calls report 0 cached tokens, so they bypass the cache and the log-probability rule could not be applied with the cache in use. As a fallback I compared 32-token greedy completions of the 64 adapter and prompt pairs. The cache served 96.0% of the shared-prefix tokens in the warm run (65,280 hits of 74,272 queried tokens). A control on fresh servers gave 0, 576, 0 and 576 cached tokens, so no cross-adapter hit, and no misroutes. The check is coarse: two identical cache-off runs agree on just 60.9% of completions, so it would catch a gross adapter swap, not a subtle one.
- mistral.rs: [PENDING].

Neither result is a correctness guarantee for an engine. One model, rank 16, four adapters at scale 0.02, 16 prompts in two prefix groups. Raw data: `followup/prefix_routing/`.

## mistral.rs: it's the launches, so I fixed the launches [MEASURED; exploratory, after the 24 hours]

mistral.rs handled the base model and one adapter up to 10 rps and fell over at 20. With 32 Zipf adapters, p95 TTFT was 123 ms at 5 rps and 24.6 s at 10 rps. At 5 rps, 6 requests came back as empty completions, which my ladder scores as failures. I checked afterwards: on the two prompts that fail every time, end-of-text really is the model's most likely first token for that adapter (fp32 PEFT reference), so these are not engine failures. vLLM and SGLang had no empty completions in their grids (0 of 30,000 adapter requests each, against 10 of 9,000 for mistral.rs), but those runs used different prompt and adapter mixes, so I cannot say whether that is a difference between the engines.

A closed-loop probe with 8 adapters loaded shows where the time goes. At 32 concurrent requests, time per output token: 12.1 ms with no LoRA, 12.8 with LoRA on and unused, 18.3 with 1 adapter, 21.6 with 4, 26.0 with 8. Throughput at 8 adapters is about half the base model's.

The source (v0.9.5, `mistralrs-quant/src/lora/dynamic/cuda.rs`) launches one kernel per active adapter at each of 196 sites per decode step. The repo has a grouped kernel for MoE LoRA and none that I found for dense layers. To see what that loop costs, I built a patched binary, gated by an environment variable, that pushes all rows through one adapter. **Its output is wrong. It's a stopwatch, not a fix.** It took about a quarter off the time per token at 8 adapters, and skipping the per-adapter bookkeeping as well changed nothing, so the launches were the cost. (That patch also called `arange` at every site, which forces a sync, so it probably understated the gain.)

So I wrote the real thing: one grouped kernel launch per site, adapters looked up per row, same arithmetic as before. This part happened after the 24 hours were up, on a different, busier pod (build without flash-attn, so the absolute numbers do not compare with anything above). What I did to make the result hard to argue with:

- Unit tests that compare the grouped path with the old loop bit for bit (mixed ranks, an adapter with no weights at a site, more than 32 adapters), and 128 end-to-end completions (16 prompts, 8 adapters, temperature 0) identical to the old build.
- A balanced benchmark: old build, new build and a second copy of the old build, in all 6 orderings, 150 requests per run. My first A/B always ran the old build first, and the identical-code control cells came out up to 23% apart. Order matters on a loaded host (load average 25 to 250).

Median time per output token, closed loop, 6 blocks:

| Concurrency | Adapters | Before (ms) | After (ms) | Time per token | Throughput | Faster in |
|---|---|---|---|---|---|---|
| 8 | 8 | 36.0 | 19.8 | -45% | +59% | 6 of 6 blocks |
| 32 | 8 | 52.8 | 38.3 | -28% | +32% | 6 of 6 blocks |
| 32 | 4 | 43.5 | 36.6 | -15% | +17% | 5 of 6 blocks |

At 4 adapters one block went the other way, so I would not read more into that row. With 0 or 1 adapters the code is unchanged, and those cells still moved by up to 20% from noise. That is the honest size of the error bar here.

It does not close the gap. At 32 concurrent requests, 8 adapters are still about 2.7 times slower per token than no adapters (38.3 against 14.1 ms), and I cannot explain the rest. The new kernel is still a per-row dot product, with no weight-tile reuse like vLLM's and SGLang's Triton kernels, which keep adapters stacked per layer. That redesign would help more at large batches, and I offered to do it if the maintainer prefers.

One scare: the new build returned a few more empty completions than the old one on two requests. The first-token logprobs showed near-ties between end-of-text and a comma, and the old build flips on the same requests under different batching. Not a bug, but I did look.

## Things I got wrong [MEASURED, painfully]

- I retracted a mistral.rs finding because my diagnostic ran the base model and the tuning data used 8 adapters. The rerun reproduced the slowdown. I corrected the changelog.
- I "estimated mistral.rs capacity" from a concurrency-8 closed loop. That doesn't measure capacity. Withdrawn.
- I called mistral.rs's empty completions "unexplained failures", then "not engine failures", before checking either. They are end-of-text first tokens, and whether the other engines differ is still untested.
- My first A/B of the new kernel was order-confounded. The controls said so, and I rebuilt the design.
- The network volume lost its contents mid-study, so I reran the first vLLM grid; the summaries agreed to within a few percent.
- I wrote "size for the working set, not the instantaneous count" before checking it. The analysis showed the reverse for 1 to 10 s windows.
- The prefix-routing control was initially run on a server whose cache earlier calls had written, so a hit could not show crossing. The plan was amended before any pass and the control rerun on fresh servers.
- A rebase on another machine deleted follow-up records from the changelog and plans. They were restored unchanged, and a check now blocks staged changes that delete records.
- The first SGLang LoRA session failed my drift check and was rerun. The discarded numbers matched the rerun closely: the check was cautious, not wrong.
- I wrote that SGLang and vLLM had "same slots, same traffic" and called the SGLang cliff a configuration cliff. SGLang's default decode CUDA graph range on this 24 GB GPU stops at a batch of 48, with my 0.85 memory fraction and with SGLang's own default, and with or without LoRA; vLLM's default covers far more. I first guessed the short range came from my memory setting; the check showed it did not. Raising the SGLang limit with one flag took 40 rps p95 TTFT from 13.5 s to 2.1-3.1 s on 16 slots (3 starts, shared host, screening only). The two defaults were not equivalent, and the SGLang part of the "slots" story is at least partly this. Rerun on the dedicated machine is not done.

## Upstream

All of this is open and unmerged, and none of it has had a maintainer reply yet:

- mistral.rs #2567, the logprobs layout differing from the OpenAI format, with a draft fix in #2570. I ran a build of master before and after the change against a live server; only the `top_logprobs` entries changed.
- mistral.rs #2569, the question about grouping the dense LoRA kernel, with the draft PR #2571 behind it.

Searching first saved me from two duplicate reports. If these merge I will update this section; until then it says draft.

## What I'd do with this

- Count how many adapters your traffic touches at once, then set slots to at least that. Watch for a latency jump, not a gradual slope.
- Don't assume slots are free. I didn't measure the memory.
- If you serve many adapters on mistral.rs today, expect the per-adapter launch cost, and watch #2571 before you tune around it.
- One GPU and one model is a reason to run your own check, not a reason to believe me.

## Disclosure

AI coding tools helped with analysis scripts, code reading and drafts. The design, the checks and the conclusions are mine to defend, and anything I haven't read line by line is marked as such.

Repo, raw data, scripts, versions and exact commands: https://github.com/brianosaurus/multilora-serving-study
