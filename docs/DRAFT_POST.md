# DRAFT. Not for publication until the author has read it and every number is checked against the raw files.

Tags: [MEASURED] in the raw data and repeated. [EXPLORATORY] run after the plan was fixed, logged in the changelog. [HYPOTHESIS] not tested. The longer, unsarcastic version is `DRAFT_POST_long.md`.

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

So the advice is dull: size the slots to the number of adapters active at the same time. This is a configuration cliff, not a bug. I looked for a bug. I'd have been happier with a bug.

Why SGLang falls so much harder past the cliff, I don't know. Reading its source suggested three suspects: no anti-starvation by default, synchronous adapter swaps, and eviction policy. I tried the drain threshold (1 s, my pick), overlap loading and FIFO eviction at 16 slots. Drain lowered 40-rps p95 by about 16%, to about 10 s. The others did nothing useful. Suspects mostly acquitted, culprit at large.

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

## mistral.rs: it's the adapters, and it's a quarter of the story [MEASURED / EXPLORATORY]

mistral.rs handled the base model and one adapter up to 10 rps and fell over at 20. With 32 Zipf adapters, p95 TTFT was 123 ms at 5 rps (with 6 failed requests) and 24.6 s at 10 rps.

A closed-loop probe with 8 adapters loaded shows where it goes. At 32 concurrent requests, time per output token: 12.1 ms with no LoRA, 12.8 with LoRA on and unused, 18.3 with 1 adapter, 21.6 with 4, 26.0 with 8. Throughput at 8 adapters is about half the base model's. One run of 150 requests per cell, so a millisecond here or there is noise.

The source (v0.9.5, `mistralrs-quant/src/lora/dynamic/cuda.rs`) launches one kernel per active adapter at each of 196 sites per decode step. The repo has a grouped kernel for MoE LoRA and none that I found for dense layers. To see what that loop costs, I built a patched binary, gated by an environment variable, that pushes all rows through one adapter. **Its output is wrong. It's a stopwatch, not a fix.** With 8 adapters, time per token (median of 3):

| | Normal | One kernel call | Also skip per-adapter lookups |
|---|---|---|---|
| concurrency 8 | 17.1 ms | 12.6 | 12.5 |
| concurrency 32 | 26.0 | 19.0 | 18.8 |

About a quarter of the time per token (throughput up about 35%), and the bookkeeping around the launch turned out innocent. Roughly 4 ms per token of LoRA cost remains and I can't explain it. I couldn't count kernel launches: Nsight Compute can't read GPU counters inside the container, so the launch count itself is unmeasured. There's also a warm-up wrinkle: the first 150-request run after switching traffic type read about 35% slow, then recovered. Cause unknown. Single runs right after a change of traffic can lie to you.

## Things I got wrong [MEASURED, painfully]

- I retracted a mistral.rs finding because my diagnostic ran the base model and the tuning data used 8 adapters. The rerun reproduced the slowdown. I corrected the changelog.
- I "estimated mistral.rs capacity" from a concurrency-8 closed loop. That doesn't measure capacity. Withdrawn.
- The network volume lost its contents mid-study. The first vLLM grid was rerun; its summaries agreed with the rerun to within a few percent.
- The first SGLang LoRA session failed my drift check and was rerun. The discarded numbers matched the rerun closely, which says the drift check was cautious, not wrong, and is mildly reassuring for everything else.

## Upstream

I filed mistral.rs #2567 (logprobs layout differs from the OpenAI format; a local fix exists, no PR open). Searching first saved me from two duplicate reports. The mistral.rs per-adapter cost isn't root-caused to a fix. I have a measurement and a hypothesis, not a pull request.

## What I'd do with this

- Count how many adapters your traffic touches at once, then set slots to at least that. Watch for a latency jump, not a gradual slope.
- Don't assume slots are free. I didn't measure the memory.
- One GPU and one model is a reason to run your own check, not a reason to believe me.

## Disclosure

AI coding tools helped with analysis scripts, code reading and drafts. The design, the checks and the conclusions are mine to defend, and anything I haven't read line by line is marked as such.

Repo, raw data, scripts, versions and exact commands: https://github.com/brianosaurus/multilora-serving-study
