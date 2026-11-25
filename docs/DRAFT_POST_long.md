# DRAFT (not for publication until the author has read it and every number is checked against the raw files)

Status marks: [MEASURED] = in the raw data and reproduced; [EXPLORATORY] = run after the pre-registered plan, logged in the changelog; [PENDING] = not finished; [HYPOTHESIS] = not tested. Anything marked PENDING or HYPOTHESIS is either filled in or cut before publishing.

# What happens when your LoRA adapters outnumber your slots

One RTX 4090, one 1.5B model (Qwen2.5-1.5B-Instruct), three open-source engines (vLLM 0.34.0, SGLang 0.5.23, mistral.rs v0.9.5), and many LoRA adapters. I ran this in one day and logged every change to the plan as I went. All raw files, scripts and the changelog are in the repo.

## What was measured, and its limits

- Open-loop Poisson arrivals, 256-token prompts, 128 output tokens, 500 requests per run (the first 50 excluded), 3 repeats per step. A rate "passes" if the median p95 time-to-first-token (TTFT) of the 3 repeats is 500 ms or less.
- Adapters are synthetic (rank 16, all linear layers, random weights). They show serving cost, not the quality of a real fine-tune. Correctness was not certified for any engine: both pre-registered checks lacked the power to tell a correct engine from a wrong one, and I report that as a failure of the checks.
- One model, one GPU, one day, 3 repeats, a coarse rate ladder (5, 10, 20, 40, 80, 160, 320 rps). I do not rank engines whose ranges overlap, and I do not extrapolate beyond these adapter counts.
- Each engine had a small tuning pass on a separate workload. The three trials were within a few percent of each other for vLLM, so defaults were used.

## 1. With enough slots, LoRA costs little [MEASURED]

Last passing rate (rps) with 16 slots: vLLM with LoRA off 80, with LoRA enabled and no adapter used 80, with one adapter 80. At the next rate (160) one adapter gave about 20% lower throughput than LoRA off (10.3k against 12.9k tok/s). SGLang passed to 320 with LoRA off and enabled, and to 160 with one adapter. Enabling LoRA without using an adapter made no measurable difference on vLLM.

## 2. The cliff: more adapters than slots [MEASURED]

With 32 adapters sharing 16 slots, the last passing rate fell sharply on both engines:

| Workload (16 slots) | vLLM | SGLang |
|---|---|---|
| 32 adapters, Zipf popularity | 20 rps | 20 rps |
| 32 adapters, uniform popularity | 10 rps | 10 rps |

Past the cliff the two engines differ a lot. At 40 rps with Zipf popularity, p95 TTFT was about 1.0 to 1.6 s on vLLM and about 12 s on SGLang (11.8 to 13.5 s across repeats).

## 3. Slots set the cliff [MEASURED, and EXPLORATORY for SGLang]

vLLM, 32 adapters, p95 TTFT range over 3 repeats:

| | 8 slots | 16 slots | 32 slots |
|---|---|---|---|
| uniform, 20 rps | 2471 to 2937 ms | 496 to 937 ms | 93 to 99 ms |
| uniform, 40 rps | not run | 2277 to 3080 ms | 142 to 151 ms |
| Zipf, 40 rps | 4669 to 5794 ms | 957 to 1564 ms | 136 to 146 ms |

SGLang, Zipf with 32 adapters (16 against 32 slots): at 40 rps, 11.8 to 13.8 s against 94 to 108 ms; at 80 rps, 9.7 to 9.8 s against 126 to 131 ms. Throughput at 80 rps rose from about 4,100 to about 5,800 to 6,300 tok/s.

The practical rule for both engines: size the slot count to the number of adapters with requests in flight at the same time. I did not find a defect here. It is a configuration cliff, and on SGLang the failure past it is much more severe than on vLLM, with latency going from about 100 ms to about 12 s without a warning. Why SGLang degrades so much more past the cliff is not explained. From reading its source I tested three scheduling flags (anti-starvation drain, overlap loading, FIFO eviction) at 16 slots. None removed the collapse (drain at 1 s lowered 40-rps p95 by about 16%, to about 10 s).

Working-set analysis [EXPLORATORY, no new runs]: I checked that advice against the request logs of the main grid (32 adapters, 16 slots, Zipf and uniform traffic, every rate, vLLM and SGLang, 3 repeats). A cell is healthy if its p95 TTFT was under 100 ms. The count of adapters with a request in flight (sent, not finished) was above 16 for at most 9% of the time in every healthy cell and for 72% or more of the time in every degraded cell. Distinct adapters over longer windows do not separate them: over 1 s windows the count was above 16 for 67 to 69% of the time at Zipf 20 rps and 32 to 33% at uniform 10 rps, both healthy, and over 5 and 10 s windows it was above 16 for 99 to 100% of the time in cells such as Zipf 10 rps and uniform 10 rps, also healthy. Caveat: the in-flight count includes requests waiting in the engine's queue, so in degraded cells it is partly a result of the slowdown, not just a cause. This is correlation, not mechanism. It covers 16 slots, one model and one GPU. Details: `followup/working_set/RESULTS.md`.

A relevant change landed after the version I tested. SGLang PR #46500 (a LoRADrainer revamp, merged 2026-10-29) describes this situation. It is on main but not in v0.5.23: the drainer file at the v0.5.23 tag is identical to the file before that PR. A comparison against main is planned. [PENDING]

### 128 adapters on 32 slots [EXPLORATORY]

With four times more adapters than slots the cliff returns, at lower rates than with 32 adapters on 32 slots. p95 TTFT, median (min to max) over 3 repeats, in ms:

| Engine | Popularity | 10 rps | 20 rps | 40 rps | 80 rps |
|---|---|---|---|---|---|
| vLLM | Zipf | 70 (66–72) | 113 (98–114) | 3,011 (2,424–3,057) | 5,537 (5,076–5,627) |
| vLLM | uniform | 81 (77–88) | 815 (564–1,520) | 5,440 (4,817–5,480) | not run |
| SGLang | Zipf | 63 (61–69) | 82 (72–5,806) | 11,661 (11,199–13,399) | 12,890 (11,856–13,169) |
| SGLang | uniform | 88 (76–90) | 351 (250–1,032) | 17,408 (16,593–17,741) | 17,544 (17,331–17,898) |

Last passing rate: vLLM 20 rps (Zipf) and 10 rps (uniform); SGLang 20 rps on both. At 20 rps the ranges overlap the limit on several rows (SGLang Zipf had one repeat at 5.8 s), so I do not rank the engines there. Past the cliff SGLang is again much worse than vLLM at similar throughput. I did not test 64 slots or measure memory per slot.

## 4. Replication: does SGLang's csgmv backend beat triton? [MEASURED]

SGLang's documentation claims 20 to 80% lower latency with csgmv than with the triton backend. The threshold I fixed in advance was at least 20% lower median p95 at concurrency 32 and above, with non-overlapping ranges. Result: not held on this setup. At concurrency 128, csgmv was 3.6% lower on end-to-end p95 and 4.7% lower on TTFT p95 (ranges did not overlap), and throughput was about 4.5% higher. At concurrency 32 the TTFT difference was 11% but the ranges overlapped. This is one small model on one GPU. It does not show the documentation is wrong elsewhere.

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

## 5. mistral.rs [MEASURED; cause PENDING]

mistral.rs handled the base model up to 10 rps and one adapter up to 10 rps, but failed at 20 rps for both. With 32 adapters under Zipf popularity p95 TTFT was 123 ms at 5 rps but 6 requests failed (the ladder counts any error as a violation), and at 10 rps it was 24.6 s. The reduced plan did not run mistral.rs with uniform popularity.

A closed-loop probe with 8 adapters loaded shows where the time goes. At 32 concurrent requests, time per output token was 12.1 ms with no LoRA, 12.8 ms with LoRA enabled and no adapter used, 18.3 ms with 1 adapter, 21.6 ms with 4 and 26.0 ms with 8. Throughput with 8 adapters was about half the base model's. At concurrency 8 it was 831 against 427 tok/s. Each cell is one run of 150 requests, so differences of a millisecond or so are noise. In 8-adapter runs 1 to 2 requests per run failed with a "no tokens" error, which is unexplained.

### Where the time goes [EXPLORATORY, timing experiment]

Reading the source (v0.9.5, `mistralrs-quant/src/lora/dynamic/cuda.rs`), the dense LoRA path loops over the adapters active in the batch and launches one kernel per adapter at each of 196 sites per decode step (28 layers × 7 modules). The repo has a grouped kernel for MoE LoRA but I found none for dense layers, and no existing issue or pull request about this.

To measure how much that loop costs I built a patched binary, gated by an environment variable, that runs a single adapter over all rows. **Its outputs are wrong by design, so it is a timing measurement and not a fix.** Time per output token, median of 3 repeats, 8 adapters in a uniform mix (ms):

| Setup | Normal | One kernel call | One call and no per-adapter lookups |
|---|---|---|---|
| concurrency 8 | 17.1 | 12.6 | 12.5 |
| concurrency 32 | 26.0 | 19.0 | 18.8 |
| concurrency 32, no adapter used | 14.9 | 15.3 | 15.0 |

Collapsing the launches cut time per token by about a quarter (throughput up about 35%: 445 to 606 tok/s at concurrency 8 and 1,007 to 1,355 at concurrency 32), and skipping the per-adapter lookups and layout checks as well changed nothing measurable. About 4 ms per token of LoRA cost remains at 8 adapters even with one call, which I have not explained. I could not count kernel launches directly (Nsight Compute is blocked in this container, error ERR_NVGPUCTRPERM), so the launch count itself is not measured. A separate effect showed up in the 1-adapter runs: the first 150-request run after a change in traffic type was slower. On a fresh server, a 1-adapter run directly after a no-adapter run took 24.7 ms per token and the next identical run took 18.0 ms, while 1-adapter traffic from the start gave 18.4 to 18.8 ms. I did not find the cause. It means single runs right after a change of traffic can read about 35% slow, so the sequence of runs matters. The 1-adapter and 2-adapter cells of the scaling probe may include some of this effect, and the normal-versus-patched comparisons are not affected because both ran in the same order after an 8-adapter warm-up.

## 6. Prefix caching with adapters [MEASURED on vLLM]

On vLLM, shared-prefix traffic with 32 adapters behaved like plain 32-adapter traffic (last passing rate 10 rps), so no separate cost shows up. SGLang keeps a separate prefix cache per adapter (read in its source). I did not isolate it.

Routing with prefix caching on [FLAGGED]: the host was not quiet during these passes (external CPU use 14 to 17 cores against the plan's limit of 2). Routing correctness does not depend on speed, but the plan's rule applies, so they stay flagged until the SGLang radix-cache passes are repeated in a quiet window (planned, [PENDING]). The prompts share prefixes across adapters; noise 0.0204, margin threshold 0.0611, all four power checks passed on them.

- SGLang 0.5.23, radix cache on, 16 in flight: 64 of 64 requests routed correctly in the cold pass (fresh server) and the warm pass straight after (smallest margins 0.111 and 0.113). The cache served 95.6% (cold) and 100% (warm) of the shared-prefix tokens. A control on fresh servers showed same-adapter reuse of a 579-token prefix (579 cached) and none across adapters (0 cached on the opening call of the other adapter).
- vLLM 0.34.0, prefix caching on: prompt-logprob calls report 0 cached tokens, so they bypass the cache and the log-probability rule could not be applied with the cache in use. As a fallback I compared 32-token greedy completions of the 64 adapter and prompt pairs. The cache served 96.0% of the shared-prefix tokens in the warm run (65,280 hits of 74,272 queried tokens). A control on fresh servers gave 0, 576, 0 and 576 cached tokens, so no cross-adapter hit, and no misroutes. The check is coarse: two identical cache-off runs agree on just 60.9% of completions, so it would catch a gross adapter swap, not a subtle one.
- mistral.rs: [PENDING].

Neither result is a correctness guarantee for an engine. One model, rank 16, four adapters at scale 0.02, 16 prompts in two prefix groups. Raw data: `followup/prefix_routing/`.

## 7. Things I got wrong along the way

- I once retracted a mistral.rs finding because my diagnostic ran the base model only. The tuning data used 8 adapters. The re-run reproduced the slowdown and I corrected the changelog.
- I estimated mistral.rs's capacity from a closed-loop run at concurrency 8, which does not measure capacity. I withdrew that figure.
- I wrote "size for the working set, not the instantaneous count" before checking it. The analysis showed the reverse for 1 to 10 s windows.
- The prefix-routing control was initially run on a server whose cache earlier calls had written, so a hit could not show crossing. The plan was amended before any pass and the control rerun on fresh servers.
- A rebase on another machine deleted follow-up records from the changelog and plans. They were restored unchanged, and a check now blocks staged changes that delete records.
- A network volume lost its contents mid-study, so the first vLLM grid was rerun. Its per-step summaries agree with the rerun to within a few percent.
- The first SGLang LoRA session failed my drift check. I added a second warm-up and reran it. The discarded session's numbers matched the rerun closely.
- I wrote that SGLang and vLLM had "same slots, same traffic" and called the SGLang cliff a configuration cliff. In my launch the SGLang decode CUDA graphs covered batches of at most 48, and vLLM's covered far more. Raising the SGLang limit with one flag took 40 rps p95 TTFT from 13.5 s to 2.1 to 3.1 s on 16 slots (3 starts, shared host, screening only). The engines were not at the same setting, so part of the SGLang slots story is this. Not yet rerun on the dedicated machine.

## 8. Bugs and upstream work

I filed mistral.rs #2567 (the `/v1/completions` logprobs layout differs from the OpenAI format; a local fix exists but no pull request is open). Searching upstream first showed two other candidates were already reported or fixed. The mistral.rs throughput finding is not yet root-caused.

## What I would do with this

- Count the adapters your traffic touches at once and set slots to at least that number. Watch for a latency jump, not a gradual rise.
- Do not assume more slots are free: I did not measure the memory cost per slot.
- Treat a single-GPU, one-model result as a reason to run your own check.

## Disclosure

AI coding tools were used throughout, including for analysis scripts, code reading and drafts. The study design, the checks, and the conclusions are mine to defend, and anything I have not reviewed line by line is marked as such.

## Appendix

Repo, raw data, scripts, versions, flags and exact commands: https://github.com/brianosaurus/multilora-serving-study (the repo's latest pushed commit does not yet contain the replication, sweep and probe results above).
