# Differential study: results so far (AI-drafted, not reviewed line by line)

**Status:** Axis 1 phase 1 (Qwen2.5-1.5B-Instruct only) is done. Gate 2 is not reached: Llama and Mistral models, Axis 1 cells C1-C5, and Axes 2-4 are not run. Nothing here is a ranked finding yet. Machine: runpod2 (`bc2389eb52de`), RTX 4090.

## A correction made on the way (see CHANGELOG)
The first Qwen reference was generated with Hugging Face's default generation config, which applies Qwen2.5-1.5B's `repetition_penalty=1.1` even with `do_sample=False`: 8.9% of its tokens were not the raw top-1. All first-pass numbers (including an apparent 17x logprob deviation for mistral.rs) are **superseded** and kept in `results/raw/superseded_hfdefault_ref/` for the record; they must not be quoted. The corrected reference sets `repetition_penalty=1.0`; its repeat is exact (64/64 identical generations, 0.0 logprob difference) and 0.7% of its tokens are not the top-1 of its own full-forward pass (decode-vs-prefill numerics at near-ties).

## Cells run (Qwen2.5-1.5B, defaults + the flags needed to serve; 64 prompts)
Method: vLLM and SGLang are teacher-forced (engine scores the reference's tokens). mistral.rs has no prompt logprobs, so it uses generate-and-compare over the prefix before the first token mismatch; its numbers are a different method and not pooled.
D = mean |engine logprob - reference logprob| per token (nats), mean over prompts, 95% bootstrap CI; N_e = the engine's own noise floor (max of run-to-run spread and D on 16 trivially short prompts).

| Engine | N_e | C0 baseline | C7 32 concurrent | C6 cache on | C6 cache off | Flagged by the pre-registered rule |
|---|---|---|---|---|---|---|
| vLLM 0.34.0 | 0.0218 | 0.0073 | 0.0070 | 0.0072 | 0.0072 | none |
| SGLang 0.5.23 | 0.0199 | 0.0076 | 0.0075 | 0.0072 | 0.0073 | none |
| mistral.rs v0.9.5 | 0.0207 | 0.0068 | 0.0086 | 0.0069 | 0.0067 | none |

- **Cache hits are valid** in every cache-on cell, confirmed by the engine's own counters (vLLM 2,048 tokens hit; SGLang 4,603 cached tokens; mistral.rs 64/64 lookups, 3,744 tokens reused) and zero in every cache-off cell. (vLLM does not use the prefix cache for requests with `prompt_logprobs`, so teacher-forced vLLM requests show 0 cached tokens; its cache cell uses generation mode.)
- **Greedy generation with explicit neutral sampling parameters** matches the raw reference exactly on 47-50 of 64 prompts for all three engines; the mismatches start at a median position of 37-46 with gap at most 0.125-0.25 nats (near-ties). Cache on vs off does not change this beyond 2 prompts for mistral.rs (47 vs 49 identical), which is within the noise of this sample size.
- **Concurrency (C7)** has no measurable effect on D for any engine.
- **No cell exceeds its flag threshold.** All three engines agree with transformers bf16 to about 0.007 nats per token on these cells.

## C8: what a request with only `temperature=0` produces (defaults)
Greedy generation vs the raw reference (repetition_penalty=1.0) and vs the HF-default reference (repetition_penalty=1.1 applied):

| Engine | identical to raw ref | identical to HF-default ref | reading |
|---|---|---|---|
| vLLM | 9/64 (55 mismatch, first mismatch median pos 8, median gap 1.37 nats) | 27/64 (+14 length-only) | behaves like HF's default: applies the model's generation_config penalty |
| mistral.rs | 9/64 (same statistics) | 26/64 (+13 length-only) | same as vLLM |
| SGLang | **50/64** (same as its neutral run) | 9/64 | does **not** apply the generation_config repetition penalty by default |

So with the same request body, SGLang returns the raw greedy output and vLLM and mistral.rs return penalized output matching Hugging Face's default behavior. Not yet established: whether this is each engine's documented default (design choice) or an unintended difference. That check (docs and source) is pending; it is not ranked yet. Related prior work reported an analogous default-parameter difference for another engine (arXiv 2605.19537, Ollama).

## Limits
One model; one run per cell (repeat spread measured separately: vLLM 0.0, SGLang 0.0066, mistral.rs 0.0055); synthetic-but-fixed prompts; 64 prompts so CIs are about ±0.001 nats; "identical" counts are small samples.
