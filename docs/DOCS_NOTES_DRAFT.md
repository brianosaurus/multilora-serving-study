# Draft docs notes for vLLM and SGLang (NOT opened anywhere; for the author's review)

Both rest on one benchmark: Qwen2.5-1.5B-Instruct, one RTX 4090, synthetic rank-16 adapters, open-loop Poisson load, 3 repeats, 500 requests per run. Raw data: `results/raw/sweep_m8`, `sweep_m32`, `main/vllm` (vLLM) and `results/raw/sgl_flags`, `main_rerun1/sglang` (SGLang) in https://github.com/brianosaurus/multilora-serving-study. If a PR is opened, its description follows the author's rules (no "Test plan" section on mistral.rs; AI-tool help disclosed; own root cause and verification stressed).

## vLLM: docs/features/lora.md, new short subsection (suggested after the paragraph that lists `max_loras`)

### Sizing `max_loras`

`max_loras` is the number of adapters that can be active in one batch. If requests in flight name more distinct adapters than `max_loras`, requests for the other adapters wait, and time to first token can rise sharply. In one benchmark (Qwen2.5-1.5B-Instruct, one RTX 4090, 32 adapters, uniform traffic at 20 requests per second), the p95 time to first token was 2.5 to 2.9 s with `--max-loras 8`, 0.5 to 0.9 s with 16, and about 0.1 s with 32. As a starting point, set `--max-loras` to at least the number of adapters you expect to be active at the same time. Larger values reserve more GPU memory.

Check before opening: the first sentence ("number of adapters that can be active in one batch") against vLLM's own wording of `max_loras`; the claim that the others wait, which I read in `vllm/v1/core/sched/scheduler.py` (the per-step cap near lines 1130 to 1139 of 0.34.0) and have not re-read on current main; and whether the docs prefer to avoid benchmark numbers (some maintainers do).

## SGLang: docs/docs/advanced_features/lora.mdx, extend the existing paragraph

Current text (about line 434): "When requests target more adapters than fit in the pool, requests for additional adapters wait for a slot. To improve fairness under skewed traffic, set `--lora-drain-wait-threshold` to a positive value. Its default, `0`, disables draining."

Suggested addition after the first sentence: "The wait can be long. In one benchmark (Qwen2.5-1.5B-Instruct, one RTX 4090, 32 adapters with Zipf-distributed traffic at 40 requests per second), the p95 time to first token was about 12 s with `--max-loras-per-batch 16` and about 0.1 s with 32. A drain threshold of 1 s lowered it by about 16% in the same test, to about 10 s, so setting `--max-loras-per-batch` to at least the number of adapters active at once is the more effective change."

Check before opening: the 16% figure is from 3 repeats at one setting (1 s is my pick), so I would drop that sentence if the maintainers prefer docs without tuning claims; the doc says `max_loras_per_batch` counts the base-model "None" entry, which my 32-adapter runs did not exercise, so "at least the number of adapters active at once" may need "plus one if base-model requests are mixed in".

## Not claimed in either note

The cause of the engines' differing behavior past the limit, any memory cost per slot, any result at other models, GPUs or adapter ranks.
