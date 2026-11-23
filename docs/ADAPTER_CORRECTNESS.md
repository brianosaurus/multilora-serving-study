# Adapter correctness, and the grouped-kernel check: what was measured and what it shows

**Bottom line.** Adapter correctness is **not certified** for vLLM 0.34.0, SGLang 0.5.23 or mistral.rs v0.9.5. Both checks I fixed in advance turned out to be unable to tell a correct engine from a wrong one, and a known-correct implementation failed them too. What the evidence supports is narrower, and it is listed below.

The adapters are synthetic (rank 16, all linear layers, seeded random weights, B scale 0.05). They produce incoherent text, so they exercise serving mechanics and routing between adapters, not the quality of a useful fine-tune. Nothing here says anything about real fine-tuned adapters.

## What is supported

- **The adapter is applied in all three engines.** No engine's output resembles the base model's.
- **vLLM and SGLang follow the PEFT reference closely at the first token** (table below). This is evidence, not certification.
- **mistral.rs deviates more at the first token, and the deviation is consistent with a precision difference.** An unmerged PEFT run in fp32 is as far from the bf16 reference (mean distance 0.717) as mistral.rs is (0.72). So the data show no adapter bug in any engine and cannot show there is none.

## What was checked

All checks use 4 adapters (seeds 0 to 3), 16 fixed prompts, greedy decoding, and the same 64 requests in serial mode and in mixed-batch mode (requests for all 4 adapters sent together). The reference is Hugging Face transformers plus PEFT.

| Check | vLLM | SGLang | mistral.rs | Known-correct stand-in |
|---|---|---|---|---|
| 1. Output text matches the right adapter's reference, serial / mixed (pre-registered, bar 90%) | 71.9% / 68.8% | 65.6% / 65.6% | 42.2% / 35.9% | **43.8%** |
| 2. First-token log-probabilities close to the right variant, serial / mixed (post-hoc replacement, bar 90%) | 98.4% / 96.9% | 98.4% / 96.9% | 78.1% / 79.7% | **84.4%** (below the bar) |
| Mean distance to the correct variant, serial / mixed | 0.34 / 0.38 | 0.35 / 0.35 | 0.72 / 0.72 | 0.68 |

The stand-in is the adapter merged into the base weights in bf16, a mathematically equivalent implementation.

- **Check 1 lacks power.** With these chaotic adapters, outputs diverge from the reference within one or two tokens even when the math is equivalent, and the stand-in scores 43.8%. Every engine failing it is therefore not evidence about the engines.
- **Check 2 was added after seeing check 1's results,** so it is weaker evidence than a criterion fixed beforehand. The rule was written down first: the stand-in had to score at least 90% on it before it could certify any engine. It scored 84.4%, so the metric was declared invalid and nothing is certified by it. The engine scores in the table are reported for the record.
- **The bf16 reference is not ground truth for these adapters.** Merging into bf16 weights rounds away much of a small adapter perturbation. An fp32 PEFT run, the same math in higher precision, scores 84.4% against the bf16 reference with mean distance 0.717, about as far as mistral.rs. That vLLM and SGLang sit closer probably reflects shared bf16 numerics with the reference (an inference, not tested).

## The mistral.rs response layout and the first-token check

The first-token check reads each engine's top-20 log-probabilities. mistral.rs v0.9.5 returns `top_logprobs` entries with the token id in `token` and the decoded text in `bytes`, the reverse of the OpenAI format and of the top-level entry in the same response (issue #2567, draft fix in PR #2570). The harness parser first did not handle that layout, so mistral.rs was reported "not checkable"; the parser was fixed and the engine rerun, and no mistral.rs result had been seen before the fix. The layout therefore sits underneath the mistral.rs first-token numbers above.

Checked against a live server (CPU build of upstream `master` at `21da642`, Qwen2.5-0.5B-Instruct, `/v1/completions`, `logprobs: 3`, first position): before the fix, `top_logprobs[0]` was `{"token": 12095, "bytes": " Paris"}`; after, `{"token": " Paris", "bytes": [32, 80, 97, 114, 105, 115]}`. The top-level entry was identical in both, and only the `top_logprobs` entries changed.

## Follow-up checks after the 24 hours (exploratory, not part of the study)

1. **mistral.rs empty completions, checked against an fp32 PEFT reference.** mistral.rs returned empty completions ("no tokens") on a few requests with adapter 4. For the two prompts that do so in every run, the reference ranks end-of-text first for adapter 4 (log-probability -3.68 and -3.03), so the empty output is the model's own choice. A third prompt is a genuine near-tie in the reference (end-of-text rank 2, 0.2 nats behind a comma) and flips with batch composition in both a patched and an unpatched build. One prompt is a gap I did not explain: the reference puts end-of-text at rank 9 (-5.44), while mistral.rs's own log-probabilities put it near the top (-3.96). It is consistent with bf16 against fp32 numerics on a flat distribution but I did not show that. This is one adapter and seven prompts.
2. **The grouped dense-LoRA kernel (mistral.rs draft PR #2571) matches the existing per-adapter path bit for bit** in unit tests (fp16 and bf16, mixed ranks, an adapter with no weights at a site, more than 32 adapters) and in 128 end-to-end completions (16 prompts, 8 adapters, temperature 0). That shows the new path equals the old one. It does not show that the old one is correct. The full LoRA test module passes on the GPU (81 tests, 0 failed), and `cargo fmt --check` and `clippy -D warnings` pass. Running with `CUDA_LAUNCH_BLOCKING=1` did not remove the few extra empty completions seen on two requests, which argues against a timing race; the near-tie analysis in item 1 and the same flips in the unpatched build are the better explanation.
3. **Empty-completion counts across engines are not comparable.** vLLM and SGLang recorded 0 empty completions in 30,000 adapter requests each, mistral.rs 10 in 9,000, but the runs used different prompt and adapter mixes. Whether the engines really differ was not tested.

## The grouped kernel's speed (exploratory, after the 24 hours)

The existing CUDA path launched a shrink and an expand kernel for every active adapter at every one of 196 sites per decode step (28 layers by 7 modules). The grouped kernel launches one pair per site. Measured on a different, busier pod than the study (RTX 4090, build without flash-attn, so the absolute numbers do not compare with anything above), Qwen2.5-1.5B-Instruct, 8 rank-16 adapters, closed loop, 150 requests per run.

Design: three binaries (the old build, the new build and a second copy of the old build as a same-versus-same control) in all 6 orderings, so each binary ran in each position twice. An earlier 3-repeat run always started with the old build and its identical-code control cells came out up to 23% apart, so I discarded it as order-confounded and kept the data. No cell was throttled (lowest GPU clock 2,520 MHz). The shared host's load average ranged from 25 to 250.

Median time per output token over the 6 blocks, and the median of the per-block changes:

| Concurrency | Adapters | Before (ms) | After (ms) | Time per token | Throughput | Per-block range | Faster in |
|---|---|---|---|---|---|---|---|
| 8 | 8 | 36.0 | 19.8 | -45% | +59% | -35% to -51% | 6 of 6 blocks |
| 32 | 8 | 52.8 | 38.3 | -28% | +32% | -19% to -51% | 6 of 6 blocks |
| 32 | 4 | 43.5 | 36.6 | -15% | +17% | +27% to -46% | 5 of 6 blocks |
| 32 | 1 (same code) | 36.3 | 30.9 | -19% | | +62% to -44% | 5 of 6 blocks |
| 32 | 0 (same code) | 14.1 | 14.8 | +2% | | -21% to +12% | 3 of 6 blocks |

- **At 8 adapters the new build was faster in all 6 blocks at both concurrencies** (exact two-sided sign test p = 0.031, the smallest possible with 6 blocks). At 4 adapters the direction held but was not established (5 of 6, p = 0.22).
- **The identical-code rows are the error bar.** The 1-adapter and no-adapter paths are the same code in both builds, yet they moved by up to 19% in the median and by far more in single blocks. The same-versus-same control differed from the old build by +5% (range -4% to +32%) at concurrency 8 and +8.5% (range -8% to +54%) at concurrency 32. So single-block percentages are noisy and only the consistent sign across blocks is solid.
- **It does not close the gap.** At concurrency 32, 8 adapters are still about 2.7 times slower per token than no adapters (38.3 against 14.1 ms), and I cannot explain the rest.
- **Limits:** one model, one GPU type, rank 16, a noisy shared host. The kernel is still a per-row dot product with no weight-tile reuse, unlike the stacked-weight Triton kernels in vLLM and SGLang. A larger batch or prefill workload was not measured.
- **An earlier, cruder estimate agrees:** a timing-only patch that sent all rows through one adapter (wrong outputs by design) took about 27% off at 8 adapters on the study's pod.

## What this does not show

- Correctness for any real fine-tuned adapter, other ranks, other models, other precisions, or quantized weights.
- Anything about the 128-adapter runs beyond the first four adapters, which were the ones checked against a reference. Adapters 0 to 3 were verified byte-identical to the calibration set the reference was computed from.
- That any engine is wrong. Nor that any is right.

## Where the data is

- Pre-registered criteria and the amendment: `docs/PLAN.md`, section "Correctness check".
- Every change and its reason: `docs/CHANGELOG.md`.
- Raw results: `results/raw/correctness/` and `results/raw/calibration/`; scripts `scripts/check_engine.py`, `scripts/first_token.py`, `scripts/noise_floor.py`, `scripts/reference_outputs.py`.
- The follow-up reference, near-tie and output-check data: `results/raw/correctness_followup/`.
- The grouped-kernel benchmark, the `CUDA_LAUNCH_BLOCKING` runs, the scripts and the commit as a patch: `results/raw/kernel_followup/`.
