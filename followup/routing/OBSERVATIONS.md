# Observations after Gate A1 (post-hoc, exploratory, not a verdict)

These were observed after the pre-registered rule failed two of its four power checks (`PLAN.md`, Gate A1). They are descriptions of one reference run, not a certification of any engine and not a pass of the rule. Data: `results/raw/refs_out.json`, `refs_out_power.json`, `noise_diag_A1_A2.txt`, `a3_checks_out.txt`, `preview_metrics.txt`.

Setup: Qwen2.5-1.5B-Instruct, the study's adapters (seeds 0 to 3, B scale 0.05), 16 prompts, 128 forced tokens, fp32 PEFT as the reference, bf16 PEFT as the stand-in for a correct implementation.

## What was observed

- **Nearest candidate.** For the correct bf16 stand-in, the right adapter was the nearest candidate (smallest distance to the fp32 reference, among the four adapters and the base) on **64 of 64 requests**. The margin between the nearest wrong candidate and the right one had a minimum of 0.449 and a median of 2.058 nats (mean absolute per-token log-probability difference).
- **Misrouting is detected.** A stand-in serving adapter j under adapter i's name (192 requests, all ordered pairs) was routed correctly by the rule on 0 of 192, and the base model served under an adapter's name on 0 of 64.
- **The margin rule, as fixed, is not satisfiable by a correct implementation at this scale.** Noise (the median bf16-vs-fp32 distance for the right adapter) is 0.970, so k = 3 requires a margin above 2.909; 10 of 64 requests cleared it. Three of ten candidate pairs are separated by less than that.

## Why the noise is large (A1 to A3)

- The bf16-vs-fp32 distance for the **base model with no adapter** is 0.0156 (median over 16 prompts, range 0.0085 to 0.0225). With the study's adapters it is 0.970, about 60 times larger. The adapters make the network numerically fragile.
- It is not a few outlier positions. Pooled over 8,192 positions: median 0.477, 90th percentile 2.67, 99th percentile 6.58, maximum 13.65 nats; 31.4% of positions differ by more than 1 nat and 2.6% by more than 5; the top 5% of positions contribute 27.3% of the sum.
- The largest differences are tokens the fp32 model ranks 1 to 3 (logit 14 to 18) that the bf16 model ranks between 71 and 41,757 (its logit for them is 5 to 18 lower). Logit magnitudes are about 20 to 30, where bf16 spacing is 0.125, so rounding of the final logits cannot explain it; it is amplification through the layers.
- bf16 is not even self-consistent here: the same position computed in the same bf16 model from a full teacher-forced sequence or from the prefix alone differed by up to about 2 nats.
- Alignment and dtype checks found no bug: position t is log p(token t | tokens before t) (teacher-forced fp32 equals a prefix-only forward to 3 decimals), both runs used identical token-id arrays, the fp32 run is fp32 throughout with TF32 matmul off, the bf16 run has bf16 base weights and logits with fp32 adapter weights (the PEFT default), log-softmax is computed in fp32 in both, and attention is `sdpa` in both.

## What this does and does not show

It shows that the measurement has power in the weaker sense (a correct implementation is nearest to the right adapter and swapped or missing adapters are never accepted), and that the strict margin of the plan cannot be met because correct implementations are as noisy as the separation between adapters. It does not show that any engine routes correctly. No engine was run.
