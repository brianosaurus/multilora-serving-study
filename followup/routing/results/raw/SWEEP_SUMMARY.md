# Calibration sweep: result (routing follow-up, plan: `../../PLAN.md`)

Rule unchanged: k = 3 and the four power checks. Stand-in for a correct implementation: PEFT bf16; reference: PEFT fp32; 16 prompts, 128 forced tokens, adapter seeds 0 to 3. No engine was run. Passing the checks does not certify any engine.

| B scale | Noise (median bf16-vs-fp32) | Threshold (3 x noise) | Check 1: correct stand-in routed correctly | Margin min / median | Check 2: swapped adapters accepted | Check 3: base under an adapter name accepted | Check 4: pairs above threshold (smallest pair) | Passes all four |
|---|---|---|---|---|---|---|---|---|
| 0.01 | 0.0167 | 0.0500 | 64 / 64 | 0.061 / 0.131 | 0 / 192 | 0 / 64 | 10 / 10 (0.141) | yes |
| 0.02 | 0.0201 | 0.0604 | 64 / 64 | 0.171 / 0.343 | 0 / 192 | 0 / 64 | 10 / 10 (0.340) | yes |
| 0.05 (the study's) | 0.970 | 2.91 | 10 / 64 | 0.449 / 2.058 | 0 / 192 | 0 / 64 | 7 / 10 (1.954) | no |
| 0.1 | 10.08 | 30.2 | 0 / 64 | 0.806 / 2.577 | 0 / 192 | 0 / 64 | 0 / 10 (8.57) | no |

Protocol: the smallest passing scale is used, which is **0.01**. Its smallest request margin (0.061) clears its threshold (0.050) by a factor of 1.2; 0.02 clears its threshold by 2.8. The scale 0.02 run was restarted after a pause; its first, interrupted attempt is not used.

Caveat: at 0.01 and 0.02 the study's own calibration found the adapters changed the output within 8 tokens on too few of 16 prompts, so these are weak adapters. They share the study's seeds, not its weights; checksums are in each `sweep_s*/log_and_checksums.txt`.
