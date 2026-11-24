# 3a result: routing with prefix caching on (pod runs 2026-11-24, 06:35 to 06:48 UTC; plan and amendments in PLAN.md)

**Flag, under the plan's quiet rule:** external CPU use read 14 to 17 cores before and after every pass (and 6 to 11 cores afterwards with nothing of ours running), against the plan's limit of 2 cores. No continuous sampler ran during the passes. By the plan these passes are flagged. Routing correctness does not depend on speed, but the plan says a flagged run is reported as flagged and repeated in a quiet window; that repeat has not been done. Raw data: `results/raw/`; analysis text: `results/raw/analysis_3a.txt`; produced by `analyze_passes.py`.

Fixed numbers (from the power gate in PLAN.md): noise 0.0204, margin threshold 0.0611. All four power checks passed on these prompts.

## SGLang 0.5.23 (radix cache on, probe passes, 16 in flight)

| Pass | Routed correctly | Margin min / median | Right-adapter distance median / max | Shared-prefix tokens reported cached |
|---|---|---|---|---|
| cold (fresh server) | 64 of 64 | 0.111 / 0.308 | 0.0196 / 0.0596 | 95.6% |
| warm (same server, straight after) | 64 of 64 | 0.113 / 0.305 | 0.0193 / 0.0437 | 100% |

No incomplete requests, no errors. Control on fresh servers: same-adapter reuse of a 579-token prefix (579 cached), none across adapters (0 cached on the first call of the other adapter), for probe and plain calls.

## vLLM 0.34.0 (prefix caching on; fallback comparison, because probe calls do not read the cache)

- Probe calls (`prompt_logprobs`) report 0 cached tokens, so the log-probability rule could not be applied with the cache in use. The fallback was run instead: 32-token greedy completions of the 64 (adapter, prompt) pairs, 16 in flight.
- Cache used: 96.0% of the shared-prefix tokens reported cached in the warm run; counters over the cache server's runs: 65,280 hits of 74,272 queried tokens.
- Control: plain calls on a fresh server: 0, 576, 0, 576 cached tokens (same adapter reuses, the other adapter gets 0). No cross-adapter hit.
- Agreement of text: runs on the no-cache server agree with each other on 60.9% of completions (f0); the warm cache run agrees with the no-cache runs on 60.9% (fw). Misroutes by the plan's definition: 0.
- Rule from the plan (fw at least f0 minus 1/64, cache used, no misroute): met.

## What this means, and what it does not

No misrouting was seen on these requests for SGLang (log-probability rule, 64 of 64 in both passes) or vLLM (completion comparison). The vLLM check has low resolution: even two runs of the same server without a cache agree on only 61% of completions at these adapter scales, so a small contamination effect would not show; it could catch a gross swap of adapters, not a subtle one. The vLLM probe-call path never touches the cache, so the log-probability rule says nothing about vLLM's cached path. This does not certify either engine. One model, rank 16, four adapters at scale 0.02, 16 prompts in two prefix groups; forced-token and 32-token completions, not long decodes.
