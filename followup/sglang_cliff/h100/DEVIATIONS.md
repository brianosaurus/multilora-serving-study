# H100 check: the plan, its changes and what was dropped

Moved from the paper (section 6.9, brian.biz/multilora). Plan: `h100_amendment.md` (committed 2026-11-26 21:01 -0800, before any H100 data).

- **Scope as planned.** TP=1, TP=2 and TP=4; open-loop and saturating SGLang runs; a vLLM open-loop reference; GSM8K at TP=1 for both builds and at TP=4 for the patched build.
- **TP=4 was dropped, at the author's request.** `run.log` reads "SKIPPED (dropped by the author)" for the TP=4 open-loop, saturating, vLLM and GSM8K runs. The artifacts record no reason beyond the request, and none is supplied here. TP=4 is the case where the 20 to 35 GiB branch uses a range of 160 and the 80 GB tier uses 512, so it is the least covered configuration.
- **Added after the TP=1 results, not pre-registered:** one vLLM TP=1 saturating run (13,211 chunks/s), outside the decision rule.
- **Invalid original attempt kept.** The original TP=1 saturating runs hit the pod's 1,024 open-file limit on the client (2,280 of 3,000 requests failed with a connection error in both builds, none on the server). The data are kept as `results/invalid_fdlimit_*`; the script now raises the limit to 65,536 and the reruns are the reported numbers. An earlier TP=1 attempt over the network volume was also aborted (`results/aborted_networkfs_*`) because imports took over 15 minutes per server start; the reported runs used local-disk copies of the environments.
- **Result by the rule:** both TP=1 and TP=2 passed. Predictions h1, h2, h3 and h4 held as written (h3 made no claim about which build is faster).
- **Post-hoc, not in the plan:** a paired GSM8K analysis (`analysis/gsm8k_paired.py`), described in the paper's section 8 and the changelog.
