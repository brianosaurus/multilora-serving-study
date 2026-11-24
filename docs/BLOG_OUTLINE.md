# Blog post outline (draft, not for publication)

Status key: SOLID = measured and reproduced. PENDING = being measured now. IF = only if the data supports it. Nothing here goes to brian.biz, LinkedIn, or a resume until the author has read the draft.

Working title: "What happens when your LoRA adapters outnumber your slots" (change if the sweep says otherwise)

## 1. The question (SOLID)
One RTX 4090, three open-source engines (vLLM, SGLang, mistral.rs), many LoRA adapters. Why this matters: one base model, many fine-tunes, one GPU. State up front: single day, one model, synthetic adapters, 3 repeats.

## 2. How the study was run (SOLID)
Pre-registered plan, changelog of every change, open-loop Poisson load, 500 ms p95 TTFT as the line, reference runs to catch drift, raw data in the repo. One short paragraph on what went wrong and how it was handled: correctness criteria lacked power, a network volume lost data mid-run, and the grid was rerun.

## 3. Result: the cliff (SOLID for vLLM)
vLLM last passing rate by workload: LoRA off 80 rps; LoRA enabled with no adapter 80; one adapter 80 (about 20% lower throughput at 160); 32 adapters Zipf 20; 32 adapters uniform 10. Include the table with min-max ranges, and say the ladder is coarse. Enabling LoRA costs nothing measurable.

## 4. Why: slots (SOLID for the effect, mechanism read from source only)
The `max_loras` sweep (replication B, 8/16/32 slots) shows slot count sets the cliff on vLLM; SGLang with 32 slots shows the same. 128 adapters on 32 slots brings the cliff back at lower rates. Advice: size slots to the adapters active at once. Why SGLang degrades far worse than vLLM past the cliff is NOT explained (drain, FIFO, overlap flags did not remove it).

## 5. The other engines (SOLID)
SGLang and mistral.rs measured. Replication A: csgmv vs triton NOT HELD (about 4 to 5% on this setup). mistral.rs: cost grows with distinct adapters per batch; a timing-only patch that collapses the per-adapter launches cuts time per token about 25%; bookkeeping is not the cost. Launch count not measured (Nsight Compute blocked). 1-adapter anomaly open.

## 6. Prefix caching with adapters (SOLID for vLLM, PENDING others)
On vLLM, shared-prefix traffic with 32 adapters behaves like plain 32-adapter traffic: no separable extra cost. SGLang keeps a separate prefix cache per adapter by design (from its issue history). Keep this short.

## 7. What I could not establish (SOLID)
Adapter correctness not certified for any engine (both pre-registered criteria lacked power; the first-token metric failed its own validity test). Synthetic adapters test serving cost, not fine-tune quality. One model, one GPU, one day.

## 8. Bugs and upstream work (SOLID so far; mistral.rs issue draft pending the author's review)
Searching upstream first avoided two duplicate reports (an SGLang ninja lookup already had two open PRs; a mistral.rs memory 503 was already fixed). Filed: mistral.rs #2567 (logprobs layout). IF B6 is root-caused: the mistral.rs concurrency finding. Otherwise one honest paragraph saying where it stands. Describe the bug work as found-while-benchmarking, not as a campaign.

## 9. Takeaways (IF, written last)
Three or four rules a reader can act on, each tied to a number in section 3 or 5. No "first", "only", or "best". Disclose that AI tooling helped with analysis and code, and that the design, checks, and conclusions are the author's.

## Appendix
Repo link, raw data, scripts, versions, flags, and the exact commands.
