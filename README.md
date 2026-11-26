# multilora-serving-study

Status: the pre-registered study has run, and follow-ups have been added since. The write-up is at https://brian.biz/multilora. The sections below this one were written before any run and are kept as the plan; they are not a description of the results. Every plan, amendment and correction is recorded in [docs/CHANGELOG.md](docs/CHANGELOG.md), which is append-only.

Read results with these limits in mind:

- One RTX 4090, one base model (Qwen2.5-1.5B-Instruct), synthetic rank-16 adapters. Nothing here transfers to other GPUs, larger models or real adapters without re-measuring.
- Results from the pre-registered plan are separated from results run after the plan was fixed. The later ones are labeled **EXPLORATORY** on the page.
- Runs on a shared host are labeled **FLAGGED**. They are screening results, not page-grade timing. The dedicated-machine rerun is still pending.
- The pre-registered correctness gate failed, so correctness is not certified for any engine (see [docs/ADAPTER_CORRECTNESS.md](docs/ADAPTER_CORRECTNESS.md)).
- Follow-ups went beyond the original scope (for example a grouped LoRA kernel for mistral.rs, under upstream review). The "Not in scope" line below describes the original plan only.

## Where the evidence is

| Topic | Plan | Results and data |
|---|---|---|
| Pre-registered grid (vLLM, SGLang, mistral.rs) | [docs/PLAN.md](docs/PLAN.md) | [results/raw/](results/raw/) (layout in [results/raw/README.md](results/raw/README.md)), [results/INTERIM_vLLM.md](results/INTERIM_vLLM.md) |
| Slot count against KV capacity, SGLang 128-adapter rerun | [followup/pod_items/PLAN.md](followup/pod_items/PLAN.md) | [followup/pod_items/RESULTS.md](followup/pod_items/RESULTS.md), [followup/pod_items/results/B/TABLE.md](followup/pod_items/results/B/TABLE.md) |
| SGLang latency cliff past the slot limit: the decode CUDA graph range, drainer, scheduling and eviction checks, and what is still unexplained | [followup/sglang_cliff/PLAN.md](followup/sglang_cliff/PLAN.md) | [followup/sglang_cliff/RESULTS.md](followup/sglang_cliff/RESULTS.md); raw runs in `followup/sglang_cliff/results/`; unposted upstream draft in [followup/sglang_cliff/draft-issue.md](followup/sglang_cliff/draft-issue.md) |
| csgmv against triton backend (SGLang's documented 20 to 80%) | [followup/csgmv_ab/PLAN.md](followup/csgmv_ab/PLAN.md) | summarized on the page; raw data under [results/raw/](results/raw/) |
| Adapters in flight against the working set | see [docs/CHANGELOG.md](docs/CHANGELOG.md) | [followup/working_set/RESULTS.md](followup/working_set/RESULTS.md) |
| Routing and prefix-cache checks | [followup/routing/PLAN.md](followup/routing/PLAN.md), [followup/prefix_routing/PLAN.md](followup/prefix_routing/PLAN.md) | [followup/prefix_routing/RESULTS.md](followup/prefix_routing/RESULTS.md), [followup/routing/OBSERVATIONS.md](followup/routing/OBSERVATIONS.md) |
| mistral.rs grouped LoRA kernel (rerun, incomplete) | [followup/kernel_rerun/PLAN.md](followup/kernel_rerun/PLAN.md) | [followup/kernel_rerun/results/](followup/kernel_rerun/results/) (B2 ended incomplete: disk full; recorded in [docs/BUGS.md](docs/BUGS.md)) |
| Dedicated-machine session | [followup/dedicated_session/PLAN.md](followup/dedicated_session/PLAN.md) | pending |

Open items, stated plainly: the cause of the roughly 50 ms stall after each SGLang adapter load, the trigger of SGLang's slow state, and the dedicated-machine rerun are not established.

## Question

How do current open-source inference engines behave when serving many LoRA adapters on one consumer GPU, and do published claims about multi-LoRA serving hold on this hardware?

## Scope

- Engines: SGLang and vLLM first, then mistral.rs in the same outcome-level comparison. mistral.rs is left out of the `max_loras` replication only, because it has no equivalent flag.
- Hardware: one NVIDIA RTX 4090 (24 GB). Results do not transfer to other GPUs without re-measuring.
- Time: the whole study is planned to run within one day, so the grid is deliberately small. See the Day plan in [docs/PLAN.md](docs/PLAN.md).
- Not in scope: a new serving system, kernel work, adapter training.

## Approach

1. **Replicate before comparing.** Start with two published claims and report whether they hold here:
   - SGLang's documentation says its `csgmv` LoRA backend gives 20% to 80% lower latency than the Triton backend. The docs do not state the workload or hardware.
   - A public vLLM sweep (see prior art) found `max_loras=16` gave the best throughput for 10 to 500 adapters under skewed traffic.
2. **Matched comparison.** Same base model, same adapters (rank and target modules), same prompts, same sampling settings, same traffic (uniform and skewed), and one TTFT limit used to define peak throughput. Each engine gets the same tuning budget from a documented list of settings, chosen on a separate tuning workload and measured on a held-out one. Each engine is also reported relative to its own no-adapter baseline, so "slower overall" is separated from "pays more for adapters".
3. **Correctness first.** Before any timing counts, check that each engine actually applies the adapter (outputs or logprobs against a PEFT reference).
4. **Measurement hygiene.** Warm-up and cache flushing, fixed request counts, repeated runs, variance and confidence intervals reported. Record engine versions and commit hashes, every flag, and the hardware state.

## Secondary goal

Find reproducible, fixable bugs in the engines. Observations are logged in [docs/BUGS.md](docs/BUGS.md), with what has and has not been verified for each. This is exploratory and is kept separate from the pre-registered measurements.

## Ground rules

- Negative and inconclusive results are reported as plainly as positive ones.
- No claims of being first, only, or comprehensive. The evidence stands on its own.
- Scripts and raw data are published so anyone can rerun or challenge them.
- AI coding tools are used in this project. The design choices, the checks, and the conclusions are the author's to defend.

## Prior art

Found by a web search on 2026-11-21. These are summaries and have **not** yet been checked against the papers.

- S-LoRA: https://arxiv.org/pdf/2311.03285
- Punica: https://arxiv.org/pdf/2310.18547
- dLoRA: https://www.cs.princeton.edu/~ravian/COS597_F24/papers/dlora.pdf
- FastLibra (TTFT-limited peak throughput definition): https://arxiv.org/pdf/2505.03756
- LoRAForge (vLLM only; says it does not rank engines): https://arxiv.org/pdf/2611.01256
- LoRazor (vLLM, adapter diversity sweep): https://arxiv.org/pdf/2611.10049
- EpiLoRA (compares vLLM, SGLang, TensorRT-LLM against the authors' own system): https://arxiv.org/pdf/2611.03899
- A vLLM-only reproducible sweep: https://github.com/priyaltaneja/multi-lora-serving-benchmark
- SGLang LoRA documentation: https://docs.sglang.io/advanced_features/lora.html
- vLLM and AWS multi-LoRA for MoE: https://vllm.ai/blog/multi-lora

See [docs/PLAN.md](docs/PLAN.md) for the decisions still open.

## License

Code is under Apache-2.0 (see `LICENSE`). A license for the data will be chosen before any results are published.
