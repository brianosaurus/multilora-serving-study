# multilora-serving-study

Status: planning. There are no results yet. Nothing here should be read as a finding.

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
