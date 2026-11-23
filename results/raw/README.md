# Raw results: what each folder is

Every file is one loadgen run (per-request records plus a summary) unless noted. Versions: vLLM 0.34.0, SGLang 0.5.23, mistral.rs v0.9.5 (commit 465191d). One RTX 4090, Qwen2.5-1.5B-Instruct, synthetic rank-16 adapters. See `docs/CHANGELOG.md` for every change and its reason.

Pre-registered grid (open-loop Poisson, 3 repeats, 500 requests, 50 warm-up excluded):
- `main/vllm`, `main/mistralrs`, `main/sglang` (LoRA off only), `main_rerun1/sglang` (SGLang LoRA workloads, the valid session).
- `main_discarded/sglang_lora_session1`: the first SGLang LoRA session. It failed the 5% drift check on its start reference run and was rerun. Kept for transparency; its numbers match the rerun closely.
- `main_firstrun_logs`: per-step summary lines of the first vLLM grid, whose raw files were lost with the network volume (see changelog). The rerun in `main/vllm` is the one reported.
- `tuning`, `calibration`, `correctness`: earlier gates.

Replications:
- `sweep_m8`, `sweep_m32` (vLLM `max_loras` 8 and 32; the 16-slot data is `main/vllm`).
- `repA` (SGLang csgmv against triton, closed loop).

Exploratory runs, not pre-registered (labelled as such in the changelog):
- `exp1` (128 adapters on 32 slots, vLLM and SGLang), `sgl_flags` (SGLang scheduling flags and 32 slots).
- `b6_scale`, `onecall`, `onecall_rep`, `onecall2`, `a1_clean`: mistral.rs closed-loop probes and the timing-only patch experiment (`patches/mrs_onecall_timing_experiment.diff`; outputs of the patched binary are wrong by design).
- `run_summaries`: the console summary of each driver script. Driver scripts are in `scripts/exploratory/`.

Added after the study's hard stop (exploratory, a different pod, see `docs/ADAPTER_CORRECTNESS.md`):
- `correctness_followup`: fp32 PEFT reference first-token logprobs for the flaky adapter-4 prompts, the near-tie test (`nt`), and the 128-completion output check (`outputs`), with their scripts.
- `kernel_followup`: the grouped dense-LoRA kernel benchmarks (`ab` is the first, order-confounded run kept for the record; `ab2` is the balanced run the draft PR cites), the `CUDA_LAUNCH_BLOCKING` runs (`sync`), driver scripts, the test and lint logs, and the commit as a patch.

Not included: server logs. Nsight Compute launch counts do not exist (counters blocked in the container).
