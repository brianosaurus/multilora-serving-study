# Standing prompt: the multi-LoRA serving study

Use this as standing context for any work on this study. It is a snapshot taken on 2026-11-21; the sources of truth are `docs/PLAN.md` (the design) and `docs/CHANGELOG.md` (every change, with the reason). If this file and those disagree, they win. Update this file when the state changes.

## What the study is

A reproducible, one-day measurement of how current open-source inference engines serve many LoRA adapters on one consumer GPU.

- Question: how do SGLang, vLLM, and mistral.rs behave when serving many LoRA adapters on one RTX 4090 (24 GB), and do published claims about multi-LoRA serving hold on this hardware?
- Engines and pinned versions: vLLM 0.34.0, SGLang 0.5.23, mistral.rs v0.9.5. No upgrades mid-study.
- Hardware: one NVIDIA RTX 4090, one machine, one engine process at a time, nothing else on the GPU.
- Time: the whole study, from first setup to published write-up, must fit in under 24 hours wall-clock. See the Day plan in `docs/PLAN.md` for the reduced grid, the three gates, the cut order, and the hard stop at hour 24. Hour 0 is 2026-11-21 23:25:58 UTC (when the engine installs began, as the author specified); the hard stop is 2026-11-22 23:25:58 UTC.
- Purpose: a credible, reproducible study that its author can publish as research. Credibility comes from method and honesty, not from the result being favorable.

## Secondary goal

Find reproducible, fixable bugs in the engines, which the author can fix upstream. Log every anomaly in `docs/BUGS.md` with what is verified and what is not, and prefer root-causing from source over guessing. This is exploratory and separate from the pre-registered measurements.

## Ground rules (non-negotiable)

- Report negative and inconclusive results as plainly as positive ones. A difference inside the run-to-run spread is "inconclusive", not a win.
- Never claim to be first, only, or comprehensive. Let the evidence stand. Describe the work as what it is: a reproducible benchmark study with these engines on this hardware.
- Do not overstate: open means open, merged means merged. Describe reviews as reviews and co-authorship as co-authorship.
- AI coding tools are used throughout. The design choices, the checks, and the conclusions are the author's to defend. Anything the author has not reviewed line by line is marked as such.
- Nothing leaves the author's machines without being shown to him first: no pushes, no posts, no comments, no publishing. Showing the exact text first is required, and approval for one action does not extend to the next.
- Look before deleting, and verify the machine's identity before any action on it. The ssh alias for the GPU machine can be re-pointed; confirm the hostname matches the one recorded in your notes.
- Pre-registration: the decisions in `docs/PLAN.md` were fixed before any run. After the first run, any change is logged in `docs/CHANGELOG.md` with the reason, and the repository history is never rewritten.

## The design in brief (details and exact numbers are in `docs/PLAN.md`)

- Model: Qwen2.5-1.5B-Instruct. Adapters: synthetic, built with PEFT, rank 16, all linear layers, a distinct seed each, B matrices nonzero.
- Correctness first: before any timing counts, verify that each engine actually applies the adapter, by comparing outputs or logprobs against a PEFT reference.
- Replications come before comparisons. A: SGLang's documentation claims its `csgmv` backend gives 20% to 80% lower latency than Triton. B: a public vLLM sweep found `max_loras=16` best for 10 to 500 adapters under skewed traffic. Both have numeric thresholds for "held", "partly held", and "not held", fixed in advance.
- Matched comparison: same model, adapters, prompts, sampling, traffic, and TTFT limit (p95 TTFT of 500 ms or less defines peak throughput). Open-loop Poisson arrivals, uniform and Zipf adapter popularity. Replications follow the original claim's method.
- Relative metric, two levels, both always reported: (1) within each engine, adapter versus no adapter (baseline B0 with LoRA disabled, B1 with LoRA enabled but no adapter selected); overhead factor is adapter over baseline for latencies and baseline over adapter for throughput; (2) across engines, the overhead factors compared with each other, with absolute results beside them. Engines are not ranked when their ranges overlap.
- Fairness: every engine gets the same tuning budget from a documented list of settings, chosen on a separate tuning workload and measured on a held-out one. Every setting used is recorded. The tuning settings list is still open and must be committed before the first timed run.
- Hygiene: warm-up and cache flushing, fixed request counts, repeated runs with min-max ranges, GPU idle check before every run, clocks and temperature recorded, thermally throttled runs discarded, and a reference run at the start and end of every session (a session whose reference differs by more than 5% is discarded and rerun).
- Storage rule: nothing read during a timed run comes from the network volume. The model checkpoint and all adapter files are copied to RAM-backed storage first, and compiler and JIT caches stay on local disk. Warm-up and the first requests of every run are excluded from results.
- mistral.rs is in the outcome-level comparison but has no `--max-loras` equivalent, so it is excluded from replication B. Whether it batches requests with different adapters is unknown and is one of the things the study measures.

## Machine and paths (no addresses or credentials here)

- RTX 4090 (24 GB), 128 CPU cores, 251 GB RAM. Local disk 30 GB. A network-mounted volume at `/workspace` with a measured quota of about 47 GiB (reads about 660 MB/s, small files about 17 times slower than local). RAM-backed `/dev/shm` of 19 GB, mounted noexec: use it for data only.
- The machine was wiped of all old data and is dedicated to this study.
- Engine environments: `/root/study/venvs/vllm` and `/root/study/venvs/sglang`. Package lists: `/workspace/env/*-freeze.txt`. Logs: `/root/study/logs`. mistral.rs source and build: `/workspace/build`. Model: `/workspace/models/Qwen2.5-1.5B-Instruct`, with a verified copy in `/dev/shm/models`.
- CUDA toolkit with `nvcc` 12.8 is at `/usr/local/cuda/bin` (not on the default PATH). A Rust toolchain is under `/root/.rustup`.

## Known traps

- SGLang 0.5.23 depends on `sglang-kernel==0.4.10` and a pre-release `flash-attn-4>=4.0.0b18`. Installing it needs that package named explicitly so the resolver allows the pre-release for it only. Mixing in the older `sgl-kernel` 0.3.21 breaks with an ABI error.
- In zsh, a loop variable named `path` overwrites `PATH`; zsh also aborts a whole command when one glob has no match (use `find`).
- On macOS, `sed -i` needs an empty-string argument (`sed -i ''`); the GNU form fails silently. Verify edits landed.
- Several engine processes on one GPU at once cause flaky host-memory errors and invalid timings. One engine at a time.

## State as of this snapshot

- Repository created (private) with the plan, the fairness protocol, the Day plan, the storage rules, and the changelog, pushed.
- The machine is clean. vLLM 0.34.0 (torch 2.14.0) and SGLang 0.5.23 (torch 2.14.1) are installed and see the GPU. mistral.rs v0.9.5 is building with CUDA and flash-attn. The model is downloaded, copied to RAM-backed storage, and checksum-verified.
- Gate 1 passed at about hour 0.6: every engine answered a base-model request (vLLM 90 s cold start, SGLang 96 s, mistral.rs 25 s; mistral.rs needs an explicit `--kv-cache 8GiB`). A tools environment with PEFT exists.
- Correctness-check criteria are fixed in `docs/PLAN.md`. The adapter generator, the PEFT reference script, and the 16 check prompts are written; calibration of the adapter scale and the engine checks (Gate 2) come next.

## What comes next, in order

1. Calibrate the adapter scale against the PEFT reference, generate the adapters, and write the per-engine adapter-selection configs.
2. Run the correctness check on each engine, serial and mixed-batch (Gate 2).
3. Write the load generator and harness (open-loop Poisson, Zipf and uniform popularity, the rate ladder, the metrics in the plan), run smoke tests, fix and commit the tuning settings list (Gate 3).
4. Run measurements in the order the plan gives, cutting in the plan's order if a gate is missed.
5. Analyze with the relative metric at both levels, then write the results, with every limit stated, for the author to review before anything is published.

## Output expectations

- Raw run outputs in `results/raw/`, analysis code in `analysis/`, scripts in `scripts/`, configurations in `configs/`. Every run records its full command line, versions, commit hashes, GPU state, and storage layout.
- The write-up states what was measured, what was not, the engine versions, the sample sizes (3 repeats), and that this is a single-day study on one GPU. It does not rank engines whose ranges overlap, and it does not extrapolate beyond the tested adapter counts.
