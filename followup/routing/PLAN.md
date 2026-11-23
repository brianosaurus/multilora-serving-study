# Follow-up A: adapter routing correctness (plan, fixed before any engine run)

Exploratory follow-up after the study's hard stop. It is separate from the pre-registered study and does not change any of its results. This plan was written before any measurement for this follow-up exists; changes after the first engine run are logged in `docs/CHANGELOG.md` with the reason.

## Question

Does each request get the right adapter? This is not "does the generated text match a reference". The study's two earlier criteria asked the second question and lacked power with these adapters (see `docs/ADAPTER_CORRECTNESS.md`).

## Fixed inputs

- Engines and versions (the study's pins): vLLM 0.34.0, SGLang 0.5.23, mistral.rs v0.9.5. Engine settings follow the study's `scripts/run_main.sh`, except as noted below.
- Model: Qwen2.5-1.5B-Instruct. Adapters: the study's seeds 0 to 3 (`adapter_0000` to `adapter_0003`, rank 16, B scale 0.05). Their SHA-256 checksums were confirmed equal to `results/raw/adapters_manifest.json`, as was the base-weights checksum.
- Candidates: the four adapters and the base model (five).
- Prompts: the 16 prompts in `configs/check_prompts.json`, tokenized once with the model's tokenizer, no chat template, as in the study's reference.
- Reference: PEFT unmerged, fp32, greedy, 128 new tokens, for every (prompt, candidate): the reference continuation `T[p,c]`. A second reference, PEFT unmerged in bf16, is computed on the same tokens (it is the stand-in for a correct implementation and gives the noise).

## Measurement

For a request (prompt p, adapter a) served by an implementation E, and for each candidate c, force E to read the sequence prompt(p) + `T[p,c]` and record E's log-probability of each of the 128 forced tokens. The fp32 reference gives the log-probability of the same tokens under candidate c, from its own run. Then

`d(request, c) = mean over the 128 positions of | logp_E(token) - logp_fp32,c(token) |`.

If E applies adapter a correctly, d is small for c = a (numerical noise only) and larger for c != a. Adapter and base differences show up as large d.

Method per engine:
- vLLM: `/v1/completions` with the token ids as the prompt, `max_tokens` 1, `prompt_logprobs` 0 (the logprob of each prompt token), model name = adapter name.
- SGLang: `/generate` with `input_ids`, `return_logprob` true, `logprob_start_len` = prompt length, `lora_path` = adapter name; read `input_token_logprobs`.
- mistral.rs: no prompt-logprob output exists in v0.9.5 (its `echo_prompt` returns only the text). Method: one request per position, with the decoded prefix text as the prompt, `max_tokens` 1, `logprobs` set to the largest value accepted, and the reference token's logprob read from the returned top list. Weaknesses, stated in advance: (1) a forced token outside the returned top list has no logprob, so it is given the smallest returned logprob, which makes that token look more probable than it is and understates d at those positions; the count of such positions is reported; (2) decoding to text and re-tokenizing may not give back the same ids, so positions where the re-encoded prefix differs from the intended ids are dropped and counted; (3) its prefix cache may reuse a prefix across requests, which is part of what is tested, not removed.

Modes: serial (one call at a time) and mixed batch (concurrency 16, all four adapters' calls in flight together, order shuffled with a fixed seed), as in the study.

## Decision rule (fixed)

- `d_right = d(request, a)`. `d_wrong = min over c != a of d(request, c)`, including the base candidate.
- `noise` = the median over prompts of d(bf16 PEFT request, c = a), pooled over the four adapters. Per-adapter values are reported too.
- A request is routed correctly if `d_right < d_wrong` and `d_wrong - d_right > k * noise`, with `k = 3`.
- Why 3: `noise` is a median, so individual correct requests sit above and below it. A margin of three noise units is meant to exceed that spread so that a correct engine clears it, while a wrong adapter (whose distance is the separation between two adapters, see power check 4) is far larger. It is checked, not assumed: power check 1 must show a correct implementation clearing k = 3 on every request. `k` is not changed after any engine result.
- An engine passes a mode if 100% of its requests are routed correctly. Any misrouted request is a finding and is reported individually.
- The verdict for an engine is "inconclusive" if any power check fails, because the margin rule then cannot be trusted.

## Power checks (run before any engine, results shown to the author)

1. bf16 PEFT, as the stand-in for a correct implementation, must pass the rule against the fp32 reference on all 64 requests (16 prompts by 4 adapters).
2. Misrouting must be detected: the bf16 stand-in serving adapter j's weights under adapter i's name, for every ordered pair i != j, must fail the rule on every request (stand-in for an engine serving the wrong adapter).
3. The bf16 stand-in with the adapter disabled (the base model), served under each adapter's name, must fail on every request.
4. The candidates must be separated: the median over prompts of the distance between each pair of different candidate references (on the second candidate's tokens, the first candidate's fp32 logprobs against the second's) must exceed `k * noise` for all 10 pairs. If it does not, the adapters are strengthened (for example a larger B scale) and the change is logged before any engine run.

Checks 2 and 3 use the PEFT stand-in because an engine-level swap needs a serving environment; they test whether the measurement can detect misrouting at all. An engine-level confirmation (for example a deliberately swapped adapter name in vLLM) can be added after the author approves the engine runs.

## Stops and reporting

- Gate A1: this plan and the power-check results. No engine runs before the author approves.
- Gate A2: per engine and mode: requests routed correctly out of total, the margin distribution (minimum and median of `d_wrong - d_right`), `d_right` against noise, and every misrouted request with its prompt, adapter and the candidate nearest to it. If any engine misroutes, reproduce it with a minimal script before anything else and log it in `docs/BUGS.md`.

## Limits stated in advance

Synthetic adapters, one model, rank 16, four adapters, 16 prompts. A pass means no misrouting was seen on these requests and this setup; it does not certify any engine. The fp32 reference is itself not ground truth for numerics, which is why the rule compares candidates against each other and uses a noise margin.

## Implementation notes (added before the first reference run, 2026-11-23)

- The reference continuation is generated with `min_new_tokens` = `max_new_tokens` = 128 so that every continuation has exactly 128 tokens; an end-of-text token is therefore not allowed to end it early. (The study's earlier reference used 32 tokens and no minimum.)
- "bf16 PEFT" uses the same loading as the study's reference script: bf16 base weights, `PeftModel.from_pretrained` defaults (PEFT may keep adapter weights in fp32 internally). "fp32 PEFT" uses fp32 base weights. Both run on the follow-up pod's GPU (the first attempt on CPU was abandoned after about 7 minutes as too slow on the loaded host: 425 s per prompt for generation; no CPU result was used); matmul TF32 stays off so the fp32 reference is true fp32. The GPU is used only for the references, with no engine running, and was idle when checked. The library versions are recorded in the output.
- `noise` is the median of 64 values (16 prompts by 4 adapters) of `d(bf16 stand-in serving adapter a, prompt, c = a)`.
- Power check 4 is evaluated per ordered pair (20 pairs, base included) and the 10 unordered pairs are judged by the smaller of their two directions, which is the stricter reading of the wording above.
- Distances use all 128 forced positions of each continuation, no weighting.
- Scripts: `scripts/refs.py` (continuations and teacher-forced log-probability matrices, fp32 and bf16), `scripts/analyze.py` (the decision rule and the four power checks). Outputs go to `results/raw/`.

## Calibration sweep (added 2026-11-23 about 05:05 UTC, after Gate A1 failed power checks 1 and 4; option 1 chosen by the author)

Gate A1 at the study's adapter scale (B standard deviation 0.05) gave noise 0.970 and failed checks 1 and 4 at k = 3 (10 of 64 requests routed correctly by the bf16 stand-in; 3 of 10 candidate pairs below k times noise). The plan's remedy for check 4 is to change the adapters before any engine run. Because noise may grow with the scale, the sweep covers weaker and stronger adapters.

Protocol, fixed before any run:
- Scales (B standard deviation) to try: 0.01, 0.02 and 0.1, each with seeds 0 to 3, rank 16, alpha 32, float16, generated by the study's `scripts/make_adapters.py`. The 0.05 result already exists. No other scale is tried without the author's approval.
- For each scale, run the same references and the same analysis with k = 3 (the rule is unchanged): all four power checks must pass.
- Choice: among the scales that pass all four checks, use the smallest. If no scale passes, the routing follow-up is reported as inconclusive and stops there; no further changes to the rule or the metric are made without the author's approval.
- The scale chosen this way replaces the study's adapters for the routing engine runs. Those adapters are therefore not the study's (they share the seeds but not the weights); the report says so. Their checksums are recorded.
- Known risk, from the study's own calibration (`results/raw/calibration/`): at scales 0.01 and 0.02 the adapters changed the output within 8 tokens on too few of 16 prompts, so base against adapter separation (check 4) may fail there.
- The GPU is used for the references only, with no engine running. A kernel rebuild is running on the CPU at the same time; it does not touch the GPU until its unit-test step.

## Amendment 1: scale for the engine runs (author's decision, 2026-11-23 about 06:35 UTC pod clock)

The sweep protocol says to use the smallest passing scale, which is 0.01. The author instead chose **0.02** for the engine runs. This is a departure from the protocol and is recorded as one. The rule (k = 3, four power checks, 100% to pass) is unchanged.

Reason: a smoke test of vLLM against itself (12 repeated calls, same prompt, adapter and candidate) gave a mean absolute difference per call of 0.015 to 0.021 between passes, the size of the stand-in's own noise (0.0167 at scale 0.01). At scale 0.01 the stand-in's smallest request margin (0.061) is only 1.2 times the threshold (0.050), so engine noise alone could make a correct engine fail the rule. At scale 0.02 the smallest margin (0.171) is 2.8 times the threshold (0.0604). Scale 0.02's four power checks passed in the sweep. The fixed numbers for the engine analysis are those of the 0.02 stand-in run: noise 0.0201, threshold 0.0604 (`results/raw/sweep_s0.02/refs_s0.02_power.json`). Adapters: seeds 0 to 3 at B scale 0.02, generated by `scripts/make_adapters.py`; checksums in `results/raw/sweep_s0.02/log_and_checksums.txt`. They are not the study's adapters.

## A2 implementation notes (added before the first engine run)

- Probe: `scripts/engine_probe.py`; analysis: `scripts/analyze_engine.py` (inputs `refs_s0.02.json` and its power file). 16 prompts by 4 served adapters by 5 candidates = 320 calls per engine and mode, each with exactly the forced tokens of one candidate's continuation and the served adapter selected. Calls are shuffled with seed 7. Serial: one call at a time. Mixed: 16 calls in flight, so all four adapters are in the batch together.
- A request counts only if all five of its calls return 128 values. Any incomplete request is reported and counts as not routed correctly; an engine passes a mode only with 64 of 64 routed correctly.
- vLLM 0.34.0: `vllm serve` with `--max-model-len 4096 --gpu-memory-utilization 0.85 --enable-lora --max-loras 16 --max-cpu-loras 64 --max-lora-rank 16 --lora-modules` (the four adapters), prefix caching left at its default; token ids as the prompt, `prompt_logprobs` = 1, the forced token's entry read by id. SGLang 0.5.23: `--enable-lora --max-lora-rank 16 --max-loras-per-batch 16`, the seven target modules, `--lora-paths` (the four adapters), `--disable-radix-cache` so the whole sequence is recomputed; `/generate` with `input_ids`, `return_logprob`, `logprob_start_len` = prompt length, `lora_path`. mistral.rs v0.9.5, the old build without flash-attn is not used: the rebuilt old binary with `cuda flash-attn` is, with `--kv-cache 8GiB --max-seqs 256 --lora-max-adapters 64` and the four adapters via `--lora`; one request per forced position (decoded prefix as text, `max_tokens` 1, `logprobs` 20, the adapter named).
- mistral.rs weaknesses and a stop rule: a forced token missing from the top-20 list gets the smallest returned logprob, and a prefix that does not re-encode to the same ids is dropped. If more than 5% of positions are missing or dropped in a mode, the verdict for that mode is "inconclusive (method)" and not a pass or a fail.
- Self-consistency context, reported but not part of the rule: each engine's serial pass is repeated once and the mean absolute difference between the two passes is reported next to the stand-in noise.
- One engine on the GPU at a time; before each engine the GPU must be idle; load average, GPU clocks and temperature are recorded at the start and end of each probe run. The B1 unit tests (which use the GPU) are not run at the same time as an engine.
- If any engine misroutes (nearest candidate is not the right adapter), I stop and reproduce it with a minimal script, and log it in `docs/BUGS.md`, before anything else.

### Correction to the A2 notes above (2026-11-23 about 06:40 UTC pod clock, before any engine run)

The mistral.rs sentence in the A2 notes says the rebuilt old binary with `cuda flash-attn` would be used. That is wrong for this follow-up: the rebuilt trees used for the kernel work (`followup/kernel_rerun/`) are upstream `master` at `21da642`, which is the 0.10.0 development line (the crates compile as 0.10.0), not the study's pinned mistral.rs v0.9.5. The routing test pins v0.9.5. A separate clean build of the v0.9.5 tag (commit `465191d`, the commit the study's binary was built from) with `cargo build --release -p mistralrs-cli --features "cuda flash-attn"` is made for the mistral.rs probe, and its checksum is recorded. The mistral.rs probe runs last, so that build is ready in time. If the v0.9.5 build cannot be made, the mistral.rs result would be reported for `master` and labeled as a different version.
