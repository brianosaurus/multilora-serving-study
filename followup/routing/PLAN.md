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
