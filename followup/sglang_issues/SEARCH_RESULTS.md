# Search of sgl-project/sglang (read-only, 2026-11-23)

Queries (issues, PRs and discussions; open and closed): lora slots, max_loras_per_batch, lora eviction, lora starvation, lora TTFT, lora drain, lora-drain-wait-threshold, csgmv, csgmv triton latency, lora overlap loading, multi lora latency, chunked sgmv, csgmv backend, lora eviction policy, lora overlap loader, multi-lora throughput, LoRADrainer. GitHub's search is noisy; entries below are the ones that bear on the drafts. No existing issue or discussion reports what either draft reports.

## Related to Issue A (degradation when active adapters exceed slots)

| # | Kind | State | Why it matters |
|---|---|---|---|
| [17913](https://github.com/sgl-project/sglang/pull/17913) | PR | merged 2026-05-02 | "add LoRADrainer to address high P99 TTFT" (Llama-3.1-8B, 6 adapters on 3 slots). The feature behind `--lora-drain-wait-threshold`. |
| [46500](https://github.com/sgl-project/sglang/pull/46500) | PR | merged 2026-10-29 | "revamp LoRADrainer for better performance". Describes the same situation ("more LoRA adapters are busy than there are slots", 1% of requests waiting about 90 s at 4 adapters on 3 slots). **Not in v0.5.23:** `lora_drainer.py` at the v0.5.23 tag is byte-identical to the file before this PR (blob `6af2f025e9c6`); `main` has the revamped one (`457523aeaec7`). The latest release is v0.5.23. |
| [15129](https://github.com/sgl-project/sglang/pull/15129) | PR | closed | First draining PR ("a request never scheduling" with `max_loras_per_batch = 1`). |
| [25078](https://github.com/sgl-project/sglang/pull/25078), [25865](https://github.com/sgl-project/sglang/issues/25865), [23141](https://github.com/sgl-project/sglang/issues/23141) | PR, issues | closed | Admission accounting against `max_loras_per_batch` (crashes), not latency. |
| [21380](https://github.com/sgl-project/sglang/issues/21380) | issue | closed | Adapter unload leaking a pool slot (slot exhaustion). |
| [25417](https://github.com/sgl-project/sglang/issues/25417), [27181](https://github.com/sgl-project/sglang/pull/27181) | issue, PR | closed | Per-request adapter load pipelining (CPU to GPU); related to overlap loading. |
| [51328](https://github.com/sgl-project/sglang/pull/51328) | PR | open (2026-11-18) | "Prepare dynamic adapters and upload buffers off the scheduler". Related to swap cost. |
| [31305](https://github.com/sgl-project/sglang/pull/31305), [46403](https://github.com/sgl-project/sglang/pull/46403) | PRs | open | Paged LoRA memory pool with page-level eviction; eviction and batch-prep slot rescans. |

**Duplicate?** Not a duplicate of an existing issue: none reports measured latency with 32 adapters on 16 slots or compares it with vLLM. But the behavior is known and has been worked on (#17913, #46500), and the test I ran used a version that predates #46500. Issue A should therefore not be posted as is; see the recommendation in the draft.

## Related to Issue B (csgmv against triton)

| # | Kind | State | Why it matters |
|---|---|---|---|
| [10286](https://github.com/sgl-project/sglang/pull/10286) | PR | merged 2025-09-15 | Introduced csgmv. Its benchmark is the source of the docs' "20% to 80%": Llama-3.1-8B-Instruct, **one adapter**, ShareGPT, request rates 2, 8, 16 and unlimited, `--disable-radix-cache`; median end-to-end latency -9%, -40%, -70%, -81%; median ITL -12% to -79%; median TTFT 0% to -79%. |
| [10592](https://github.com/sgl-project/sglang/pull/10592), [10704](https://github.com/sgl-project/sglang/pull/10704) | PRs | merged, closed | Follow-ups (dynamic chunking, +10%). |
| [24007](https://github.com/sgl-project/sglang/pull/24007), [51199](https://github.com/sgl-project/sglang/pull/51199), [51099](https://github.com/sgl-project/sglang/issues/51099) | PR, PR, issue | merged, merged, closed | csgmv with virtual experts; an int64 row-offset fix in the csgmv kernels. Not performance reports. |

**Duplicate?** No. Nothing compares csgmv with triton on other settings or asks what the documented range was measured on. The template says general questions belong in Discussions, and this is a documentation question, so the draft is written for the feature-request template (Discussion is the alternative).
