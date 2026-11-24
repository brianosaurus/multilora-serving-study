#!/usr/bin/env python3
"""cellinfo.py RUN_JSON -> tab-separated: ttft_p95 e2e_p50 errors tok_s max_send_lag n_not_128_chunks measured
(output length control: loadgen sends ignore_eos and max_tokens 128 to vLLM and SGLang; every measured request must report 128 chunks)"""
import json, sys
d = json.load(open(sys.argv[1])); s = d["summary"]
reqs = [r for r in d["requests"] if r.get("i", 0) >= d["config"]["warmup"]]
bad = sum(1 for r in reqs if r.get("output_chunks") != d["config"]["output_tokens"] and r.get("e2e_s") is not None)
print("\t".join(str(x) for x in [s["ttft_ms"]["p95"], s["e2e_ms"]["p50"], s["errors"], s["output_tokens_per_s"], s["max_send_lag_s"], bad, s["measured"]]))
