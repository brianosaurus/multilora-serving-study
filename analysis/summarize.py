#!/usr/bin/env python3
"""Summarize ladder results: per engine, workload and rate, the median and min-max
across repeats of p95 TTFT, plus throughput and errors; then the last passing rate
(median p95 TTFT <= LIMIT ms; the ladder stops after two consecutive violations).

Usage: summarize.py RESULTS_ROOT > summary.md
Sources (valid data only; the first SGLang LoRA session was discarded for drift):
  vllm      main/vllm
  sglang    b0 from main/sglang, LoRA workloads from main_rerun1/sglang
  mistralrs main/mistralrs
"""
import glob, json, os, re, statistics as st, sys
from collections import defaultdict

LIMIT = 500.0
root = sys.argv[1]
SRC = [
    ("vllm", "main/vllm", None),
    ("sglang", "main/sglang", {"b0"}),
    ("sglang", "main_rerun1/sglang", "lora"),
    ("mistralrs", "main/mistralrs", None),
]
data = defaultdict(list)  # (engine, workload, rate) -> [summary]
for eng, sub, only in SRC:
    for f in glob.glob(os.path.join(root, sub, "*_r*_rep*.json")):
        m = re.search(r"_(\w+?)_r(\d+)_rep(\d+)\.json$", f)
        if not m:
            continue
        wl, rate = m.group(1), int(m.group(2))
        if only == "lora" and wl == "b0":
            continue
        if isinstance(only, set) and wl not in only:
            continue
        data[(eng, wl, rate)].append(json.load(open(f))["summary"])

WL = ["b0", "b1", "one", "zipf32", "uni32", "pfx_base", "pfx32"]
print("| Engine | Workload | Rate | p95 TTFT median (min-max) ms | tok/s mean | errors |")
print("|---|---|---|---|---|---|")
last = {}
for eng in ("vllm", "sglang", "mistralrs"):
    for wl in WL:
        rates = sorted(r for (e, w, r) in data if e == eng and w == wl)
        passing = None; failed = False
        for r in rates:
            s = data[(eng, wl, r)]
            p = [x["ttft_ms"]["p95"] for x in s]
            med = st.median(p)
            tps = sum(x["output_tokens_per_s"] for x in s) / len(s)
            err = sum(x["errors"] for x in s)
            print(f"| {eng} | {wl} | {r} | {med:.0f} ({min(p):.0f}-{max(p):.0f}) | {tps:.0f} | {err} |")
            if med <= LIMIT and not failed:
                passing = r
            else:
                failed = True
        last[(eng, wl)] = passing
print("\nLast passing rate (median p95 TTFT <= 500 ms, contiguous from the lowest rate):\n")
print("| Workload | vllm | sglang | mistralrs |")
print("|---|---|---|---|")
for wl in WL:
    print("| " + wl + " | " + " | ".join(str(last.get((e, wl), "-")) for e in ("vllm", "sglang", "mistralrs")) + " |")
