#!/usr/bin/env python3
"""Every number for the two SGLang issue drafts, read from the raw files in results/raw (stdlib only).
Prints markdown tables and, under each, the raw file path pattern(s) the numbers come from."""
import glob, json, os, statistics as st
R = "results/raw"
def runs(pat): return [json.load(open(f))["summary"] for f in sorted(glob.glob(pat))]
def rng(pat, key=("ttft_ms", "p95")):
    v = [s[key[0]][key[1]] for s in runs(pat)]
    return (min(v), st.median(v), max(v), len(v)) if v else None
def tps(pat):
    r = runs(pat); return sum(s["output_tokens_per_s"] for s in r) / len(r) if r else None
def errs(pat): return sum(s["errors"] for s in runs(pat))
def last_pass(pat_fmt, rates):
    last = None
    for r in rates:
        x = rng(pat_fmt.format(r=r))
        if x is None: break
        if x[1] <= 500 and errs(pat_fmt.format(r=r)) == 0: last = r
        else: break
    return last
fmt = lambda x: "-" if x is None else f"{x[1]:.0f} ({x[0]:.0f}-{x[2]:.0f})"
RATES = [5, 10, 20, 40, 80]
print("## A1. Last passing rate under a 500 ms p95 TTFT bar (median of 3 repeats; 32 adapters, 16 slots, open loop 256 in / 128 out)")
print("| Workload | vLLM 0.34.0 | SGLang 0.5.23 |\n|---|---|---|")
for w, name in (("zipf32", "Zipf popularity"), ("uni32", "uniform popularity")):
    v = last_pass(f"{R}/main/vllm/vllm_{w}_r{{r}}_rep*.json", RATES); s = last_pass(f"{R}/main_rerun1/sglang/sglang_{w}_r{{r}}_rep*.json", RATES)
    print(f"| {name} | {v} rps | {s} rps |")
print(f"files: {R}/main/vllm/vllm_{{zipf32,uni32}}_r*_rep*.json ; {R}/main_rerun1/sglang/sglang_{{zipf32,uni32}}_r*_rep*.json\n")
print("## A2. p95 TTFT in ms, median (min-max) over 3 repeats, 32 adapters, 16 slots")
print("| Workload | Rate | vLLM | SGLang | vLLM tok/s | SGLang tok/s |\n|---|---|---|---|---|---|")
for w in ("zipf32", "uni32"):
    for r in RATES:
        pv, ps = f"{R}/main/vllm/vllm_{w}_r{r}_rep*.json", f"{R}/main_rerun1/sglang/sglang_{w}_r{r}_rep*.json"
        if rng(pv) or rng(ps): print(f"| {w} | {r} rps | {fmt(rng(pv))} | {fmt(rng(ps))} | {tps(pv) and round(tps(pv))} | {tps(ps) and round(tps(ps))} |")
print(f"files: as A1\n")
print("## A3. 32 adapters, 32 slots (Zipf): p95 TTFT ms")
print("| Rate | vLLM max_loras 8 | vLLM 16 | vLLM 32 | SGLang 16 slots | SGLang 32 slots |\n|---|---|---|---|---|---|")
for r in (20, 40, 80):
    print(f"| {r} rps | {fmt(rng(f'{R}/sweep_m8/vllm/vllm_zipf32_r{r}_rep*.json'))} | {fmt(rng(f'{R}/main/vllm/vllm_zipf32_r{r}_rep*.json'))} | {fmt(rng(f'{R}/sweep_m32/vllm/vllm_zipf32_r{r}_rep*.json'))} | {fmt(rng(f'{R}/sgl_flags/control/sglang_zipf32_r{r}_rep*.json'))} | {fmt(rng(f'{R}/sgl_flags/slots32/sglang_zipf32_r{r}_rep*.json'))} |")
print(f"files: {R}/sweep_m8/vllm, {R}/main/vllm, {R}/sweep_m32/vllm, {R}/sgl_flags/control, {R}/sgl_flags/slots32 (the SGLang 16-slot control here is a fresh same-session run; the main-grid value is in A2)\n")
print("## A4. SGLang at 16 slots, Zipf, 32 adapters: scheduling options (p95 TTFT ms, median (min-max); tok/s)")
print("| Variant | 20 rps | 40 rps | 80 rps | tok/s at 40 | tok/s at 80 |\n|---|---|---|---|---|---|")
for v, name in (("control", "defaults"), ("drain1", "--lora-drain-wait-threshold 1.0"), ("overlap2", "--enable-lora-overlap-loading --max-loaded-loras 32"), ("fifo", "--lora-eviction-policy fifo"), ("slots32", "--max-loras-per-batch 32 (32 slots)")):
    p = lambda r: f"{R}/sgl_flags/{v}/sglang_zipf32_r{r}_rep*.json"
    print(f"| {name} | {fmt(rng(p(20)))} | {fmt(rng(p(40)))} | {fmt(rng(p(80)))} | {round(tps(p(40)))} | {round(tps(p(80)))} |")
print(f"files: {R}/sgl_flags/<variant>/sglang_zipf32_r*_rep*.json ; first overlap attempt failed at startup (assertion on --max-loaded-loras) and was rerun as overlap2\n")
print("## A5. 128 adapters on 32 slots (exploratory), p95 TTFT ms median (min-max)")
print("| Workload | Rate | vLLM (max_loras 32) | SGLang (max_loras_per_batch 32) |\n|---|---|---|---|")
for w in ("zipf128", "uni128"):
    for r in RATES:
        pv, ps = f"{R}/exp1/vllm/vllm_{w}_r{r}_rep*.json", f"{R}/exp1/sglang/sglang_{w}_r{r}_rep*.json"
        if rng(pv) or rng(ps): print(f"| {w} | {r} rps | {fmt(rng(pv))} | {fmt(rng(ps))} |")
print(f"files: {R}/exp1/{{vllm,sglang}}/*_{{zipf128,uni128}}_r*_rep*.json\n")
print("## B1. csgmv vs triton, closed loop, 32 Zipf adapters, 16 slots, 3 repeats, 500 requests per run (50 warm-up excluded)")
print("| Concurrency | Metric | triton median (min-max) | csgmv median (min-max) | change in median | ranges overlap |\n|---|---|---|---|---|---|")
for c in (8, 32, 128):
    for key, label in ((("ttft_ms", "p95"), "TTFT p95 ms"), (("ttft_ms", "p50"), "TTFT p50 ms"), (("tpot_ms", "p95"), "TPOT p95 ms"), (("e2e_ms", "p95"), "end-to-end p95 ms")):
        t, g = rng(f"{R}/repA/sglang_triton_c{c}_rep*.json", key), rng(f"{R}/repA/sglang_csgmv_c{c}_rep*.json", key)
        ov = "yes" if not (g[2] < t[0] or t[2] < g[0]) else "no"
        print(f"| {c} | {label} | {t[1]:.1f} ({t[0]:.1f}-{t[2]:.1f}) | {g[1]:.1f} ({g[0]:.1f}-{g[2]:.1f}) | {(g[1]-t[1])/t[1]*100:+.1f}% | {ov} |")
    print(f"| {c} | output tokens/s (mean) | {tps(f'{R}/repA/sglang_triton_c{c}_rep*.json'):.0f} | {tps(f'{R}/repA/sglang_csgmv_c{c}_rep*.json'):.0f} | {(tps(f'{R}/repA/sglang_csgmv_c{c}_rep*.json')/tps(f'{R}/repA/sglang_triton_c{c}_rep*.json')-1)*100:+.1f}% | |")
print(f"files: {R}/repA/sglang_{{triton,csgmv}}_c{{8,32,128}}_rep{{0,1,2}}.json ; references: {R}/repA/ref_{{triton,csgmv}}_{{start,end}}.json")
for be in ("triton", "csgmv"):
    a, b = [json.load(open(f"{R}/repA/ref_{be}_{w}.json"))["summary"]["e2e_ms"]["p50"] for w in ("start", "end")]
    print(f"reference run drift, {be}: e2e p50 start {a:.0f} ms, end {b:.0f} ms ({(b-a)/a*100:+.1f}%)")
