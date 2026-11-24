#!/usr/bin/env python3
"""Phase 1 tables from results/phase1/r*/<cfg>/zipf32_r{20,40}.json. Applies the reading rule in PLAN.md."""
import glob, json, os, statistics as st, sys
D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "results/phase1")
LIM = 500.0
cells = {}
for f in sorted(glob.glob(f"{D}/r*/*/zipf32_r*.json")):
    rnd, cfg, name = f.split("/")[-3:]
    rate = int(name.split("_r")[1].split(".")[0])
    d = json.load(open(f)); s = d["summary"]; ok = [x for x in d["requests"][d["config"]["warmup"]:] if "error" not in x]
    cells.setdefault((cfg, rate), []).append({
        "round": rnd, "p95": s["ttft_ms"]["p95"], "p50": s["ttft_ms"]["p50"], "errors": s["errors"], "tok": s["output_tokens_per_s"],
        "max": max((x["ttft_s"] for x in ok), default=None), "lag": s["max_send_lag_s"], "n": s["ok"]})
cfgs = ["v23-def", "v23-d1", "v23-d02", "main-def", "main-d1", "main-d02"]
print("FLAGGED: shared host, screening run\n")
for rate in (20, 40):
    print(f"rate {rate} rps  (p95 TTFT ms per round; pass = p95<=500 and 0 errors)")
    for c in cfgs:
        v = cells.get((c, rate), [])
        if not v: print(f"  {c:9s} no data"); continue
        p = [round(x["p95"]) for x in v]; ps = [x["p95"] <= LIM and x["errors"] == 0 for x in v]
        print(f"  {c:9s} p95 {p} median {round(st.median(p))}  pass {sum(ps)}/{len(ps)}  max_ttft_s {[round(x['max'],1) for x in v]}  errors {[x['errors'] for x in v]}  tok/s {[round(x['tok']) for x in v]}  lag>0.05s {sum(x['lag']>0.05 for x in v)}")
print("\nReading rule at 40 rps")
def stats(c):
    v = cells.get((c, 40), []); return [x["p95"] for x in v], [x["p95"] <= LIM and x["errors"] == 0 for x in v]
for fl in ("def", "d1", "d02"):
    a, ap = stats(f"v23-{fl}"); b, bp = stats(f"main-{fl}")
    if not a or not b: print(f"  {fl}: missing data"); continue
    ratio = st.median(a) / st.median(b)
    disjoint = max(b) < min(a)
    removes = all(bp) and not all(ap) and len(b) == 3
    reduces = ratio >= 2 and disjoint
    print(f"  {fl:4s}: v23 median {round(st.median(a))} ms, main median {round(st.median(b))} ms, ratio {ratio:.2f}x, ranges disjoint {disjoint}, main passes {sum(bp)}/{len(bp)}, v23 passes {sum(ap)}/{len(ap)} -> " + ("REMOVES" if removes else "REDUCES" if reduces else "no material effect"))
