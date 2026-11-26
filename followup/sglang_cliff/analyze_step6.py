#!/usr/bin/env python3
"""Step 6 (PLAN.md option 17): part T (graph range under contention) and part U (decode step cost by batch size). Usage: analyze_step6.py results/phase2"""
import glob, json, os, statistics as st, sys

D = sys.argv[1]


def events(cell):
    ev = []
    for f in glob.glob(f"{cell}/trace/*.jsonl"):
        for l in open(f):
            try: ev.append(json.loads(l))
            except ValueError: pass
    return ev


def marks(cell):
    return [json.loads(l) for l in open(f"{cell}/marks.jsonl") if l.strip()] if os.path.exists(f"{cell}/marks.jsonl") else []


print("FLAGGED: shared host; runpod 64bba14d4768; step 6 (PLAN.md option 17)\n")
print("Part T: 14 busy loops, 20 rps")
T = {}
for cell in sorted(glob.glob(f"{D}/T_main-16-k14-*")):
    m = next((x for x in marks(cell) if int(x.get("rate", 0)) == 20), None)
    if not m or not os.path.exists(f"{cell}/zipf32_r20.json"): continue
    dec = [e for e in events(cell) if e["e"] == "batch" and "DECODE" in e["mode"].upper() and m["t_start"] <= e["t"] <= m["t_end"]]
    p95 = json.load(open(f"{cell}/zipf32_r20.json"))["summary"]["ttft_ms"]["p95"]
    p95_40 = json.load(open(f"{cell}/zipf32_r40.json"))["summary"]["ttft_ms"]["p95"] if os.path.exists(f"{cell}/zipf32_r40.json") else None
    big = sum(e["n"] > 48 for e in dec) / max(len(dec), 1)
    arm = "cg256" if "-cg256-" in cell else "stock"; T[os.path.basename(cell)] = (arm, p95, big)
    print(f"  {os.path.basename(cell):30s} p95 20 rps {p95:8.0f} ms   40 rps {p95_40 or float('nan'):8.0f} ms   decode steps above 48: {big:6.1%}   max batch {max((e['n'] for e in dec), default=0)}")
cg = [v for v in T.values() if v[0] == "cg256"]; sk = [v for v in T.values() if v[0] == "stock"]
if cg and sk:
    ok = sum(p > 1000 for _, p, _ in cg) <= 1 and all(p > 1000 or b > 0 for _, p, b in sk)
    print(f"(q) cg256 starts above 1 s: {sum(p > 1000 for _, p, _ in cg)} of {len(cg)} (<=1); every stock start above 1 s or with steps above 48: "
          f"{all(p > 1000 or b > 0 for _, p, b in sk)}: {'HOLDS' if ok else 'fails'}")

print("\nPart U: decode step wall time (ms, median) by batch size; LoRA prep CPU per run_batch call (us)")
B = [(1, 48), (49, 64), (65, 96), (97, 999)]
U = {}
for cell in sorted(glob.glob(f"{D}/U[1-4]_*")):
    ev = events(cell)
    for m in marks(cell):
        t0, t1 = m["t_start"], m["t_end"]
        dec = [e for e in ev if e["e"] == "batch" and "DECODE" in e["mode"].upper() and t0 <= e["t"] <= t1]
        ph = [e for e in ev if e["e"] == "ph" and t0 <= e["t"] < t1]
        rb = sum(e["p"].get("run_batch", [0])[0] for e in ph)
        prep = {n: 1e6 * sum(e["p"].get(n, [0, 0, 0])[2] for e in ph) / max(rb, 1) for n in ("LoRAManager.prepare_lora_batch", "ChunkedSgmvLoRABackend.prepare_lora_batch")}
        row = {}
        for lo, hi in B:
            d = [1000 * e["dur"] for e in dec if lo <= e["n"] <= hi]
            row[(lo, hi)] = (st.median(d), len(d)) if d else (None, 0)
        key = (os.path.basename(cell), m["adapters"], m["conc"]); U[key] = (row, prep)
        print(f"  {key[0]:18s} adapters {m['adapters']:2d} conc {m['conc']:3d}  " + "  ".join(
            f"{lo}-{hi if hi < 999 else '+'}: {'n/a' if v is None else f'{v:6.1f}'} ({n})" for (lo, hi), (v, n) in row.items())
              + f"   prep mgr {prep['LoRAManager.prepare_lora_batch']:6.0f}  backend {prep['ChunkedSgmvLoRABackend.prepare_lora_batch']:6.0f}")


def med_over(cellpre, ads, lo, hi):
    vals = []
    for (c, a, _), (row, _) in U.items():
        if c.startswith(cellpre) and a == ads and row[(lo, hi)][0] is not None: vals += [row[(lo, hi)][0]] * row[(lo, hi)][1]
    return st.median(vals) if vals else None


for lo, hi in [(49, 64), (65, 96)]:
    l, n = med_over("U1_", 32, lo, hi), med_over("U3_", 0, lo, hi)
    if l and n:
        print(f"(r) batch {lo}-{hi}: LoRA eager {l:.1f} ms vs no-LoRA eager {n:.1f} ms, ratio {l / n:.1f} (>=3): {'HOLDS' if l >= 3 * n else 'fails'}")
        preps = [p for (c, a, _), (row, p) in U.items() if c.startswith("U1_") and a == 32]
        pc = st.median(p["LoRAManager.prepare_lora_batch"] for p in preps) / 1000  # inclusive of the backend call if preps else None
        if pc is not None:
            print(f"(s) batch {lo}-{hi}: LoRA prep CPU {pc:.2f} ms per step of excess {l - n:.1f} ms: {pc / (l - n):.0%} (>=50%): {'HOLDS' if pc >= .5 * (l - n) else 'fails'}")
