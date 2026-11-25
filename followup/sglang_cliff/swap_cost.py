#!/usr/bin/env python3
"""Option 7 (PLAN.md): decode-step intervals with and without an adapter load in between. Usage: swap_cost.py results/phase2"""
import bisect, glob, json, statistics as st, sys
D = sys.argv[1] if len(sys.argv) > 1 else "results/phase2"
for cell in ["H_main-16-cg256-1", "H_main-16-cg256-2", "H_main-16-cg256-3", "E_main-16-U1"]:
    ev = [json.loads(l) for f in glob.glob(f"{D}/{cell}/trace/*.jsonl") for l in open(f)]
    m = {int(json.loads(l)["rate"]): json.loads(l) for l in open(f"{D}/{cell}/marks.jsonl") if l.strip()}[40]
    t0, t1 = m["t_start"], m["t_end"]
    bat = sorted([e for e in ev if e["e"] == "batch" and t0 <= e["t"] <= t1 + 5], key=lambda e: e["t"])
    lod = sorted([e for e in ev if e["e"] == "load" and e["loads"] and t0 <= e["t"] <= t1 + 5], key=lambda e: e["t"])
    lt = [e["t"] for e in lod]
    dec = [b for b in bat if "DECODE" in b["mode"].upper()]
    with_l, no_l = [], []
    for a, b in zip(dec, dec[1:]):
        iv = b["t"] - a["t"]
        if iv > 5: continue
        i = bisect.bisect_left(lt, a["t"])
        (with_l if i < len(lt) and lt[i] < b["t"] else no_l).append((iv, a["n"]))
    busy = max(b["t"] + b["dur"] for b in bat) - min(b["t"] for b in bat)
    base = st.median(x for x, _ in no_l); ex = sum(max(x - base, 0) for x, _ in with_l)
    print(f"{cell}: decode intervals {len(no_l)} without load, {len(with_l)} with load; load events {len(lod)}, loads {sum(len(e['loads']) for e in lod)}")
    print(f"  median interval no-load {1000*base:.1f} ms (mean batch {st.mean(n for _, n in no_l):.0f}), with-load {1000*st.median(x for x, _ in with_l):.1f} ms; "
          f"load call time {sum(e['dur'] for e in lod):.2f} s = {100*sum(e['dur'] for e in lod)/busy:.1f}% of {busy:.1f} s; excess {ex:.2f} s = {100*ex/busy:.1f}%")
