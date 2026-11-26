#!/usr/bin/env python3
"""Trigger step 1 / option 12 (PLAN.md): client-path check for the slow state. Usage: harness_check.py results/phase2"""
import glob, json, os, statistics as st, sys
from collections import Counter
D = sys.argv[1] if len(sys.argv) > 1 else "results/phase2"
SLOW = {"B_main-16", "E_main-16-P1"}


def pct(a, q):
    a = sorted(a); return a[min(len(a) - 1, int(round(q / 100 * (len(a) - 1))))] if a else None


def analyse(cell, rate=20):
    ev = [json.loads(l) for f in glob.glob(f"{cell}/trace/*.jsonl") for l in open(f)]
    m = {int(json.loads(l)["rate"]): json.loads(l) for l in open(f"{cell}/marks.jsonl") if l.strip()}[rate]
    t0, t1 = m["t_start"], m["t_end"]
    req = sorted(json.load(open(f"{cell}/zipf32_r{rate}.json"))["requests"], key=lambda r: r["t_send"])
    req = [r for r in req if r.get("t_done") is not None]
    arr = sorted([e for e in ev if e["e"] == "arrive" and t0 <= e["t"] <= t1 and not str(e["rid"]).startswith("HEALTH")], key=lambda e: e["t"])
    out = {"checks": []}
    if len(arr) != len(req): out["checks"].append(f"arrival count {len(arr)} != loadgen {len(req)}")
    n = min(len(arr), len(req))
    pairs = Counter((x["lora"], r["adapter"]) for x, r in zip(arr[:n], req[:n]))
    u2n = {}
    for (u, nm), c in sorted(pairs.items(), key=lambda kv: -kv[1]): u2n.setdefault(u, nm)
    if len(set(u2n.values())) != len(u2n): out["checks"].append("adapter map is not one-to-one")
    au, rn = Counter(x["lora"] for x in arr), Counter(r["adapter"] for r in req)
    if any(au[u] != rn.get(nm, -1) for u, nm in u2n.items()): out["checks"].append("per-adapter counts differ")
    byname = {}
    for r in req: byname.setdefault(r["adapter"], []).append(r)
    adm = {}
    for b in sorted([e for e in ev if e["e"] == "batch" and t0 <= e["t"] <= t1 + 5], key=lambda e: e["t"]):
        for rid in b.get("rids", []): adm.setdefault(rid, b["t"])
    seen = Counter(); pr = []
    for x in arr:
        nm = u2n.get(x["lora"]); i = seen[x["lora"]]; seen[x["lora"]] += 1
        if nm is None or i >= len(byname.get(nm, [])): continue
        r = byname[nm][i]; pr.append({"arr": x["t"], "send": r["t_send"], "rid": x["rid"], "adm": adm.get(x["rid"])})
    pr.sort(key=lambda p: p["send"])
    d = [p["arr"] - p["send"] for p in pr]
    out.update(n=len(pr), med_ms=1000 * st.median(d), p99_ms=1000 * pct(d, 99), first50_med_ms=1000 * st.median(d[:50]),
               send0_minus_tstart=pr[0]["send"] - t0, arr0_minus_send0=pr[0]["arr"] - pr[0]["send"], arr0_minus_tstart=min(p["arr"] for p in pr) - t0)
    waits = sorted([(p["arr"], p["adm"] - p["arr"], p["arr"] - p["send"]) for p in pr if p["adm"] is not None])
    first = next(((a, w, dl) for a, w, dl in waits if w > 1), None)
    out["first_gt1"] = None if first is None else {"t_after_first_arrival": first[0] - waits[0][0], "wait_s": first[1], "client_delay_ms": 1000 * first[2]}
    return out


res = {}
for cell in sorted(glob.glob(f"{D}/*_*")):
    if not os.path.isdir(f"{cell}/trace") or not os.path.exists(f"{cell}/zipf32_r20.json"): continue
    try: res[os.path.basename(cell)] = analyse(cell)
    except Exception as ex: res[os.path.basename(cell)] = {"checks": [f"error: {ex!r}"]}
print("FLAGGED: existing shared-host data; descriptive client-path check (PLAN.md trigger step 1)\n")
print(f"{'start':26s}{'n':>5s}{'med ms':>9s}{'p99 ms':>9s}{'f50 med':>9s}{'send0-t0':>10s}{'arr0-send0 ms':>15s}{'arr0-t0':>9s}  first wait>1s (t after 1st arrival, wait, client delay ms)  checks")
for k, o in res.items():
    if "med_ms" not in o: print(f"{k:26s} EXCLUDED: {o['checks']}"); continue
    fg = o["first_gt1"]; fgs = "none" if fg is None else f"{fg['t_after_first_arrival']:.1f}s {fg['wait_s']:.1f}s {fg['client_delay_ms']:.1f}"
    print(f"{k + (' *SLOW*' if k in SLOW else ''):26s}{o['n']:5d}{o['med_ms']:9.2f}{o['p99_ms']:9.1f}{o['first50_med_ms']:9.2f}{o['send0_minus_tstart']:10.2f}{1000*o['arr0_minus_send0']:15.1f}{o['arr0_minus_tstart']:9.2f}  {fgs:28s} {o['checks'] or ''}")
ok = {k: o for k, o in res.items() if "med_ms" in o and not o["checks"]}
slow = {k: o for k, o in ok.items() if k in SLOW}; norm = {k: o for k, o in ok.items() if k not in SLOW}
print(f"\nvalid windows: {len(ok)} (slow {len(slow)}, normal {len(norm)}); excluded: {[k for k in res if k not in ok]}")
if slow and norm:
    nm = st.median(o["med_ms"] for o in norm.values())
    a = all(o["med_ms"] <= 20 and o["p99_ms"] <= 250 for o in ok.values()) and all(o["med_ms"] <= 2 * nm for o in slow.values())
    b1 = all(o["arr0_minus_send0"] * 1000 <= 50 for o in ok.values())
    dn = st.median(o["arr0_minus_tstart"] for o in norm.values())
    diffs = {k: (o["arr0_minus_tstart"] - dn, o["send0_minus_tstart"] - st.median(x["send0_minus_tstart"] for x in norm.values())) for k, o in slow.items()}
    b2 = all(t > 0 and s / t >= 0.9 for t, s in diffs.values()) if all(t > 0 for t, s in diffs.values()) else None
    c = all(o["first_gt1"] is not None and o["first_gt1"]["client_delay_ms"] <= 250 for o in slow.values())
    print(f"normal median client delay {nm:.2f} ms; extra time to first arrival vs normal median, and the part from load-generator start-up (send0-t0): {diffs}")
    print(f"(a) {'HOLDS' if a else 'fails'}  (b1) {'HOLDS' if b1 else 'fails'}  (b2) {'HOLDS' if b2 else ('fails' if b2 is False else 'n/a')}  (c) {'HOLDS' if c else 'fails'}")
