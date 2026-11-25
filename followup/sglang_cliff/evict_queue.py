#!/usr/bin/env python3
"""Option 11 (PLAN.md): at each eviction, did the evicted adapter have waiting requests? Usage: evict_queue.py results/phase2"""
import bisect, glob, json, os, sys
D = sys.argv[1] if len(sys.argv) > 1 else "results/phase2"
SLOW = {("B_main-16", 20), ("E_main-16-P1", 20), ("B_main-16", 40), ("E_main-16-U4", 40)}


def analyse(cell, rate):
    ev = [json.loads(l) for f in glob.glob(f"{cell}/trace/*.jsonl") for l in open(f)]
    m = {int(json.loads(l)["rate"]): json.loads(l) for l in open(f"{cell}/marks.jsonl") if l.strip()}[rate]
    t0, t1 = m["t_start"], m["t_end"]
    arr = sorted([e for e in ev if e["e"] == "arrive" and t0 <= e["t"] <= t1 and not str(e["rid"]).startswith("HEALTH")], key=lambda e: e["t"])
    a0 = arr[0]["t"]; cut = a0 + 5
    adm = {}
    for b in sorted([e for e in ev if e["e"] == "batch" and t0 <= e["t"] <= t1 + 5], key=lambda e: e["t"]):
        for rid in b.get("rids", []): adm.setdefault(rid, b["t"])
    byu = {}
    for a in arr: byu.setdefault(a["lora"], []).append((a["t"], adm.get(a["rid"], float("inf"))))
    cnt = sorted(((len(v), u) for u, v in byu.items()), reverse=True)
    top4 = {u for _, u in cnt[:4]}; rank = {u: i + 1 for i, (_, u) in enumerate(cnt)}
    rows = []
    for e in sorted([e for e in ev if e["e"] == "load" and e["evicts"] and cut <= e["t"] <= t1 + 5], key=lambda e: e["t"]):
        for u in e["evicts"]:
            w = sum(1 for ta, ts in byu.get(u, []) if ta <= e["t"] < ts)
            rows.append((e["t"] - a0, u, rank.get(u, 99), w))
    return dict(rows=rows, top4=top4, n=len(rows), waiting=sum(1 for r in rows if r[3] >= 1), top4_waiting=[(round(r[0], 1), r[2], r[3]) for r in rows if r[1] in top4])


res = {}
for cell in sorted(glob.glob(f"{D}/[BDEHK]*_*")):
    tag = os.path.basename(cell)
    if not glob.glob(f"{cell}/trace/*.jsonl") or not os.path.exists(f"{cell}/marks.jsonl"): continue
    if tag.split("_")[1].startswith(("main-32", "v23-32", "main-20", "main-24", "main-28")): continue
    for rate in (20, 40):
        try: res[(tag, rate)] = analyse(cell, rate)
        except Exception as ex: print("ERR", tag, rate, repr(ex))
grp = lambda t: "drainer" if t.startswith("D_") else ("raised" if "cg256" in t else "stock")
print("FLAGGED: existing traced screening starts; 'after 5 s' = 5 s after the first arrival\n")
print(f"{'start':24s}{'rate':>5s}{'group':>9s}{'slow':>5s}{'evicts':>7s}{'with waiting':>13s}{'share':>7s}{'max waiting':>12s}")
for (t, r), o in sorted(res.items(), key=lambda kv: (kv[0][1], grp(kv[0][0]), kv[0][0])):
    print(f"{t:24s}{r:5d}{grp(t):>9s}{'YES' if (t, r) in SLOW else '':>5s}{o['n']:7d}{o['waiting']:13d}{(100 * o['waiting'] / o['n'] if o['n'] else 0):6.0f}%{max([x[3] for x in o['rows']] or [0]):12d}")
print()
for rate in (20, 40):
    for name, pred in (("slow", lambda k: k in SLOW), ("normal stock", lambda k: k not in SLOW and grp(k[0]) == "stock"), ("drainer", lambda k: grp(k[0]) == "drainer"), ("raised", lambda k: grp(k[0]) == "raised")):
        sel = [o for k, o in res.items() if k[1] == rate and pred(k)]
        if sel:
            n = sum(o["n"] for o in sel); w = sum(o["waiting"] for o in sel)
            print(f"{rate} rps {name:13s}: starts {len(sel)}, evictions {n}, with waiting {w} ({100 * w / n if n else 0:.0f}%)")
print("\nTop-4 adapter evictions in slow starts: (seconds after first arrival, popularity rank, waiting count)")
for (t, r), o in sorted(res.items()):
    if (t, r) in SLOW: print(f"  {t} @{r}: " + (" ".join(f"({a},r{b},w{c})" for a, b, c in o["top4_waiting"]) or "none"))
