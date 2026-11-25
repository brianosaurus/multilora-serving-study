#!/usr/bin/env python3
"""Option 10 (PLAN.md): slot-capture test for the slow state. Usage: slot_capture.py results/phase2"""
import bisect, glob, json, os, statistics as st, sys
D = sys.argv[1] if len(sys.argv) > 1 else "results/phase2"
SLOW = {("B_main-16", 20), ("E_main-16-P1", 20), ("B_main-16", 40), ("E_main-16-U4", 40)}


def analyse(cell, rate):
    ev = [json.loads(l) for f in glob.glob(f"{cell}/trace/*.jsonl") for l in open(f)]
    m = {int(json.loads(l)["rate"]): json.loads(l) for l in open(f"{cell}/marks.jsonl") if l.strip()}[rate]
    t0, t1 = m["t_start"], m["t_end"]
    req = sorted(json.load(open(f"{cell}/zipf32_r{rate}.json"))["requests"], key=lambda r: r["t_send"])
    req = [r for r in req if r.get("t_done") is not None]
    arr = sorted([e for e in ev if e["e"] == "arrive" and t0 <= e["t"] <= t1 and not str(e["rid"]).startswith("HEALTH")], key=lambda e: e["t"])
    out = {"checks": []}
    if len(arr) != len(req): out["checks"].append(f"arrival count {len(arr)} != loadgen {len(req)}")
    a0 = arr[0]["t"]
    from collections import Counter
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
    iv, seen = {}, Counter()
    for x in arr:
        nm = u2n.get(x["lora"]); i = seen[x["lora"]]; seen[x["lora"]] += 1
        if nm is None or i >= len(byname.get(nm, [])) or x["rid"] not in adm: continue
        r = byname[nm][i]; iv.setdefault(x["lora"], []).append((adm[x["rid"]], x["t"] + r["t_done"] - r["t_send"]))
    busy = {}
    for u, l in iv.items():
        l.sort(); seg = []
        for s0, e0 in l:
            if seg and s0 <= seg[-1][1]: seg[-1][1] = max(seg[-1][1], e0)
            else: seg.append([s0, e0])
        busy[u] = ([x for x, _ in seg], [y for _, y in seg])
    has = lambda u, t: u in busy and (lambda i: i >= 0 and t < busy[u][1][i])(bisect.bisect_right(busy[u][0], t) - 1)
    # resident replay from the start of the trace
    lod = sorted([e for e in ev if e["e"] == "load" and (e["loads"] or e["evicts"])], key=lambda e: e["t"])
    R, hist, mx = set(), [], 0
    for e in lod:
        R.difference_update(e["evicts"]); R.update(e["loads"]); mx = max(mx, len(R)); hist.append((e["t"], frozenset(R)))
    if mx > 16: out["checks"].append(f"resident set reached {mx} > 16")
    ht = [h[0] for h in hist]
    res_at = lambda t: (hist[bisect.bisect_right(ht, t) - 1][1] if bisect.bisect_right(ht, t) else frozenset())
    cut = a0 + 5
    after = [e for e in lod if cut <= e["t"] <= t1 + 5]
    out["changes"] = len(after); out["loads"] = sum(len(e["loads"]) for e in after); out["evicts"] = sum(len(e["evicts"]) for e in after)
    dec = sorted([e for e in ev if e["e"] == "batch" and "DECODE" in e["mode"].upper() and cut <= e["t"] <= t1], key=lambda e: e["t"])
    full = [all(has(u, b["t"]) for u in res_at(b["t"])) for b in dec if res_at(b["t"])]
    out["share_all_running"] = sum(full) / len(full) if full else None
    # slot time by popularity rank
    cnt = {}
    for r in req: cnt[r["adapter"]] = cnt.get(r["adapter"], 0) + 1
    rank = {a: i + 1 for i, (a, _) in enumerate(sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0])))}
    held = {}
    tl = [b["t"] for b in dec]
    for i in range(len(tl) - 1):
        dt = tl[i + 1] - tl[i]
        if dt > 5: continue
        for u in res_at(tl[i]): held[u] = held.get(u, 0) + dt
    tot = sum(dt for dt in [tl[i + 1] - tl[i] for i in range(len(tl) - 1)] if dt <= 5) or 1
    out["held"] = sorted(((rank.get(u2n.get(u), 99), v / tot) for u, v in held.items()), key=lambda x: -x[1])
    out["distinct"] = len(held)
    return out


rows = {}
for cell in sorted(glob.glob(f"{D}/[BDEHK]*_*")):
    tag = os.path.basename(cell)
    if not glob.glob(f"{cell}/trace/*.jsonl") or not os.path.exists(f"{cell}/marks.jsonl"): continue
    if tag.split("_")[1].startswith(("main-32", "v23-32", "main-20", "main-24", "main-28")): continue
    for rate in (20, 40):
        try: rows[(tag, rate)] = analyse(cell, rate)
        except Exception as ex: rows[(tag, rate)] = {"checks": [f"error {ex!r}"]}
def grp(tag):
    if tag.startswith("D_"): return "drainer"
    if "cg256" in tag: return "raised"
    return "stock"
print("FLAGGED: existing traced screening starts; 'after 5 s' = 5 s after the first arrival\n")
print(f"{'start':24s}{'rate':>5s}{'group':>9s}{'slow':>5s}{'changes':>8s}{'loads':>6s}{'evict':>6s}{'all running':>12s}{'distinct':>9s}  checks")
for (tag, rate), o in sorted(rows.items(), key=lambda kv: (kv[0][1], grp(kv[0][0]), kv[0][0])):
    if "changes" not in o: print(f"{tag:24s}{rate:5d}  {o['checks']}"); continue
    s = "YES" if (tag, rate) in SLOW else ""
    print(f"{tag:24s}{rate:5d}{grp(tag):>9s}{s:>5s}{o['changes']:8d}{o['loads']:6d}{o['evicts']:6d}{100*o['share_all_running']:11.0f}%{o['distinct']:9d}  {'; '.join(o['checks'])}")
print()
for rate in (20, 40):
    sl = [o for (t, r), o in rows.items() if r == rate and (t, r) in SLOW and "changes" in o]
    nm = [o for (t, r), o in rows.items() if r == rate and (t, r) not in SLOW and grp(t) == "stock" and "changes" in o and not o["checks"]]
    f = lambda L, k: (min(o[k] for o in L), max(o[k] for o in L)) if L else None
    print(f"{rate} rps: slow n={len(sl)} changes {f(sl,'changes')} evicts {f(sl,'evicts')} share-all-running {[round(100*o['share_all_running']) for o in sl]}%")
    print(f"        normal stock n={len(nm)} changes {f(nm,'changes')} evicts {f(nm,'evicts')} share-all-running {sorted(round(100*o['share_all_running']) for o in nm)}%")
print("\nSlot-time share by popularity rank (rank:share of decode time resident), slow starts:")
for (tag, rate), o in sorted(rows.items()):
    if (tag, rate) in SLOW and "held" in o: print(f"  {tag} @{rate}: " + " ".join(f"r{r}:{100*v:.0f}%" for r, v in o["held"][:16]))
