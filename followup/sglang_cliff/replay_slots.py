#!/usr/bin/env python3
"""Offline replay of drain-victim policies (PLAN.md, option 4). Usage: replay_slots.py results/phase2"""
import heapq, json, os, statistics as st, sys
D = sys.argv[1] if len(sys.argv) > 1 else "results/phase2"
SLOTS, DRAIN_AFTER, PREFILL, WIN = 16, 1.0, 0.05, 30.0
CELLS = ["E_main-16-U1", "E_main-16-U5", "E_main-16-P2", "E_main-16-P3", "E_main-16-P6", "E_main-16-U4"]


def pct(a, q): a = sorted(a); return a[min(len(a) - 1, int(round(q / 100 * (len(a) - 1))))]


def load(cell):
    rq = json.load(open(f"{D}/{cell}/zipf32_r40.json"))["requests"]
    out = [dict(a=r["adapter"], t=r["t_send"], dec=max(r["t_done"] - r["t_send"] - r["ttft_s"], 0.0), ttft=r["ttft_s"])
           for r in rq if r.get("ttft_s") is not None and r.get("t_done") is not None]
    return sorted(out, key=lambda r: r["t"])


def sim(reqs, policy):
    arr_by = {}
    for r in reqs: arr_by.setdefault(r["a"], []).append(r["t"])
    ev = []; seq = 0
    def push(t, kind, x): nonlocal seq; seq += 1; heapq.heappush(ev, (t, seq, kind, x))
    for i, r in enumerate(reqs): push(r["t"], "arr", i)
    waiting = []; run = {}; ends = {}; drain = {}; starver_victim = {}; waits = [None] * len(reqs); last = 0.0
    def rate(a, now):
        ts = arr_by.get(a, []); n = sum(1 for t in ts if now - WIN < t <= now); return n / min(WIN, max(now, 1e-9))
    def victim(now):
        cands = [a for a in run if a not in drain]
        if not cands: return None
        tte = {a: max(ends[a]) - now for a in cands}
        if policy == "R1": key = lambda a: (tte[a], a)
        elif policy == "R2": key = lambda a: (rate(a, now) * tte[a], tte[a], a)
        elif policy == "R3": key = lambda a: (rate(a, now), tte[a], a)
        else: key = lambda a: (sum(1 for t in arr_by[a] if now < t <= now + tte[a]), tte[a], a)
        return min(cands, key=key)
    def admit(now):
        changed = True
        while changed:
            changed = False
            for i in list(waiting):
                a = reqs[i]["a"]
                if a in drain: continue
                if a in run or len(run) < SLOTS:
                    waiting.remove(i); run[a] = run.get(a, 0) + 1; waits[i] = now - reqs[i]["t"]
                    e = now + reqs[i]["dec"]; ends.setdefault(a, []).append(e); push(e, "done", (a, e)); changed = True; break
    def fix_drains(now):
        for v, s in list(drain.items()):
            if v not in run or s in run or not any(reqs[i]["a"] == s for i in waiting): drain.pop(v, None)
        for sv in [s for s, v in starver_victim.items() if v not in drain]: starver_victim.pop(sv, None)
        if policy == "F": return
        for i in waiting:
            a = reqs[i]["a"]
            if a in run or a in starver_victim or now - reqs[i]["t"] < DRAIN_AFTER or len(run) < SLOTS: continue
            v = victim(now)
            if v is not None: drain[v] = a; starver_victim[a] = v
    while ev:
        now, _, kind, x = heapq.heappop(ev); last = now
        if kind == "arr":
            waiting.append(x); push(now + DRAIN_AFTER, "chk", None)
        elif kind == "done":
            a, e = x; ends[a].remove(e); run[a] -= 1
            if run[a] == 0: run.pop(a); ends.pop(a)
        fix_drains(now); admit(now); fix_drains(now)
    t = [w + PREFILL for w in waits]
    return {"p95": pct(t, 95), "mean_wait": st.mean(waits), "makespan": last}


rows = {}
for c in CELLS:
    reqs = load(c)
    meas = [r["ttft"] for r in reqs]
    rows[c] = {"measured": {"p95": pct(meas, 95), "mean_wait": st.mean(meas) - PREFILL}}
    for p in ("F", "R1", "R2", "R3", "R4"): rows[c][p] = sim(reqs, p)
print("FLAGGED: replay of screening data; decode time per request fixed at its measured value\n")
print(f"{'cell':14s}{'policy':9s}{'p95 TTFT s':>12s}{'mean wait s':>13s}{'makespan s':>12s}")
for c in CELLS:
    for p in ("measured", "F", "R1", "R2", "R3", "R4"):
        r = rows[c][p]; print(f"{c[2:]:14s}{p:9s}{r['p95']:12.2f}{r['mean_wait']:13.2f}{r.get('makespan', float('nan')):12.1f}")
normal = CELLS[:5]
ok_p = sum(abs(rows[c]["F"]["p95"] / rows[c]["measured"]["p95"] - 1) <= 0.25 for c in normal)
ok_m = sum(abs(rows[c]["F"]["mean_wait"] / rows[c]["measured"]["mean_wait"] - 1) <= 0.25 for c in normal)
print(f"\nV1 model check: F p95 within 25% of measured in {ok_p}/5 cells; mean wait in {ok_m}/5 (need >= 4 each)")
med = lambda f: st.median(f(c) for c in normal)
print(f"V2 R1 vs F p95: {100*(med(lambda c: rows[c]['R1']['p95']/rows[c]['F']['p95'])-1):+.0f}% (need <= -15%)")
for p in ("R2", "R3", "R4"):
    print(f"V3/V4 {p} vs R1 p95: {100*(med(lambda c: rows[c][p]['p95']/rows[c]['R1']['p95'])-1):+.0f}%  mean wait {100*(med(lambda c: rows[c][p]['mean_wait']/rows[c]['R1']['mean_wait'])-1):+.0f}%  makespan {100*(med(lambda c: rows[c][p]['makespan']/rows[c]['R1']['makespan'])-1):+.0f}%")
