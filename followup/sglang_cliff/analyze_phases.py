#!/usr/bin/env python3
"""Trigger step 4 (PLAN.md option 15): scheduler-loop phase timing over the 20 rps window. Usage: analyze_phases.py results/phase2 [prefix]"""
import glob, json, os, statistics as st, sys

D = sys.argv[1]
PRE = sys.argv[2] if len(sys.argv) > 2 else "R_main-16-"
PH = ("ingest_requests", "get_next_batch_to_run", "run_batch", "process_batch_result")


def spearman(x, y):
    if len(x) < 3: return None
    def rk(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]: j += 1
            for q in range(i, j + 1): r[o[q]] = (i + j) / 2 + 1
            i = j + 1
        return r
    a, b = rk(x), rk(y); ma, mb = st.mean(a), st.mean(b)
    n = sum((p - ma) * (q - mb) for p, q in zip(a, b)); d = (sum((p - ma) ** 2 for p in a) * sum((q - mb) ** 2 for q in b)) ** .5
    return n / d if d else None


rows = {}
for cell in sorted(glob.glob(f"{D}/{PRE}*")):
    tag = os.path.basename(cell)
    if not (os.path.exists(f"{cell}/marks.jsonl") and os.path.exists(f"{cell}/zipf32_r20.json")): continue
    m = next((json.loads(l) for l in open(f"{cell}/marks.jsonl") if l.strip() and int(json.loads(l)["rate"]) == 20), None)
    if not m: continue
    t0, t1 = m["t_start"], m["t_end"]
    ev = []
    for f in glob.glob(f"{cell}/trace/*.jsonl"):
        for l in open(f):
            try: ev.append(json.loads(l))
            except ValueError: pass
    ph = sorted((e for e in ev if e["e"] == "ph" and t0 <= e["t"] < t1), key=lambda e: e["t"])
    loads = sum(len(e["loads"]) for e in ev if e["e"] == "load" and t0 <= e["t"] <= t1)
    dur = max(t1 - t0, 1e-9)
    wall = sum(e["p"].get(p, [0, 0, 0])[1] for e in ph for p in PH); cpu = sum(e["p"].get(p, [0, 0, 0])[2] for e in ph for p in PH)
    g = [(e["p"]["get_next_batch_to_run"][2] / e["p"]["get_next_batch_to_run"][0], e["qlen"]) for e in ph
         if e["p"].get("get_next_batch_to_run", [0])[0]]
    per = {p: (sum(e["p"].get(p, [0, 0, 0])[1] for e in ph), sum(e["p"].get(p, [0, 0, 0])[2] for e in ph)) for p in PH}
    q = [e["qlen"] for e in ph]; first = next((i for i, v in enumerate(q) if v > 50), None)
    p95 = json.load(open(f"{cell}/zipf32_r20.json"))["summary"]["ttft_ms"]["p95"]
    rows[tag] = dict(p95=p95, slow=p95 > 10000, buckets=len(ph), ivcs=st.median([e["ivcs"] for e in ph if e["ivcs"] is not None] or [0]),
                     offcpu=(wall - cpu) / wall if wall else None, loop_busy=wall / dur, gnb_cpu=st.median([a for a, _ in g]) if g else None,
                     rho=spearman([a for a, _ in g], [b for _, b in g]), loads_s=loads / dur, qmax=max(q, default=0),
                     sustained=(all(v > 50 for v in q[first:]) if first is not None else None), per=per)

print("FLAGGED: shared host; runpod 64bba14d4768; trigger step 4 (PLAN.md option 15); 20 rps window\n")
print(f"{'start':24s}{'p95 ms':>9s}{'ivcs/s':>8s}{'offCPU':>8s}{'loop/s':>8s}{'gnb cpu us':>11s}{'rho':>6s}{'loads/s':>8s}{'qmax':>6s}{'sust':>6s}")
for t, r in rows.items():
    f = lambda v, fmt: "n/a" if v is None else format(v, fmt)
    print(f"{t:24s}{r['p95']:9.0f}{r['ivcs']:8.0f}{f(r['offcpu'], '8.1%')}{r['loop_busy']:8.2f}{f(r['gnb_cpu'] and r['gnb_cpu'] * 1e6, '11.1f')}"
          f"{f(r['rho'], '6.2f')}{r['loads_s']:8.2f}{r['qmax']:6d}{str(r['sustained']):>6s}")
print("\nPer-phase wall / CPU seconds over the window:")
for t, r in rows.items():
    print(f"  {t:24s} " + "  ".join(f"{p.split('_')[0]}{'_' + p.split('_')[-1] if p.count('_') > 1 else ''} {w:.2f}/{c:.2f}" for p, (w, c) in r["per"].items()))

k0 = [r for t, r in rows.items() if "-k0-" in t]; slow = {t: r for t, r in rows.items() if "-k14-" in t and r["slow"]}
print(f"\nk=0 starts: {len(k0)}; slow k=14 starts: {list(slow)}")
if k0:
    p95m = st.median(r["p95"] for r in k0)
    print(f"overhead check: k=0 median p95 {p95m:.0f} ms against 83 ms (part Q): {'HOLDS' if abs(p95m / 83 - 1) <= 0.2 else 'fails'}")
if k0 and slow:
    iv0 = st.median(r["ivcs"] for r in k0); off0 = max(r["offcpu"] or 0 for r in k0); g0 = st.median(r["gnb_cpu"] for r in k0 if r["gnb_cpu"]); l0 = st.median(r["loads_s"] for r in k0)
    for t, r in slow.items():
        print(f"{t}:")
        print(f"  (k) preemption: ivcs {r['ivcs']:.0f} vs {iv0:.0f} (x5 needed), off-CPU {r['offcpu']:.1%} (>=30%) vs k=0 max {off0:.1%} (<10%): "
              f"{'HOLDS' if r['ivcs'] >= 5 * max(iv0, 1e-9) and (r['offcpu'] or 0) >= .3 and off0 < .1 else 'fails'}")
        print(f"  (l) work grows with queue: rho {r['rho']}, gnb CPU/call {r['gnb_cpu'] * 1e6:.1f} us vs {g0 * 1e6:.1f} us (x3): "
              f"{'HOLDS' if (r['rho'] or 0) >= .5 and r['gnb_cpu'] >= 3 * g0 else 'fails'}")
        print(f"  (m) adapter churn: loads/s {r['loads_s']:.2f} vs {l0:.2f} (x3): {'HOLDS' if r['loads_s'] >= 3 * max(l0, 1e-9) else 'fails'}")
        print(f"  (n) self-sustaining: queue>50 first reached, then stays: {r['sustained']}: {'HOLDS' if r['sustained'] else 'fails'}")
elif k0:
    print("no slow k=14 start: (k) to (n) not tested")
