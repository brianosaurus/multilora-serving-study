#!/usr/bin/env python3
"""Trigger step 2 / option 13 (PLAN.md): split client-to-scheduler delay into client->front end, front end, hand-off+scheduler.
Usage: trigger2.py results/phase2 [smoke]"""
import glob, json, os, statistics as st, sys
D = sys.argv[1] if len(sys.argv) > 1 else "results/phase2"


def pct(a, q):
    a = sorted(a); return a[min(len(a) - 1, int(round(q / 100 * (len(a) - 1))))] if a else None


def events(cell):
    ev = {}
    for f in glob.glob(f"{cell}/trace/*.jsonl"):
        for l in open(f):
            try: e = json.loads(l)
            except ValueError: continue
            if e.get("e") in ("recv", "sent", "arrive") and e.get("rid"): ev.setdefault(e["rid"], {}).setdefault(e["e"], e["t"])
    return ev


def comps(cell, fname):
    d = json.load(open(f"{cell}/{fname}")); ev = events(cell)
    rows = []
    for r in d["requests"]:
        rid = r.get("rid"); x = ev.get(rid, {})
        if rid and "t_send_abs" in r and all(k in x for k in ("recv", "sent", "arrive")):
            rows.append({"i": r["i"], "send": r["t_send_abs"], "c2f": x["recv"] - r["t_send_abs"], "fe": x["sent"] - x["recv"], "h2s": x["arrive"] - x["sent"], "tot": x["arrive"] - r["t_send_abs"], "arr": x["arrive"]})
    return d, rows


if len(sys.argv) > 2 and sys.argv[2] == "smoke":
    cell = f"{D}/M0_smoke"; d, rows = comps(cell, "smoke.json")
    sent = [r["rid"] for r in d["requests"] if r.get("rid")]; ev = events(cell)
    print(f"requests with rid: {len(sent)}; trace rids matching: {sum(1 for r in sent if r in ev)}; with recv+sent+arrive: {len(rows)}")
    if rows: print("median ms  c2f %.2f  fe %.2f  h2s %.2f" % tuple(1000 * st.median(r[k] for r in rows) for k in ("c2f", "fe", "h2s")))
    sys.exit(0)
res = {}
for cell in sorted(glob.glob(f"{D}/{os.environ.get('CELLS', 'M_main-16-')}*")):
    tag = os.path.basename(cell); marks = {int(json.loads(l)["rate"]): json.loads(l) for l in open(f"{cell}/marks.jsonl") if l.strip()}
    for rate in (20, 40):
        if not os.path.exists(f"{cell}/zipf32_r{rate}.json"): continue
        d, rows = comps(cell, f"zipf32_r{rate}.json"); s = d["summary"]; n_all = len(d["requests"])
        rows.sort(key=lambda r: r["send"]); a0 = min((r["arr"] for r in rows), default=None)
        o = {"n": len(rows), "n_all": n_all, "p95": s["ttft_ms"]["p95"], "tok_s": s["output_tokens_per_s"]}
        for k in ("c2f", "fe", "h2s", "tot"): o[k + "_med"] = 1000 * st.median(r[k] for r in rows); o[k + "_p99"] = 1000 * pct([r[k] for r in rows], 99)
        o["t0_minus_tstart"] = d["t0_abs"] - marks[rate]["t_start"]
        o["send0_minus_tstart"] = rows[0]["send"] - marks[rate]["t_start"] if rows else None
        o["arr0_minus_tstart"] = a0 - marks[rate]["t_start"] if a0 else None
        res[(tag, rate)] = o
print("FLAGGED: shared host; runpod 64bba14d4768; trigger step 2 (PLAN.md option 13)\n")
hdr = ["n/all", "p95 ms", "c2f med", "c2f p99", "fe med", "fe p99", "h2s med", "h2s p99", "tot med", "tot p99", "t0-tst", "send0-tst", "arr0-tst"]
print(f"{'start@rate':22s}" + "".join(h.rjust(10) for h in hdr))
for (t, r), o in sorted(res.items(), key=lambda kv: (kv[0][1], int(kv[0][0].rsplit('-', 1)[1]))):
    vals = [f"{o['n']}/{o['n_all']}", f"{o['p95']:.0f}"] + [f"{o[k]:.1f}" for k in ("c2f_med", "c2f_p99", "fe_med", "fe_p99", "h2s_med", "h2s_p99", "tot_med", "tot_p99")] + [f"{o[k]:.2f}" if o[k] is not None else "n/a" for k in ("t0_minus_tstart", "send0_minus_tstart", "arr0_minus_tstart")]
    print(f"{t + '@' + str(r):22s}" + "".join(v.rjust(10) for v in vals))
r20 = {t: o for (t, r), o in res.items() if r == 20}
slow = {t: o for t, o in r20.items() if o["p95"] > 10000}; norm = {t: o for t, o in r20.items() if o["p95"] <= 10000}
print(f"\n20 rps: slow starts {sorted(slow)}; normal {len(norm)}")
if r20:
    a = all(o["c2f_med"] <= 5 and o["c2f_p99"] <= 50 for o in r20.values())
    print(f"(a) client->front end median<=5 ms and p99<=50 ms in every 20 rps start: {'HOLDS' if a else 'fails'}")
    if slow and norm:
        m = {k: st.median(o[k] for o in norm.values()) for k in ("c2f_med", "fe_med", "h2s_med", "tot_med")}
        for t, o in slow.items():
            ex = o["tot_med"] - m["tot_med"]; eh = o["h2s_med"] - m["h2s_med"]
            print(f"(b) {t}: extra median delay {ex:.1f} ms, of which hand-off+scheduler {eh:.1f} ms ({(eh / ex * 100 if ex > 0 else float('nan')):.0f}%): {'HOLDS' if ex > 0 and eh / ex >= 0.8 else 'fails'}")
    else: print("(b) not tested: no slow start" if not slow else "(b) not tested: no normal start")
    xs = [o["arr0_minus_tstart"] for o in r20.values() if o["arr0_minus_tstart"] is not None]; ys = [o["send0_minus_tstart"] for o in r20.values() if o["send0_minus_tstart"] is not None]
    if len(xs) > 2:
        vx, vy = max(xs) - min(xs), max(ys) - min(ys)
        print(f"(c) range of t_start->first arrival {vx:.2f} s; range of t_start->first send {vy:.2f} s; share {vy / vx * 100 if vx else float('nan'):.0f}%: {'HOLDS' if vx and vy / vx >= 0.9 else 'fails'}")
    nn = [o["p95"] for o in norm.values()]
    if nn: med = st.median(nn); print(f"(d) normal 20 rps p95 within 20% of their median {med:.0f} ms: {'HOLDS' if all(abs(x - med) <= 0.2 * med for x in nn) else 'fails'} ({sorted(round(x) for x in nn)})")
