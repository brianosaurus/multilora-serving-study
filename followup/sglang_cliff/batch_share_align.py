#!/usr/bin/env python3
"""Option 9.2 (d), clock-aligned. batch_share.py compares the first slow request (loadgen clock, t_send) with the batch timeline (clock of marks.jsonl t_start, which precedes the load generator's start by its start-up time). Here both are put on the loadgen clock: offset = first decode call after t_start minus the first request's t_send. Usage: batch_share_align.py results/phase2"""
import glob, json, statistics as st, sys
D = sys.argv[1] if len(sys.argv) > 1 else "results/phase2"
for tag, rate in (("B_main-16", 20), ("E_main-16-P1", 20), ("B_main-16", 40), ("E_main-16-U4", 40)):
    c = f"{D}/{tag}"
    ev = [json.loads(l) for f in glob.glob(f"{c}/trace/*.jsonl") for l in open(f)]
    m = {int(json.loads(l)["rate"]): json.loads(l) for l in open(f"{c}/marks.jsonl") if l.strip()}[rate]
    dec = sorted([e for e in ev if e["e"] == "batch" and "DECODE" in e["mode"].upper() and m["t_start"] <= e["t"] <= m["t_end"] + 5], key=lambda e: e["t"])
    reqs = json.load(open(f"{c}/zipf32_r{rate}.json"))["requests"]
    off = dec[0]["t"] - min(r["t_send"] for r in reqs if "t_send" in r) - m["t_start"]
    bk = {}
    for b in dec: bk.setdefault(int((b["t"] - m["t_start"]) - off), []).append(b["n"])
    tl = {k: st.mean(v) for k, v in bk.items()}
    first_over48 = next((k for k in sorted(tl) if tl[k] > 48), None)
    sustained = next((k for k in sorted(tl) if all(tl.get(j, 0) > 48 for j in range(k, k + 5))), None)
    slow = sorted(r["t_send"] for r in reqs if r.get("ttft_s") is not None and r["ttft_s"] > 1.0)
    print(f"{tag} {rate} rps: first request over 1 s TTFT sent at {slow[0]:.1f} s (loadgen clock); mean batch above 48 from {first_over48} s, for 5 s running from {sustained} s; mean batch in the second of the first slow request: {tl.get(int(slow[0]), 0):.0f}")
