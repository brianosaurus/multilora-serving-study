#!/usr/bin/env python3
"""Trigger step 5 (PLAN.md option 16): timed functions inside batch selection, 20 rps window. Usage: analyze_subphases.py results/phase2 [prefix]
Inclusive times: get_next_batch_to_run contains get_new_batch_prefill, update_running_batch, process_pending_chunked_abort and
_process_hicache_events; get_new_batch_prefill contains calc_priority, reorder, can_schedule_lora_req and add_one_req.
'gnb other' and 'prefill other' are the parent minus its timed children (the untimed remainder)."""
import glob, json, os, statistics as st, sys


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


D = sys.argv[1]; PRE = sys.argv[2] if len(sys.argv) > 2 else "S_main-16-"
GNB_KIDS = ["get_new_batch_prefill", "update_running_batch", "process_pending_chunked_abort", "_process_hicache_events"]
PF_KIDS = ["SchedulePolicy.calc_priority", "IndexedReqList.reorder", "can_schedule_lora_req", "PrefillAdder.add_one_req"]
res = {}
for cell in sorted(glob.glob(f"{D}/{PRE}*")):
    if not (os.path.exists(f"{cell}/marks.jsonl") and os.path.exists(f"{cell}/zipf32_r20.json")): continue
    tag = os.path.basename(cell)
    m = next(json.loads(l) for l in open(f"{cell}/marks.jsonl") if l.strip() and int(json.loads(l)["rate"]) == 20)
    t0, t1 = m["t_start"], m["t_end"]; dur = t1 - t0
    ph = []
    for f in glob.glob(f"{cell}/trace/*.jsonl"):
        for l in open(f):
            try:
                e = json.loads(l)
                if e["e"] == "ph" and t0 <= e["t"] < t1: ph.append(e)
            except ValueError: pass
    def tot(n, i): return sum(e["p"].get(n, [0, 0, 0])[i] for e in ph)
    gnb_calls = tot("get_next_batch_to_run", 0)
    cpu = {n: tot(n, 2) for n in ["get_next_batch_to_run"] + GNB_KIDS + PF_KIDS}
    cpu["gnb other"] = cpu["get_next_batch_to_run"] - sum(cpu[k] for k in GNB_KIDS)
    cpu["prefill other"] = cpu["get_new_batch_prefill"] - sum(cpu[k] for k in PF_KIDS)
    calls = {n: tot(n, 0) for n in GNB_KIDS + PF_KIDS}
    rho = {}
    for n in list(cpu):
        xs, qs = [], []
        for e in ph:
            g = e["p"].get("get_next_batch_to_run", [0, 0, 0])[0]
            if not g: continue
            if n == "gnb other": v = e["p"]["get_next_batch_to_run"][2] - sum(e["p"].get(k, [0, 0, 0])[2] for k in GNB_KIDS)
            elif n == "prefill other": v = e["p"].get("get_new_batch_prefill", [0, 0, 0])[2] - sum(e["p"].get(k, [0, 0, 0])[2] for k in PF_KIDS)
            else: v = e["p"].get(n, [0, 0, 0])[2]
            xs.append(v / g); qs.append(e["qlen"])
        rho[n] = spearman(xs, qs)
    p95 = json.load(open(f"{cell}/zipf32_r20.json"))["summary"]["ttft_ms"]["p95"]
    res[tag] = dict(p95=p95, gnb_calls=gnb_calls, per_call={n: 1e6 * v / max(gnb_calls, 1) for n, v in cpu.items()}, rho=rho,
                    calls_s={n: v / dur for n, v in calls.items()}, qmax=max((e["qlen"] for e in ph), default=0))

print("FLAGGED: shared host; runpod 64bba14d4768; trigger step 5 (PLAN.md option 16); 20 rps window")
print("CPU microseconds per get_next_batch_to_run call (inclusive; see docstring), Spearman with queue length in brackets\n")
names = list(next(iter(res.values()))["per_call"]) if res else []
print(f"{'':32s}" + "".join(f"{t.replace('S_main-16-', ''):>18s}" for t in res))
print(f"{'p95 TTFT ms':32s}" + "".join(f"{r['p95']:18.0f}" for r in res.values()))
print(f"{'max queue':32s}" + "".join(f"{r['qmax']:18d}" for r in res.values()))
for n in names:
    print(f"{n:32s}" + "".join(f"{r['per_call'][n]:10.1f} [{(r['rho'][n] if r['rho'][n] is not None else float('nan')):5.2f}]" for r in res.values()))
print("\ncalls per second")
for n in GNB_KIDS + PF_KIDS:
    print(f"{n:32s}" + "".join(f"{r['calls_s'][n]:18.0f}" for r in res.values()))

k0 = [r for t, r in res.items() if "-k0-" in t]; k14 = {t: r for t, r in res.items() if "-k14-" in t}
if k0 and k14:
    b = k0[0]; t, s = max(k14.items(), key=lambda kv: kv[1]["p95"])
    inc = s["per_call"]["get_next_batch_to_run"] - b["per_call"]["get_next_batch_to_run"]
    leaves = ["update_running_batch", "process_pending_chunked_abort", "_process_hicache_events", "gnb other", "prefill other"] + PF_KIDS
    share = {n: (s["per_call"][n] - b["per_call"][n]) / inc if inc > 0 else None for n in leaves}
    top = max(share, key=lambda n: share[n] or -1)
    print(f"\nslowest k=14 start: {t} (p95 {s['p95']:.0f} ms); increase in get-next-batch CPU per call: {inc:.1f} us")
    for n in sorted(leaves, key=lambda n: -(share[n] or 0)): print(f"  {n:32s} share {share[n]:6.1%}  rho {s['rho'][n]}")
    print(f"(o) {top}: share {share[top]:.1%} (>=50%), rho {s['rho'][top]} (>=0.5): {'HOLDS' if share[top] >= .5 and (s['rho'][top] or 0) >= .5 else 'fails'}")
    c0 = b["calls_s"]["can_schedule_lora_req"]; c1 = s["calls_s"]["can_schedule_lora_req"]
    print(f"(p) can_schedule_lora_req calls/s {c1:.0f} vs {c0:.0f} (x5): {'HOLDS' if c1 >= 5 * max(c0, 1e-9) else 'fails'}")
