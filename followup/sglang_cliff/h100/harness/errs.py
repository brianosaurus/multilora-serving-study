import json, collections, sys
for tag in ("tp1_stock_sat", "tp1_patched_sat"):
    d = json.load(open(f"/workspace/h100_run/results/{tag}/sat_c1024.json"))
    recs = d["requests"]
    c = collections.Counter()
    for r in recs:
        e = r.get("error") or r.get("err")
        if e:
            c[str(e)[:140]] += 1
    print(tag, "records", len(recs), "keys", sorted(recs[0].keys())[:12])
    for k, v in c.most_common(4):
        print("  ", v, k)
    ok = [r for r in recs if not (r.get("error") or r.get("err"))]
    if ok:
        print("   first ok t_send", ok[0].get("t_send"), "last ok t_done", max(r.get("t_done", 0) or 0 for r in ok))
