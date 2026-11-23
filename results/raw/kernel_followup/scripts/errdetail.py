import json,glob,collections
for b in ("base","grouped"):
    msgs=collections.Counter(); where=collections.Counter(); n=0
    for f in sorted(glob.glob(f"/root/k/ab2/a8_c*_{b}_b*.json")):
        d=json.load(open(f))
        for r in d["requests"]:
            if r.get("error") or not r.get("ok",True):
                n+=1; msgs[str(r.get("error"))[:90]]+=1; where[r.get("adapter")]+=1
    print(b,"failed requests:",n); print("  messages:",dict(msgs)); print("  by adapter:",dict(where))
d=json.load(open(sorted(glob.glob("/root/k/ab2/a8_c8_grouped_b1.json"))[0])); print("keys in a request record:",list(d["requests"][0].keys()))
