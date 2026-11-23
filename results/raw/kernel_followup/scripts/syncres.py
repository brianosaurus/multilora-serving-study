import json,glob,collections,re
res=collections.defaultdict(lambda: collections.defaultdict(int)); runs=collections.Counter()
for f in sorted(glob.glob("/root/k/sync/*.json")):
    m=re.match(r".*/(\w+?)_blk(\d)_rep(\d)_(\d)\.json",f)
    if not m: continue
    key=(m[1],"blocking" if m[2]=="1" else "normal"); runs[key]+=1
    for r in json.load(open(f))["requests"]:
        if r.get("error") or r.get("ttft_s") is None: res[key][r["i"]]+=1
for key in sorted(runs):
    print("%-8s %-9s runs=%d  failing request index -> times failed: %s"%(key[0],key[1],runs[key],dict(sorted(res[key].items()))))
