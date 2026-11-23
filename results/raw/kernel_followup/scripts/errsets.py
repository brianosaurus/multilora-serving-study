import json,glob,collections
for cell in ("a8_c8","a8_c32"):
    sets={}
    for b in ("base","base2","grouped"):
        s=collections.Counter()
        for f in sorted(glob.glob(f"/root/k/ab2/{cell}_{b}_b*.json")):
            d=json.load(open(f))
            for r in d["requests"]:
                if r.get("error") or r.get("ttft_s") is None: s[r["i"]]+=1
        sets[b]=s
    allk=set().union(*[set(s) for s in sets.values()])
    print(cell,"| failing request indices (count of 6 blocks each build failed on it)")
    print("   idx   base base2 grouped")
    for k in sorted(allk): print("   %4d  %4d %5d %7d"%(k,sets["base"][k],sets["base2"][k],sets["grouped"][k]))
    print("   totals: base",sum(sets["base"].values()),"base2",sum(sets["base2"].values()),"grouped",sum(sets["grouped"].values()))
