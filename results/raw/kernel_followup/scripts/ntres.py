import json,collections
tags=["base","grouped","base2","grouped2"]
d={t:json.load(open(f"/root/k/nt/{t}.json")) for t in tags}
print("target idx (adapter): empty first-token count out of 12, per build")
print("%-16s %5s %8s %6s %9s"%("idx (adapter)","base","grouped","base2","grouped2"))
for k in d["base"]:
    row=[sum((tr["text"] or "")=="" for tr in d[t][k]["trials"]) for t in tags]
    print("%-16s %5d %8d %6d %9d"%(k+" ("+d["base"][k]["adapter"][-1]+")",*row))
print()
print("same text per trial across builds (identical companion mixes), fraction of 12 trials where base == grouped:")
for k in d["base"]:
    a=[tr["text"] for tr in d["base"][k]["trials"]]; b=[tr["text"] for tr in d["grouped"][k]["trials"]]; c=[tr["text"] for tr in d["base2"][k]["trials"]]
    print("  idx %-4s base-vs-grouped %2d/12   base-vs-base2 %2d/12"%(k,sum(x==y for x,y in zip(a,b)),sum(x==y for x,y in zip(a,c))))
print()
print("first-token top logprobs for the flaky targets (base, trial 0 and a trial that differs):")
for k in ("32","106"):
    for t in ("base","grouped"):
        trials=d[t][k]["trials"]
        e=[i for i,tr in enumerate(trials) if (tr["text"] or "")==""]; n=[i for i,tr in enumerate(trials) if (tr["text"] or "")!=""]
        for label,idxs in (("empty",e[:1]),("nonempty",n[:1])):
            for i in idxs: print("  idx %s %-8s %-8s trial %d text=%r top=%s"%(k,t,label,i,trials[i]["text"],trials[i]["top"]))
