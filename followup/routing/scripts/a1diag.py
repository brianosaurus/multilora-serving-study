import json, numpy as np, statistics as st
d=json.load(open("/root/k/routing/refs_out.json")); C=d["candidates"]; P=len(d["prompts"])
L32={v:np.array(d["L32"][v]) for v in C}; LBF={v:np.array(d["Lbf"][v]) for v in C}
REF={c:np.array([L32[c][p][i] for p in range(P)]) for i,c in enumerate(C)}
dist=lambda L,v,p,ci: float(np.mean(np.abs(L[v][p][ci]-REF[C[ci]][p])))
ADP=C[1:]
rows=[]
for a in ADP:
    for p in range(P):
        r=dist(LBF,a,p,C.index(a)); w=min((dist(LBF,a,p,ci),C[ci]) for ci in range(len(C)) if C[ci]!=a)
        rows.append((a,p,r,w[0],w[1]))
nearest=sum(r<w for _,_,r,w,_ in rows)
print(f"correct implementation: nearest candidate is the right one (no margin) on {nearest}/{len(rows)} requests")
margins=np.array([w-r for _,_,r,w,_ in rows]); noise=st.median([r for _,_,r,_,_ in rows])
print(f"noise {noise:.3f}; margin (d_wrong - d_right) min {margins.min():.3f}, 5th pct {np.percentile(margins,5):.3f}, median {np.median(margins):.3f}")
print("largest k for which check 1 passes on all 64 requests: k <", round(float(margins.min()/noise),2))
pair=[]
import itertools
for c1,c2 in itertools.combinations(C,2):
    d12=st.median([float(np.mean(np.abs(L32[c1][p][C.index(c2)]-REF[c2][p]))) for p in range(P)])
    d21=st.median([float(np.mean(np.abs(L32[c2][p][C.index(c1)]-REF[c1][p]))) for p in range(P)]); pair.append(min(d12,d21))
print("largest k for which check 4 passes on all 10 pairs: k <", round(min(pair)/noise,2))
print("\nrequests where a WRONG candidate is nearer than the right one, or margin < noise:")
for a,p,r,w,wc in sorted(rows,key=lambda x:x[3]-x[2])[:8]: print(f"   {a} prompt {p:2d}: d_right {r:.3f} d_wrong {w:.3f} nearest wrong = {wc}  margin {w-r:+.3f}")
print("\nper-adapter: median d_right / median d_wrong / median margin")
for a in ADP:
    rr=[x for x in rows if x[0]==a]; print(f"   {a}: {st.median([x[2] for x in rr]):.3f} / {st.median([x[3] for x in rr]):.3f} / {st.median([x[3]-x[2] for x in rr]):.3f}")
# robust alternative views (descriptive only)
fp=[(a,p,float(np.median(np.abs(LBF[a][p][C.index(a)]-REF[a][p])))) for a in ADP for p in range(P)]
print(f"\nmedian-over-positions (instead of mean) of bf16-vs-fp32 distance, median over requests: {st.median([x[2] for x in fp]):.3f} (mean-based noise is {noise:.3f})")
print("fraction of positions where the forced token's fp32 logprob is above -1 (peaked), pooled:", round(float(np.mean([REF[a][p]>-1 for a in ADP for p in range(P)])),3))
