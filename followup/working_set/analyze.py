"""Working-set analysis of the main grid's request logs (results/raw/main, 16 slots, 32 adapters).
Instantaneous: distinct adapters with a request in flight at time t, [t_send, t_done), sampled every 10 ms.
Window w: distinct adapters with a request in flight at any time in (t-w, t], i.e. t in [t_send, t_done+w).
All requests (including the 50 warm-up requests) form the history; statistics use only the measured span, from the first measured request's send time (i >= 50) to the last completion. Requests without t_done or adapter are excluded.
Files: vLLM from results/raw/main/vllm, SGLang (valid session) from results/raw/main_rerun1/sglang."""
import json,glob,re,statistics as st
roots={'vllm':'results/raw/main/vllm','sglang':'results/raw/main_rerun1/sglang'}
def series(iv,t0,t1,step=0.01):
    ev=[]
    for s,e,a in iv: ev.append((s,1,a)); ev.append((e,-1,a))
    ev.sort(key=lambda x:(x[0],x[1]))
    cnt={}; n=0; j=0; out=[]; t=t0-1e9 if False else t0
    while t<t1:
        while j<len(ev) and ev[j][0]<=t:
            _,d,a=ev[j]; c=cnt.get(a,0); cnt[a]=c+d
            if c==0 and d==1: n+=1
            if c+d==0: n-=1
            j+=1
        out.append(n); t+=step
    return out
def q(x,p): x=sorted(x); return x[min(len(x)-1,int(p*len(x)))]
def stats(x): return [st.median(x),q(x,.95),max(x),sum(v>16 for v in x)/len(x)]
res={}
for eng in ('vllm','sglang'):
  for dist in ('zipf32','uni32'):
    for f in sorted(glob.glob(f'{roots[eng]}/{eng}_{dist}_r*_rep*.json')):
        m=re.search(r'_r(\d+)_rep(\d+)',f); rate=int(m[1]); rep=int(m[2])
        d=json.load(open(f)); r=[x for x in d['requests'] if x.get('i',0)>=50]
        ok=[x for x in d['requests'] if x.get('t_done') is not None and x.get('adapter')]
        ms=[x for x in ok if x.get('i',0)>=50]
        t0=min(x['t_send'] for x in ms); t1=max(x['t_done'] for x in ms)
        sm=d['summary']; row=dict(rep=rep,n=len(r),used=len(ms),err=sm['errors'],ttft_p95=sm['ttft_ms']['p95'])
        for name,w in (('inst',0),('w1',1),('w5',5),('w10',10)):
            row[name]=stats(series([(x['t_send'],x['t_done']+w,x['adapter']) for x in ok],t0,t1))
        res.setdefault((eng,dist,rate),[]).append(row)
json.dump({'|'.join(map(str,k)):v for k,v in res.items()},open('followup/working_set/working_set.json','w'),indent=1)
def med(rows,key,i): return st.median(r[key][i] for r in rows)
L=['| engine | traffic | rps | reps | TTFT p95 ms (median of reps) | errors | inst med/p95/max/>16 | 1 s med/p95/max/>16 | 5 s | 10 s |','|---|---|---|---|---|---|---|---|---|---|']
for k in sorted(res,key=lambda k:(k[0],k[1],k[2])):
    rows=res[k]
    def f(key): return ' / '.join([f"{med(rows,key,0):.0f}",f"{med(rows,key,1):.0f}",f"{med(rows,key,2):.0f}",f"{100*med(rows,key,3):.0f}%"])
    L.append(f"| {k[0]} | {k[1]} | {k[2]} | {len(rows)} | {st.median(r['ttft_p95'] for r in rows):.0f} | {sum(r['err'] for r in rows)} | {f('inst')} | {f('w1')} | {f('w5')} | {f('w10')} |")
open('followup/working_set/table.md','w').write('\n'.join(L)+'\n')
print('\n'.join(L))
