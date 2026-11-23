#!/usr/bin/env bash
cd /root/rerun; L=b3_more.log; : > $L; PY=/root/e/sglang/bin/python
say() { echo "$(date -u +%T) $*" >> $L; }
say "host $(hostname); GPU procs $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l); load $(cut -d' ' -f1-3 /proc/loadavg)"
for r in 1 2 3 4; do for tag in new old; do
  b=mrs-old-fa; [ $tag = new ] && b=mrs-new-fa
  CMD=(/root/rerun/$b serve -m /root/k/model -p 1234 --kv-cache 8GiB --max-seqs 256 --lora-max-adapters 64)
  for i in 0 1 2 3 4 5 6 7; do CMD+=(--lora "adapter_000$i=/root/k/adapters/adapter_000$i"); done
  "${CMD[@]}" > b3out/server_${tag}_r$r.log 2>&1 &
  SP=$!; for i in $(seq 1 100); do [ "$(curl -s -o /dev/null -w %{http_code} -m 3 localhost:1234/v1/models)" = 200 ] && break; sleep 3; done
  $PY /root/k/outputs.py b3out/${tag}_r$r.json >> $L 2>&1; say "$tag run $r done"
  kill $SP; sleep 6; kill -9 $SP 2>/dev/null; sleep 3
done; done
$PY - >> $L <<PY
import json, itertools
runs = {}
for t in ("old", "new"):
    runs[t] = [json.load(open("b3out/old.json" if (t == "old") else "b3out/new.json"))] + [json.load(open(f"b3out/{t}_r{r}.json")) for r in range(1, 5)]
runs["old"].append(json.load(open("b3out/old_again.json")))
same = lambda a, b: sum(a[k] == b[k] for k in a)
oo = [same(a, b) for a, b in itertools.combinations(runs["old"], 2)]
nn = [same(a, b) for a, b in itertools.combinations(runs["new"], 2)]
on = [same(a, b) for a in runs["old"] for b in runs["new"]]
f = lambda v: f"n={len(v)} pairs, identical completions per pair: min {min(v)}, median {sorted(v)[len(v)//2]}, max {max(v)} of 128; pairs with 128/128: {sum(x==128 for x in v)}"
print("old vs old (different launches of the same binary):", f(oo))
print("new vs new:", f(nn))
print("old vs new:", f(on))
keys = list(runs["old"][0])
def varies(rs): return {k for k in keys if len({r[k] for r in rs}) > 1}
print("completions that vary across launches of old:", len(varies(runs["old"])), "| of new:", len(varies(runs["new"])), "| of all old+new pooled:", len(varies(runs["old"] + runs["new"])))
print("completions that vary only across builds (stable within old and within new, different between):", len([k for k in keys if len({r[k] for r in runs['old']}) == 1 and len({r[k] for r in runs['new']}) == 1 and runs['old'][0][k] != runs['new'][0][k]]))
PY
say B3_MORE_DONE
