#!/usr/bin/env bash
# B3: 128 end-to-end completions (16 prompts x 8 adapters, 8 distinct adapters in flight, temperature 0), old vs new flash-attn builds
cd /root/rerun; L=b3.log; : > $L; PY=/root/e/sglang/bin/python; mkdir -p b3out
say() { echo "$(date -u +%T) $*" >> $L; }
say "host $(hostname); GPU procs $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l); load $(cut -d' ' -f1-3 /proc/loadavg)"
say "binaries: old $(sha256sum mrs-old-fa | cut -c1-16)  new $(sha256sum mrs-new-fa | cut -c1-16)"
for tag in old new old_again; do
  b=mrs-old-fa; [ $tag = new ] && b=mrs-new-fa
  CMD=(/root/rerun/$b serve -m /root/k/model -p 1234 --kv-cache 8GiB --max-seqs 256 --lora-max-adapters 64)
  for i in 0 1 2 3 4 5 6 7; do CMD+=(--lora "adapter_000$i=/root/k/adapters/adapter_000$i"); done
  "${CMD[@]}" > b3out/server_$tag.log 2>&1 &
  SP=$!; for i in $(seq 1 100); do [ "$(curl -s -o /dev/null -w %{http_code} -m 3 localhost:1234/v1/models)" = 200 ] && break; sleep 3; done
  $PY /root/k/outputs.py b3out/$tag.json >> $L 2>&1; say "$tag done: $(python3 -c "import json;d=json.load(open('b3out/$tag.json'));print(len(d),'completions,',sum(1 for v in d.values() if v.startswith('ERR')),'errors,',sum(1 for v in d.values() if v.strip()=='' ),'empty')" 2>&1 | tail -1)"
  kill $SP; sleep 6; kill -9 $SP 2>/dev/null; sleep 3
done
python3 - >> $L <<PY
import json
o=json.load(open("b3out/old.json")); n=json.load(open("b3out/new.json")); o2=json.load(open("b3out/old_again.json"))
same=lambda a,b: sum(a[k]==b[k] for k in a)
print(f"old vs new: {same(o,n)}/{len(o)} identical; old vs old (repeat): {same(o,o2)}/{len(o)} identical; new vs old (repeat): {same(n,o2)}/{len(o)} identical")
for k in [k for k in o if o[k]!=n[k]][:5]: print("  differs:", repr(k[:70]), "| old:", repr(o[k][:50]), "| new:", repr(n[k][:50]))
PY
say B3_DONE
