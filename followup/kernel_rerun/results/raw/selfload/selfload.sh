#!/usr/bin/env bash
cd /root/rerun; PY=/root/e/sglang/bin/python; O=selfload.txt; : > $O
echo "before: load $(cut -d' ' -f1-3 /proc/loadavg)" >> $O
CMD=(/root/rerun/mrs-new-fa serve -m /root/k/model -p 1234 --kv-cache 8GiB --max-seqs 256 --lora-max-adapters 64)
for i in 0 1 2 3 4 5 6 7; do CMD+=(--lora "adapter_000$i=/root/k/adapters/adapter_000$i"); done
"${CMD[@]}" > selfload_server.log 2>&1 &
SP=$!; for i in $(seq 1 100); do [ "$(curl -s -o /dev/null -w %{http_code} -m 3 localhost:1234/v1/models)" = 200 ] && break; sleep 3; done
echo "server up: load $(cut -d' ' -f1-3 /proc/loadavg)" >> $O
( for i in $(seq 1 40); do l1=$(head -1 /proc/stat); sleep 2; l2=$(head -1 /proc/stat); echo "$(cut -d' ' -f1 /proc/loadavg) $(python3 - "$l1" "$l2" <<PY
import sys
a=list(map(int,sys.argv[1].split()[1:9])); b=list(map(int,sys.argv[2].split()[1:9])); d=[y-x for x,y in zip(a,b)]; t=sum(d)
print("%.2f" % (128*(t-d[3]-d[4])/t))
PY
)"; done ) > selfload_samples.txt &
SAMP=$!
$PY /root/k/scripts/loadgen.py --engine mistralrs --url http://localhost:1234 --tokenizer /root/k/model --requests 600 --warmup 20 --seed 5 --concurrency 32 --adapters 8 --popularity uniform --out selfload_run.json > selfload_loadgen.txt 2>&1
wait $SAMP
kill $SP; sleep 5; kill -9 $SP 2>/dev/null
echo "after: load $(cut -d' ' -f1-3 /proc/loadavg)" >> $O
$PY - >> $O <<PY
import statistics as st
rows=[l.split() for l in open("selfload_samples.txt") if l.strip()]
la=[float(r[0]) for r in rows]; cores=[float(r[1]) for r in rows]
print(f"{len(rows)} samples while a 32-concurrent, 8-adapter run was active: 1-minute load average min {min(la):.2f} max {max(la):.2f}; host CPU busy (cores) median {st.median(cores):.2f} max {max(cores):.2f}")
PY
echo SELFLOAD_DONE >> $O
