#!/usr/bin/env bash
# Balanced A/B: base, grouped, and a second copy of base (same-vs-same control), all 6 orderings, GPU state logged
set -u
cd /root/k; cp -n mrs-base mrs-base2
M=/root/k/model; A=/root/k/adapters; PYL=/root/k/venv/bin/python; SC=/root/k/scripts; OUT=/root/k/ab2; mkdir -p $OUT
S=/root/k/ab2.summary; : > $S
gpu() { nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw,clocks_throttle_reasons.active --format=csv,noheader | tr -d ' ' | tr '\n' ' '; echo "load=$(cut -d' ' -f1 /proc/loadavg)"; }
orders=("base grouped base2" "grouped base2 base" "base2 base grouped" "grouped base base2" "base base2 grouped" "base2 grouped base")
blk=0
for ord in "${orders[@]}"; do
  blk=$((blk+1)); pos=0
  for bin in $ord; do
    pos=$((pos+1))
    CMD=(/root/k/mrs-$bin serve -m $M -p 1234 --kv-cache 8GiB --max-seqs 256 --lora-max-adapters 64)
    for i in $(seq 0 7); do CMD+=(--lora "adapter_$(printf %04d $i)=$A/adapter_$(printf %04d $i)"); done
    setsid nohup "${CMD[@]}" > /root/k/server2_${bin}_$blk.log 2>&1 < /dev/null &
    SP=$!; for i in $(seq 1 100); do [ "$(curl -s -o /dev/null -w %{http_code} -m 3 localhost:1234/v1/models)" = 200 ] && break; sleep 3; done
    lg() { n=$1; shift; r=$($PYL $SC/loadgen.py --engine mistralrs --url http://localhost:1234 --tokenizer $M --requests 150 --warmup 20 --seed 5 --out $OUT/${n}_${bin}_b$blk.json "$@" 2>&1 | tail -1 | $PYL -c "import sys,json;d=json.loads(sys.stdin.read());print(d['ok'],d['errors'],'tpot_p50',round(d['tpot_ms']['p50'],2),'ttft_p95',round(d['ttft_ms']['p95']),'tok/s',d['tok_per_s'])" 2>&1 | tail -1); echo "blk$blk pos$pos $bin $n: $r | gpu $(gpu)" >> $S; }
    lg warm --concurrency 8 --adapters 8 --popularity uniform
    lg a8_c8 --concurrency 8 --adapters 8 --popularity uniform
    lg a0_c32 --concurrency 32 --adapters 0
    lg a1_c32 --concurrency 32 --adapters 1
    lg a4_c32 --concurrency 32 --adapters 4 --popularity uniform
    lg a8_c32 --concurrency 32 --adapters 8 --popularity uniform
    kill $SP; sleep 6; kill -9 $SP 2>/dev/null; pkill -9 -f "mrs-$bin serve"; sleep 4
  done
done
echo AB2_DONE >> $S
