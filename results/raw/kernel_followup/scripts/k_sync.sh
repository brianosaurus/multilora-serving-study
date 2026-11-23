#!/usr/bin/env bash
# Are the extra empty completions in the grouped build a timing (race) bug? Compare normal vs CUDA_LAUNCH_BLOCKING=1.
set -u
cd /root/k; M=/root/k/model; A=/root/k/adapters; PYL=/root/k/venv/bin/python; SC=/root/k/scripts; OUT=/root/k/sync; mkdir -p $OUT; S=/root/k/sync.summary; : > $S
for rep in 1 2; do for combo in "base 0" "grouped 0" "base 1" "grouped 1"; do
  set -- $combo; bin=$1; blk=$2
  CMD=(/root/k/mrs-$bin serve -m $M -p 1234 --kv-cache 8GiB --max-seqs 256 --lora-max-adapters 64)
  for i in $(seq 0 7); do CMD+=(--lora "adapter_$(printf %04d $i)=$A/adapter_$(printf %04d $i)"); done
  CUDA_LAUNCH_BLOCKING=$blk setsid nohup "${CMD[@]}" > /root/k/server4_${bin}_$blk.log 2>&1 < /dev/null &
  SP=$!; for i in $(seq 1 100); do [ "$(curl -s -o /dev/null -w %{http_code} -m 3 localhost:1234/v1/models)" = 200 ] && break; sleep 3; done
  for r in 1 2 3; do
    $PYL $SC/loadgen.py --engine mistralrs --url http://localhost:1234 --tokenizer $M --requests 150 --warmup 20 --seed 5 --concurrency 8 --adapters 8 --popularity uniform --out $OUT/${bin}_blk${blk}_rep${rep}_$r.json > /dev/null 2>&1
    echo "rep$rep $bin launch_blocking=$blk run$r done" >> $S
  done
  kill $SP; sleep 6; kill -9 $SP 2>/dev/null; pkill -9 -f "mrs-$bin serve"; sleep 4
done; done
echo SYNC_DONE >> $S
