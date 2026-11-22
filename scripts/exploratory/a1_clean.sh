#!/usr/bin/env bash
cd /root/study
M=/dev/shm/models/Qwen2.5-1.5B-Instruct; A=/dev/shm/adapters; PYL=/root/study/venvs/vllm/bin/python; SC=/root/study/scripts
OUT=/root/study/results/a1_clean; mkdir -p $OUT; S=logs/a1_clean.summary; : > $S
start() { CMD=(/root/study/bin/mistralrs serve -m $M -p 1234 --kv-cache 8GiB --max-seqs 256 --lora-max-adapters 64)
  for i in $(seq 0 7); do CMD+=(--lora "adapter_$(printf %04d $i)=$A/adapter_$(printf %04d $i)"); done
  setsid nohup "${CMD[@]}" > logs/a1_clean_$1.log 2>&1 < /dev/null &
  SP=$!; for i in $(seq 1 100); do [ "$(curl -s -o /dev/null -w %{http_code} -m 3 localhost:1234/v1/models)" = 200 ] && break; sleep 3; done; }
stop() { kill $SP; sleep 6; pkill -9 -f "[b]in/mistralrs serve"; sleep 4; }
lg() { tag=$1; n=$2; shift 2; echo "$tag $n: $($PYL $SC/loadgen.py --engine mistralrs --url http://localhost:1234 --tokenizer $M --requests 150 --warmup 20 --seed 5 --out $OUT/${tag}_$n.json "$@" 2>&1 | tail -1 | $PYL -c "import sys,json;d=json.loads(sys.stdin.read());print(d[\"ok\"],d[\"errors\"],\"tpot_p50\",round(d[\"tpot_ms\"][\"p50\"],1),\"ttft_p95\",round(d[\"ttft_ms\"][\"p95\"]),\"tok/s\",d[\"tok_per_s\"])" 2>&1 | tail -1)" >> $S; }
# server 1: only 1-adapter traffic first, then 8-adapter traffic, then 1-adapter again
start s1
lg s1 warm --concurrency 8 --adapters 1
for k in 1 2 3; do lg s1 first_a1_c32_$k --concurrency 32 --adapters 1; done
lg s1 a8_c32 --concurrency 32 --adapters 8 --popularity uniform
for k in 1 2 3; do lg s1 after_a1_c32_$k --concurrency 32 --adapters 1; done
stop
# server 2: no-adapter traffic first, then 1-adapter
start s2
lg s2 warm --concurrency 8 --adapters 0
lg s2 a0_c32 --concurrency 32 --adapters 0
for k in 1 2; do lg s2 after_a0_a1_c32_$k --concurrency 32 --adapters 1; done
stop
echo DONE >> $S
