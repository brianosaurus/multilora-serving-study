#!/usr/bin/env bash
cd /root/study
M=/dev/shm/models/Qwen2.5-1.5B-Instruct; A=/dev/shm/adapters; PYL=/root/study/venvs/vllm/bin/python; SC=/root/study/scripts
OUT=/root/study/results/onecall_rep; mkdir -p $OUT; S=logs/onecall_rep.summary; : > $S
for rep in 1 2 3; do for v in 0 1; do
  CMD=(/root/study/bin/mistralrs-onecall serve -m $M -p 1234 --kv-cache 8GiB --max-seqs 256 --lora-max-adapters 64)
  for i in $(seq 0 7); do CMD+=(--lora "adapter_$(printf %04d $i)=$A/adapter_$(printf %04d $i)"); done
  MRS_LORA_ONECALL=$v setsid nohup "${CMD[@]}" > logs/onecall_rep_$v.log 2>&1 < /dev/null &
  SP=$!
  for i in $(seq 1 100); do [ "$(curl -s -o /dev/null -w %{http_code} -m 3 localhost:1234/v1/models)" = 200 ] && break; sleep 3; done
  lg() { n=$1; shift; echo "rep$rep gate=$v $n: $($PYL $SC/loadgen.py --engine mistralrs --url http://localhost:1234 --tokenizer $M --requests 150 --warmup 20 --seed 5 --out $OUT/${n}_v${v}_r$rep.json "$@" 2>&1 | tail -1 | $PYL -c "import sys,json;d=json.loads(sys.stdin.read());print(d[\"ok\"],d[\"errors\"],\"tpot_p50\",round(d[\"tpot_ms\"][\"p50\"],1),\"ttft_p95\",round(d[\"ttft_ms\"][\"p95\"]),\"tok/s\",d[\"tok_per_s\"])" 2>&1 | tail -1)" >> $S; }
  lg warm --concurrency 8 --adapters 8 --popularity uniform
  lg a8_c8 --concurrency 8 --adapters 8 --popularity uniform
  lg a0_c32 --concurrency 32 --adapters 0
  lg a1_c32 --concurrency 32 --adapters 1
  lg a8_c32 --concurrency 32 --adapters 8 --popularity uniform
  kill $SP; sleep 6; pkill -9 -f "[m]istralrs-onecall serve"; sleep 4
done; done
echo DONE >> $S
