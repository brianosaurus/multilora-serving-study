#!/usr/bin/env bash
# B6: does cost grow with the number of distinct adapters in a batch, or is it a fixed cost of the LoRA path?
M=/dev/shm/models/Qwen2.5-1.5B-Instruct; A=/dev/shm/adapters; PYL=/root/study/venvs/vllm/bin/python; SC=/root/study/scripts
OUT=/root/study/results/b6_scale; mkdir -p $OUT; export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
S=/root/study/logs/b6_scale.summary; : > $S
start() { # lora(0|1)
  CMD=(/root/study/bin/mistralrs serve -m $M -p 1234 --kv-cache 8GiB --max-seqs 256)
  if [ $1 = 1 ]; then CMD+=(--lora-max-adapters 64); for i in $(seq 0 7); do CMD+=(--lora "adapter_$(printf %04d $i)=$A/adapter_$(printf %04d $i)"); done; fi
  setsid nohup "${CMD[@]}" > /root/study/logs/b6_scale_$1.log 2>&1 < /dev/null &
  SP=$!
  for i in $(seq 1 100); do [ "$(curl -s -o /dev/null -w %{http_code} -m 3 localhost:1234/v1/models)" = 200 ] && break; sleep 3; done
}
lg() { n=$1; shift; echo "$n: $($PYL $SC/loadgen.py --engine mistralrs --url http://localhost:1234 --tokenizer $M --requests 150 --warmup 20 --seed 5 --out $OUT/$n.json "$@" 2>&1 | tail -1 | $PYL -c "import sys,json;d=json.loads(sys.stdin.read());print(d[\"ok\"],d[\"errors\"],\"tpot_p50\",round(d[\"tpot_ms\"][\"p50\"],1),\"ttft_p95\",round(d[\"ttft_ms\"][\"p95\"]),\"tok/s\",d[\"tok_per_s\"])" 2>&1 | tail -1)" >> $S; }
start 0
lg warm --concurrency 8 --adapters 0
for c in 8 32; do lg base_c$c --concurrency $c --adapters 0; done
kill $SP; sleep 6; kill -9 $SP 2>/dev/null; sleep 3
start 1
lg warm1 --concurrency 8 --adapters 1
for c in 8 32; do for n in 0 1 2 4 8; do lg lora_a${n}_c$c --concurrency $c --adapters $n --popularity uniform; done; done
kill $SP; sleep 6; kill -9 $SP 2>/dev/null
echo DONE >> $S
