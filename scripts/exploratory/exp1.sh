#!/usr/bin/env bash
# Exploratory (not pre-registered): 128 adapters on 32 slots, vLLM and SGLang, Zipf and uniform popularity
cd /root/study
M=/dev/shm/models/Qwen2.5-1.5B-Instruct; A=/dev/shm/adapters; PYL=/root/study/venvs/vllm/bin/python
OUT=/root/study/results/exp1; mkdir -p $OUT; export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
S=logs/exp1.summary; : > $S
W=results/tuning/winners.json
tr() { python3 -c "import json;print(json.load(open(\"$W\"))[\"$1\"][\"winner\"])"; }
names() { for i in $(seq 0 127); do printf "adapter_%04d " $i; done; }
stop() { kill $SP 2>/dev/null; sleep 6; pkill -9 -f "[v]llm serve"; pkill -9 -f "[l]aunch_server"; sleep 5; }
wait_up() { for i in $(seq 1 200); do [ "$(curl -s -o /dev/null -w %{http_code} -m 3 localhost:$1/health)" = 200 ] && break; sleep 3; done; }
for e in vllm sglang; do
  T=$(tr $e)
  if [ $e = vllm ]; then P=8000
    CMD=(/root/study/venvs/vllm/bin/vllm serve $M --served-model-name qwen --port 8000 --max-model-len 4096 --gpu-memory-utilization 0.85)
    [ "$T" = T2 ] && CMD+=(--max-num-seqs 256)
    CMD+=(--enable-lora --max-loras 32 --max-cpu-loras 128 --max-lora-rank 16 --lora-modules)
    for n in $(names); do CMD+=("$n=$A/$n"); done
    BIN=/root/study/venvs/vllm/bin
  else P=30000
    CMD=(/root/study/venvs/sglang/bin/python -m sglang.launch_server --model-path $M --served-model-name qwen --port 30000 --mem-fraction-static 0.85 --context-length 4096)
    [ "$T" = T2 ] && CMD+=(--max-running-requests 256)
    CMD+=(--enable-lora --max-lora-rank 16 --max-loras-per-batch 32 --lora-target-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj --lora-paths)
    for n in $(names); do CMD+=("$n=$A/$n"); done
    BIN=/root/study/venvs/sglang/bin
  fi
  echo "$e start $(date -u +%T) trial=$T" >> $S
  PATH=$BIN:$PATH setsid nohup "${CMD[@]}" > logs/exp1_$e.log 2>&1 < /dev/null &
  SP=$!
  wait_up $P
  for s in 998 997; do $PYL scripts/loadgen.py --engine $e --url http://localhost:$P --tokenizer $M --rate 40 --requests 300 --warmup 0 --adapters 0 --seed $s --out $OUT/warm_${e}_$s.json >/dev/null 2>&1; done
  for w in zipf128 uni128; do
    mkdir -p $OUT/$e
    $PYL scripts/run_ladder.py --engine $e --url http://localhost:$P --tokenizer $M --workload $w --rates 5,10,20,40,80 --repeats 3 --requests 500 --warmup 50 --outdir $OUT/$e 2>&1 | cut -c1-230 >> $S
  done
  stop
  echo "$e done $(date -u +%T)" >> $S
done
echo DONE >> $S
