#!/usr/bin/env bash
# Tuning runs (see "Tuning protocol" in docs/PLAN.md): per engine, trials T1/T2/T3, each at
# the planned rates on the tuning workload (adapters 100-107, Zipf, seed 1000).
# One engine at a time on the GPU. Environment overrides: ENGINES, TRIALS, RATES, NREQ, WARM, OUT.
set -u
PYL=/root/study/venvs/vllm/bin/python
M=/dev/shm/models/Qwen2.5-1.5B-Instruct
A=/dev/shm/adapters
OUT=${OUT:-/workspace/study/results/tuning}
LOGS=/root/study/logs
SC=/root/study/scripts
ENGINES=${ENGINES:-"vllm sglang mistralrs"}
TRIALS=${TRIALS:-"T1 T2 T3"}
RATES=${RATES:-"10 20 40"}
NREQ=${NREQ:-300}
WARM=${WARM:-50}
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
mkdir -p "$OUT"

hyg() { nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw,utilization.gpu,memory.used,clocks_throttle_reasons.active --format=csv,noheader | tr -s ' '; }
names() { for i in $(seq 100 107); do printf "adapter_%04d " "$i"; done; }

launch_cmd() { # engine trial -> sets CMD, PORT, HEALTH, EPATH
  eng=$1; trial=$2; CMD=(); EPATH=""
  case $eng in
    vllm)
      PORT=8000; HEALTH=/health; EPATH=/root/study/venvs/vllm/bin
      CMD=(/root/study/venvs/vllm/bin/vllm serve $M --served-model-name qwen --port 8000 --max-model-len 4096
           --gpu-memory-utilization 0.85 --enable-lora --max-loras 16 --max-cpu-loras 32 --max-lora-rank 16)
      [ "$trial" = T2 ] && CMD+=(--max-num-seqs 256)
      [ "$trial" = T3 ] && CMD+=(--max-num-batched-tokens 2048)
      CMD+=(--lora-modules); for n in $(names); do CMD+=("$n=$A/$n"); done ;;
    sglang)
      PORT=30000; HEALTH=/health; EPATH=/root/study/venvs/sglang/bin
      CMD=(/root/study/venvs/sglang/bin/python -m sglang.launch_server --model-path $M --served-model-name qwen
           --port 30000 --mem-fraction-static 0.85 --context-length 4096 --enable-lora --max-lora-rank 16
           --max-loras-per-batch 16 --lora-target-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj)
      [ "$trial" = T2 ] && CMD+=(--max-running-requests 256)
      [ "$trial" = T3 ] && CMD+=(--chunked-prefill-size 2048)
      CMD+=(--lora-paths); for n in $(names); do CMD+=("$n=$A/$n"); done ;;
    mistralrs)
      PORT=1234; HEALTH=/v1/models
      CMD=(/root/study/bin/mistralrs serve -m $M -p 1234 --kv-cache 8GiB --lora-max-adapters 64)
      [ "$trial" = T2 ] && CMD+=(--max-seqs 256)
      [ "$trial" = T3 ] && CMD+=(--max-num-batched-tokens 2048)
      for n in $(names); do CMD+=(--lora "$n=$A/$n"); done ;;
  esac
}

stop_engine() {
  kill "$1" 2>/dev/null; sleep 6
  pkill -9 -f "[v]llm serve|[l]aunch_server|[r]elease/mistralrs" 2>/dev/null; sleep 3
}

run_trial() {
  eng=$1; trial=$2
  launch_cmd $eng $trial
  echo "=== $eng $trial start $(date -u +%T) | GPU before: $(hyg)"
  PATH=$EPATH:$PATH nohup setsid "${CMD[@]}" > "$LOGS/tune_${eng}_$trial.log" 2>&1 < /dev/null &
  pid=$!
  t0=$(date +%s); ready=0
  for i in $(seq 1 140); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' -m 3 "http://localhost:$PORT$HEALTH")" = 200 ] && { ready=1; break; }
    kill -0 $pid 2>/dev/null || break
    sleep 3
  done
  if [ $ready = 1 ]; then
    echo "  ready after $(( $(date +%s) - t0 )) s"
    for r in $RATES; do
      echo "  rate $r:"
      $PYL $SC/loadgen.py --engine $eng --url http://localhost:$PORT --tokenizer $M --rate $r \
        --requests $NREQ --warmup $WARM --adapters 8 --adapter-offset 100 --popularity zipf --seed 1000 \
        --out "$OUT/${eng}_${trial}_r$r.json" 2>&1 | tail -1 | cut -c1-300
      echo "    GPU after: $(hyg)"
    done
  else
    echo "  NOT READY; last log lines:"; tail -6 "$LOGS/tune_${eng}_$trial.log" | cut -c1-200
  fi
  stop_engine $pid
  echo "  stopped; GPU used: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
}

for e in $ENGINES; do for t in $TRIALS; do run_trial $e $t; done; done
echo TUNING_DONE
