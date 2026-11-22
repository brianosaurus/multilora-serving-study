#!/usr/bin/env bash
# Main measurements (Day plan in docs/PLAN.md). Per engine, two server sessions:
#   B0  : LoRA disabled                        -> workload b0
#   LORA: LoRA enabled, 32 adapters registered -> workloads b1, one, zipf32, uni32
# Each session starts and ends with a reference run (base-model requests) to detect drift.
# Each engine uses its winning tuning trial from results/tuning/winners.json.
# Environment overrides: ENGINES (run order), RATES, REPEATS, NREQ, WARM, OUT, WORKLOADS.
set -u
PYL=/root/study/venvs/vllm/bin/python
M=/dev/shm/models/Qwen2.5-1.5B-Instruct
A=/dev/shm/adapters
OUT=${OUT:-/root/study/results/main}   # local disk: the network volume lost data during the first run
WIN=${WIN:-/root/study/results/tuning/winners.json}
LOGS=/root/study/logs
SC=/root/study/scripts
ENGINES=${ENGINES:-"vllm sglang mistralrs"}
RATES=${RATES:-"5,10,20,40,80,160,320"}
REPEATS=${REPEATS:-3}
NREQ=${NREQ:-500}
WARM=${WARM:-50}
LORA_WORKLOADS=${LORA_WORKLOADS:-"b1 one zipf32 uni32 pfx_base pfx32"}
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
mkdir -p "$OUT"

hyg() { nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw,utilization.gpu,memory.used,clocks_throttle_reasons.active --format=csv,noheader | tr -s ' '; }
names() { for i in $(seq 0 31); do printf "adapter_%04d " "$i"; done; }
winner() { python3 -c "import json;print(json.load(open('$WIN'))['$1']['winner'])" 2>/dev/null || echo T1; }

launch_cmd() { # engine trial lora(0|1) -> CMD PORT HEALTH EPATH
  eng=$1; trial=$2; lora=$3; CMD=(); EPATH=""
  case $eng in
    vllm)
      PORT=8000; HEALTH=/health; EPATH=/root/study/venvs/vllm/bin
      CMD=(/root/study/venvs/vllm/bin/vllm serve $M --served-model-name qwen --port 8000 --max-model-len 4096 --gpu-memory-utilization 0.85)
      [ "$trial" = T2 ] && CMD+=(--max-num-seqs 256)
      [ "$trial" = T3 ] && CMD+=(--max-num-batched-tokens 2048)
      if [ "$lora" = 1 ]; then
        CMD+=(--enable-lora --max-loras ${VLLM_MAX_LORAS:-16} --max-cpu-loras 64 --max-lora-rank 16 --lora-modules)
        for n in $(names); do CMD+=("$n=$A/$n"); done
      fi ;;
    sglang)
      PORT=30000; HEALTH=/health; EPATH=/root/study/venvs/sglang/bin
      CMD=(/root/study/venvs/sglang/bin/python -m sglang.launch_server --model-path $M --served-model-name qwen
           --port 30000 --mem-fraction-static 0.85 --context-length 4096)
      [ "$trial" = T2 ] && CMD+=(--max-running-requests 256)
      [ "$trial" = T3 ] && CMD+=(--chunked-prefill-size 2048)
      if [ "$lora" = 1 ]; then
        CMD+=(--enable-lora --max-lora-rank 16 --max-loras-per-batch 16
              --lora-target-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj --lora-paths)
        for n in $(names); do CMD+=("$n=$A/$n"); done
      fi ;;
    mistralrs)
      PORT=1234; HEALTH=/v1/models
      CMD=(/root/study/bin/mistralrs serve -m $M -p 1234 --kv-cache 8GiB)
      [ "$trial" = T2 ] && CMD+=(--max-seqs 256)
      [ "$trial" = T3 ] && CMD+=(--max-num-batched-tokens 2048)
      if [ "$lora" = 1 ]; then
        CMD+=(--lora-max-adapters 64); for n in $(names); do CMD+=(--lora "$n=$A/$n"); done
      fi ;;
  esac
}

stop_engine() { kill "$1" 2>/dev/null; sleep 6; pkill -9 -f "[v]llm serve|[l]aunch_server|[r]elease/mistralrs" 2>/dev/null; sleep 3; }

ref_run() { # engine tag
  $PYL $SC/loadgen.py --engine $1 --url http://localhost:$PORT --tokenizer $M --rate 20 --requests 200 --warmup 50 \
    --adapters 0 --seed 999 --out "$OUT/$1/ref_${SESSION}_$2.json" 2>&1 | tail -1 | cut -c1-200
}

drift() { # engine
  python3 - "$OUT/$1/ref_${SESSION}_start.json" "$OUT/$1/ref_${SESSION}_end.json" <<'PY'
import json, sys
a, b = (json.load(open(p))["summary"] for p in sys.argv[1:3])
x, y = a["e2e_ms"]["p50"], b["e2e_ms"]["p50"]
d = (y - x) / x * 100
print(f"  reference drift: median e2e {x:.1f} -> {y:.1f} ms ({d:+.1f}%) {'DISCARD SESSION' if abs(d) > 5 else 'ok'}")
PY
}

run_session() { # engine session lora workloads...
  eng=$1; SESSION=$2; lora=$3; shift 3
  trial=$(winner $eng)
  launch_cmd $eng $trial $lora
  mkdir -p "$OUT/$eng"
  echo "=== $eng $SESSION (trial $trial, lora=$lora) start $(date -u +%T) | GPU: $(hyg)"
  PATH=$EPATH:$PATH nohup setsid "${CMD[@]}" > "$LOGS/main_${eng}_$SESSION.log" 2>&1 < /dev/null &
  pid=$!; t0=$(date +%s); ready=0
  for i in $(seq 1 200); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' -m 3 "http://localhost:$PORT$HEALTH")" = 200 ] && { ready=1; break; }
    kill -0 $pid 2>/dev/null || break
    sleep 3
  done
  if [ $ready = 1 ]; then
    echo "  ready after $(( $(date +%s) - t0 )) s"
    # warm-up pass (not recorded in results): avoids penalising the first rate after start
    $PYL $SC/loadgen.py --engine $eng --url http://localhost:$PORT --tokenizer $M --rate 10 --requests 150 \
      --warmup 0 --adapters 0 --seed 998 --out "$OUT/$eng/warmup_${SESSION}.json" > /dev/null 2>&1
    # second warm-up at a higher rate so large batches are exercised before measuring
    $PYL $SC/loadgen.py --engine $eng --url http://localhost:$PORT --tokenizer $M --rate 40 --requests 400 \
      --warmup 0 --adapters 0 --seed 997 --out "$OUT/$eng/warmup2_${SESSION}.json" > /dev/null 2>&1
    echo "  reference start: $(ref_run $eng start)"
    for w in "$@"; do
      echo "  workload $w $(date -u +%T) | GPU: $(hyg)"
      $PYL $SC/run_ladder.py --engine $eng --url http://localhost:$PORT --tokenizer $M --workload $w \
        --rates $RATES --repeats $REPEATS --requests $NREQ --warmup $WARM --outdir "$OUT/$eng" 2>&1 | cut -c1-260
    done
    echo "  reference end:   $(ref_run $eng end)"
    drift $eng
  else
    echo "  NOT READY; last log lines:"; tail -6 "$LOGS/main_${eng}_$SESSION.log" | cut -c1-200
  fi
  stop_engine $pid
  echo "  stopped; GPU used: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
}

for e in $ENGINES; do
  [ "${SKIP_B0:-0}" = 1 ] || run_session $e b0 0 b0
  run_session $e lora 1 $LORA_WORKLOADS
done
echo MAIN_DONE
