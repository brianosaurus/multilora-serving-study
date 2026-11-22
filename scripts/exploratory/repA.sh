#!/usr/bin/env bash
# Replication A: SGLang csgmv vs triton, closed loop, 32 adapters Zipf, 16 slots, 3 repeats
cd /root/study
M=/dev/shm/models/Qwen2.5-1.5B-Instruct; A=/dev/shm/adapters; PYL=/root/study/venvs/vllm/bin/python
OUT=/root/study/results/repA; mkdir -p $OUT
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
TRIAL=$(python3 -c "import json;print(json.load(open(\"/root/study/results/tuning/winners.json\"))[\"sglang\"][\"winner\"])")
for be in triton csgmv; do
  CMD=(/root/study/venvs/sglang/bin/python -m sglang.launch_server --model-path $M --served-model-name qwen --port 30000 --mem-fraction-static 0.85 --context-length 4096)
  [ "$TRIAL" = T2 ] && CMD+=(--max-running-requests 256)
  [ "$TRIAL" = T3 ] && CMD+=(--chunked-prefill-size 2048)
  CMD+=(--enable-lora --max-lora-rank 16 --max-loras-per-batch 16 --lora-backend $be --lora-target-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj --lora-paths)
  for i in $(seq 0 31); do CMD+=("adapter_$(printf %04d $i)=$A/adapter_$(printf %04d $i)"); done
  echo "repA $be start $(date -u +%T) trial=$TRIAL" >> logs/repA.log
  PATH=/root/study/venvs/sglang/bin:$PATH nohup setsid "${CMD[@]}" > logs/repA_$be.log 2>&1 < /dev/null &
  pid=$!
  for i in $(seq 1 200); do [ "$(curl -s -o /dev/null -w %{http_code} -m 3 localhost:30000/health)" = 200 ] && break; kill -0 $pid 2>/dev/null || break; sleep 3; done
  for s in 998 997; do $PYL scripts/loadgen.py --engine sglang --url http://localhost:30000 --tokenizer $M --rate 40 --requests 300 --warmup 0 --adapters 0 --seed $s --out $OUT/warm_${be}_$s.json >/dev/null 2>&1; done
  $PYL scripts/loadgen.py --engine sglang --url http://localhost:30000 --tokenizer $M --rate 20 --requests 200 --warmup 50 --adapters 0 --seed 999 --out $OUT/ref_${be}_start.json | tail -1 | cut -c1-160 >> logs/repA.log
  for c in 8 32 128; do for r in 0 1 2; do
    $PYL scripts/loadgen.py --engine sglang --url http://localhost:30000 --tokenizer $M --concurrency $c --requests 500 --warmup 50 --adapters 32 --popularity zipf --seed $((100+r)) --out $OUT/sglang_${be}_c${c}_rep$r.json | tail -1 | cut -c1-200 >> logs/repA.log
  done; done
  $PYL scripts/loadgen.py --engine sglang --url http://localhost:30000 --tokenizer $M --rate 20 --requests 200 --warmup 50 --adapters 0 --seed 999 --out $OUT/ref_${be}_end.json | tail -1 | cut -c1-160 >> logs/repA.log
  kill $pid; sleep 6; pkill -9 -f "[l]aunch_server"; sleep 4
  echo "repA $be done $(date -u +%T)" >> logs/repA.log
done
echo REPA_DONE >> logs/repA.log
