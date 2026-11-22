#!/usr/bin/env bash
# Exploratory (not pre-registered): SGLang LoRA scheduling flags under Zipf-32 open-loop load, 16 slots
cd /root/study
while ! grep -q DONE logs/b6_scale.summary 2>/dev/null; do sleep 10; done
M=/dev/shm/models/Qwen2.5-1.5B-Instruct; A=/dev/shm/adapters; PYL=/root/study/venvs/vllm/bin/python
OUT=/root/study/results/sgl_flags; mkdir -p $OUT; export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
S=logs/sgl_flags.summary; true
TRIAL=$(python3 -c "import json;print(json.load(open(\"/root/study/results/tuning/winners.json\"))[\"sglang\"][\"winner\"])")
run() { name=$1; shift
  CMD=(/root/study/venvs/sglang/bin/python -m sglang.launch_server --model-path $M --served-model-name qwen --port 30000 --mem-fraction-static 0.85 --context-length 4096)
  [ "$TRIAL" = T2 ] && CMD+=(--max-running-requests 256)
  CMD+=(--enable-lora --max-lora-rank 16 --max-loras-per-batch 16 --lora-target-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj "$@" --lora-paths)
  for i in $(seq 0 31); do CMD+=("adapter_$(printf %04d $i)=$A/adapter_$(printf %04d $i)"); done
  echo "$name start $(date -u +%T) extra: $*" >> $S
  PATH=/root/study/venvs/sglang/bin:$PATH setsid nohup "${CMD[@]}" > logs/sgl_flags_$name.log 2>&1 < /dev/null &
  SP=$!
  for i in $(seq 1 200); do [ "$(curl -s -o /dev/null -w %{http_code} -m 3 localhost:30000/health)" = 200 ] && break; sleep 3; done
  for s in 998 997; do $PYL scripts/loadgen.py --engine sglang --url http://localhost:30000 --tokenizer $M --rate 40 --requests 300 --warmup 0 --adapters 0 --seed $s --out $OUT/warm_${name}_$s.json >/dev/null 2>&1; done
  mkdir -p $OUT/$name
  $PYL scripts/run_ladder.py --engine sglang --url http://localhost:30000 --tokenizer $M --workload zipf32 --rates 20,40,80 --repeats 3 --requests 500 --warmup 50 --outdir $OUT/$name 2>&1 | cut -c1-240 >> $S
  kill $SP; sleep 6; pkill -9 -f "[l]aunch_server"; sleep 4
  echo "$name done $(date -u +%T)" >> $S
}
run overlap2 --enable-lora-overlap-loading --max-loaded-loras 32
echo DONE >> $S
