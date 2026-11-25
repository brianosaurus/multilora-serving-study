#!/usr/bin/env bash
# Phase 2 runner (plan: followup/sglang_cliff/PLAN.md, "Phase 2"). Usage: HOST_EXPECT=48bd787e389e ./run_phase2.sh A|B|C
R=/root/msf_study; D=$R/followup/sglang_cliff/results/phase2; mkdir -p $D; L=$D/run.log; PART=${1:?part A|B|C}
say() { echo "$(date -u +%T) $*" | tee -a $L; }
HOST_EXPECT=${HOST_EXPECT:-48bd787e389e}
[ "$(hostname)" = "$HOST_EXPECT" ] || { echo "REFUSED: host $(hostname) is not $HOST_EXPECT"; exit 2; }
M=/dev/shm/models/Qwen2.5-1.5B-Instruct; A=/dev/shm/adapters; PYL=/root/envs/sglang/bin/python; TR=$R/followup/sglang_cliff/trace
busy() { [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" != 0 ] || pgrep -f "[r]un_rerun.sh|[r]un_session.sh|[r]un_phase1.sh" >/dev/null; }
declare -A ENVOF=( [v23]=/root/envs/sglang [main]=/root/envs/sglang_main )
SP=""
start_sglang() { # tag eng slots trace(0|1)
  local tag=$1 eng=$2 slots=$3 trace=$4 O=$D/$1; mkdir -p $O; E=${ENVOF[$eng]}
  busy && { say "REFUSED $tag: GPU or another job busy"; return 1; }
  say "start $tag host $(hostname) load $(cut -d' ' -f1-3 /proc/loadavg) external CPU: $($PYL $R/followup/kernel_rerun/extload.py check 10 2>&1 | tail -1)"
  CMD=($E/bin/python -m sglang.launch_server --model-path $M --served-model-name qwen --port 30000 --mem-fraction-static 0.85 --context-length 4096 --enable-lora --max-lora-rank 16 --max-loras-per-batch $slots --lora-target-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj --lora-paths)
  for i in $(seq 0 31); do CMD+=("adapter_$(printf %04d $i)=$A/adapter_$(printf %04d $i)"); done
  if [ $trace = 1 ]; then PP="PYTHONPATH=$TR SGLANG_CLIFF_TRACE_DIR=$O/trace"; else PP="SGLANG_CLIFF_NOTRACE=1"; fi
  setsid nohup env PATH=$E/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda $PP "${CMD[@]}" > $O/server.log 2>&1 < /dev/null & SP=$!
  ok=0; for i in $(seq 1 600); do [ "$(curl -s -o /dev/null -w %{http_code} -m 3 localhost:30000/health)" = 200 ] && { ok=1; break; }; kill -0 $SP 2>/dev/null || break; sleep 3; done
  say "$tag ready=$ok trace=$trace"; [ $ok = 1 ]
}
stop_server() { kill $SP 2>/dev/null; sleep 8; pkill -9 -f "[s]glang.launch_server|[s]glang::|[v]llm serve|[V]LLM::EngineCore" 2>/dev/null; sleep 5; }
load() { # tag engine url rate seed
  local O=$D/$1; echo "{\"rate\":$4,\"t_start\":$(date +%s.%N)," >> $O/marks.tmp
  $PYL $R/scripts/loadgen.py --engine $2 --url $3 --tokenizer $M --rate $4 --requests 500 --warmup 50 --adapters 32 --adapter-offset 0 --popularity zipf --seed $5 --out $O/zipf32_r$4.json | cut -c1-300 | tee -a $L
  echo "\"t_end\":$(date +%s.%N)}" >> $O/marks.tmp; tr -d '\n' < $O/marks.tmp >> $O/marks.jsonl; echo >> $O/marks.jsonl; rm -f $O/marks.tmp
}
warm() { for s in 998 997; do $PYL $R/scripts/loadgen.py --engine $1 --url $2 --tokenizer $M --rate 40 --requests 300 --warmup 0 --adapters 0 --seed $s --out $D/$3/warm_$s.json >/dev/null 2>&1; done; }
case $PART in
A) # tracing overhead: main, 16 slots, 20 rps; U P P U U P
  n=0; for v in U P P U U P; do n=$((n+1)); tag=A_${n}_$v; tr=0; [ $v = P ] && tr=1
    start_sglang $tag main 16 $tr && { warm sglang http://localhost:30000 $tag; load $tag sglang http://localhost:30000 20 1; [ $tr = 1 ] && say "$tag trace lines: $(cat $D/$tag/trace/*.jsonl 2>/dev/null | wc -l)"; }
    stop_server; say "done $tag"; done;;
B) for cfg in v23-16 main-16 main-32 v23-32; do eng=${cfg%%-*}; sl=${cfg#*-}; tag=B_$cfg
    start_sglang $tag $eng $sl 1 && { warm sglang http://localhost:30000 $tag; for rate in 20 40; do load $tag sglang http://localhost:30000 $rate 1; done; say "$tag trace lines: $(cat $D/$tag/trace/*.jsonl | wc -l)"; }
    stop_server; say "done $tag"; done;;
C) tag=C_vllm-16; O=$D/$tag; mkdir -p $O; E=/root/envs/vllm
  busy && { say "REFUSED $tag: busy"; exit 1; }
  say "start $tag host $(hostname) load $(cut -d' ' -f1-3 /proc/loadavg) external CPU: $($PYL $R/followup/kernel_rerun/extload.py check 10 2>&1 | tail -1)"
  CMD=($E/bin/vllm serve $M --served-model-name qwen --port 8000 --max-model-len 4096 --gpu-memory-utilization 0.85 --enable-lora --max-loras 16 --max-cpu-loras 64 --max-lora-rank 16 --lora-modules)
  for i in $(seq 0 31); do CMD+=("adapter_$(printf %04d $i)=$A/adapter_$(printf %04d $i)"); done
  setsid nohup env PATH=$E/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda "${CMD[@]}" > $O/server.log 2>&1 < /dev/null & SP=$!
  ok=0; for i in $(seq 1 900); do [ "$(curl -s -o /dev/null -w %{http_code} -m 3 localhost:8000/health)" = 200 ] && { ok=1; break; }; kill -0 $SP 2>/dev/null || break; sleep 3; done
  say "$tag ready=$ok"
  if [ $ok = 1 ]; then warm vllm http://localhost:8000 $tag
    for rate in 20 40; do curl -s localhost:8000/metrics > $O/metrics_before_r$rate.txt; load $tag vllm http://localhost:8000 $rate 1; curl -s localhost:8000/metrics > $O/metrics_after_r$rate.txt; done; fi
  stop_server; say "done $tag";;
esac
echo "DONE_$PART" >> $L
