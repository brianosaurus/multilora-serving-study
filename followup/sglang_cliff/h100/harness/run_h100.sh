#!/usr/bin/env bash
# Phase 3 runs for the H100 check (plan: ../h100_amendment.md). Usage: run_h100.sh PART TP
#   PART: lat (stock/patched, 3 starts, 20 and 40 rps), sat (closed loop c1024 per build),
#         vllm (3 starts, 20 and 40 rps), gsm (GSM8K; builds in $GSM_BUILDS, default "stock patched")
set -uo pipefail
[ "$(hostname)" = f70961d23335 ] || { echo "REFUSED: host $(hostname)"; exit 9; }
PART=${1:?part}; TP=${2:?tp}
# The pod's default open-file limit (1024) made the concurrency-1024 client fail to connect
# (ClientConnectorError); raise it for this shell and the processes it starts.
ulimit -n 65536
# tp=4 dropped at the author's request on 2026-11-27 (1.5B model; little value at tp=4).
[ "$TP" = 4 ] && { echo "$(date -u +%FT%T) SKIPPED $PART tp=4 (dropped by the author)" | tee -a /workspace/h100_run/run.log; exit 0; }
export HF_HOME=/workspace/hf
R=/workspace/h100_run; D=$R/results; mkdir -p $D; L=$R/run.log
M=/workspace/hf/models/Qwen2.5-1.5B-Instruct; A=/workspace/hf/adapters; H=$R/harness
# Environments run from local-disk copies of /workspace/envs (see to_local.sh; /dev/shm is noexec): imports over the
# network filesystem took over 15 minutes per server start.
ENVS=/opt/fast/envs
PYL=$ENVS/vllm/bin/python
GPUS=$(seq -s, 0 $((TP - 1)))
say() { echo "$(date -u +%FT%T) $*" | tee -a $L; }
SP=""; MON=""

gpus_free() { [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ]; }

monitors() { # dir
  # Container CPU (cgroup usage_usec against the 88.4-CPU quota), one sample per second.
  ( while :; do echo "$(date +%s.%N) $(awk '/^usage_usec/{print $2}' /sys/fs/cgroup/cpu.stat)"; sleep 1; done ) \
    > $1/cpu_usage.txt 2>&1 & MON="$!"
  nvidia-smi dmon -s um -d 1 > $1/dmon.txt 2>&1 & MON="$MON $!"
}

stop_all() {
  [ -n "$SP" ] && kill $SP 2>/dev/null; sleep 8
  # Kill the server's process group by id; a pattern kill can match unrelated shells.
  [ -n "$SP" ] && kill -9 -- -$SP 2>/dev/null; sleep 5
  [ -n "$MON" ] && kill $MON 2>/dev/null; MON=""; SP=""
  for i in $(seq 1 30); do gpus_free && return 0; sleep 2; done
  say "WARNING: GPUs not free after stop"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv | tee -a $L; return 1
}

wait_ready() { # port
  for i in $(seq 1 600); do
    [ "$(curl -s -o /dev/null -w %{http_code} -m 3 localhost:$1/health)" = 200 ] && return 0
    kill -0 $SP 2>/dev/null || return 1; sleep 3
  done; return 1
}

start_sglang() { # tag build
  local O=$D/$1; mkdir -p $O; local E=$ENVS/sglang-$2
  gpus_free || { say "REFUSED $1: GPUs busy"; return 1; }
  monitors $O
  local CMD=($E/bin/python -m sglang.launch_server --model-path $M --served-model-name qwen --port 30000
    --tp $TP --mem-fraction-static 0.85 --context-length 4096 --enable-lora --max-lora-rank 16
    --max-loras-per-batch 16 --lora-target-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj --lora-paths)
  for i in $(seq 0 31); do CMD+=("adapter_$(printf %04d $i)=$A/adapter_$(printf %04d $i)"); done
  say "start $1 ($2, tp=$TP, GPUs $GPUS) load $(cut -d' ' -f1-3 /proc/loadavg)"
  setsid nohup env CUDA_VISIBLE_DEVICES=$GPUS PATH=$E/bin:$PATH "${CMD[@]}" > $O/server.log 2>&1 < /dev/null & SP=$!
  wait_ready 30000 || { say "$1 NOT READY"; tail -20 $O/server.log >> $L; return 1; }
  say "$1 ready | $(grep -oE 'max_running_requests defaults to [0-9]+ \(was [0-9]+\)' $O/server.log | head -1) | $(grep -oE 'max_running_requests=[0-9]+' $O/server.log | sort -u | tr '\n' ' ')| KV $(grep -oE '#tokens: [0-9]+' $O/server.log | head -1) | decode graph $(grep -oE 'decode CUDA graph end\. bs=\[[0-9, ]*\]' $O/server.log | head -1 | grep -oE '[0-9]+\]$') max, $(grep -oE 'decode CUDA graph end.*mem usage=[0-9.]+ GB' $O/server.log | grep -oE 'mem usage=[0-9.]+ GB' | head -1)"
}

start_vllm() { # tag
  local O=$D/$1; mkdir -p $O; local E=$ENVS/vllm
  gpus_free || { say "REFUSED $1: GPUs busy"; return 1; }
  monitors $O
  local CMD=($E/bin/vllm serve $M --served-model-name qwen --port 8000 --tensor-parallel-size $TP
    --max-model-len 4096 --gpu-memory-utilization 0.85 --enable-lora --max-loras 16 --max-cpu-loras 64
    --max-lora-rank 16 --lora-modules)
  for i in $(seq 0 31); do CMD+=("adapter_$(printf %04d $i)=$A/adapter_$(printf %04d $i)"); done
  say "start $1 (vllm, tp=$TP) load $(cut -d' ' -f1-3 /proc/loadavg)"
  setsid nohup env CUDA_VISIBLE_DEVICES=$GPUS PATH=$E/bin:$PATH "${CMD[@]}" > $O/server.log 2>&1 < /dev/null & SP=$!
  wait_ready 8000 || { say "$1 NOT READY"; tail -20 $O/server.log >> $L; return 1; }
  say "$1 ready | $(grep -oE 'max_num_seqs=[0-9]+|GPU KV cache size: [0-9,]+ tokens' $O/server.log | sort -u | tr '\n' ' ')"
}

warm() { # engine url tag
  for s in 998 997; do
    $PYL $H/loadgen.py --engine $1 --url $2 --tokenizer $M --rate 40 --requests 300 --warmup 0 \
      --adapters 0 --seed $s --out $D/$3/warm_$s.json > /dev/null 2>&1
  done
}

load() { # tag engine url rate
  $PYL $H/loadgen.py --engine $2 --url $3 --tokenizer $M --rate $4 --requests 500 --warmup 50 \
    --adapters 32 --popularity zipf --seed 1 --out $D/$1/zipf32_r$4.json | cut -c1-400 | tee -a $L
}

saturate() { # tag engine url
  $PYL $H/loadgen.py --engine $2 --url $3 --tokenizer $M --concurrency 1024 --requests 3000 --warmup 300 \
    --adapters 32 --popularity zipf --seed 1 --out $D/$1/sat_c1024.json | cut -c1-400 | tee -a $L
}

case $PART in
lat)
  for rep in 1 2 3; do for b in stock patched; do tag=tp${TP}_${b}_lat$rep
    start_sglang $tag $b && { warm sglang http://localhost:30000 $tag; for r in 20 40; do load $tag sglang http://localhost:30000 $r; done; }
    stop_all; say "done $tag"; done; done ;;
sat)
  for b in stock patched; do tag=tp${TP}_${b}_sat
    start_sglang $tag $b && { warm sglang http://localhost:30000 $tag; saturate $tag sglang http://localhost:30000; }
    stop_all; say "done $tag"; done ;;
vllm)
  for rep in 1 2 3; do tag=tp${TP}_vllm_lat$rep
    start_vllm $tag && { warm vllm http://localhost:8000 $tag; for r in 20 40; do load $tag vllm http://localhost:8000 $r; done; }
    stop_all; say "done $tag"; done ;;
vllmsat)
  # Added after the tp=1 results, at the author's request (not pre-registered).
  tag=tp${TP}_vllm_sat
  start_vllm $tag && { warm vllm http://localhost:8000 $tag; saturate $tag vllm http://localhost:8000; }
  stop_all; say "done $tag" ;;
gsm)
  for b in ${GSM_BUILDS:-stock patched}; do tag=tp${TP}_${b}_gsm8k
    start_sglang $tag $b && /workspace/envs/sgleval/bin/sgl-eval run gsm8k --base-url http://127.0.0.1:30000/v1 \
      --model qwen --temperature 0 --max-tokens 512 --seed 0 --num-threads 128 --out-dir $D/$tag/sgleval > $D/$tag/sgleval.log 2>&1
    stop_all; say "done $tag $(grep -h '"score"' $D/$tag/sgleval/*/metrics.json 2>/dev/null | head -1)"; done ;;
*) echo "unknown part $PART"; exit 2 ;;
esac
say "DONE_${PART}_tp$TP"
