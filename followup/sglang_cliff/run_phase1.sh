#!/usr/bin/env bash
# Phase 1 runner (plan: followup/sglang_cliff/PLAN.md). Host must equal HOST_EXPECT (default runpod 126ab78ffbd7; runpod2 run: HOST_EXPECT=48bd787e389e PYL=/root/envs/sglang/bin/python).
# Usage: run_phase1.sh [--wait]   (--wait: wait for the GPU and other study jobs instead of refusing)
R=/root/msf_study; D=$R/followup/sglang_cliff/results/phase1; mkdir -p $D; L=$D/run.log
say() { echo "$(date -u +%T) $*" | tee -a $L; }
HOST_EXPECT=${HOST_EXPECT:-126ab78ffbd7}
[ "$(hostname)" = "$HOST_EXPECT" ] || { echo "REFUSED: host $(hostname) is not $HOST_EXPECT"; exit 2; }
M=/dev/shm/models/Qwen2.5-1.5B-Instruct; A=/dev/shm/adapters; PYL=${PYL:-/root/envs/vllm/bin/python}
busy() { [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" != 0 ] || pgrep -f "run_rerun.sh|run_session.sh|run_item|run_cold_pass.sh|run_slots_memory" >/dev/null; }
# adapter checksums against the manifest
$PYL - <<'P' || { say "REFUSED: adapter checksum mismatch"; exit 3; }
import json,hashlib,sys
m=json.load(open("/root/msf_study/results/raw/adapters_manifest.json"))["adapters"][:32]
for a in m:
    h=hashlib.sha256(open(f"/dev/shm/adapters/{a['name']}/adapter_model.safetensors","rb").read()).hexdigest()
    if h!=a["sha256"]: print("MISMATCH",a["name"]); sys.exit(1)
print("adapters ok")
P
declare -A ENVOF=( [v23]=/root/envs/sglang [main]=/root/envs/sglang_main )
declare -A FLAG=( [def]="" [d1]="--lora-drain-wait-threshold 1.0" [d02]="--lora-drain-wait-threshold 0.2" )
ORDERS=( "v23-def main-def v23-d1 main-d1 v23-d02 main-d02" "main-d02 v23-d02 main-d1 v23-d1 main-def v23-def" "v23-d1 main-d02 main-def v23-d02 main-d1 v23-def" )
for r in 1 2 3; do
  for cfg in ${ORDERS[$((r-1))]}; do
    eng=${cfg%%-*}; fl=${cfg#*-}; E=${ENVOF[$eng]}; O=$D/r$r/$cfg; mkdir -p $O
    if busy; then
      if [ "$1" = --wait ]; then until ! busy; do sleep 30; done; sleep 20; busy && { say "REFUSED $r $cfg: busy again"; continue; }
      else say "REFUSED $r $cfg: GPU or another study job busy"; continue; fi
    fi
    say "start r$r $cfg host $(hostname) load $(cut -d' ' -f1-3 /proc/loadavg) external CPU: $($PYL $R/followup/kernel_rerun/extload.py check 10 2>&1 | tail -1)"
    CMD=($E/bin/python -m sglang.launch_server --model-path $M --served-model-name qwen --port 30000 --mem-fraction-static 0.85 --context-length 4096 --enable-lora --max-lora-rank 16 --max-loras-per-batch 16 --lora-target-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj ${FLAG[$fl]} --lora-paths)
    for i in $(seq 0 31); do CMD+=("adapter_$(printf %04d $i)=$A/adapter_$(printf %04d $i)"); done
    setsid nohup env PATH=$E/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda "${CMD[@]}" > $O/server.log 2>&1 < /dev/null & SP=$!
    ok=0; for i in $(seq 1 600); do [ "$(curl -s -o /dev/null -w %{http_code} -m 3 localhost:30000/health)" = 200 ] && { ok=1; break; }; kill -0 $SP 2>/dev/null || break; sleep 3; done
    say "r$r $cfg ready=$ok"
    if [ $ok = 1 ]; then
      for s in 998 997; do $PYL $R/scripts/loadgen.py --engine sglang --url http://localhost:30000 --tokenizer $M --rate 40 --requests 300 --warmup 0 --adapters 0 --seed $s --out $O/warm_$s.json >/dev/null 2>&1; done
      for rate in 20 40; do
        $PYL $R/scripts/loadgen.py --engine sglang --url http://localhost:30000 --tokenizer $M --rate $rate --requests 500 --warmup 50 --adapters 32 --adapter-offset 0 --popularity zipf --seed $r --out $O/zipf32_r${rate}.json | cut -c1-300 | tee -a $L
      done
      grep -m1 -o "lora_drain_wait_threshold=[0-9.]*" $O/server.log > $O/effective_args.txt 2>/dev/null
    fi
    kill $SP 2>/dev/null; sleep 8; pkill -9 -f "[s]glang.launch_server|[s]glang::" 2>/dev/null; sleep 5
    say "done r$r $cfg"
  done
done
echo DONE >> $L
