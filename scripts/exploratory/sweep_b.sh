#!/usr/bin/env bash
cd /root/study
for m in 32 8; do
  echo "sweep max_loras=$m start $(date -u +%T)" >> logs/sweep.log
  ENGINES=vllm SKIP_B0=1 LORA_WORKLOADS="uni32 zipf32" VLLM_MAX_LORAS=$m OUT=/root/study/results/sweep_m$m bash scripts/run_main.sh > logs/sweep_m$m.log 2>&1
  echo "sweep max_loras=$m done $(date -u +%T)" >> logs/sweep.log
done
echo SWEEP_DONE >> logs/sweep.log
