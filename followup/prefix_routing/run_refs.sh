#!/usr/bin/env bash
# 3a references: fp32 continuations and stand-in power checks on the new prompts (plan: followup/prefix_routing/PLAN.md). Pod only.
cd /workspace/study/followup/prefix_routing; mkdir -p results/raw; L=results/raw/refs_run.log; : > $L
PY=/workspace/envs/vllm/bin/python; S=/workspace/study/followup/routing/scripts
say() { echo "$(date -u +%T) $*" >> $L; }
say "host $(hostname); GPU compute procs: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l); extload $($PY /workspace/study/followup/kernel_rerun/extload.py check 10)"
[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] || { say "REFUSED: GPU not idle"; exit 1; }
for i in 0 1 2 3; do echo "$(sha256sum /dev/shm/adapters_s0.02/adapter_000$i/adapter_model.safetensors | cut -c1-64)"; done > results/raw/adapter_sha256.txt
MODEL_DIR=/dev/shm/models/Qwen2.5-1.5B-Instruct ADAPTER_DIR=/dev/shm/adapters_s0.02 $PY $S/refs.py prompts.json results/raw/refs_prefix.json 8 > results/raw/refs_prefix.log 2>&1
say "references done (exit $?)"
$PY $S/analyze.py results/raw/refs_prefix.json > results/raw/analyze_prefix.txt 2>&1
say "analysis done: $(tail -1 results/raw/analyze_prefix.txt | cut -c1-160)"
echo REFS_DONE >> $L
