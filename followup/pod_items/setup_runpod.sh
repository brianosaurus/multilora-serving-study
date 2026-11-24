#!/usr/bin/env bash
# Setup on `runpod` (host 126ab78ffbd7; 2026-11-24). Writes only to NEW directories under /workspace (envs, src, bin, setup, msf_study, uvcache);
# the existing /workspace/study, claude-scratch, kvc-diff and sglmain are not touched. Models and adapters go to /dev/shm (19 GB).
# Mirrors followup/pod_runpod2/SETUP.md; versions that resolve are recorded in /workspace/setup/*.txt.
set -u; W=/workspace; mkdir -p $W/envs $W/src $W/bin $W/setup; L=$W/setup/setup.log; : > $L
say() { echo "$(date -u +%T) $*" >> $L; }
export UV_CACHE_DIR=$W/uvcache
say "host $(hostname); $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
cd $W/envs
(uv venv vllm --python 3.12 && uv pip install --python vllm/bin/python vllm==0.34.0 peft) > $W/setup/vllm_install.log 2>&1 &
(uv venv sglang --python 3.12 && uv pip install --python sglang/bin/python "sglang[all]==0.5.23" peft) > $W/setup/sglang_install.log 2>&1 &
( [ -x /root/.cargo/bin/cargo ] || { curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal > $W/setup/rustup.log 2>&1; } ) &
wait; say "envs and rust installed; $(/root/.cargo/bin/rustc --version 2>&1)"
mkdir -p /dev/shm/models
$W/envs/vllm/bin/python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen2.5-1.5B-Instruct',local_dir='/dev/shm/models/Qwen2.5-1.5B-Instruct')" > $W/setup/model.log 2>&1
say "model sha256 $(sha256sum /dev/shm/models/Qwen2.5-1.5B-Instruct/model.safetensors | cut -c1-64) (study: dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee)"
cd $W/msf_study
$W/envs/vllm/bin/python scripts/make_adapters.py --base /dev/shm/models/Qwen2.5-1.5B-Instruct --out /dev/shm/adapters_128 --count 128 --b-std 0.05 > $W/setup/make_adapters_128.log 2>&1
ln -sfn /dev/shm/adapters_128 /dev/shm/adapters
$W/envs/vllm/bin/python scripts/make_adapters.py --base /dev/shm/models/Qwen2.5-1.5B-Instruct --out /dev/shm/adapters_s0.02_32 --count 32 --b-std 0.02 > $W/setup/make_adapters_0.02_32.log 2>&1
mkdir -p /dev/shm/adapters_s0.02; for i in 0 1 2 3; do ln -sfn /dev/shm/adapters_s0.02_32/adapter_000$i /dev/shm/adapters_s0.02/adapter_000$i; done
$W/envs/vllm/bin/python - >> $L 2>&1 <<PY
import json
a={x["name"]:x["sha256"] for x in json.load(open("$W/msf_study/results/raw/adapters_manifest.json"))["adapters"]}
b={x["name"]:x["sha256"] for x in json.load(open("/dev/shm/adapters_128/manifest.json"))["adapters"]}
print("0.05 adapters identical to the study manifest:",sum(a[k]==b.get(k) for k in a),"of",len(a))
c={x["name"]:x["sha256"] for x in json.load(open("$W/msf_study/followup/routing/results/raw/sweep_s0.02/adapters_s0.02/manifest.json"))["adapters"]}
d={x["name"]:x["sha256"] for x in json.load(open("/dev/shm/adapters_s0.02_32/manifest.json"))["adapters"]}
print("0.02 adapters (seeds 0-3) identical to the routing sweep:",sum(c[k]==d.get(k) for k in c),"of",len(c))
PY
for e in vllm sglang; do echo "== $e" >> $W/setup/versions.txt; uv pip list --python $W/envs/$e/bin/python 2>/dev/null | grep -i -E "^(vllm|sglang|sglang-kernel|torch|peft|transformers|flashinfer-python|triton) " >> $W/setup/versions.txt; done
say "SETUP_PHASE1_DONE"
