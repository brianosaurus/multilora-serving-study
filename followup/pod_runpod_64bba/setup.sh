#!/usr/bin/env bash
# Setup on `runpod` host 64bba14d4768 (new pod) for trigger step 2 (followup/sglang_cliff/PLAN.md, option 13).
# Writes only to new paths: /root/src/sglang-main, /root/envs/sglang_main, /dev/shm/{models,adapters*}, /root/setup, /workspace/uvcache.
set -u; [ "$(hostname)" = 64bba14d4768 ] || { echo "REFUSED: host $(hostname)"; exit 2; }
S=/root/setup; mkdir -p $S /root/src /root/envs; L=$S/setup.log; : > $L
say() { echo "$(date -u +%T) $*" >> $L; }
export UV_CACHE_DIR=/workspace/uvcache
say "host $(hostname); $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader); nvcc: $(ls /usr/local/cuda/bin/nvcc 2>&1)"
[ -d /root/src/sglang-main/.git ] || git clone -q --filter=blob:none https://github.com/sgl-project/sglang /root/src/sglang-main >> $L 2>&1; git -C /root/src/sglang-main checkout -q 0e9cff98f5b7 >> $L 2>&1
say "sglang source at $(git -C /root/src/sglang-main log -1 --format='%H %cs')"
(uv venv /root/envs/sglang_main --python 3.12 && uv pip install --python /root/envs/sglang_main/bin/python -e /root/src/sglang-main/python peft) > $S/sglang_main_install.log 2>&1
say "install exit $?; import: $(/root/envs/sglang_main/bin/python -c 'import sglang;print(sglang.__version__)' 2>&1 | tail -1)"
uv pip list --python /root/envs/sglang_main/bin/python 2>/dev/null | grep -i -E "^(sglang|sglang-kernel|torch|peft|transformers|flashinfer-python|triton|aiohttp|numpy) " > $S/versions.txt
mkdir -p /dev/shm/models
/root/envs/sglang_main/bin/python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen2.5-1.5B-Instruct',local_dir='/dev/shm/models/Qwen2.5-1.5B-Instruct')" > $S/model.log 2>&1
say "model sha256 $(sha256sum /dev/shm/models/Qwen2.5-1.5B-Instruct/model.safetensors | cut -c1-64) (study: dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee)"
cd /root/msf_study && /root/envs/sglang_main/bin/python scripts/make_adapters.py --base /dev/shm/models/Qwen2.5-1.5B-Instruct --out /dev/shm/adapters --count 32 --b-std 0.05 > $S/make_adapters.log 2>&1
/root/envs/sglang_main/bin/python - >> $L 2>&1 <<PY
import json
a={x["name"]:x["sha256"] for x in json.load(open("/root/msf_study/results/raw/adapters_manifest.json"))["adapters"]}
b={x["name"]:x["sha256"] for x in json.load(open("/dev/shm/adapters/manifest.json"))["adapters"]}
print("adapters identical to the study manifest:",sum(a[k]==b.get(k) for k in a if k in b),"of",len(b),"made;",len(a),"in manifest")
PY
say "SETUP_DONE"
