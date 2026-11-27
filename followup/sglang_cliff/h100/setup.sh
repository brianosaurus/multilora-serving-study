#!/usr/bin/env bash
# Phase 2 setup for the H100 check (see ../h100_amendment.md). Everything under /workspace.
set -euo pipefail
[ "$(hostname)" = f70961d23335 ] || { echo "REFUSED: host $(hostname)"; exit 9; }
export HF_HOME=/workspace/hf UV_CACHE_DIR=/workspace/uvcache
R=/workspace/h100_run; mkdir -p $R/logs /workspace/envs /workspace/src
log() { echo "$(date -u +%T) $*"; }

# SGLang's source build compiles Rust extensions; keep the toolchain under /workspace.
export RUSTUP_HOME=/workspace/rust/rustup CARGO_HOME=/workspace/rust/cargo
if [ ! -x $CARGO_HOME/bin/cargo ]; then
  log "rust toolchain"
  curl -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path --profile minimal > /dev/null
fi
export PATH=$CARGO_HOME/bin:$PATH
cargo --version

log "clone sglang"
if [ ! -d /workspace/src/sglang/.git ]; then
  git clone -q --filter=blob:none https://github.com/sgl-project/sglang.git /workspace/src/sglang
fi
cd /workspace/src/sglang
git fetch -q origin a9a27e9255863d44e15a2965b9cf550ffcd19e9b
git fetch -q https://github.com/brianosaurus/sglang.git decode-graph-range
git rev-parse FETCH_HEAD
for b in stock:a9a27e9255863d44e15a2965b9cf550ffcd19e9b patched:9009897b217ddd63e08d1a713c7997fc02e41d19; do
  name=${b%%:*}; rev=${b#*:}
  [ -d /workspace/src/sglang-$name ] || git worktree add -q --detach /workspace/src/sglang-$name $rev
  log "sglang-$name at $(git -C /workspace/src/sglang-$name rev-parse --short=10 HEAD)"
done
git -C /workspace/src/sglang diff --stat a9a27e9255863d44e15a2965b9cf550ffcd19e9b 9009897b217ddd63e08d1a713c7997fc02e41d19

for name in stock patched; do
  E=/workspace/envs/sglang-$name
  if [ ! -x $E/bin/python ]; then
    log "env sglang-$name"
    rm -rf $E; uv venv -q -p 3.12 $E
    VIRTUAL_ENV=$E uv pip install -q -e "/workspace/src/sglang-$name/python" 2>&1 | tail -5
  fi
  $E/bin/python -c "import sglang, torch; print('sglang-$name', sglang.__file__, torch.__version__)"
  VIRTUAL_ENV=$E uv pip freeze > $R/logs/freeze_sglang-$name.txt
done
diff <(grep -v '^-e\|sglang' $R/logs/freeze_sglang-stock.txt) <(grep -v '^-e\|sglang' $R/logs/freeze_sglang-patched.txt) \
  && log "stock and patched environments resolve to the same dependencies" || log "WARNING: dependency sets differ"

if [ ! -x /workspace/envs/vllm/bin/vllm ]; then
  log "env vllm 0.34.0"
  uv venv -q -p 3.12 /workspace/envs/vllm
  VIRTUAL_ENV=/workspace/envs/vllm uv pip install -q vllm==0.34.0 peft 2>&1 | tail -5
fi
/workspace/envs/vllm/bin/python -c "import vllm, torch; print('vllm', vllm.__version__, torch.__version__)"
VIRTUAL_ENV=/workspace/envs/vllm uv pip freeze > $R/logs/freeze_vllm.txt

if [ ! -x /workspace/envs/sgleval/bin/sgl-eval ]; then
  log "env sgl-eval 0.1.3"
  uv venv -q -p 3.12 /workspace/envs/sgleval
  VIRTUAL_ENV=/workspace/envs/sgleval uv pip install -q sgl-eval==0.1.3 2>&1 | tail -5
fi

log "model"
/workspace/envs/vllm/bin/python -c "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen2.5-1.5B-Instruct', local_dir='/workspace/hf/models/Qwen2.5-1.5B-Instruct'))"

log "adapters"
M=/workspace/hf/models/Qwen2.5-1.5B-Instruct
if [ ! -f /workspace/hf/adapters/adapter_0031/adapter_model.safetensors ]; then
  /workspace/envs/vllm/bin/python $R/harness/make_adapters.py --base $M --out /workspace/hf/adapters --count 32 --b-std 0.05
fi
/workspace/envs/vllm/bin/python $R/harness/check_adapters.py /workspace/hf/adapters $R/harness/adapters_manifest.json $M
log "SETUP_DONE"
