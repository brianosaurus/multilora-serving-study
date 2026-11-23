#!/usr/bin/env bash
cd /root/k; L=/root/k/lint.log; : > $L
while ! grep -q FINAL_DONE /root/k/final.log; do sleep 15; done
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda CUDA_COMPUTE_CAP=89
export CARGO_BUILD_JOBS=6 CUDAFORGE_THREADS=2 TMPDIR=/root/k/tmp CARGO_TARGET_DIR=/root/k/target
cd /root/b3/mistral.rs
cargo fmt --all -- --check >> $L 2>&1; echo "fmt exit $?" >> $L
cargo clippy --release -p mistralrs-quant --features cuda --lib --tests -- -D warnings > /root/k/clippy.log 2>&1; echo "clippy exit $?" >> $L
grep -E "^(warning|error)" -A8 /root/k/clippy.log | grep -E "lora/dynamic|dynamic_lora|^error|^warning: unused" | head -12 >> $L
echo LINT_DONE >> $L
