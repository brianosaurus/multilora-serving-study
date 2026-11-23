#!/usr/bin/env bash
set -u
cd /root/k; L=/root/k/final.log; : > $L
say() { echo "$(date -u +%T) $*" >> $L; }
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda CUDA_COMPUTE_CAP=89
export CARGO_BUILD_JOBS=6 CUDAFORGE_THREADS=2 TMPDIR=/root/k/tmp CARGO_TARGET_DIR=/root/k/target
cd /root/b3/mistral.rs && git checkout -q -- . && git apply --check /root/k/grouped_fmt.patch >> $L 2>&1 || { say "PATCH DOES NOT APPLY"; echo FINAL_DONE >> $L; exit 1; }
git apply /root/k/grouped_fmt.patch; say "formatted patch applied: $(git diff --stat | tail -1)"
say "fmt check"; cargo fmt --all -- --check >> $L 2>&1; say "fmt exit $?"
say "clippy start"; cargo clippy --release -p mistralrs-quant --features cuda --lib --tests -- -D warnings > /root/k/clippy.log 2>&1; say "clippy exit $?"
grep -E "^(warning|error)" -A6 /root/k/clippy.log | grep -E "lora/dynamic|dynamic_lora|^error" | head -10 >> $L
say "test start"; cargo test --release -p mistralrs-quant --features cuda --lib lora::dynamic > /root/k/final_test.log 2>&1; say "test exit $?"
grep -E "test result|cuda_grouped|FAILED|panicked" /root/k/final_test.log | head -8 >> $L
echo FINAL_DONE >> $L
