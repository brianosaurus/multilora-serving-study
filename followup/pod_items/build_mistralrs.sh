#!/usr/bin/env bash
# Item D builds (plan: followup/pod_items/PLAN.md). Old = master 21da642; new = old + results/raw/kernel_followup/grouped_kernel_commit_600cdef.patch.
# Sequential, empty target dir each, deleted after the binary is copied to /workspace/bin (small pod disk). Flags as followup/kernel_rerun/BUILD.md.
L=/workspace/setup/mistralrs_build.log; : > $L; mkdir -p /workspace/bin /workspace/src
say() { echo "$(date -u +%T) $*" >> $L; }
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda CUDA_COMPUTE_CAP=89 CARGO_BUILD_JOBS=8 CUDAFORGE_THREADS=4
unset RUSTFLAGS
say "rustc $(rustc --version); cargo $(cargo --version); nvcc $(nvcc --version | tail -1); host $(hostname)"
cd /workspace/src && [ -d mistral.rs ] || git clone -q https://github.com/EricLBuehler/mistral.rs.git mistral.rs
cd mistral.rs && git checkout -q 21da642fdd2210c30ca4028e1eabdd28e9c602fd && say "base $(git rev-parse HEAD)"
for t in old new; do
  rm -rf /workspace/src/tree_$t; mkdir -p /workspace/src/tree_$t; git -C /workspace/src/mistral.rs archive HEAD | tar -x -C /workspace/src/tree_$t
  if [ $t = new ]; then (cd /workspace/src/tree_$t && git init -q . && git apply /workspace/study/results/raw/kernel_followup/grouped_kernel_commit_600cdef.patch 2>&1 | head -3 >> $L; say "patch applied: $(git status --short | wc -l) files changed"); fi
  say "building $t"; s=$(date +%s)
  (cd /workspace/src/tree_$t && TMPDIR=/workspace/tmp_$t CARGO_TARGET_DIR=/workspace/target_$t cargo build --release -p mistralrs-cli --features "cuda flash-attn" > /workspace/setup/mistralrs_build_$t.log 2>&1; echo "exit $?" >> $L)
  b=/workspace/target_$t/release/mistralrs; [ -x $b ] || b=$(ls /workspace/target_$t/release/mistralrs* 2>/dev/null | grep -v '\.d$' | head -1)
  if [ -x "$b" ]; then cp $b /workspace/bin/mistralrs_$t; say "$t built in $(( $(date +%s) - s )) s: $(stat -c %s /workspace/bin/mistralrs_$t) bytes sha256 $(sha256sum /workspace/bin/mistralrs_$t | cut -c1-64)"; else say "$t: binary not found"; fi
  rm -rf /workspace/target_$t /workspace/tmp_$t; say "target removed; $(df -h / | tail -1)"
done
echo BUILD_DONE >> $L
