#!/usr/bin/env bash
# Pause any patched-tree cargo/rustc until the target copy finishes, then resume it.
while ! grep -q "CP_" /workspace/h100_run/logs_cptarget.txt 2>/dev/null; do
  for p in $(pgrep -f "cargo|rustc"); do
    if readlink /proc/$p/cwd 2>/dev/null | grep -q sglang-patched; then
      kill -STOP $p 2>/dev/null && echo "$(date -u +%T) paused $p" >> /workspace/h100_run/logs_hold.txt
    fi
  done
  sleep 2
done
for p in $(pgrep -f "cargo|rustc"); do kill -CONT $p 2>/dev/null; done
echo "$(date -u +%T) released ($(cat /workspace/h100_run/logs_cptarget.txt))" >> /workspace/h100_run/logs_hold.txt
