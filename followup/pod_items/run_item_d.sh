#!/usr/bin/env bash
# Item D (Amendment 6). Six starts: old new old new old new. Stops starting new ones after 2 h.
D=/root/msf_study/followup/pod_items/results/D; mkdir -p $D; L=$D/run.log; : > $L
say() { echo "$(date -u +%T) $*" >> $L; }
M=/dev/shm/models/Qwen2.5-1.5B-Instruct; A=/dev/shm/adapters_s0.05_8; PY=/root/envs/sglang_main/bin/python
t0=$(date +%s); n=0
say "host $(hostname); old $(sha256sum /root/bin/mistralrs_old | cut -c1-16) new $(sha256sum /root/bin/mistralrs_new | cut -c1-16)"
for tag in old new old new old new; do
  n=$((n+1)); [ $(( $(date +%s) - t0 )) -lt 7200 ] || { say "NOT RUN: start $n ($tag), 2 h GPU limit reached"; continue; }
  for i in $(seq 1 24); do [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] && break; sleep 5; done
  [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] || { say "REFUSED start $n ($tag): GPU not idle"; continue; }
  say "start $n $tag; load $(cut -d' ' -f1-3 /proc/loadavg); external CPU: $($PY /root/msf_study/followup/kernel_rerun/extload.py check 10 2>&1 | tail -1)"
  CMD=(/root/bin/mistralrs_$tag serve -m $M -p 1234 --kv-cache 8GiB --max-seqs 256 --lora-max-adapters 64)
  for i in 0 1 2 3 4 5 6 7; do CMD+=(--lora "adapter_000$i=$A/adapter_000$i"); done
  setsid nohup "${CMD[@]}" > $D/server_${n}_$tag.log 2>&1 < /dev/null & SP=$!
  ok=0; for i in $(seq 1 1200); do [ "$(curl -s -o /dev/null -w %{http_code} -m 3 localhost:1234/v1/models)" = 200 ] && { ok=1; break; }; kill -0 $SP 2>/dev/null || break; sleep 3; done
  say "start $n $tag ready=$ok"
  [ $ok = 1 ] && { $PY /root/msf_study/followup/pod_items/item_d_requests.py $D/start${n}_$tag.json >> $D/requests.log 2>&1; say "start $n $tag requests exit $?"; }
  kill $SP 2>/dev/null; sleep 6; kill -9 $SP 2>/dev/null; sleep 4
done
say "elapsed $(( $(date +%s) - t0 )) s"; echo DONE >> $L
