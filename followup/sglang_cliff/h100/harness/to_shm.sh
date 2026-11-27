#!/usr/bin/env bash
# Copy the run environments to /dev/shm (RAM disk) so server starts do not import over the network FS.
set -euo pipefail
[ "$(hostname)" = f70961d23335 ] || exit 9
mkdir -p /dev/shm/envs /dev/shm/src/sglang-stock /dev/shm/src/sglang-patched
echo "$(date -u +%T) copy start"
cp -a /workspace/envs/sglang-stock /dev/shm/envs/ &
cp -a /workspace/envs/sglang-patched /dev/shm/envs/ &
cp -a /workspace/envs/vllm /dev/shm/envs/ &
cp -a /workspace/src/sglang-stock/python /dev/shm/src/sglang-stock/ &
cp -a /workspace/src/sglang-patched/python /dev/shm/src/sglang-patched/ &
wait
echo "$(date -u +%T) copy done"
for n in sglang-stock sglang-patched vllm; do
  E=/dev/shm/envs/$n
  grep -rlI "/workspace/envs/$n" $E/bin | xargs -r sed -i "s#/workspace/envs/$n#$E#g"
  grep -lI "/workspace" $E/lib/python3.12/site-packages/*.pth $E/lib/python3.12/site-packages/__editable__* 2>/dev/null \
    | xargs -r sed -i "s#/workspace/src/$n#/dev/shm/src/$n#g"
done
for n in sglang-stock sglang-patched; do
  /dev/shm/envs/$n/bin/python -c "import sglang, sys; print(\"$n\", sglang.__file__, sys.prefix)"
done
/dev/shm/envs/vllm/bin/python -c "import vllm, sys; print(\"vllm\", vllm.__version__, vllm.__file__, sys.prefix)"
head -1 /dev/shm/envs/vllm/bin/vllm
echo "$(date -u +%T) SHM_DONE"
