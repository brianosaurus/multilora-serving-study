#!/usr/bin/env bash
# /dev/shm is noexec on this pod; move the environment copies to the local root disk (/opt/fast).
[ "$(hostname)" = f70961d23335 ] || exit 9
F=/opt/fast; mkdir -p $F
echo "$(date -u +%T) copy start"
cp -a /dev/shm/envs $F/ & cp -a /dev/shm/src $F/ & wait
echo "$(date -u +%T) copy done"
for n in sglang-stock sglang-patched vllm; do
  E=$F/envs/$n
  grep -rlI -e "/dev/shm/envs/$n" -e "/workspace/envs/$n" $E/bin 2>/dev/null | xargs -r sed -i -e "s#/dev/shm/envs/$n#$E#g" -e "s#/workspace/envs/$n#$E#g"
  for f in $E/lib/python3.12/site-packages/*.pth $E/lib/python3.12/site-packages/__editable__*; do
    [ -f "$f" ] && sed -i -e "s#/dev/shm/src/$n#$F/src/$n#g" -e "s#/workspace/src/$n#$F/src/$n#g" "$f"
  done
done
for n in sglang-stock sglang-patched; do
  $F/envs/$n/bin/python -c "import sglang, sys; print(\"$n\", sglang.__file__, sys.prefix)" 2>&1 | tail -1
done
$F/envs/vllm/bin/python -c "import vllm, sys; print(\"vllm\", vllm.__version__, sys.prefix)" 2>&1 | tail -1
head -1 $F/envs/vllm/bin/vllm
echo "remaining old paths in bin: $(grep -rlI -e /dev/shm/envs -e /workspace/envs $F/envs/*/bin 2>/dev/null | wc -l)"
echo "$(date -u +%T) LOCAL_DONE"
