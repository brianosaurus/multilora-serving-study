#!/usr/bin/env bash
# Sequential full cross-entropy suite: stock then patched, separate processes, separate Triton caches.
for v in stock patched; do
  cd /root/fa/$v
  rm -f /root/fa/raw/mem_$v.csv
  echo "$(date -u +%T) start $v" >> /root/fa/raw/suite.log
  FA_SRC=/root/fa/$v FA_MEMCSV=/root/fa/raw/mem_$v.csv PYTHONPATH=/root/fa/harness TRITON_CACHE_DIR=/root/fa/tcache_suite_$v     /root/fa/venv/bin/python -m pytest -p fa_noext tests/losses/test_cross_entropy.py -v -rA --durations=0 --junitxml=/root/fa/raw/junit_$v.xml -p no:cacheprovider > /root/fa/raw/pytest_$v.log 2>&1
  echo "$(date -u +%T) done $v exit $?" >> /root/fa/raw/suite.log
done
echo SUITE_DONE >> /root/fa/raw/suite.log
