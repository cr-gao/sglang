#!/bin/bash
# Usage: run_tests.sh <tag> <pytest args...>
TAG=$1; shift
T=/root/autodl-tmp/sglang-sana-video2-sp-20261109
PY=/root/autodl-tmp/sglang-sana-video2-20260928/env/bin/python
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1
cd "$T/worktree"; export PYTHONPATH=$PWD/python
"$PY" -m pytest "$@" -q -p no:cacheprovider > "$T/logs/$TAG.log" 2>&1
echo "exit=$?" >> "$T/logs/$TAG.log"
tail -40 "$T/logs/$TAG.log"
