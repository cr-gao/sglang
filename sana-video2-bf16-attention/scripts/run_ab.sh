#!/bin/bash
# A/B: quality=exact vs quality=high on the same worktree. Usage: run_ab.sh <tag> <width> <height> <frames> <steps> <repeats> [tasks...]
set -eu
TAG=$1; W=$2; H=$3; F=$4; STEPS=$5; REP=$6; shift 6; TASKS=${*:-t2v ti2v}
TASK=/root/autodl-tmp/sglang-sana-video2-bf16attn-20261108
PY=/root/autodl-tmp/sglang-sana-video2-20260928/env/bin/python
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1
cd "$TASK/worktree"; export PYTHONPATH=$PWD/python
for Q in exact high; do
  OUT=$TASK/artifacts/$TAG/$Q; mkdir -p "$OUT"
  SGLANG_TORCH_PROFILER_DIR=$OUT/traces "$PY" "$TASK/bench_quality.py" --quality $Q --output-dir "$OUT" \
    --width $W --height $H --frames $F --steps $STEPS --repeats $REP --tasks $TASKS ${PROFILE_FLAGS:-} \
    > "$TASK/logs/$TAG-$Q.log" 2>&1
  echo "$Q exit=$?" >> "$TASK/logs/$TAG.status"
done
for T in $TASKS; do
  "$PY" "$TASK/compare_videos.py" "$TASK/artifacts/$TAG/exact/exact-$T-measure1.mp4" "$TASK/artifacts/$TAG/high/high-$T-measure1.mp4" \
    > "$TASK/logs/$TAG-compare-$T.json"
done
echo done >> "$TASK/logs/$TAG.status"
