#!/bin/bash
# A/B: one GPU vs two GPUs (sequence parallel) on the same worktree.
# Usage: run_sp_ab.sh <tag> <width> <height> <frames> <steps> <repeats> [tasks...]
set -u
TAG=$1; W=$2; H=$3; F=$4; STEPS=$5; REP=$6; shift 6; TASKS=${*:-t2v ti2v}
TASK=/root/autodl-tmp/sglang-sana-video2-sp-20261109
PY=/root/autodl-tmp/sglang-sana-video2-20260928/env/bin/python
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1
cd "$TASK/worktree"; export PYTHONPATH=$PWD/python
rm -f "$TASK/logs/$TAG.status"
for N in ${GPUS:-1 2}; do
  OUT=$TASK/artifacts/$TAG/gpu$N; mkdir -p "$OUT"
  SGLANG_TORCH_PROFILER_DIR=$OUT/traces "$PY" "$TASK/bench_sp.py" --num-gpus $N --output-dir "$OUT" \
    --width $W --height $H --frames $F --steps $STEPS --repeats $REP --tasks $TASKS ${PROFILE_FLAGS:-} \
    > "$TASK/logs/$TAG-gpu$N.log" 2>&1
  echo "gpu$N exit=$?" >> "$TASK/logs/$TAG.status"
done
for T in $TASKS; do
  "$PY" /root/autodl-tmp/sglang-sana-video2-bf16attn-20261108/compare_videos.py \
    "$TASK/artifacts/$TAG/gpu1/$T-measure1.mp4" "$TASK/artifacts/$TAG/gpu2/$T-measure1.mp4" \
    > "$TASK/logs/$TAG-compare-$T.json" 2>&1
done
echo done >> "$TASK/logs/$TAG.status"
