#!/bin/bash
# Waits for p720, then: (1) scratch fp16 variant at 480p (worktree-test), (2) strict-numerics exact control at 480p (worktree).
set -u
T=/root/autodl-tmp/sglang-sana-video2-bf16attn-20261108
PY=/root/autodl-tmp/sglang-sana-video2-20260928/env/bin/python
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1
until grep -q done $T/logs/p720.status 2>/dev/null; do sleep 60; done
cd $T/worktree-test && git checkout -q 737a842506 && git log --oneline -1 > $T/logs/fu-fp16.head
PYTHONPATH=$PWD/python "$PY" $T/bench_quality.py --quality high --output-dir $T/artifacts/p480-fp16/high --repeats 1 --tasks t2v ti2v > $T/logs/p480-fp16-high.log 2>&1
echo "fp16 exit=$?" >> $T/logs/followups.status
for t in t2v ti2v; do
  "$PY" $T/compare_videos.py $T/artifacts/p480b/exact/exact-$t-measure1.mp4 $T/artifacts/p480-fp16/high/high-$t-measure1.mp4 > $T/logs/p480-fp16-compare-$t.json
done
cd $T/worktree && git log --oneline -1 > $T/logs/fu-strict.head
PYTHONPATH=$PWD/python "$PY" $T/bench_quality.py --quality exact --strict-numerics --output-dir $T/artifacts/p480-strict/exact --repeats 1 --tasks t2v > $T/logs/p480-strict-exact.log 2>&1
echo "strict exit=$?" >> $T/logs/followups.status
"$PY" $T/compare_videos.py $T/artifacts/p480b/exact/exact-t2v-measure1.mp4 $T/artifacts/p480-strict/exact/exact-t2v-measure1.mp4 > $T/logs/p480-strict-compare-t2v.json
echo done >> $T/logs/followups.status
