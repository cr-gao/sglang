#!/bin/bash
# Default (auto) performance mode with only num_gpus set: one GPU vs two GPUs.
T=/root/autodl-tmp/sglang-sana-video2-sp-20261109
PY=/root/autodl-tmp/sglang-sana-video2-20260928/env/bin/python
cd $T/worktree; export PYTHONPATH=$PWD/python HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1
TAG=p480-auto; rm -f $T/logs/$TAG.status
for N in 1 2; do
  mkdir -p $T/artifacts/$TAG/gpu$N
  $PY $T/bench_sp.py --num-gpus $N --output-dir $T/artifacts/$TAG/gpu$N --tasks t2v ti2v --steps 50 \
    --performance-mode auto > $T/logs/$TAG-gpu$N.log 2>&1
  echo "gpu$N exit=$?" >> $T/logs/$TAG.status
done
for t in t2v ti2v; do
  $PY /root/autodl-tmp/sglang-sana-video2-bf16attn-20261108/compare_videos.py \
    $T/artifacts/$TAG/gpu1/$t-measure1.mp4 $T/artifacts/$TAG/gpu2/$t-measure1.mp4 > $T/logs/$TAG-compare-$t.json 2>&1
done
echo done >> $T/logs/$TAG.status
