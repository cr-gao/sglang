#!/bin/bash
# FP32 transformer on two GPUs with parallel VAE decode disabled; compared with the one-GPU FP32 run.
T=/root/autodl-tmp/sglang-sana-video2-sp-20261109
PY=/root/autodl-tmp/sglang-sana-video2-20260928/env/bin/python
cd $T/worktree; export PYTHONPATH=$PWD/python HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1
TAG=p480-fp32-nopd; rm -f $T/logs/$TAG.status; mkdir -p $T/artifacts/$TAG/gpu2
$PY $T/bench_sp.py --num-gpus 2 --output-dir $T/artifacts/$TAG/gpu2 --tasks t2v ti2v --steps 50 \
  --transformer-precision fp32 --no-parallel-decode > $T/logs/$TAG-gpu2.log 2>&1
echo "gpu2 exit=$?" >> $T/logs/$TAG.status
for t in t2v ti2v; do
  $PY /root/autodl-tmp/sglang-sana-video2-bf16attn-20261108/compare_videos.py \
    $T/artifacts/p480-fp32/gpu1/$t-measure1.mp4 $T/artifacts/$TAG/gpu2/$t-measure1.mp4 > $T/logs/$TAG-compare-$t.json 2>&1
done
echo done >> $T/logs/$TAG.status
