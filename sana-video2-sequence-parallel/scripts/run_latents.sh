#!/bin/bash
# Dump denoised latents for one-GPU and two-GPU runs (needs the SP_DUMP_DIR debug hook in the stage).
T=/root/autodl-tmp/sglang-sana-video2-sp-20261109
PY=/root/autodl-tmp/sglang-sana-video2-20260928/env/bin/python
cd $T/worktree; export PYTHONPATH=$PWD/python HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1
rm -f $T/logs/latents.status
for CFG in "p480-fp32-lat --transformer-precision=fp32" "p480-bf16-lat"; do
  set -- $CFG; TAG=$1; FL=${2:-}
  for N in 1 2; do
    mkdir -p $T/artifacts/$TAG/gpu$N
    SP_DUMP_DIR=$T/artifacts/$TAG/gpu$N $PY $T/bench_sp.py --num-gpus $N --output-dir $T/artifacts/$TAG/gpu$N \
      --tasks t2v ti2v --steps 50 $FL > $T/logs/$TAG-gpu$N.log 2>&1
    echo "$TAG gpu$N exit=$?" >> $T/logs/latents.status
  done
done
cp $T/stage.orig.py $T/worktree/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/sana_video2.py
echo done >> $T/logs/latents.status
