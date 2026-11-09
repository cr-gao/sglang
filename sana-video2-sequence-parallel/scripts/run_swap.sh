#!/bin/bash
# One-GPU decode of saved latents: the denoising result is replaced by a saved latent file
# (needs the SP_LOAD_DIR debug hook in the stage). Control: one-GPU latents; test: two-GPU latents.
T=/root/autodl-tmp/sglang-sana-video2-sp-20261109
PY=/root/autodl-tmp/sglang-sana-video2-20260928/env/bin/python
F=$T/worktree/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/sana_video2.py
cp $F $T/stage.orig.py
$PY - <<PYEOF
p = "$F"; s = open(p).read()
old = """                callback=on_step,
            )
        return batch
"""
new = """                callback=on_step,
            )
        import os
        if os.environ.get("SP_LOAD_DIR") and not batch.is_warmup:
            batch.latents = torch.load(os.path.join(os.environ["SP_LOAD_DIR"], f"latents-rank0-{is_ti2v}.pt")).to(batch.latents)
        return batch
"""
assert old in s; open(p, "w").write(s.replace(old, new))
PYEOF
cd $T/worktree; export PYTHONPATH=$PWD/python HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1
rm -f $T/logs/swap.status
for SRC in gpu1 gpu2; do
  OUT=$T/artifacts/p480-fp32-swap/from-$SRC; mkdir -p $OUT
  SP_LOAD_DIR=$T/artifacts/p480-fp32-lat/$SRC $PY $T/bench_sp.py --num-gpus 1 --output-dir $OUT \
    --tasks t2v ti2v --steps 3 > $T/logs/p480-fp32-swap-$SRC.log 2>&1
  echo "$SRC exit=$?" >> $T/logs/swap.status
done
cp $T/stage.orig.py $F
CMP=/root/autodl-tmp/sglang-sana-video2-bf16attn-20261108/compare_videos.py
for t in t2v ti2v; do
  $PY $CMP $T/artifacts/p480-fp32-lat/gpu1/$t-measure1.mp4 $T/artifacts/p480-fp32-swap/from-gpu1/$t-measure1.mp4 > $T/logs/swap-control-$t.json 2>&1
  $PY $CMP $T/artifacts/p480-fp32-swap/from-gpu1/$t-measure1.mp4 $T/artifacts/p480-fp32-swap/from-gpu2/$t-measure1.mp4 > $T/logs/swap-test-$t.json 2>&1
  $PY $CMP $T/artifacts/p480-fp32-lat/gpu2/$t-measure1.mp4 $T/artifacts/p480-fp32-swap/from-gpu2/$t-measure1.mp4 > $T/logs/swap-decode2-$t.json 2>&1
done
echo done >> $T/logs/swap.status
