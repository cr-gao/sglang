# SANA-Video 2.0 sequence parallelism — evidence

Evidence for the pull request "[Feat] diffusion/sana-video2: Support sequence parallelism"
(sgl-project/sglang#49405). The code under test is the branch
`perf/sana-video2-sequence-parallel` at 84646ecd1c.

Setup: 2x A800-SXM4-80GB (NVLink), torch 2.13.0+cu130, official 5B checkpoint, all components
resident, no torch.compile, no breakable CUDA graph, CFG 8, seed 42, prompt
"A red tram moves slowly through a sunlit city square."; TI2V conditions on a first frame.
"2 GPUs" is `num_gpus=2` with defaults (`sp_degree=2`). Every measured request follows a
2-step warmup request. One measured request per cell.

## Speed (BF16, 50 steps, saved request wall time)

| size | task | 1 GPU (s) | 2 GPUs (s) | speedup | logs |
|---|---|---:|---:|---:|---|
| 832x480, 81f | T2V | 35.92 | 20.64 | 1.74x | `logs/p480-gpu1.log`, `logs/p480-gpu2.log` |
| 832x480, 81f | TI2V | 36.15 | 21.69 | 1.67x | same |
| 1280x736, 193f | T2V | 363.21 | 184.70 | 1.97x | `logs/p720-gpu1.log`, `logs/p720-gpu2.log` |
| 1280x736, 193f | TI2V | 363.87 | 190.45 | 1.91x | same |

1280x736 / 193 frames: peak allocated 33.2 GB -> 27.0 GB per rank, peak reserved
48.0 GB -> 35.4 GB.

## Agreement with one GPU

One transformer forward, real weights, 832x480 / 81-frame latent shape
(`scripts/forward_compare.py`, `logs/forward-fp32-480p.log`, `logs/forward-bf16-480p.log`):

| dtype | case | relative L2 | max abs |
|---|---|---:|---:|
| FP32 | T2V | 1.51e-6 | 2.43e-5 |
| FP32 | TI2V | 1.47e-6 | 1.43e-5 |
| BF16 | T2V | 1.17e-2 | 0.094 |
| BF16 | TI2V | 1.13e-2 | 0.188 |

Denoised latents after 50 steps, 832x480 / 81 frames (`scripts/run_latents.sh`,
`scripts/compare_latents.py`, `logs/latents-compare.jsonl`). The latents were dumped with a
temporary hook in the denoising stage that is not part of the branch; the latent files are
not uploaded.

| transformer dtype | T2V relative L2 | TI2V relative L2 |
|---|---:|---:|
| FP32 | 2.18e-4 | 4.79e-5 |
| BF16 | 8.81e-2 | 4.17e-2 |

Decoded videos, 2 GPUs vs 1 GPU (`scripts/compare_videos.py`, `logs/*-compare-*.json`):

| transformer dtype | size | task | SSIM mean / worst | PSNR mean / worst (dB) |
|---|---|---|---|---|
| BF16 | 832x480, 81f | T2V | 0.924 / 0.896 | 27.0 / 25.1 |
| BF16 | 832x480, 81f | TI2V | 0.951 / 0.927 | 29.6 / 27.1 |
| BF16 | 1280x736, 193f | T2V | 0.852 / 0.787 | 23.2 / 20.7 |
| BF16 | 1280x736, 193f | TI2V | 0.913 / 0.866 | 27.1 / 23.9 |
| FP32 | 832x480, 81f | T2V | 0.967 / 0.952 | 33.4 / 30.8 |
| FP32 | 832x480, 81f | TI2V | 0.970 / 0.955 | 33.6 / 30.9 |

## Why FP32 videos stop at ~33 dB

| comparison (FP32 transformer, 832x480, 81f) | T2V SSIM mean | TI2V SSIM mean | logs |
|---|---:|---:|---|
| 2 GPUs vs 1 GPU, parallel VAE decode on (default) | 0.9673 | 0.9704 | `logs/p480-fp32-compare-*.json` |
| 2 GPUs vs 1 GPU, parallel VAE decode off | 0.9672 | 0.9705 | `logs/p480-fp32-nopd-compare-*.json` |
| same latents decoded twice on one GPU | 1.0 (identical frames) | 1.0 (identical frames) | `logs/swap-control-*.json` |
| same one-GPU decode path, 1-GPU latents vs 2-GPU latents | 0.9673 | 0.9705 | `logs/swap-test-*.json` |
| same 2-GPU latents, parallel decode vs one-GPU decode | 0.9675 | 0.9705 | `logs/swap-decode2-*.json` |

Latents that are 2e-4 apart decode to videos that are 33 dB apart on an identical decode
path, so the video metric cannot resolve agreement below that level. Parallel VAE decode is
not the cause. Why the decode is this sensitive was not investigated.

## Visual check

`panels/panel{480,720}-{t2v,ti2v}.jpg`: 1 GPU | 2 GPUs | 8x absolute difference at four frames
(first, two middle, last). In all four panels both sides show the same scene, layout and
motion, with no seams, blocks or corrupted regions; the differences sit on edges and fine
texture. Only these still frames were inspected; the videos were not played back.

## Unit tests

`logs/red.log` (before the implementation: 3 failed, 1 passed), `logs/red2.log` (frame
alignment test before its fix), `logs/green1.log`, `logs/green2.log`, `logs/final.log`
(52 passed). `logs/green3.log` has one failure in the unrelated SANA-Video 2B test
`test_sana_prompt_mask_is_cast_once_per_forward` (`FileNotFoundError: 'ninja'` on this box).

`logs/p480-gpu2-89frames.log` is the first two-GPU run, before frame alignment to the GPU
count was disabled: it produced 89 frames for an 81-frame request.

## Files

- `videos/`: BF16 outputs, `{480p,720p}-{1gpu,2gpu}-{t2v,ti2v}.mp4`.
- `panels/`: frame panels built by `scripts/frame_panel.py`.
- `scripts/`: benchmark, comparison and driver scripts as run.
- `logs/`: raw logs and comparison reports.
