# SANA-Video 2.0 BF16-input softmax attention at `quality=high`: evidence

Companion to the SGLang PR that adds `SANA_VIDEO2_SOFTMAX_ATTENTION`. Code under test:
`perf/sana-video2-bf16-attention` head `1b3c643fb0` on main `65abad034b`.

Hardware and environment: one NVIDIA A800-SXM4-80GB (AutoDL), torch 2.13.0+cu130, triton 3.7.1,
flashinfer 0.6.16.post3, diffusers 0.37.0, transformers 5.12.1, huggingface-hub 1.33.0, no flash-attn
package (SDPA dispatches PyTorch's built-in flash kernel). Checkpoints: `SANA_Video_2.0_5B_720p.pth`,
`gemma-2-2b-it`, LTX-2.3 VAE. All components resident, `--performance-mode manual`, no torch.compile,
no breakable CUDA graphs, `warmup_mode=off`; each arm runs 1 warmup request then the measured ones.

Request: prompt "A red tram moves slowly through a sunlit city square.", CFG 8, seed 42, fps 24,
flow shift 12 (model default). TI2V conditions on `pr-first-frame.png` (an earlier 832x480 T2V frame).

## Videos (`videos/`)

`<profile>-<quality>-<task>.mp4`, the first measured request of each arm. Each arm is deterministic:
the warmup and all measured requests of an arm share one decoded-RGB SHA-256 (see `logs/*.log`).

## Panels (`panels/`)

`frame_panel.py` output: exact | high | 8x amplified |diff|, at frames 0/40/80 (480p) and 0/96/192 (720p).

## Speed

| profile | task | exact e2e s | high e2e s | e2e | exact denoise s | high denoise s | ms/step exact -> high | peak reserved MiB |
|---|---|---:|---:|---:|---:|---:|---|---:|
| 832x480, 81 frames, 50 steps (n=3) | T2V | 35.910 | 30.890 | -13.98% | 34.578 | 29.571 | 691.6 -> 591.4 | 21142 (both) |
| 832x480, 81 frames, 50 steps (n=3) | TI2V | 36.008 | 30.895 | -14.20% | 34.658 | 29.554 | 693.2 -> 591.1 | 21142 (both) |
| 1280x736, 193 frames, 50 steps (n=1) | T2V | 364.068 | 172.365 | -52.66% | 357.635 | 166.974 | 7152.7 -> 3339.5 | 48004 (both) |
| 1280x736, 193 frames, 50 steps (n=1) | TI2V | 363.511 | 172.577 | -52.52% | 357.088 | 167.004 | 7141.8 -> 3340.1 | 48004 (both) |

e2e is wall time around `DiffGenerator.generate` for a saved request (text encoding, latent preparation,
denoising, VAE decode, MP4 encode and save). Denoise is the `[SanaVideo2DenoisingStage] finished in`
stage time from the server log, i.e. the 50-step DiT loop (CFG-batched forward plus sampler math). The
stage timer is not CUDA-synchronized unless `SGLANG_DIFFUSION_SYNC_STAGE_PROFILING=1`, which these runs
did not set, so queued GPU work at the stage tail can be charged to the decode stage instead (720p decode
reads 1.15 s exact vs 0.67 s high). The loop is GPU-bound, so the denoise figures are within about one
queue depth of the GPU time; e2e carries no such caveat.

Per anchor layer (`attn_micro.py`, batch 2, bf16, real RoPE, 10 timed iterations after 3 warmups):

| shape | tokens | exact ms | high ms | exact top kernel | high top kernel |
|---|---:|---:|---:|---|---|
| 480p 81f | 4290 | 21.29 | 6.68 | `fmha_cutlassF_f32_aligned_32x128_gmem_sm80` 14.30 ms | `pytorch_flash::flash_fwd_kernel<...256,128,64...>` 2.09 ms |
| 720p 193f | 23000 | 554.65 | 78.06 | same, 529.30 ms | same, 54.07 ms |

## Quality (high vs exact, per frame; `compare_videos.py`: skimage-default SSIM reimplemented with a 7x7 uniform window per channel, RGB PSNR)

| profile | task | SSIM mean / min (frame) | PSNR mean / min dB (frame) | frames with SSIM < 0.92 | frames with PSNR < 24 |
|---|---|---|---|---:|---:|
| 480p 81f | T2V | 0.9222 / 0.8982 (80) | 26.82 / 25.34 (80) | 25 / 81 | 0 |
| 480p 81f | TI2V | 0.9512 / 0.9282 (80) | 29.71 / 27.27 (80) | 0 / 81 | 0 |
| 720p 193f | T2V | 0.8712 / 0.8293 (192) | 23.76 / 22.05 (192) | 193 / 193 | 139 |
| 720p 193f | TI2V | 0.9132 / 0.8667 (192) | 26.98 / 24.09 (192) | 130 / 193 | 0 |

Per-frame arrays are in `logs/p480b-compare-*.json` and `logs/p720-compare-*.json`.

## Side experiments

- fp16 attention inputs (scratch, not in the PR; `logs/p480-fp16-high.log`, `logs/p480-fp16-compare-*.json`):
  480p T2V SSIM mean/min 0.9253/0.9032, PSNR 27.02/25.66; TI2V 0.9513/0.9271, 29.71/27.34; 595 ms/step.
- `quality=exact` with `--allow-cudnn-tf32 false --allow-bf16-reduced-precision-reduction false` vs the
  default `exact` (`logs/p480-strict-*.{log,json}`): 480p T2V SSIM mean/min 0.8150/0.7788, PSNR 21.09/20.14.
  Reported as context only; it is not a like-for-like perturbation.
- Unit tests at `1b3c643fb0` (`logs/tests-full.log`): `test_sites.py`, `test_sana_video2.py`,
  `test_sana_video2_pipeline.py`: 55 passed. Red/green logs for each test commit: `logs/tests-red*.log`,
  `logs/tests-green*.log`.

## Reproduction

`scripts/run_ab.sh <tag> <width> <height> <frames> <steps> <repeats> [tasks...]` runs both arms with
`scripts/bench_quality.py` and compares the first measured videos with `scripts/compare_videos.py`;
`scripts/analyze_ab.py <tag>` prints the tables. Paths at the top of the scripts point at the AutoDL layout.
