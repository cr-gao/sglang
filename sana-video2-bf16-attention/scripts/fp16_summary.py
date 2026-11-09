import json, re
T = "/root/autodl-tmp/sglang-sana-video2-bf16attn-20261108"
for t in ("t2v", "ti2v"):
    for tag in ("p480b", "p480-fp16"):
        c = json.load(open(f"{T}/logs/{tag}-compare-{t}.json")); s = c["per_frame_ssim"]
        print(f"{tag:10s} {t:5s} SSIM mean/min {c['ssim_mean']:.4f}/{c['ssim_min']:.4f}  PSNR mean/min {c['psnr_mean']:.2f}/{c['psnr_min']:.2f}  frames<0.92: {sum(v < 0.92 for v in s)}  last5 {[round(v, 3) for v in s[-5:]]}")
log = open(f"{T}/logs/p480-fp16-high.log").read()
rows = [json.loads(l) for l in log.splitlines() if l.startswith('{"event": "request"')]
stages = [float(m) for m in re.findall(r"\[SanaVideo2DenoisingStage\] finished in ([\d.]+)", log)]
for r, st in zip(rows, stages):
    print("fp16", r["task"], r["label"], "e2e", round(r["elapsed_saved_request_s"], 2), "denoise", st, "ms/step", round(st * 20, 1), r["decoded_rgb_sha256"][:12])
