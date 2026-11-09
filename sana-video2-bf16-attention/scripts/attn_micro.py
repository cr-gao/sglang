import torch, time, json
from torch.profiler import profile, ProfilerActivity
from sglang.kernels.ops.diffusion import SANA_VIDEO2_SOFTMAX_ATTENTION
from sglang.multimodal_gen.runtime.models.dits.sana_video2 import GatedSoftmaxAttention, SanaVideo2RotaryPosEmbed
torch.manual_seed(0)
attn = GatedSoftmaxAttention(2560, 256).cuda().bfloat16().eval()
rope = SanaVideo2RotaryPosEmbed(256, (1, 1, 1))
for name, (f, h, w) in {"480p81f": (11, 15, 26), "720p193f": (25, 23, 40)}.items():
    x = torch.randn(2, f * h * w, 2560, device="cuda", dtype=torch.bfloat16)
    freqs = rope((f, h, w), x.device)
    out = {}
    for q in ("exact", "high"):
        if q == "high": SANA_VIDEO2_SOFTMAX_ATTENTION.mount(attn)
        else: SANA_VIDEO2_SOFTMAX_ATTENTION.unmount(attn)
        with torch.no_grad():
            for _ in range(3): y = attn(x, freqs)
            torch.cuda.synchronize(); t = time.perf_counter()
            for _ in range(10): y = attn(x, freqs)
            torch.cuda.synchronize(); ms = (time.perf_counter() - t) * 100
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                attn(x, freqs); torch.cuda.synchronize()
        kern = sorted(((e.device_time_total, e.key) for e in prof.key_averages() if e.device_time_total > 0), reverse=True)[:3]
        out[q] = (y.float(), ms, kern)
    d = (out["high"][0] - out["exact"][0]).abs()
    rel = (d.norm() / out["exact"][0].norm()).item()
    print(json.dumps({"shape": name, "tokens": f*h*w, "layer_ms_exact": round(out["exact"][1],2), "layer_ms_high": round(out["high"][1],2),
      "speedup": round(out["exact"][1]/out["high"][1],2), "max_abs_diff": d.max().item(), "rel_l2": rel}))
    for q in ("exact","high"):
        for us, k in out[q][2]: print(f"   {q:5s} {us/1000:8.2f} ms  {k[:90]}")
