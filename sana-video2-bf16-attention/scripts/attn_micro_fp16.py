import torch, time, json
import torch.nn.functional as F
from sglang.multimodal_gen.runtime.models.dits.sana_video2 import GatedSoftmaxAttention, SanaVideo2RotaryPosEmbed, _apply_rope
torch.manual_seed(0)
attn = GatedSoftmaxAttention(2560, 256).cuda().bfloat16().eval()
rope = SanaVideo2RotaryPosEmbed(256, (1, 1, 1))
def qkv(x, freqs):
    b, n, c = x.shape
    q, k, v = attn.qkv(x).reshape(b, n, 3, c).unbind(2)
    q = attn.q_norm(q).reshape(b, n, attn.heads, attn.dim).transpose(1, 2)
    k = attn.k_norm(k).reshape(b, n, attn.heads, attn.dim).transpose(1, 2)
    v = v.reshape(b, n, attn.heads, attn.dim).transpose(1, 2)
    return _apply_rope(q, freqs), _apply_rope(k, freqs), v
def timed(fn):
    for _ in range(3): fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(10): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) * 100
for name, (f, h, w) in {"480p81f": (11, 15, 26), "720p193f": (25, 23, 40)}.items():
    x = torch.randn(2, f * h * w, 2560, device="cuda", dtype=torch.bfloat16)
    freqs = rope((f, h, w), x.device)
    with torch.no_grad():
        q, k, v = qkv(x, freqs)
        ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float())
        out = {}
        for dt in (torch.bfloat16, torch.float16):
            o = F.scaled_dot_product_attention(q.to(dt), k.to(dt), v.to(dt)).to(torch.bfloat16).float()
            out[str(dt)] = (o, timed(lambda: F.scaled_dot_product_attention(q.to(dt), k.to(dt), v.to(dt))))
        ref_b = ref.to(torch.bfloat16).float()
    print(json.dumps({"shape": name, "max_abs_v": v.abs().max().item(), "max_abs_q": q.abs().max().item(), "max_abs_out": ref.abs().max().item(),
        **{dt: {"rel_l2_vs_fp32": ((o - ref).norm() / ref.norm()).item(), "rel_l2_vs_fp32_rounded_bf16": ((o - ref_b).norm() / ref_b.norm()).item(),
                "frac_elements_differ_from_bf16_rounded_ref": (o != ref_b).float().mean().item(), "attn_ms": round(ms, 2)} for dt, (o, ms) in out.items()}}))
