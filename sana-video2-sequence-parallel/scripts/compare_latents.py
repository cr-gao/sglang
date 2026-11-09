"""Relative L2 of the dumped two-GPU denoised latents against the one-GPU ones."""

import json

import torch

for tag in ("p480-fp32-lat", "p480-bf16-lat"):
    for ti2v in (False, True):
        a = torch.load(f"artifacts/{tag}/gpu1/latents-rank0-{ti2v}.pt")
        b0 = torch.load(f"artifacts/{tag}/gpu2/latents-rank0-{ti2v}.pt")
        b1 = torch.load(f"artifacts/{tag}/gpu2/latents-rank1-{ti2v}.pt")
        print(json.dumps({
            "tag": tag, "ti2v": ti2v, "shape": list(a.shape),
            "rel_l2_sp2_vs_sp1": float((b0 - a).norm() / a.norm()),
            "max_abs": float((b0 - a).abs().max()),
            "ranks_equal": bool(torch.equal(b0, b1)), "ref_abs_mean": float(a.abs().mean()),
        }))
