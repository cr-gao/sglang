"""One real-weight SANA-Video 2.0 transformer forward: one GPU vs two GPUs (sequence parallel).

Usage: forward_compare.py <output_dir> <dtype: bf16|fp32> <frames> <height> <width>
Prints the relative L2 and max abs difference of the SP2 output against the saved single-GPU output.
"""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

CKPT_DIR = Path("/root/autodl-tmp/models/sana-video2-sp-pinned/sana-video2-5b/checkpoints")


def _forward(rank, world_size, port, output_dir, dtype_name, frames, height, width):
    from sglang.multimodal_gen.configs.models.dits.sana_video2 import SanaVideo2Config
    from sglang.multimodal_gen.configs.pipeline_configs.sana_video2 import SanaVideo2PipelineConfig
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        maybe_init_distributed_environment_and_model_parallel,
    )
    from sglang.multimodal_gen.runtime.models.dits.sana_video2 import SanaVideo2Transformer3DModel
    from sglang.multimodal_gen.runtime.server_args import set_global_server_args

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank),
                      LOCAL_RANK=str(rank), WORLD_SIZE=str(world_size))
    torch.cuda.set_device(rank)
    set_global_server_args(SimpleNamespace(
        attention_backend=None, attention_backend_config=None, kv_gather_degree=1,
        sp_split_auto=False, pipeline_config=SanaVideo2PipelineConfig()))
    maybe_init_distributed_environment_and_model_parallel(
        tp_size=1, sp_size=world_size, ulysses_degree=world_size, ring_degree=1)
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[dtype_name]
    model = SanaVideo2Transformer3DModel(SanaVideo2Config()).eval()
    checkpoint = next(CKPT_DIR.glob("*.pth"))
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)["state_dict"]
    result = model.load_state_dict(state, strict=False)
    if rank == 0:
        print(json.dumps({"event": "load", "world_size": world_size, "missing": list(result.missing_keys),
                          "unexpected": list(result.unexpected_keys)}), flush=True)
    model = model.to(device=rank, dtype=dtype)
    model.post_load_weights()
    generator = torch.Generator().manual_seed(11)
    x = torch.randn(2, 128, frames, height, width, generator=generator).to(rank)
    y = torch.randn(2, 300, 2304, generator=generator).to(rank)
    mask = torch.ones(2, 300, dtype=torch.long, device=rank)
    mask[0, 40:] = 0
    for name in ("t2v", "ti2v"):
        timestep = torch.tensor([500.0, 500.0], device=rank)
        if name == "ti2v":
            timestep = timestep[:, None, None, None, None].repeat(1, 1, frames, 1, 1)
            timestep[:, :, 0] = 0
        with torch.no_grad():
            out = model(hidden_states=x, timestep=timestep, encoder_hidden_states=y,
                        encoder_attention_mask=mask).float()
        path = Path(output_dir) / f"{dtype_name}-{name}.pt"
        if world_size == 1:
            torch.save(out.cpu(), path)
        else:
            ref = torch.load(path, weights_only=True, map_location=f"cuda:{rank}")
            print(json.dumps({
                "event": "compare", "dtype": dtype_name, "case": name, "rank": rank,
                "shape": list(out.shape), "finite": bool(torch.isfinite(out).all()),
                "rel_l2": float((out - ref).norm() / ref.norm()),
                "max_abs": float((out - ref).abs().max()), "ref_abs_mean": float(ref.abs().mean()),
            }), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    from sglang.srt.utils.network import get_free_port_below_ephemeral

    output_dir, dtype_name = sys.argv[1], sys.argv[2]
    shape = tuple(int(v) for v in sys.argv[3:6])
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for world_size in (1, 2):
        mp.spawn(_forward, args=(world_size, get_free_port_below_ephemeral(), output_dir, dtype_name, *shape),
                 nprocs=world_size)
