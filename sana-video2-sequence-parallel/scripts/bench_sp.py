"""Run saved SANA-Video 2.0 requests on one local scheduler with a given GPU count."""

import argparse
import hashlib
import json
import time
from pathlib import Path

from sglang.multimodal_gen import DiffGenerator
from sglang.multimodal_gen.runtime.server_args import ServerArgs

MODEL_ROOT = Path("/root/autodl-tmp/models/sana-video2-sp-pinned")
IMAGE = Path("/root/autodl-tmp/sglang-sana-video2-20260928/artifacts/pr-first-frame.png")
PROMPT = "A red tram moves slowly through a sunlit city square."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-gpus", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--tasks", nargs="+", choices=("t2v", "ti2v"), default=("t2v", "ti2v"))
    parser.add_argument("--profile", action="store_true", help="one extra profiled request per task")
    parser.add_argument("--transformer-precision", default=None)
    parser.add_argument("--no-parallel-decode", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    extra = {"vae_config.use_parallel_decode": False} if args.no_parallel_decode else {}
    server_args = ServerArgs.from_kwargs(
        **extra,
        model_path=str(MODEL_ROOT / "sana-video2-5b"),
        model_id="Efficient-Large-Model/SANA-Video_2.0_5B_720p",
        component_paths={"text_encoder": str(MODEL_ROOT / "gemma-2-2b-it"),
                         "vae": str(MODEL_ROOT / "ltx-2.3-vae")},
        num_gpus=args.num_gpus,
        performance_mode="manual",
        component_residency={"all": "resident"},
        enable_torch_compile=False,
        enable_breakable_cuda_graph=False,
        warmup_mode="off",
        component_precisions=(
            {"transformer": args.transformer_precision} if args.transformer_precision else {}
        ),
    )
    print(json.dumps({"event": "configuration", "num_gpus": args.num_gpus,
                      **{k: getattr(server_args, k) for k in (
                          "sp_degree", "ulysses_degree", "ring_degree", "kv_gather_degree",
                          "sp_split_auto", "tp_size", "enable_cfg_parallel")},
                      "vae_use_parallel_decode": server_args.pipeline_config.vae_config.use_parallel_decode,
                      "width": args.width, "height": args.height, "frames": args.frames,
                      "steps": args.steps}), flush=True)
    start = time.perf_counter()
    generator = DiffGenerator.from_server_args(server_args)
    print(json.dumps({"event": "startup", "seconds": time.perf_counter() - start}), flush=True)
    try:
        for task in args.tasks:
            cases = [("warmup", 2, False)] + [(f"measure{i + 1}", args.steps, False) for i in range(args.repeats)]
            if args.profile:
                cases.append(("profile", args.steps, True))
            for label, steps, profile in cases:
                path = args.output_dir / f"{task}-{label}.mp4"
                params = {
                    "prompt": PROMPT, "width": args.width, "height": args.height,
                    "num_frames": args.frames, "fps": 24, "num_inference_steps": steps,
                    "guidance_scale": 8.0, "seed": 42,
                    "save_output": True, "output_path": str(path.parent), "output_file_name": path.name,
                }
                if profile:
                    params.update(profile=True, num_profiled_timesteps=2)
                if task == "ti2v":
                    params["image_path"] = str(IMAGE)
                start = time.perf_counter()
                result = generator.generate(params)
                elapsed = time.perf_counter() - start
                if result is None or isinstance(result, list):
                    raise AssertionError(f"Expected one result for {task}: {result}")
                output = Path(result.output_file_path)
                print(json.dumps({
                    "event": "request", "num_gpus": args.num_gpus, "task": task, "label": label,
                    "frames": args.frames, "steps": steps, "elapsed_saved_request_s": elapsed,
                    "generation_time_s": result.generation_time, "peak_memory_mb": result.peak_memory_mb,
                    "metrics": result.metrics, "output_path": str(output),
                    "mp4_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                }), flush=True)
    finally:
        generator.shutdown()


if __name__ == "__main__":
    main()
