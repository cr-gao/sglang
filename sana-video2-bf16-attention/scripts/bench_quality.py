"""Run repeated saved SANA-Video 2.0 requests at one request quality on one local scheduler."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import av

from sglang.multimodal_gen import DiffGenerator
from sglang.multimodal_gen.runtime.server_args import ServerArgs

MODEL_ROOT = Path("/root/autodl-tmp/models/sana-video2-sp-pinned")
IMAGE = Path("/root/autodl-tmp/sglang-sana-video2-20260928/artifacts/pr-first-frame.png")
PROMPT = "A red tram moves slowly through a sunlit city square."


def video_digest(path):
    digest = hashlib.sha256()
    frames = 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        width, height, fps = stream.width, stream.height, str(stream.average_rate)
        for frame in container.decode(video=0):
            digest.update(frame.to_ndarray(format="rgb24").tobytes())
            frames += 1
    return {"decoded_rgb_sha256": digest.hexdigest(), "video_width": width,
            "video_height": height, "video_fps": fps, "video_frames": frames}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality", choices=("exact", "lossless", "high"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tasks", nargs="+", choices=("t2v", "ti2v"), default=("t2v", "ti2v"))
    parser.add_argument("--profile", action="store_true", help="one extra profiled request per task")
    parser.add_argument("--profiled-steps", type=int, default=2)
    parser.add_argument("--strict-numerics", action="store_true", help="cudnn TF32 and bf16 reduced-precision reductions off")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    server_args = ServerArgs.from_kwargs(
        model_path=str(MODEL_ROOT / "sana-video2-5b"),
        model_id="Efficient-Large-Model/SANA-Video_2.0_5B_720p",
        component_paths={"text_encoder": str(MODEL_ROOT / "gemma-2-2b-it"),
                         "vae": str(MODEL_ROOT / "ltx-2.3-vae")},
        performance_mode="manual",
        component_residency={"all": "resident"},
        enable_torch_compile=False,
        enable_breakable_cuda_graph=False,
        warmup_mode="off",
        allow_cudnn_tf32=not args.strict_numerics,
        allow_bf16_reduced_precision_reduction=not args.strict_numerics,
    )
    print(json.dumps({"event": "configuration", "quality": args.quality, "width": args.width,
                      "height": args.height, "frames": args.frames, "steps": args.steps, "strict_numerics": args.strict_numerics}), flush=True)
    start = time.perf_counter()
    generator = DiffGenerator.from_server_args(server_args)
    print(json.dumps({"event": "startup", "seconds": time.perf_counter() - start}), flush=True)
    try:
        for task in args.tasks:
            cases = [("warmup" if i == 0 else f"measure{i}", False) for i in range(args.repeats + 1)]
            if args.profile:
                cases.append(("profile", True))
            for label, profile in cases:
                path = args.output_dir / f"{args.quality}-{task}-{label}.mp4"
                params = {
                    "prompt": PROMPT, "width": args.width, "height": args.height,
                    "num_frames": args.frames, "fps": 24, "num_inference_steps": args.steps,
                    "guidance_scale": 8.0, "seed": 42, "quality": args.quality,
                    "save_output": True, "output_path": str(path.parent), "output_file_name": path.name,
                }
                if profile:
                    params.update(profile=True, num_profiled_timesteps=args.profiled_steps)
                if task == "ti2v":
                    params["image_path"] = str(IMAGE)
                start = time.perf_counter()
                result = generator.generate(params)
                elapsed = time.perf_counter() - start
                if result is None or isinstance(result, list):
                    raise AssertionError(f"Expected one result for {task}: {result}")
                output = Path(result.output_file_path)
                if output != path or not output.is_file():
                    raise AssertionError(f"Missing saved output: {output} != {path}")
                metadata = video_digest(output)
                print(json.dumps({
                    "event": "request", "quality": args.quality, "task": task, "label": label,
                    "frames": args.frames, "steps": args.steps, "elapsed_saved_request_s": elapsed,
                    "generation_time_s": result.generation_time, "peak_memory_mb": result.peak_memory_mb,
                    "metrics": result.metrics, "output_path": str(output),
                    "mp4_sha256": hashlib.sha256(output.read_bytes()).hexdigest(), **metadata,
                }), flush=True)
    finally:
        generator.shutdown()


if __name__ == "__main__":
    main()
