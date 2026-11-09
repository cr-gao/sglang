"""Summarize an exact-vs-high A/B: per-task means of saved-request e2e, denoise/decode stage time, memory, hashes."""

import json
import re
import statistics
import sys
from pathlib import Path

task_dir = Path("/root/autodl-tmp/sglang-sana-video2-bf16attn-20261108")
tag = sys.argv[1]
stage_re = re.compile(r"\[(SanaVideo2DenoisingStage|DecodingStage)\] finished in ([\d.]+) seconds")


def parse(quality):
    rows, stages = [], {}
    for line in (task_dir / "logs" / f"{tag}-{quality}.log").read_text().splitlines():
        m = stage_re.search(line)
        if m:
            stages[m[1]] = float(m[2])
        if line.startswith('{"event": "request"'):
            r = json.loads(line)
            r["stages"] = stages
            stages = {}
            rows.append(r)
    return rows


summary = {}
for quality in ("exact", "high"):
    rows = parse(quality)
    for task in sorted({r["task"] for r in rows}):
        measured = [r for r in rows if r["task"] == task and r["label"].startswith("measure")]
        snap = lambda r: r["metrics"]["memory_snapshots"]["after_forward"]
        summary[quality, task] = {
            "n": len(measured),
            "e2e_s": statistics.mean(r["elapsed_saved_request_s"] for r in measured),
            "denoise_s": statistics.mean(r["stages"]["SanaVideo2DenoisingStage"] for r in measured),
            "decode_s": statistics.mean(r["stages"]["DecodingStage"] for r in measured),
            "steps": measured[0]["steps"],
            "peak_reserved_mib": max(snap(r)["peak_reserved_mb"] for r in measured),
            "peak_allocated_mib": max(snap(r)["peak_allocated_mb"] for r in measured),
            "rgb_sha_set": sorted({r["decoded_rgb_sha256"][:12] for r in measured}),
        }
for task in sorted({t for _, t in summary}):
    a, b = summary["exact", task], summary["high", task]
    print(f"== {tag} {task} (n={a['n']}/{b['n']}, {a['steps']} steps)")
    for key in ("e2e_s", "denoise_s", "decode_s"):
        print(f"  {key:10s} exact {a[key]:9.3f}  high {b[key]:9.3f}  change {100 * (b[key] / a[key] - 1):+6.2f}%")
    print(f"  denoise ms/step  exact {1000 * a['denoise_s'] / a['steps']:.1f}  high {1000 * b['denoise_s'] / b['steps']:.1f}")
    print(f"  peak reserved MiB exact {a['peak_reserved_mib']:.0f}  high {b['peak_reserved_mib']:.0f}; allocated {a['peak_allocated_mib']:.0f} / {b['peak_allocated_mib']:.0f}")
    print(f"  repeat-determinism: exact hashes {a['rgb_sha_set']}  high hashes {b['rgb_sha_set']}")
    cmp = task_dir / "logs" / f"{tag}-compare-{task}.json"
    if cmp.is_file():
        c = json.loads(cmp.read_text())
        print(f"  exact vs high: identical={c['identical']} SSIM mean/min {c['ssim_mean']:.4f}/{c['ssim_min']:.4f} (frame {c['ssim_argmin']})"
              f"  PSNR mean/min {c['psnr_mean']:.2f}/{c['psnr_min']:.2f} dB (frame {c['psnr_argmin']})  pass={c['pass_every_frame']}")
