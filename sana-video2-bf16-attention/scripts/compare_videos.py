"""Per-frame PSNR/SSIM between two MP4s (skimage-default SSIM: 7x7 uniform window, per channel)."""

import json
import sys

import av
import cv2
import numpy as np


def frames(path):
    with av.open(path) as container:
        return [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]


def ssim(a, b, win=7, k1=0.01, k2=0.03, data_range=255.0):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    np_ = win * win
    cov_norm = np_ / (np_ - 1)
    blur = lambda x: cv2.blur(x, (win, win), borderType=cv2.BORDER_REFLECT)
    ux, uy = blur(a), blur(b)
    uxx, uyy, uxy = blur(a * a), blur(b * b), blur(a * b)
    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)
    c1, c2 = (k1 * data_range) ** 2, (k2 * data_range) ** 2
    s = ((2 * ux * uy + c1) * (2 * vxy + c2)) / ((ux * ux + uy * uy + c1) * (vx + vy + c2))
    pad = (win - 1) // 2
    return float(s[pad:-pad, pad:-pad].mean())


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255.0**2 / mse))


def main(ref_path, test_path, ssim_min=0.92, psnr_min=24.0):
    ref, test = frames(ref_path), frames(test_path)
    assert len(ref) == len(test), (len(ref), len(test))
    ssims = [ssim(r, t) for r, t in zip(ref, test)]
    psnrs = [psnr(r, t) for r, t in zip(ref, test)]
    identical = all(np.array_equal(r, t) for r, t in zip(ref, test))
    report = {
        "reference": ref_path, "test": test_path, "frames": len(ref), "identical": identical,
        "ssim_mean": float(np.mean(ssims)), "ssim_min": float(np.min(ssims)),
        "ssim_argmin": int(np.argmin(ssims)),
        "psnr_mean": float(np.mean(psnrs)), "psnr_min": float(np.min(psnrs)),
        "psnr_argmin": int(np.argmin(psnrs)),
        "pass_every_frame": bool(min(ssims) >= ssim_min and min(psnrs) >= psnr_min),
        "per_frame_ssim": ssims, "per_frame_psnr": psnrs,
    }
    print(json.dumps(report))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
