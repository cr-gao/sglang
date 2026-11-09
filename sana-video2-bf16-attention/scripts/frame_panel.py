"""Side-by-side frame panel: reference | test | 8x amplified abs diff, at chosen frame indices."""

import sys

import av
import cv2
import numpy as np


def frames(path):
    with av.open(path) as c:
        return [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]


ref_path, test_path, out_path, *idx = sys.argv[1:]
ref, test = frames(ref_path), frames(test_path)
rows = []
for i in map(int, idx):
    a, b = ref[i], test[i]
    diff = np.clip(np.abs(a.astype(np.int16) - b.astype(np.int16)) * 8, 0, 255).astype(np.uint8)
    row = np.concatenate([a, b, diff], axis=1)
    cv2.putText(row, f"frame {i}: exact | high | 8x |diff|", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    rows.append(row)
panel = np.concatenate(rows, axis=0)
cv2.imwrite(out_path, cv2.cvtColor(panel, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
print(out_path, panel.shape)
