"""Score Spirula's held-out renders (eval-render-*.png vs eval-gt-*.png) beyond PSNR/SSIM.

    python -m mattersplat.score <spirula run dir>

PSNR rewards a smooth bright blur on these bright, detailed interiors (a held-out face filled by
a floater can out-score a sharp, correct one), so two structure-aware scores are added:
    lpips      LPIPS (AlexNet), lower is better; needs `pip install torch lpips` (CPU is fine)
    grad_corr  Pearson correlation of Sobel gradient magnitudes, higher is better: 0 when the
               render has no edges where the photo does
Images are compared at 512 px.
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
from PIL import Image

SIZE = 512
_lpips = None


def _load(path):
    return np.asarray(Image.open(path).convert("RGB").resize((SIZE, SIZE), Image.LANCZOS), dtype=np.float32) / 255


def grad_corr(a, b):
    def mag(x):
        g = x.mean(2)
        gx = g[1:-1, 2:] - g[1:-1, :-2] + 0.5 * (g[:-2, 2:] - g[:-2, :-2] + g[2:, 2:] - g[2:, :-2])
        gy = g[2:, 1:-1] - g[:-2, 1:-1] + 0.5 * (g[2:, :-2] - g[:-2, :-2] + g[2:, 2:] - g[:-2, 2:])
        return np.hypot(gx, gy).ravel()
    return float(np.corrcoef(mag(a), mag(b))[0, 1])


def lpips_score(a, b):
    global _lpips
    try:
        import torch
        import lpips
    except ImportError:
        return None
    if _lpips is None:
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        _lpips = lpips.LPIPS(net="alex", verbose=False)
    t = lambda x: torch.from_numpy(x).permute(2, 0, 1)[None] * 2 - 1
    with torch.no_grad():
        return float(_lpips(t(a), t(b)).item())


def score_run(run_dir: str) -> dict:
    """Per-image and mean scores; cached in <run_dir>/scores.json."""
    cache = os.path.join(run_dir, "scores.json")
    if os.path.exists(cache):
        return json.load(open(cache))
    per = []
    for gt_path in sorted(glob.glob(os.path.join(run_dir, "eval-gt-*.png"))):
        a, b = _load(gt_path), _load(gt_path.replace("eval-gt-", "eval-render-"))
        mse = float(np.mean((a - b) ** 2))
        per.append({"psnr_512": -10 * np.log10(max(mse, 1e-10)), "grad_corr": grad_corr(a, b),
                    "lpips": lpips_score(a, b)})
    mean = {k: float(np.mean([p[k] for p in per])) for k in per[0] if per[0][k] is not None} if per else {}
    out = {"mean": mean, "images": per}
    with open(cache, "w") as fh:
        json.dump(out, fh, indent=1)
    return out


if __name__ == "__main__":
    print(json.dumps(score_run(sys.argv[1])["mean"], indent=1))
