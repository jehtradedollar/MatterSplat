"""Train a Gaussian splat from a Matterport E57 with Spirula Studio.

    python -m mattersplat.train <file.e57> --out <dir> [--spirula <spirula.exe>]
        [--dataset <existing spirula e57 import>] [--iterations 30000]
        [--holdout-sweeps [N ...] | --holdout N] [--no-crop]

Steps: `spirula e57` import into <out>/dataset (skipped if already there), crop the seed cloud
to the sweep box, `spirula train 3dgs` with mattersplat.spirula.TRAIN_FLAGS, then copy the result
to <out>/splat.ply. By default every face trains. --holdout-sweeps holds out all 6 faces of the
given sweeps (no values: mattersplat.spirula.DEFAULT_HOLDOUT_SWEEPS) and --holdout N every Nth
face; the held-out rendered vs reference images land in <out>/renders/ and the scores in
train_summary.json.
spirula.exe is found from --spirula, the MATTERSPLAT_SPIRULA env var, or the default install path.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sys
import time

import numpy as np

from . import spirula


def collect_eval(run_dir: str, renders: str) -> list[str]:
    """Copy whatever eval images Spirula wrote into renders/."""
    files = [f for f in glob.glob(os.path.join(run_dir, "**", "*"), recursive=True)
             if f.lower().endswith((".png", ".jpg")) and re.search(r"eval|render|test", f, re.I)]
    os.makedirs(renders, exist_ok=True)
    out = []
    for f in files:
        dst = os.path.join(renders, os.path.relpath(f, run_dir).replace(os.sep, "__"))
        shutil.copyfile(f, dst)
        out.append(dst)
    return out


def eval_metrics(log_path: str) -> dict:
    """Last value of each psnr/ssim/lpips that appears in the training log."""
    text = open(log_path, encoding="utf-8", errors="replace").read()
    found = {}
    for key in ("psnr", "ssim", "lpips"):
        vals = re.findall(rf"\b{key}\b\s*[=:]\s*([0-9.]+)", text, re.I)
        if vals:
            found[key] = float(vals[-1])
    return found


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("e57")
    ap.add_argument("--out", required=True)
    ap.add_argument("--spirula", help="path to spirula.exe")
    ap.add_argument("--iterations", type=int, help="training steps (Spirula default 30000)")
    ap.add_argument("--dataset", help="reuse this spirula e57 import instead of importing into <out>/dataset")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--holdout-sweeps", type=int, nargs="*", help="hold out every face of these sweeps "
                   "(no values: the default 8 sweeps)")
    g.add_argument("--holdout", type=int, default=0, help="hold out every Nth face instead")
    ap.add_argument("--no-crop", action="store_true", help="keep the seed cloud's long-range points")
    args = ap.parse_args(argv)

    exe = spirula.find_spirula(args.spirula)
    os.makedirs(args.out, exist_ok=True)
    dataset = args.dataset or os.path.join(args.out, "dataset")
    t0 = time.time()
    spirula.import_e57(exe, args.e57, dataset, args.out)
    summary = {"spirula": exe, "import_minutes": (time.time() - t0) / 60}
    if not args.no_crop:
        kept, dropped, box = spirula.crop_seed(dataset)
        print(f"Seed cloud cropped to {np.round(box, 2).tolist()}: kept {kept:,}, dropped {dropped:,}")
        summary.update(seed_kept=kept, seed_dropped=dropped, crop_box=box.tolist())

    sweeps = None
    if args.holdout_sweeps is not None:
        sweeps = args.holdout_sweeps or list(spirula.DEFAULT_HOLDOUT_SWEEPS)
    res = spirula.train(exe, dataset, args.out, args.iterations, args.holdout, sweeps)
    summary.update(res)
    if not summary["metrics"]:
        summary["metrics"] = eval_metrics(os.path.join(args.out, "spirula_train.log"))
    if args.holdout or sweeps:
        summary["renders"] = len(collect_eval(os.path.join(args.out, "train"), os.path.join(args.out, "renders")))
    with open(os.path.join(args.out, "train_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"splat.ply: {res['splats']:,} splats, {res['splat_mb']:.0f} MB; trained in {res['minutes']:.1f} min, "
          f"GPU peak {res['gpu_peak_mib']} MiB; metrics {summary['metrics']}")


if __name__ == "__main__":
    sys.exit(main())
