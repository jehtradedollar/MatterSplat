"""E57 in, viewable splat out: the whole pipeline in one command.

    python -m mattersplat run <file.e57> --out <dir> [--view] [--colmap] [--spirula <exe>]
        [--iterations N] [--cap-max N] [--resolution-divisor N] [--dataset <import>]
        [--holdout-sweeps [N ...]]

1. `spirula e57` import into <out>/dataset (skipped when already there)
2. crop the seed cloud to the sweep box
3. train with mattersplat.spirula.TRAIN_FLAGS (the tuned defaults), every face by default
4. write <out>/splat.ply, the compact <out>/splat.splat and <out>/sweeps.json
5. --view: serve the viewer and open it
--colmap also writes the step 2 COLMAP model to <out>/colmap (not needed for training).
--cap-max / --resolution-divisor override the tuned splat budget (1M) and image size (2 = 2048 px);
max quality on a 12 GB card: --cap-max 3000000 --resolution-divisor 1 --iterations 60000.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from . import spirula
from .splat_convert import ply_to_splat, write_sweeps


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mattersplat run", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("e57")
    ap.add_argument("--out", required=True)
    ap.add_argument("--spirula", help="path to spirula.exe")
    ap.add_argument("--iterations", type=int, help="training steps (default 30000)")
    ap.add_argument("--cap-max", type=int, help="largest number of splats (default 1000000)")
    ap.add_argument("--resolution-divisor", type=int, help="train on images divided by this (default 2; 1 = full)")
    ap.add_argument("--dataset", help="reuse this spirula e57 import instead of <out>/dataset")
    ap.add_argument("--holdout-sweeps", type=int, nargs="*",
                    help="hold these sweeps out and score them (no values: the default 8)")
    ap.add_argument("--colmap", action="store_true", help="also export a COLMAP text model to <out>/colmap")
    ap.add_argument("--view", action="store_true", help="open the viewer when done")
    ap.add_argument("--port", type=int, default=8770)
    args = ap.parse_args(argv)

    t0 = time.time()
    exe = spirula.find_spirula(args.spirula)
    os.makedirs(args.out, exist_ok=True)
    dataset = args.dataset or os.path.join(args.out, "dataset")
    spirula.import_e57(exe, args.e57, dataset, args.out)
    kept, dropped, box = spirula.crop_seed(dataset)
    print(f"Seed cloud cropped to {np.round(box, 2).tolist()}: kept {kept:,}, dropped {dropped:,}")
    if args.colmap:
        from .colmap_export import export_colmap
        export_colmap(args.e57, os.path.join(args.out, "colmap"))

    sweeps = None
    if args.holdout_sweeps is not None:
        sweeps = args.holdout_sweeps or list(spirula.DEFAULT_HOLDOUT_SWEEPS)
    extra = {}
    if args.cap_max:
        extra["cap-max"] = args.cap_max
    if args.resolution_divisor:
        extra["train-resolution-divisor"] = args.resolution_divisor
    res = spirula.train(exe, dataset, args.out, args.iterations, holdout_sweeps=sweeps, extra=extra)
    splat = os.path.join(args.out, "splat.ply")
    ply_to_splat(splat)
    if sweeps:
        from .score import score_run
        res["scores"] = score_run(res["run_dir"])["mean"]
    write_sweeps(os.path.join(dataset, "transforms.json"), os.path.join(args.out, "sweeps.json"))
    res.update(total_minutes=(time.time() - t0) / 60, seed_kept=kept, seed_dropped=dropped, crop_box=box.tolist())
    with open(os.path.join(args.out, "train_summary.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"Done in {res['total_minutes']:.1f} min: {splat} ({res['splats']:,} splats, {res['splat_mb']:.0f} MB) "
          f"and splat.splat" + (f"; held-out metrics {res['metrics']}" if sweeps else ""))
    print(f"View it: python -m mattersplat view {splat}")
    if args.view:
        from .view import serve
        serve(splat, args.port)


if __name__ == "__main__":
    sys.exit(main())
