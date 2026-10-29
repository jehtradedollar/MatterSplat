"""Export an E57 as a COLMAP text model, then verify it from the written files.

    python -m mattersplat.to_colmap <file.e57> --out <dir> [--image-size 2048]
        [--crop xmin ymin zmin xmax ymax zmax | --no-crop] [--voxel 0.02] [--max-points 3000000]
        [--no-verify]

See mattersplat/colmap_export.py for the layout and conventions.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from .colmap_export import export_colmap
from .verify_colmap import verify


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("e57")
    ap.add_argument("--out", required=True)
    ap.add_argument("--image-size", type=int, help="resize faces so the long side is this many pixels (intrinsics scaled)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--crop", type=float, nargs=6, metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"),
                   help="world-frame box for the point cloud (default: sweep positions +3 m in XY, -2/+5 m in Z)")
    g.add_argument("--no-crop", action="store_true")
    ap.add_argument("--voxel", type=float, default=0.02, help="starting voxel size in metres")
    ap.add_argument("--max-points", type=int, default=3_000_000, help="voxel is coarsened until the cloud fits")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args(argv)

    crop = None if args.no_crop else ("auto" if args.crop is None else np.array(args.crop).reshape(2, 3))
    t0 = time.time()
    res = export_colmap(args.e57, args.out, image_size=args.image_size, crop=crop, voxel=args.voxel,
                        max_points=args.max_points)
    print(f"Exported {res.n_images} images ({res.image_size} px), {res.n_points:,} points "
          f"(voxel {res.voxel:.4f} m) in {time.time() - t0:.0f}s")
    if not args.no_verify:
        verify(args.out, e57=args.e57)


if __name__ == "__main__":
    sys.exit(main())
