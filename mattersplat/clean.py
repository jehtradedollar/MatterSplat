"""Remove noise and loose splats from a trained 3DGS splat, using the laser scan as ground truth.

    mattersplat clean <splat.ply> --e57 <scan.e57> [--out <splat_clean.ply>]
        [--min-opacity 0.02] [--max-dist 0.15] [--big 0.5] [--sor-k 16 --sor-std 3] [--no-box]

The original is never touched; by default it writes <name>_clean.ply and <name>_clean.splat beside
it plus <name>_clean.json with the counts. Filters (each counted on its own, then combined):

  faint      opacity (after sigmoid) below --min-opacity: invisible but still sorted and drawn
  box        centre outside the sweep box (sweep positions +3 m in XY, -2/+5 m in Z, as in the
             earlier steps), the long-range junk seen through openings
  big        largest axis above --big metres AND in the top 0.1% of sizes: the streaks/smears
  unscanned  centre farther than --max-dist (+ up to 0.3 m of its own size) from every scan point:
             floaters in empty air. The scan support is the voxel-thinned (--scan-voxel) valid
             E57 cloud, cached as <splat dir>/scan_support_<voxel>.npz
  isolated   statistical outlier on splat centres: mean distance to --sor-k neighbours more than
             --sor-std standard deviations above the mean
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

from .splat_convert import ply_to_splat, read_gaussian_ply


def sweep_box(sweeps_json: str, pad_xy=3.0, below=2.0, above=5.0) -> np.ndarray:
    t = np.array([s["position"] for s in json.load(open(sweeps_json))["sweeps"]])
    return np.stack([t.min(0) - [pad_xy, pad_xy, below], t.max(0) + [pad_xy, pad_xy, above]])


def scan_support(e57: str, cache: str, box: np.ndarray | None, voxel: float) -> np.ndarray:
    if os.path.exists(cache):
        return np.load(cache)["xyz"]
    from .colmap_export import build_point_cloud
    from .e57_reader import E57File
    with E57File(e57) as f:
        xyz, _, _ = build_point_cloud(f, box, voxel, max_points=10**9)
    xyz = xyz.astype(np.float32)
    np.savez(cache, xyz=xyz)
    return xyz


def write_like(src_path: str, rec: np.ndarray, out: str):
    """Write the kept records with the source PLY's header (vertex count updated)."""
    with open(src_path, "rb") as fh:
        head = fh.read(65536)
    end = head.index(b"end_header") + len(b"end_header")
    end = head.index(b"\n", end) + 1
    lines = [f"element vertex {len(rec)}" if l.startswith("element vertex") else l
             for l in head[:end].decode("ascii").splitlines()]
    with open(out, "wb") as fh:
        fh.write(("\n".join(lines) + "\n").encode("ascii"))
        fh.write(rec.tobytes())


def clean(splat: str, e57: str | None, out: str | None = None, min_opacity=0.02, max_dist=0.15, big=0.5,
          sor_k=16, sor_std=3.0, use_box=True, scan_voxel=0.03, log=print) -> dict:
    t0 = time.time()
    out = out or os.path.splitext(splat)[0] + "_clean.ply"
    v = read_gaussian_ply(splat)
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    opacity = 1 / (1 + np.exp(-v["opacity"].astype(np.float64)))
    size = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1)).max(1)
    n = len(v)
    sweeps_json = os.path.join(os.path.dirname(os.path.abspath(splat)), "sweeps.json")
    box = sweep_box(sweeps_json) if os.path.exists(sweeps_json) else None

    drop = {}
    drop["faint"] = opacity < min_opacity
    if use_box and box is not None:
        drop["box"] = ~np.all((xyz >= box[0]) & (xyz <= box[1]), 1)
    drop["big"] = (size > big) & (size > np.quantile(size, 0.999))
    if e57:
        cache = os.path.join(os.path.dirname(os.path.abspath(splat)), f"scan_support_{scan_voxel:g}.npz")
        log(f"Scan support: {cache}" + ("" if os.path.exists(cache) else " (building from the E57, a few minutes)"))
        scan = scan_support(e57, cache, box, scan_voxel)
        d, _ = cKDTree(scan).query(xyz, k=1, workers=-1)
        drop["unscanned"] = d > max_dist + np.minimum(size, 0.3)
    tree = cKDTree(xyz)
    dk, _ = tree.query(xyz, k=sor_k + 1, workers=-1)
    mean_d = dk[:, 1:].mean(1)
    drop["isolated"] = mean_d > mean_d.mean() + sor_std * mean_d.std()

    any_drop = np.zeros(n, bool)
    counts = {}
    for k, m in drop.items():
        counts[k] = {"alone": int(m.sum()), "new": int((m & ~any_drop).sum())}
        any_drop |= m
    keep = ~any_drop
    write_like(splat, v[keep], out)
    compact = ply_to_splat(out)
    res = {"input": splat, "output": out, "compact": compact, "splats_in": n, "splats_out": int(keep.sum()),
           "removed": int(any_drop.sum()), "removed_pct": 100 * any_drop.mean(), "filters": counts,
           "params": {"min_opacity": min_opacity, "max_dist": max_dist, "big": big, "sor_k": sor_k,
                      "sor_std": sor_std, "box": None if box is None or not use_box else box.tolist(),
                      "scan_voxel": scan_voxel},
           "ply_mb": os.path.getsize(out) / 1e6, "splat_mb": os.path.getsize(compact) / 1e6,
           "seconds": time.time() - t0}
    with open(os.path.splitext(out)[0] + ".json", "w") as fh:
        json.dump(res, fh, indent=1)
    for k, c in counts.items():
        log(f"  {k:10s} {c['alone']:>9,} alone, {c['new']:>9,} not caught by an earlier filter")
    log(f"Kept {res['splats_out']:,} of {n:,} ({res['removed_pct']:.2f}% removed) -> {out} "
        f"({res['ply_mb']:.0f} MB, .splat {res['splat_mb']:.0f} MB) in {res['seconds']:.0f}s")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mattersplat clean", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("splat")
    ap.add_argument("--e57", help="scan the splat came from; enables the 'unscanned' floater filter")
    ap.add_argument("--out")
    ap.add_argument("--min-opacity", type=float, default=0.02)
    ap.add_argument("--max-dist", type=float, default=0.15, help="metres from the nearest scan point")
    ap.add_argument("--big", type=float, default=0.5, help="metres; only the top 0.1%% of sizes above it go")
    ap.add_argument("--sor-k", type=int, default=16)
    ap.add_argument("--sor-std", type=float, default=3.0)
    ap.add_argument("--scan-voxel", type=float, default=0.03)
    ap.add_argument("--no-box", action="store_true", help="keep splats outside the sweep box")
    a = ap.parse_args(argv)
    clean(a.splat, a.e57, a.out, a.min_opacity, a.max_dist, a.big, a.sor_k, a.sor_std, not a.no_box, a.scan_voxel)


if __name__ == "__main__":
    sys.exit(main())
