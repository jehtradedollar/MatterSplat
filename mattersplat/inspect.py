"""Summarise an E57 and dump its images, a preview PLY and the poses.

    python -m mattersplat.inspect <file.e57> --out <dir> [--ply-fraction 0.02]

Writes:
    <out>/images/<scan>_<image>.jpg   every images2D blob, untouched
    <out>/cloud_preview.ply           random subset of valid points, world frame, RGB
    <out>/poses.json                  scans + images with poses exactly as stored
    <out>/summary.txt                 the printed summary
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
from collections import Counter

import numpy as np
from PIL import Image

from .e57_reader import E57File, image_to_dict, project_pinhole, scan_to_dict, POSE_CONVENTION


def write_ply(path: str, pts: np.ndarray, cols: np.ndarray | None):
    n = len(pts)
    props = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    if cols is not None:
        props += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    rec = np.empty(n, dtype=props)
    rec["x"], rec["y"], rec["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    if cols is not None:
        rec["red"], rec["green"], rec["blue"] = cols[:, 0], cols[:, 1], cols[:, 2]
    types = {"f4": "float", "u1": "uchar"}
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
    header += [f"property {types[t]} {name}" for name, t in props]
    header.append("end_header")
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode())
        f.write(rec.tobytes())


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())


def reprojection_check(f: E57File, scan_index: int, sample: int = 300_000) -> list[tuple[str, float]]:
    """Mean |pixel - point colour| when projecting a scan's points into its pinhole images.
    A few DN means the stored pose/intrinsics and the documented convention agree."""
    pts, cols = f.read_scan(scan_index)
    if cols is None:
        return []
    rng = np.random.default_rng(0)
    idx = rng.choice(len(pts), min(sample, len(pts)), replace=False)
    pts, cols = pts[idx], cols[idx].astype(float)
    out = []
    for im in f.images_for_scan(f.scans[scan_index]):
        if im.representation != "pinholeRepresentation" or not im.blob_format:
            continue
        img = np.asarray(Image.open(io.BytesIO(f.read_image_bytes(im.index))).convert("RGB")).astype(float)
        uv, front = project_pinhole(im, pts)
        h, w = img.shape[:2]
        ok = front & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        if ok.sum() == 0:
            continue
        px = img[uv[ok, 1].astype(int), uv[ok, 0].astype(int)]
        out.append((im.name, float(np.abs(px - cols[ok]).mean())))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("e57")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ply-fraction", type=float, default=0.02, help="fraction of valid points kept in the preview PLY")
    ap.add_argument("--no-images", action="store_true", help="skip writing image blobs")
    ap.add_argument("--no-ply", action="store_true")
    ap.add_argument("--check-scans", type=int, nargs="*", default=[0], help="scan indices to run the reprojection check on")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    t0 = time.time()
    with E57File(args.e57) as f:
        say(f"File: {args.e57}  ({os.path.getsize(args.e57) / 1e9:.2f} GB)")
        say(f"data3D scans: {len(f.scans)}")
        total = sum(s.point_count for s in f.scans)
        say(f"  points (records incl. invalid): {total:,}")
        for fields, n in Counter(tuple(s.fields) for s in f.scans).items():
            say(f"  fields ({n} scans): {', '.join(fields)}")
        sensors = Counter((s.metadata.get("sensorVendor"), s.metadata.get("sensorModel")) for s in f.scans)
        say(f"  sensors: {dict(sensors)}")
        ib = f.scans[0].metadata.get("indexBounds") if f.scans else None
        if ib:
            say(f"  scan 0 index grid: rows {ib['rowMinimum']}-{ib['rowMaximum']}, cols {ib['columnMinimum']}-{ib['columnMaximum']}")
        for s in f.scans[:3]:
            say(f"  [{s.index}] {s.name!r} n={s.point_count:,} q(wxyz)={np.round(s.rotation, 4).tolist()} t={np.round(s.translation, 3).tolist()}")
        if len(f.scans) > 3:
            say(f"  ... {len(f.scans) - 3} more (see poses.json)")

        say(f"images2D: {len(f.images)}")
        for (rep, w, h, fmt), n in Counter((i.representation, i.width, i.height, i.blob_format) for i in f.images).items():
            say(f"  {n} x {rep} {w}x{h} {fmt}")
        per_scan = Counter(i.scan_guid for i in f.images)
        say(f"  images per associated scan: {dict(Counter(per_scan.values()))}; unassociated: {per_scan.get(None, 0)}")
        if f.images:
            im = f.images[0]
            say(f"  image 0 intrinsics: {im.intrinsics}  focal_px={im.focal_px}")
        say(f"  image names: {sorted(set(i.name for i in f.images))[:12]}")
        say(f"Pose convention: {POSE_CONVENTION}")

        poses = {
            "file": os.path.abspath(args.e57),
            "pose_convention": POSE_CONVENTION,
            "scans": [scan_to_dict(s) for s in f.scans],
            "images": [],
        }
        scan_by_guid = {s.guid: s for s in f.scans}
        if not args.no_images:
            os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
        for im in f.images:
            d = image_to_dict(im)
            scan = scan_by_guid.get(im.scan_guid)
            d["scan_index"] = scan.index if scan else None
            if im.blob_format and not args.no_images:
                ext = ".jpg" if im.blob_format == "jpegImage" else ".png"
                prefix = f"scan{scan.index:03d}" if scan else "noscan"
                fname = f"{prefix}_{im.index:04d}_{_safe(im.name)}{ext}"
                with open(os.path.join(args.out, "images", fname), "wb") as fh:
                    fh.write(f.read_image_bytes(im.index))
                d["file"] = f"images/{fname}"
            poses["images"].append(d)
        with open(os.path.join(args.out, "poses.json"), "w") as fh:
            json.dump(poses, fh, indent=1)
        if not args.no_images:
            say(f"Wrote {len(f.images)} images to {os.path.join(args.out, 'images')}")

        for si in args.check_scans:
            if si < len(f.scans):
                res = reprojection_check(f, si)
                say(f"Reprojection colour error, scan {si}: " + ", ".join(f"{n}={e:.1f}" for n, e in res))

        if not args.no_ply:
            n_valid = 0
            pts_all, cols_all = [], []
            for i, pts, cols in f.iter_points(fraction=args.ply_fraction):
                pts_all.append(pts.astype(np.float32))
                if cols is not None:
                    cols_all.append(cols)
                n_valid += len(pts)
                if i % 10 == 0:
                    print(f"  read scan {i}/{len(f.scans)}", flush=True)
            pts = np.concatenate(pts_all)
            cols = np.concatenate(cols_all) if cols_all else None
            write_ply(os.path.join(args.out, "cloud_preview.ply"), pts, cols)
            say(f"Preview PLY: {len(pts):,} points ({args.ply_fraction:g} of valid), "
                f"world bounds min={np.round(pts.min(0), 2).tolist()} max={np.round(pts.max(0), 2).tolist()}")
            say(f"  estimated valid points in file: {int(n_valid / args.ply_fraction):,}")

    say(f"Done in {time.time() - t0:.0f}s")
    with open(os.path.join(args.out, "summary.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
