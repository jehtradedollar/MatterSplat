"""Check a COLMAP text model by re-reading it from disk and reprojecting points3D.

    python -m mattersplat.verify_colmap <dataset dir> [--sweeps 5]

Uses only the written files and standard COLMAP/OpenCV math (x_cam = R(q) X + t,
u = fx x/z + cx, v = fy y/z + cy), so it is independent of the exporter's in-memory poses.
For a spread of sweeps it writes into <dir>/verify/:
    <image>_overlay.jpg   photo with visible points drawn, coloured by depth
    <image>_compare.jpg   photo | z-buffered point-colour render, side by side
    summary.json          mean / median |photo - point colour| per image
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np
from PIL import Image, ImageDraw


def qvec_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
    ])


def read_cameras(path):
    cams = {}
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        assert p[1] == "PINHOLE", f"unsupported model {p[1]}"
        cams[int(p[0])] = dict(w=int(p[2]), h=int(p[3]), fx=float(p[4]), fy=float(p[5]), cx=float(p[6]), cy=float(p[7]))
    return cams


def read_images(path):
    lines = [l.rstrip("\n") for l in open(path) if not l.startswith("#")]
    images = {}
    for i in range(0, len(lines) - 1, 2):
        p = lines[i].split()
        if not p:
            continue
        images[p[9]] = dict(id=int(p[0]), q=np.array(p[1:5], float), t=np.array(p[5:8], float), cam=int(p[8]))
    return images


def read_points3d(path):
    data = np.loadtxt(path, comments="#", usecols=range(7))
    return data[:, 1:4], data[:, 4:7]


def depth_colors(d):
    """Cheap blue -> green -> red ramp over the 2nd-98th percentile."""
    lo, hi = np.percentile(d, [2, 98])
    t = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    return (np.stack([np.clip(2 * t - 1, 0, 1), 1 - np.abs(2 * t - 1), np.clip(1 - 2 * t, 0, 1)], 1) * 255).astype(np.uint8)


def project_visible(X, cam, R, t, spacing=0.05, cell=8, max_radius=8):
    """Pixel coords, depth and indices of points that pass a z-buffer.

    The cloud is voxel-thinned, so near surfaces are sparse and far points would show
    through the gaps. Each point therefore occludes a square footprint of about
    `spacing` metres (fx * spacing / z pixels, in `cell`-pixel blocks, capped)."""
    Xc = X @ R.T + t
    z = Xc[:, 2]
    front = z > 0.05
    u = cam["fx"] * Xc[front, 0] / z[front] + cam["cx"]
    v = cam["fy"] * Xc[front, 1] / z[front] + cam["cy"]
    idx = np.flatnonzero(front)
    inside = (u >= 0) & (u < cam["w"]) & (v >= 0) & (v < cam["h"])
    u, v, idx, z = u[inside], v[inside], idx[inside], z[front][inside]
    gw, gh = (cam["w"] + cell - 1) // cell, (cam["h"] + cell - 1) // cell
    gx, gy = (u // cell).astype(np.int64), (v // cell).astype(np.int64)
    rad = np.minimum(np.ceil(cam["fx"] * spacing / z / cell / 2).astype(np.int64), max_radius)
    zmin = np.full(gw * gh, np.inf)
    for r in np.unique(rad):
        m = rad == r
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                x, y = gx[m] + dx, gy[m] + dy
                ok = (x >= 0) & (x < gw) & (y >= 0) & (y < gh)
                np.minimum.at(zmin, y[ok] * gw + x[ok], z[m][ok])
    vis = z <= zmin[gy * gw + gx] * 1.05 + 2 * spacing
    return u[vis], v[vis], z[vis], idx[vis]


def rot_y(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def colour_error(photo, X, C, cam, R, t, spacing):
    u, v, z, idx = project_visible(X, cam, R, t, spacing)
    px = photo[v.astype(int), u.astype(int)].astype(float)
    return u, v, z, idx, np.abs(px - C[idx]).mean(1)


def splat_render(shape, u, v, z, colors, fx, spacing, scale, max_r=10):
    """Z-buffered render with each point drawn as a square about `spacing` metres wide, so
    sparse near surfaces still hide what is behind them."""
    h, w = shape
    x, y = (u * scale).astype(np.int64), (v * scale).astype(np.int64)
    r = np.clip((fx * spacing / z * scale / 2).astype(np.int64), 0, max_r)
    zbuf = np.full(h * w, np.inf)
    hits = []
    for dy in range(-max_r, max_r + 1):
        for dx in range(-max_r, max_r + 1):
            m = np.flatnonzero(r >= max(abs(dx), abs(dy)))
            if not len(m):
                continue
            xx, yy = x[m] + dx, y[m] + dy
            ok = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
            pix, m = yy[ok] * w + xx[ok], m[ok]
            np.minimum.at(zbuf, pix, z[m])
            hits.append((pix, m))
    img = np.zeros((h * w, 3), np.uint8)
    for pix, m in hits:
        win = z[m] <= zbuf[pix]
        img[pix[win]] = colors[m[win]]
    return img.reshape(h, w, 3)


def sweep_points(e57, sweep_index, sample=400_000):
    """Raw valid points of one sweep from the E57 (world frame), randomly sampled."""
    from .e57_reader import E57File
    with E57File(e57) as f:
        P, C = f.read_scan(sweep_index)
    sel = np.random.default_rng(sweep_index).permutation(len(P))[:sample]
    return P[sel], C[sel].astype(float)


def own_point_error(photo, P, C, cam, R, t):
    """|photo - colour| for a sweep's own points, seen from its own centre (no occlusion test
    needed: every scan point is the first hit along its ray)."""
    Xc = P @ R.T + t
    z = Xc[:, 2]
    m = z > 0.05
    u = cam["fx"] * Xc[m, 0] / z[m] + cam["cx"]
    v = cam["fy"] * Xc[m, 1] / z[m] + cam["cy"]
    g = (u >= 0) & (u < cam["w"]) & (v >= 0) & (v < cam["h"])
    px = photo[v[g].astype(int), u[g].astype(int)].astype(float)
    return u[g], v[g], z[m][g], np.abs(px - C[m][g]).mean(1)


def verify(root: str, n_sweeps: int = 5, e57: str | None = None, preview: int = 1024) -> dict:
    """Reproject into a spread of sweeps' images, using only the COLMAP files for poses.

    Two checks per image:
      own   - (needs the E57) the sweep's own raw scan points. Each was coloured from these
              faces, so a correct pose gives a few DN; the 2-degree rotated control does not.
      cloud - the exported points3D through a z-buffer. Voxel averaging (colours mixed over
              sweeps, thin studs blended with background) keeps this high even when correct,
              so read it next to its own 2-degree control.
    """
    sparse = os.path.join(root, "sparse", "0")
    cams = read_cameras(os.path.join(sparse, "cameras.txt"))
    images = read_images(os.path.join(sparse, "images.txt"))
    X, C = read_points3d(os.path.join(sparse, "points3D.txt"))
    info_path = os.path.join(root, "export_info.json")
    spacing = json.load(open(info_path))["voxel"] if os.path.exists(info_path) else 0.05
    out_dir = os.path.join(root, "verify")
    os.makedirs(out_dir, exist_ok=True)

    sweeps = sorted({re.match(r"(sweep\d+)_", n).group(1) for n in images if re.match(r"sweep\d+_", n)})
    picked = [sweeps[i] for i in np.unique(np.linspace(0, len(sweeps) - 1, n_sweeps).round().astype(int))]
    print(f"Verifying {len(picked)} sweeps ({', '.join(picked)}); cloud {len(X):,} points"
          + ("" if e57 else "; no --e57, so only the cloud check runs"))
    results = {}
    off = rot_y(2.0)
    for sweep in picked:
        own = sweep_points(e57, int(sweep[5:])) if e57 else None
        for name in sorted(n for n in images if n.startswith(sweep + "_")):
            im = images[name]
            cam = cams[im["cam"]]
            R = qvec_to_rotmat(im["q"] / np.linalg.norm(im["q"]))
            photo = np.asarray(Image.open(os.path.join(root, "images", name)).convert("RGB"))
            r = {}
            if own is not None:
                ou, ov, oz, oe = own_point_error(photo, *own, cam, R, im["t"])
                *_, oe_off = own_point_error(photo, *own, cam, off @ R, off @ im["t"])
                r.update(own_points=int(len(oe)), own_err=float(oe.mean()), own_err_rotated_2deg=float(oe_off.mean()))
            u, v, z, idx, err = colour_error(photo, X, C, cam, R, im["t"], spacing)
            *_, err_off = colour_error(photo, X, C, cam, off @ R, off @ im["t"], spacing)
            r.update(cloud_points=int(len(idx)), cloud_err=float(err.mean()) if len(err) else None,
                     cloud_err_rotated_2deg=float(err_off.mean()) if len(err_off) else None)
            results[name] = r
            print(f"  {name}: " + (f"own {r['own_err']:.1f} (2deg {r['own_err_rotated_2deg']:.1f})  " if own is not None else "")
                  + f"cloud {r['cloud_err'] or float('nan'):.1f} (2deg {r['cloud_err_rotated_2deg'] or float('nan'):.1f})")

            s = preview / max(cam["w"], cam["h"])
            size = (round(cam["w"] * s), round(cam["h"] * s))
            small = Image.fromarray(photo).resize(size, Image.LANCZOS)
            over = Image.blend(small, Image.new("RGB", size), 0.35)
            draw = ImageDraw.Draw(over)
            du, dv, dz = (ou, ov, oz) if own is not None else (u, v, z)
            sel = np.random.default_rng(0).permutation(len(du))[:30000]
            for (x, y), col in zip(np.c_[du[sel], dv[sel]] * s, depth_colors(dz)[sel]):
                draw.point((x, y), fill=tuple(int(c) for c in col))
            over.save(os.path.join(out_dir, name.replace(".jpg", "_overlay.jpg")), quality=90)

            render = splat_render((size[1], size[0]), u, v, z, C[idx].astype(np.uint8), cam["fx"], spacing, s)
            both = Image.new("RGB", (size[0] * 2, size[1]))
            both.paste(small, (0, 0))
            both.paste(Image.fromarray(render), (size[0], 0))
            both.save(os.path.join(out_dir, name.replace(".jpg", "_compare.jpg")), quality=90)

    def avg(key):
        vals = [r[key] for r in results.values() if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    summary = dict(sweeps=picked, images=results,
                   **{k: avg(k) for k in ("own_err", "own_err_rotated_2deg", "cloud_err", "cloud_err_rotated_2deg")})
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    fmt = lambda x: "n/a" if x is None else f"{x:.1f}"
    print(f"Mean colour error over {len(results)} images: own-sweep points {fmt(summary['own_err'])} "
          f"(2deg control {fmt(summary['own_err_rotated_2deg'])}), cloud {fmt(summary['cloud_err'])} "
          f"(2deg control {fmt(summary['cloud_err_rotated_2deg'])}) -> {out_dir}")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--sweeps", type=int, default=5)
    ap.add_argument("--e57", help="source E57, enables the per-sweep raw point check")
    args = ap.parse_args(argv)
    verify(args.root, args.sweeps, args.e57)


if __name__ == "__main__":
    sys.exit(main())
