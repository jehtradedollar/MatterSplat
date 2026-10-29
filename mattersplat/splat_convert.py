"""Convert a 3DGS .ply to the compact .splat format, and write the sweep list the viewer uses.

.splat (antimatter15 layout, read by @mkkellogg/gaussian-splats-3d and most web viewers), 32 bytes
per splat, sorted by size x opacity so the important ones stream in first:
    position float32 x3 | scale float32 x3 (linear, exp of the PLY log-scale)
    colour uint8 RGBA (SH DC -> sRGB, sigmoid opacity) | rotation uint8 x4 (w,x,y,z; q*128+128)
View-dependent colour (SH bands above DC) is dropped: ~8x smaller (1M splats: 248 MB -> 32 MB).

    python -m mattersplat.splat_convert <splat.ply> [<out.splat>]
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

SH_C0 = 0.28209479177387814


def read_gaussian_ply(path: str) -> np.ndarray:
    with open(path, "rb") as fh:
        raw = fh.read()
    end = raw.index(b"end_header") + len(b"end_header")
    end = raw.index(b"\n", end) + 1
    header = raw[:end].decode("ascii", "replace").splitlines()
    if "format binary_little_endian 1.0" not in header:
        raise ValueError(f"{path}: only binary little-endian PLY is supported")
    types = {"float": "<f4", "double": "<f8", "uchar": "u1", "int": "<i4", "uint": "<u4"}
    props = [(l.split()[2], types[l.split()[1]]) for l in header if l.startswith("property")]
    n = int(next(l for l in header if l.startswith("element vertex")).split()[2])
    return np.frombuffer(raw, dtype=props, count=n, offset=end)


def ply_to_splat(ply: str, out: str | None = None) -> str:
    v = read_gaussian_ply(ply)
    out = out or os.path.splitext(ply)[0] + ".splat"
    scale = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1)).astype(np.float32)
    alpha = 1 / (1 + np.exp(-v["opacity"].astype(np.float64)))
    rgb = 0.5 + SH_C0 * np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1)
    rgba = np.clip(np.concatenate([rgb, alpha[:, None]], 1) * 255, 0, 255).round().astype(np.uint8)
    q = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], 1).astype(np.float64)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    rot = np.clip(q * 128 + 128, 0, 255).round().astype(np.uint8)
    order = np.argsort(-(scale.prod(1) * alpha))
    rec = np.empty(len(v), dtype=[("p", "<f4", 3), ("s", "<f4", 3), ("c", "u1", 4), ("r", "u1", 4)])
    rec["p"] = np.stack([v["x"], v["y"], v["z"]], 1)
    rec["s"], rec["c"], rec["r"] = scale, rgba, rot
    rec[order].tofile(out)
    return out


def write_sweeps(transforms_path: str, out: str) -> dict:
    """sweeps.json for the viewer from a Spirula/nerfstudio transforms.json: one entry per
    camera centre, with the horizontal face directions (OpenGL camera -Z) seen from it."""
    tf = json.load(open(transforms_path))
    sweeps: dict[tuple, dict] = {}
    for f in tf["frames"]:
        m = np.array(f["transform_matrix"])
        key = tuple(np.round(m[:3, 3], 4))
        fwd = -m[:3, 2]
        s = sweeps.setdefault(key, {"index": len(sweeps), "position": m[:3, 3].tolist(), "directions": []})
        if abs(fwd[2]) < 0.5:  # horizontal faces only: the viewer starts looking along one
            s["directions"].append(np.round(fwd, 4).tolist())
    data = {"up": [0, 0, 1], "frame": "E57 world, metres", "sweeps": list(sweeps.values())}
    with open(out, "w") as fh:
        json.dump(data, fh, indent=1)
    return data


if __name__ == "__main__":
    print(ply_to_splat(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None))
