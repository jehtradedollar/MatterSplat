"""Export an E57 (pinhole images + scans) as a COLMAP text model.

Layout written under <out>:
    images/sweepNNN_faceK.jpg
    sparse/0/cameras.txt, images.txt, points3D.txt, points3D.ply
    cameras_frustums.ply              camera pyramids for checking placement in CloudCompare

COLMAP conventions: images.txt holds world-to-camera (qvec w,x,y,z; tvec) in the OpenCV
camera frame (+X right, +Y down, +Z forward). The E57 faces are OpenGL cameras
(see e57_reader), so R_cv = R_gl @ diag(1, -1, -1) before inverting. The world frame is the
E57 world frame unchanged (Z-up, metres).
"""

from __future__ import annotations

import io
import json
import os
import re
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .e57_reader import E57File, ImageInfo, quat_to_matrix

GL_TO_CV = np.diag([1.0, -1.0, -1.0])


def matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """(w, x, y, z) unit quaternion with w >= 0 from a rotation matrix."""
    K = np.array([
        [R[0, 0] - R[1, 1] - R[2, 2], 0, 0, 0],
        [R[1, 0] + R[0, 1], R[1, 1] - R[0, 0] - R[2, 2], 0, 0],
        [R[2, 0] + R[0, 2], R[2, 1] + R[1, 2], R[2, 2] - R[0, 0] - R[1, 1], 0],
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1], R[0, 0] + R[1, 1] + R[2, 2]],
    ]) / 3.0
    vals, vecs = np.linalg.eigh(K)  # reads the lower triangle only
    x, y, z, w = vecs[:, np.argmax(vals)]
    q = np.array([w, x, y, z])
    return -q if q[0] < 0 else q


def world_to_camera_cv(image: ImageInfo) -> tuple[np.ndarray, np.ndarray]:
    """(R_w2c, t_w2c) in the OpenCV camera convention from an E57 OpenGL camera pose."""
    R_c2w = quat_to_matrix(image.rotation) @ GL_TO_CV
    R_w2c = R_c2w.T
    return R_w2c, -R_w2c @ image.translation


def default_crop(f: E57File, pad_xy: float = 3.0, below: float = 2.0, above: float = 5.0) -> np.ndarray:
    """Axis-aligned box [min xyz, max xyz] around the sweep positions.

    Sweeps sit on a tripod ~1.3 m up, so `below` reaches past the floor and `above` covers
    the ceiling. Drops the long-range returns Matterport records through openings."""
    t = np.array([s.translation for s in f.scans])
    lo = t.min(0) - [pad_xy, pad_xy, below]
    hi = t.max(0) + [pad_xy, pad_xy, above]
    return np.stack([lo, hi])


def _voxel_reduce(keys, xyz_sum, rgb_sum, count):
    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    return (keys[starts], np.add.reduceat(xyz_sum[order], starts), np.add.reduceat(rgb_sum[order], starts),
            np.add.reduceat(count[order], starts))


def _voxel_keys(pts: np.ndarray, origin: np.ndarray, voxel: float) -> np.ndarray:
    # 21 bits per axis centred on origin: +-1M voxels, i.e. +-20 km at 2 cm
    ijk = (np.floor((pts - origin) / voxel).astype(np.int64) + (1 << 20)) & ((1 << 21) - 1)
    return (ijk[:, 0] << 42) | (ijk[:, 1] << 21) | ijk[:, 2]


def build_point_cloud(f: E57File, crop: np.ndarray | None, voxel: float, max_points: int,
                      log=print) -> tuple[np.ndarray, np.ndarray, float]:
    """Voxel-averaged coloured cloud over all scans, cropped. Coarsens the voxel until the
    cloud fits in max_points. Returns (xyz float64, rgb uint8, final voxel size)."""
    origin = crop[0] if crop is not None else None
    acc = None
    for i in range(len(f.scans)):
        pts, cols = f.read_scan(i)
        if cols is None:
            cols = np.full_like(pts, 128, dtype=np.uint8)
        if crop is not None:
            keep = np.all((pts >= crop[0]) & (pts <= crop[1]), axis=1)
            pts, cols = pts[keep], cols[keep]
        if origin is None:
            origin = pts.mean(0)
        part = _voxel_reduce(_voxel_keys(pts, origin, voxel), pts, cols.astype(np.float64),
                             np.ones(len(pts), dtype=np.int64))
        acc = part if acc is None else _voxel_reduce(*(np.concatenate([a, b]) for a, b in zip(acc, part)))
        if i % 10 == 0 or i == len(f.scans) - 1:
            log(f"  scan {i + 1}/{len(f.scans)}: {len(acc[0]):,} voxels at {voxel:g} m")
    keys, xyz_sum, rgb_sum, count = acc
    while len(keys) > max_points:
        voxel *= 1.25
        centroids = xyz_sum / count[:, None]
        keys, xyz_sum, rgb_sum, count = _voxel_reduce(_voxel_keys(centroids, origin, voxel), xyz_sum, rgb_sum, count)
        log(f"  coarsened to {voxel:.4f} m: {len(keys):,} points")
    xyz = xyz_sum / count[:, None]
    rgb = np.clip(np.rint(rgb_sum / count[:, None]), 0, 255).astype(np.uint8)
    return xyz, rgb, voxel


def write_ply(path: str, pts: np.ndarray, cols: np.ndarray | None, faces: np.ndarray | None = None):
    """Binary PLY with optional uchar RGB per vertex and triangle faces."""
    props = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if cols is not None:
        props += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    rec = np.empty(len(pts), dtype=props)
    rec["x"], rec["y"], rec["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    if cols is not None:
        rec["red"], rec["green"], rec["blue"] = cols[:, 0], cols[:, 1], cols[:, 2]
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(pts)}"]
    header += [f"property {'float' if t == '<f4' else 'uchar'} {n}" for n, t in props]
    if faces is not None:
        header += [f"element face {len(faces)}", "property list uchar int vertex_indices"]
    header.append("end_header")
    with open(path, "wb") as fh:
        fh.write(("\n".join(header) + "\n").encode())
        fh.write(rec.tobytes())
        if faces is not None:
            frec = np.empty(len(faces), dtype=[("n", "u1"), ("v", "<i4", (3,))])
            frec["n"], frec["v"] = 3, faces
            fh.write(frec.tobytes())


FACE_COLORS = np.array([[230, 60, 60], [60, 180, 60], [60, 90, 230], [230, 200, 40], [200, 60, 200], [40, 200, 200]],
                       dtype=np.uint8)


def frustum_mesh(cams: list[tuple[np.ndarray, np.ndarray, int]], fov_half_tan: float = 1.0, size: float = 0.25):
    """Pyramid per camera (apex at centre, base `size` m ahead along +Z of the OpenCV frame).
    cams: (R_c2w_cv, centre, colour index). The base's top edge (image row 0) gets a small
    notch triangle so the image 'up' direction is visible."""
    verts, cols, faces = [], [], []
    for R, c, k in cams:
        s = size * fov_half_tan
        corners = np.array([[-s, -s, size], [s, -s, size], [s, s, size], [-s, s, size], [0, -1.6 * s, size]])
        base = len(verts)
        verts += [c] + list(corners @ R.T + c)
        cols += [FACE_COLORS[k % len(FACE_COLORS)]] * 6
        faces += [[base, base + 1 + j, base + 1 + (j + 1) % 4] for j in range(4)]
        faces.append([base + 1, base + 2, base + 5])  # notch on the top (v = 0) edge
    return np.array(verts), np.array(cols, dtype=np.uint8), np.array(faces, dtype=np.int32)


@dataclass
class ExportResult:
    n_images: int
    n_points: int
    voxel: float
    crop: np.ndarray | None
    image_size: int


def _face_index(im: ImageInfo, fallback: int) -> int:
    m = re.search(r"(\d+)\s*$", im.name)
    return int(m.group(1)) if m else fallback


def export_colmap(e57_path: str, out: str, image_size: int | None = None, crop="auto", voxel: float = 0.02,
                  max_points: int = 3_000_000, jpeg_quality: int = 95, log=print) -> ExportResult:
    sparse = os.path.join(out, "sparse", "0")
    img_dir = os.path.join(out, "images")
    os.makedirs(sparse, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    with E57File(e57_path) as f:
        scan_by_guid = {s.guid: s for s in f.scans}
        pinholes = [im for im in f.images if im.representation == "pinholeRepresentation" and im.blob_format]
        if crop == "auto":
            crop = default_crop(f)
        log(f"Crop box: {None if crop is None else np.round(crop, 2).tolist()}")

        # cameras: one per distinct (size, intrinsics) after optional resize
        cam_ids, cam_lines = {}, []
        image_lines, frusta = [], []
        per_scan_counter: dict[str, int] = {}
        for n, im in enumerate(pinholes):
            k = im.intrinsics
            scale = 1.0 if not image_size else image_size / max(im.width, im.height)
            w, h = round(im.width * scale), round(im.height * scale)
            fx = k["focalLength"] / k["pixelWidth"] * scale
            fy = k["focalLength"] / k.get("pixelHeight", k["pixelWidth"]) * scale
            cx, cy = k["principalPointX"] * scale, k["principalPointY"] * scale
            key = (w, h, round(fx, 6), round(fy, 6), round(cx, 6), round(cy, 6))
            if key not in cam_ids:
                cam_ids[key] = len(cam_ids) + 1
                cam_lines.append(f"{cam_ids[key]} PINHOLE {w} {h} {fx:.10g} {fy:.10g} {cx:.10g} {cy:.10g}")
            scan = scan_by_guid.get(im.scan_guid)
            sidx = scan.index if scan else 999
            fallback = per_scan_counter.get(im.scan_guid, 0)
            per_scan_counter[im.scan_guid] = fallback + 1
            face = _face_index(im, fallback)
            name = f"sweep{sidx:03d}_face{face}.jpg"

            data = f.read_image_bytes(im.index)
            path = os.path.join(img_dir, name)
            if scale == 1.0 and im.blob_format == "jpegImage":
                with open(path, "wb") as fh:
                    fh.write(data)
            else:
                Image.open(io.BytesIO(data)).convert("RGB").resize((w, h), Image.LANCZOS).save(path, quality=jpeg_quality)

            R_w2c, t_w2c = world_to_camera_cv(im)
            q = matrix_to_quat(R_w2c)
            image_lines.append(f"{n + 1} {' '.join(f'{v:.12g}' for v in q)} {' '.join(f'{v:.12g}' for v in t_w2c)} "
                               f"{cam_ids[key]} {name}\n")
            frusta.append((R_w2c.T, im.translation, face))
            if (n + 1) % 60 == 0:
                log(f"  images {n + 1}/{len(pinholes)}")

        with open(os.path.join(sparse, "cameras.txt"), "w") as fh:
            fh.write("# Camera list with one line of data per camera:\n"
                     "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
                     f"# Number of cameras: {len(cam_lines)}\n")
            fh.write("\n".join(cam_lines) + "\n")
        with open(os.path.join(sparse, "images.txt"), "w") as fh:
            fh.write("# Image list with two lines of data per image:\n"
                     "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
                     "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
                     f"# Number of images: {len(image_lines)}, mean observations per image: 0\n")
            for line in image_lines:
                fh.write(line + "\n")  # empty POINTS2D line

        log("Building point cloud")
        xyz, rgb, voxel = build_point_cloud(f, crop, voxel, max_points, log=log)
        with open(os.path.join(sparse, "points3D.txt"), "w") as fh:
            fh.write("# 3D point list with one line of data per point:\n"
                     "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
                     f"# Number of points: {len(xyz)}, mean track length: 0\n")
            ids = np.arange(1, len(xyz) + 1)
            np.savetxt(fh, np.column_stack([ids, xyz, rgb, np.zeros(len(xyz))]),
                       fmt=["%d", "%.5f", "%.5f", "%.5f", "%d", "%d", "%d", "%.1f"])
        write_ply(os.path.join(sparse, "points3D.ply"), xyz, rgb)
        v, c, fc = frustum_mesh(frusta)
        write_ply(os.path.join(out, "cameras_frustums.ply"), v, c, fc)
        size = image_size or max(pinholes[0].width, pinholes[0].height)
        with open(os.path.join(out, "export_info.json"), "w") as fh:
            json.dump({"source": os.path.abspath(e57_path), "images": len(pinholes), "image_size": size,
                       "points": len(xyz), "voxel": voxel,
                       "crop": None if crop is None else np.asarray(crop).tolist(),
                       "world_frame": "E57 world, unchanged (Z-up, metres)"}, fh, indent=1)
        return ExportResult(len(pinholes), len(xyz), voxel, crop, size)
