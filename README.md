# MatterSplat

Matterport E57 export -> images + point cloud in 3D -> Gaussian splat (trained with Spirula Studio).

## Setup

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

## Step 1: read and inspect an E57

```
.venv\Scripts\python -m mattersplat.inspect path\to\cloud_0.e57 --out out\cloud_0
```

Prints a summary and writes `images/` (every images2D blob as stored), `cloud_preview.ply`
(2% random subset of valid points, world frame, RGB), `poses.json` and `summary.txt`.
`--check-scans 0 50` projects those scans' coloured points into their own images and reports
the mean colour error, a quick check that poses and the camera convention agree.

Library use: `mattersplat.e57_reader.E57File` (lazy: per-scan points, per-image bytes) or
`read_e57(path, fraction=...)` for a one-shot dict.

### Conventions (as stored in the E57, not converted)

- Poses are quaternion `(w, x, y, z)` + translation and map local -> world:
  `p_world = R(q) p_local + t`. Scan points are stored in the scanner frame.
- Pinhole images use OpenGL camera axes: look down -Z, +X right, +Y up. With
  `f = focalLength / pixelWidth`: `u = cx + f x / -z`, `v = cy - f y / -z`, pixel (0,0) top-left.
  For OpenCV/COLMAP use `R_cv = R(q) diag(1, -1, -1)`.

### What a Matterport Pro3 export looks like

- One data3D per sweep, each a 1800 x 3600 (0.1 deg) spherical grid stored as cartesian XYZ + RGB,
  with `cartesianInvalidState` marking ~30% of cells empty.
- Six images2D per sweep ("Skybox 0..5"): 4096 x 4096 JPEG cube faces, 90 deg pinhole
  (f = 2048 px), sharing the sweep's position. Face 0 looks up, 1-4 horizontal, 5 down; faces
  are gravity-levelled. Linked to the sweep by `associatedData3DGuid`.
- World frame is Z-up, metres.
