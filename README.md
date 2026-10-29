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

## Step 2: export a COLMAP dataset

```
.venv\Scripts\python -m mattersplat.to_colmap path\to\cloud_0.e57 --out out\cloud_0_colmap
```

Writes a COLMAP text model that Spirula (and other 3DGS trainers) read directly:

- `images/sweepNNN_faceK.jpg`: the cube faces, copied byte for byte (`--image-size 2048` resizes and scales the intrinsics).
- `sparse/0/cameras.txt`: one PINHOLE camera (4096², f = 2048, c = 2048).
- `sparse/0/images.txt`: world-to-camera poses in the OpenCV convention. The second line of each image is empty because there are no 2D observations.
- `sparse/0/points3D.txt`: the voxel-averaged coloured cloud with empty tracks. The same points are also in `points3D.ply`.
- `cameras_frustums.ply`: one pyramid per face, coloured by face index, with a notch on the image's top edge. Open it with `points3D.ply` in CloudCompare.
- `export_info.json`: crop box, voxel size and counts.
- `verify/`: overlays and `summary.json` (see below).

Options:

- `--crop xmin ymin zmin xmax ymax zmax` overrides the crop box. The default is the sweep positions padded by 3 m in XY, with 2 m below and 5 m above the scanner height; it drops long-range returns through openings. `--no-crop` keeps everything.
- `--voxel` sets the starting voxel size (default 2 cm). It is coarsened until the cloud fits `--max-points` (default 3M).
- The world frame stays the E57 frame.

Verification (`python -m mattersplat.verify_colmap <dir> --e57 <file>`, run automatically by `to_colmap`) re-reads the written files. It projects into all 6 faces of 5 sweeps spread over the site and reports two colour errors:

- **Own-sweep points:** each sweep's raw scan points, which were coloured from those same faces. This error should be a few DN, and a 2° rotated control is printed next to it.
- **Exported cloud:** checked with a footprint z-buffer. This error stays high even when the poses are correct, because 5 cm voxels average colours across sweeps and both sides of thin studs.

## Step 3: train a splat with Spirula Studio

```
.venv\Scripts\python -m mattersplat.train path\to\cloud_0.e57 --out out\cloud_0_spirula [--spirula path\to\spirula.exe]
```

Uses Spirula's own E57 import, not the COLMAP export from step 2. `spirula.exe` is found from:
1. `--spirula`
2. the `MATTERSPLAT_SPIRULA` environment variable
3. the default install path

Steps:

1. `spirula e57` writes `<out>/dataset`: the faces, `transforms.json`, 1600² depth and normal maps from the scan, and a ~500k-point seed cloud.
2. The seed cloud is cropped to the sweep box (the same default as step 2). The original is kept as `sparse_pc_full.ply`. Use `--no-crop` to skip this.
3. `spirula train 3dgs` runs with `mattersplat.spirula.TRAIN_FLAGS`, tuned for a 12 GB RTX 4070 SUPER:
   - `--train-resolution-divisor 2`: train at 2048 px
   - `--cap-max 1000000`
   - `--depth-supervision-weight 0`: the scan depth maps are loaded but left out of the loss, because 0.1 and 0.5 scored worse (see Tuning)
   - `--input-depth-is-ray-depth 0`: the maps hold planar z-depth, which matters only when depth supervision is on
   - viewer off
4. The result is copied to `<out>/splat.ply`.

Other outputs:

- By default every face trains, and `--dataset <dir>` reuses an existing import.
- `--holdout-sweeps [N ...]` keeps all 6 faces of the given sweeps out of training. With no values it uses `DEFAULT_HOLDOUT_SWEEPS`: 6, 19, 32, 45, 58, 71, 84 and 97. `--holdout N` keeps every Nth face out instead. This is a weak test, because a cube face shares no pixels with its own sweep's other faces.
- Held-out rendered and reference images go to `<out>/renders/`, and the scores go to `train_summary.json`.
- `gpu_memory.csv` logs GPU memory use (nvidia-smi, whole GPU).
- `spirula_train.log` holds the training log.

### Tuning

```
.venv\Scripts\python -m mattersplat.tune out\cloud_0_spirula\dataset --out out\cloud_0_tuning [--runs baseline depth0 ...]
```

Each variant in `tune.VARIANTS` changes one flag. All variants train on the same import with the same held-out sweeps. The command writes `results.md`/`results.json` and `compare_<view>.jpg`: one reference and one render per run for four fixed held-out views.

`mattersplat.score` adds two scores, computed at 512 px:

- LPIPS (needs the optional `torch` + `lpips` packages)
- Sobel gradient correlation

PSNR/SSIM alone reward smooth blur on these bright, detailed interiors.

Spirula shuffles its eval image order on every run, so views are matched by content (`spirula.match_eval_images`).

Results on the test file (48 held-out faces from 8 sweeps, 30k steps, 1M splats):

| run | LPIPS (lower is better) | grad corr | PSNR | SSIM | train min |
|---|---|---|---|---|---|
| depth weight 0.1 | 0.442 | 0.436 | 16.24 | 0.566 | 18.9 |
| **depth weight 0** (default now) | **0.415** | **0.459** | **16.52** | 0.568 | 16.5 |
| depth weight 0.5 | 0.494 | 0.387 | 15.81 | 0.558 | 18.5 |
| distraction-robustness mild | 0.463 | 0.425 | 16.20 | 0.566 | 17.7 |
| full resolution (divisor 1) | 0.455 | 0.433 | 16.15 | 0.649* | 53.8 |

\* Full resolution's SSIM is scored against 4096 px references, so it isn't comparable with the other rows.

Per view, depth 0 beats depth 0.1 on LPIPS in 41 of 48 views. Depth 0.5 loses in all 48.

### What a Matterport Pro3 export looks like

- One data3D per sweep, each a 1800 x 3600 (0.1 deg) spherical grid stored as cartesian XYZ + RGB,
  with `cartesianInvalidState` marking ~30% of cells empty.
- Six images2D per sweep ("Skybox 0..5"): 4096 x 4096 JPEG cube faces, 90 deg pinhole
  (f = 2048 px), sharing the sweep's position. Face 0 looks up, 1-4 horizontal, 5 down; faces
  are gravity-levelled. Linked to the sweep by `associatedData3DGuid`.
- World frame is Z-up, metres.
