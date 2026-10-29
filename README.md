# MatterSplat

Matterport E57 export -> images + point cloud in 3D -> Gaussian splat (trained with Spirula Studio).

## Setup

```
python -m venv .venv
.venv\Scripts\python -m pip install -e .            # adds the `mattersplat` command
.venv\Scripts\python -m pip install torch lpips     # optional, only for LPIPS scores in tuning
```

Training needs [Spirula Studio](https://github.com/harry7557558/spirula-studio), a Windows Vulkan build that needs no CUDA or Python. `spirula.exe` is found from:

1. `--spirula`
2. the `MATTERSPLAT_SPIRULA` environment variable
3. the default install path in `mattersplat/spirula.py`

## Quick start: E57 to splat in one command

```
.venv\Scripts\mattersplat run path\to\cloud_0.e57 --out out\cloud_0 --view
```

This command:

1. Imports the E57 with Spirula, about 2 min for a 14 GB, 103-sweep file.
2. Crops the stray long-range points out of the seed cloud.
3. Trains with the tuned defaults: 2048 px, 1M splats, depth supervision off, 30k steps. This takes about 16 min on an RTX 4070 SUPER.
4. Writes `splat.ply`, the compact `splat.splat` and `sweeps.json`.
5. With `--view`, opens the viewer.

Options:

- `--colmap` also writes the step 2 COLMAP model.
- `--holdout-sweeps` holds whole sweeps out of training and scores them.

`python -m mattersplat run ...` works the same way.

### Web viewer

```
.venv\Scripts\mattersplat view out\cloud_0\splat.ply [--ply] [--port 8770]
```

This serves `viewer/index.html` on localhost and opens the browser. It loads `splat.splat`, which is made on the fly if missing.

The viewer:

- **Library:** uses `@mkkellogg/gaussian-splats-3d` 0.4.7 and three 0.160.1 from jsDelivr, so it needs internet on first load.
- **`.splat` file:** 32 MB for 1M splats against 248 MB for the PLY, with no view-dependent colour.
- **`--ply`:** loads the full PLY instead, with spherical-harmonics degree 2.
- **Frame:** Z-up, metres, in the E57 world frame. It starts at sweep 0, eye height, facing a horizontal cube face.
- **Walk mode (default):** drag to look around.
- **Orbit mode:** orbits a point 2.5 m ahead.
- **Keys:** WASD/arrows move, Q/E go down/up, Shift is faster.
- **Sweep list:** jumps to any scan position. Reset returns to sweep 0.
- **Header:** shows FPS, load time and splat count.

The page exposes `window.mattersplat`, with `jump(sweep, face)`, `stats()` and `screenshot(name)`. Screenshots go to `<splat dir>/screenshots/`.

On the test splat (985k splats after the viewer drops near-transparent ones), Edge and Chrome on the RTX 4070 SUPER:

- **Load:** about 1.3 s for `.splat` and 2.4 s for the full PLY.
- **Frame rate:** 60 fps, the refresh cap, at every sweep tried.

## The steps individually

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
