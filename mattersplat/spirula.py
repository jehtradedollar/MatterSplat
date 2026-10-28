"""Drive Spirula Studio (https://github.com/harry7557558/spirula-studio): E57 import, seed crop, training.

`spirula e57` turns the scan into a dataset Spirula trains on directly:
    images/NNNN_Skybox_K.jpg   the 4096x4096 cube faces
    depths/*.png, normals/*.png  1600x1600 maps rendered from the scan (uint16 mm, planar z-depth;
                                 0 = no return)
    transforms.json            nerfstudio poses (OpenGL camera-to-world, E57 world frame)
    sparse_pc.ply              ~500k voxel-thinned seed points

The seed cloud keeps Matterport's long-range returns through openings (on the test file out to
-112 m / +81 m), so crop_seed() clips it to a box around the sweeps (the step 2 default box) and
keeps the original as sparse_pc_full.ply.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import threading
import time

import numpy as np

DEFAULT_SPIRULA = r"C:\Users\ReddingDrone2\Documents\DJI_Inflight_3DGS\third_party\spirula\spirula.exe"

# Tuned for an RTX 4070 SUPER (12 GB) on a Matterport Pro3 export: 618 faces trained at 2048 px.
TRAIN_FLAGS = {
    "train-resolution-divisor": 2,
    "cap-max": 1_000_000,
    "depth-supervision-weight": 0.1,  # scan depth is metric and exact; default 0 loads it but ignores it
    "input-depth-is-ray-depth": 0,  # spirula e57 writes planar z-depth (checked against the raw scan)
    "disable-viewer": 1,
    "keep-viewer-alive": 0,
}


def find_spirula(path: str | None = None) -> str:
    for cand in (path, os.environ.get("MATTERSPLAT_SPIRULA"), DEFAULT_SPIRULA, shutil.which("spirula")):
        if cand and os.path.isfile(cand):
            return cand
    raise FileNotFoundError("spirula.exe not found: pass --spirula, set MATTERSPLAT_SPIRULA, or install it at "
                            + DEFAULT_SPIRULA)


def sweep_box(transforms: dict, pad_xy: float = 3.0, below: float = 2.0, above: float = 5.0) -> np.ndarray:
    """Same box as colmap_export.default_crop, from the camera centres in transforms.json."""
    t = np.array([np.array(f["transform_matrix"])[:3, 3] for f in transforms["frames"]])
    return np.stack([t.min(0) - [pad_xy, pad_xy, below], t.max(0) + [pad_xy, pad_xy, above]])


def _read_ply(path):
    raw = open(path, "rb").read()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    header = raw[:end].decode()
    types = {"double": "<f8", "float": "<f4", "uchar": "u1"}
    props = [(l.split()[2], types[l.split()[1]]) for l in header.splitlines() if l.startswith("property")]
    return header, np.frombuffer(raw[end:], dtype=props)


def crop_seed(dataset: str, box: np.ndarray | None = None) -> tuple[int, int, np.ndarray]:
    """Clip the dataset's seed cloud to `box` (default sweep_box). Returns (kept, dropped, box)."""
    tf = json.load(open(os.path.join(dataset, "transforms.json")))
    ply = os.path.join(dataset, tf.get("ply_file_path", "sparse_pc.ply"))
    full = ply.replace(".ply", "_full.ply")
    if not os.path.exists(full):
        shutil.copyfile(ply, full)
    box = sweep_box(tf) if box is None else np.asarray(box)
    header, rec = _read_ply(full)
    xyz = np.stack([rec["x"], rec["y"], rec["z"]], 1)
    keep = np.all((xyz >= box[0]) & (xyz <= box[1]), 1)
    lines = [f"element vertex {keep.sum()}" if l.startswith("element vertex") else l for l in header.splitlines()]
    with open(ply, "wb") as fh:
        fh.write(("\n".join(lines) + "\n").encode())
        fh.write(rec[keep].tobytes())
    return int(keep.sum()), int((~keep).sum()), box


class GpuMemoryLog(threading.Thread):
    """Polls nvidia-smi for used memory on GPU 0 while training runs."""

    def __init__(self, path: str, every: float = 2.0):
        super().__init__(daemon=True)
        self.path, self.every, self.peak = path, every, 0
        self._stop = threading.Event()

    def run(self):
        with open(self.path, "w") as fh:
            fh.write("t_s,used_mib\n")
            t0 = time.time()
            while not self._stop.is_set():
                try:
                    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                                          "-i", "0"], capture_output=True, text=True, timeout=10).stdout.strip()
                    used = int(out.splitlines()[0])
                    self.peak = max(self.peak, used)
                    fh.write(f"{time.time() - t0:.0f},{used}\n")
                    fh.flush()
                except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
                    pass
                self._stop.wait(self.every)

    def stop(self):
        self._stop.set()
        self.join()


def run(cmd: list[str], log_path: str, echo=print) -> int:
    """Run a command, tee-ing its output to log_path; returns the exit code."""
    echo("$ " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace")
        last = 0.0
        for line in proc.stdout:
            log.write(line)
            # progress lines repeat every few hundred ms; echo at most every 30 s
            if not line.startswith("step") or time.time() - last > 30:
                echo(line.rstrip())
                last = time.time() if line.startswith("step") else last
        return proc.wait()


def import_e57(spirula: str, e57: str, dataset: str, log_dir: str, echo=print) -> None:
    if os.path.exists(os.path.join(dataset, "transforms.json")):
        echo(f"Dataset already at {dataset}, skipping spirula e57")
        return
    code = run([spirula, "e57", e57, dataset], os.path.join(log_dir, "spirula_e57.log"), echo)
    if code:
        raise RuntimeError(f"spirula e57 failed ({code}), see spirula_e57.log")


def train(spirula: str, dataset: str, out: str, iterations: int | None = None, holdout: int = 8,
          extra: dict | None = None, echo=print) -> dict:
    """Train 3DGS on a Spirula dataset. holdout N > 0 keeps every Nth image for evaluation and
    writes rendered vs reference images for them; 0 trains on everything."""
    flags = dict(TRAIN_FLAGS)
    if iterations:
        flags["num-iterations"] = iterations
    if holdout:
        flags.update({"eval-mode": "interval", "eval-interval": holdout, "save-eval-images": 1})
    flags.update(extra or {})
    run_dir = os.path.join(out, "train")
    cmd = [spirula, "train", "3dgs", "--data", dataset, "--output-dir-prefix", out, "--output-dir-name", "train"]
    for k, v in flags.items():
        cmd += [f"--{k}", str(v)]
    gpu = GpuMemoryLog(os.path.join(out, "gpu_memory.csv"))
    gpu.start()
    t0 = time.time()
    try:
        code = run(cmd, os.path.join(out, "spirula_train.log"), echo)
    finally:
        gpu.stop()
    minutes = (time.time() - t0) / 60
    if code:
        raise RuntimeError(f"spirula train failed ({code}) after {minutes:.1f} min; GPU peak {gpu.peak} MiB")
    ckpts = sorted(glob.glob(os.path.join(run_dir, "step-*.ckpt")))
    splat = os.path.join(ckpts[-1], "splat.ply")
    shutil.copyfile(splat, os.path.join(out, "splat.ply"))
    with open(splat, "rb") as fh:
        head = fh.read(2000).decode(errors="replace")
    n = int(head.split("element vertex")[1].split()[0])
    return {"minutes": minutes, "gpu_peak_mib": gpu.peak, "splats": n, "checkpoint": ckpts[-1],
            "splat_mb": os.path.getsize(splat) / 1e6, "flags": flags}
