"""A/B training runs on one Spirula import, scored on the same held-out sweeps.

    python -m mattersplat.tune <dataset> --out <dir> [--runs baseline depth0 ...] [--spirula <exe>]

Each run changes one thing from mattersplat.spirula.TRAIN_FLAGS (see VARIANTS) and trains into
<out>/<run>/. Finished runs (with train_summary.json) are skipped, so the command can be re-run to
add variants. After every run it rewrites <out>/results.md + results.json and, for a fixed set of
held-out views, <out>/compare_<view>.jpg (reference, then one render per run).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

from PIL import Image, ImageDraw

from . import score, spirula

VARIANTS = {
    "baseline": {},
    "depth0": {"depth-supervision-weight": 0},
    "depth0.5": {"depth-supervision-weight": 0.5},
    "distract_mild": {"distraction-robustness": "mild"},
    "fullres": {"train-resolution-divisor": 1},
}

# held-out faces shown side by side: horizontal faces of four held-out sweeps across the site
COMPARE_VIEWS = ("0116_Skybox_2", "0272_Skybox_2", "0430_Skybox_4", "0583_Skybox_1")


def other_gpu_jobs() -> list[str]:
    """Command lines of spirula.exe processes already running (they share the GPU)."""
    ps = ("Get-CimInstance Win32_Process -Filter \"name='spirula.exe'\" | "
          "ForEach-Object { $_.CommandLine }")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True,
                             timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [" ".join(l.split())[:200] for l in out.splitlines() if l.strip()]


def write_results(out: str, runs: list[str]):
    rows = []
    for name in runs:
        p = os.path.join(out, name, "train_summary.json")
        if os.path.exists(p):
            s = json.load(open(p))
            s["scores"] = score.score_run(s["run_dir"])["mean"]
            rows.append((name, s))
    with open(os.path.join(out, "results.json"), "w") as fh:
        json.dump({n: s for n, s in rows}, fh, indent=1)
    lines = ["| run | change | LPIPS (lower better) | grad corr | PSNR | cc PSNR | SSIM | cc SSIM | train min | "
             "GPU peak MiB (whole GPU) | splats | MB | other GPU jobs |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, s in rows:
        m = s.get("metrics", {})
        f = lambda k, d=2: f"{m[k]:.{d}f}" if k in m else "-"
        change = ", ".join(f"{k}={v}" for k, v in VARIANTS.get(name, {}).items()) or "-"
        sc = s["scores"]
        g = lambda k: f"{sc[k]:.3f}" if k in sc else "-"
        lines.append(f"| {name} | {change} | {g('lpips')} | {g('grad_corr')} | {f('psnr')} | {f('cc_psnr')} | {f('ssim', 3)} | {f('cc_ssim', 3)} | "
                     f"{s['minutes']:.1f} | {s['gpu_peak_mib']} | {s['splats']:,} | {s['splat_mb']:.0f} | "
                     f"{len(s.get('other_gpu_jobs', []))} |")
    with open(os.path.join(out, "results.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    for view in COMPARE_VIEWS:
        tiles = []
        for name, s in rows:
            names = [os.path.splitext(n)[0].removeprefix("eval_") for n in s.get("eval_names") or []]
            if view not in names:
                continue
            i = names.index(view)
            if not tiles:
                tiles.append(("reference " + view, os.path.join(s["run_dir"], f"eval-gt-{i:05d}.png"), None))
            psnr = (json.load(open(os.path.join(s["run_dir"], "metrics.json"))).get("psnr") or [None] * (i + 1))[i]
            tiles.append((name, os.path.join(s["run_dir"], f"eval-render-{i:05d}.png"), psnr))
        if not tiles:
            continue
        size = 640
        sheet = Image.new("RGB", (size * len(tiles), size + 24), "white")
        draw = ImageDraw.Draw(sheet)
        for k, (label, path, psnr) in enumerate(tiles):
            sheet.paste(Image.open(path).convert("RGB").resize((size, size)), (k * size, 24))
            draw.text((k * size + 6, 5), label + (f"  {psnr:.1f} dB" if psnr is not None else ""), fill="black")
        sheet.save(os.path.join(out, f"compare_{view}.jpg"), quality=90)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", help="a spirula e57 import (seed already cropped)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--spirula")
    ap.add_argument("--iterations", type=int)
    args = ap.parse_args(argv)

    exe = spirula.find_spirula(args.spirula)
    for name in args.runs:
        run_out = os.path.join(args.out, name)
        if os.path.exists(os.path.join(run_out, "train_summary.json")):
            print(f"== {name}: done already")
            continue
        others = other_gpu_jobs()
        print(f"== {name}: {VARIANTS[name] or 'baseline flags'}; other spirula jobs on the GPU: {len(others)}")
        for o in others:
            print("   " + o)
        t0 = time.time()
        try:
            res = spirula.train(exe, args.dataset, run_out, args.iterations,
                                holdout_sweeps=spirula.DEFAULT_HOLDOUT_SWEEPS, extra=VARIANTS[name])
        except RuntimeError as e:
            print(f"== {name} failed after {(time.time() - t0) / 60:.1f} min: {e}")
            with open(os.path.join(run_out, "failed.txt"), "w") as fh:
                fh.write(str(e) + "\n")
            continue
        res["other_gpu_jobs"] = others
        with open(os.path.join(run_out, "train_summary.json"), "w") as fh:
            json.dump(res, fh, indent=1)
        write_results(args.out, list(VARIANTS))


if __name__ == "__main__":
    sys.exit(main())
