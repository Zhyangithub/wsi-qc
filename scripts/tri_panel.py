<<<<<<< HEAD

import argparse, os, csv
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def read_qc_csv(path):
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if "qc_score" in r:
                s = float(r["qc_score"])
            else:
                s = float(r.get("score", 0))
            rows.append({"x": int(float(r["x"])), "y": int(float(r["y"])), "score": s})
    return rows

def pick_tile(tile_dir, qc_rows, mode="median"):
    scores = np.array([r["score"] for r in qc_rows], dtype=float)
    if scores.size == 0:
        raise SystemExit("qc.csv contains no rows")
    if mode == "median":
        target = np.median(scores)
        idx = np.argmin(np.abs(scores - target))
    elif mode == "best":
        idx = np.argmax(scores)
    else:
        idx = np.argmin(scores)
    r = qc_rows[idx]
    fname = f"x{r['x']:07d}_y{r['y']:07d}.png"
    p = Path(tile_dir) / fname
    return str(p), r

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", required=True, help="results/.../preview.jpg from tiling")
    ap.add_argument("--qc_csv", required=True, help="results/.../qc.csv")
    ap.add_argument("--tile_dir", required=True, help="results/.../tiles")
    ap.add_argument("--norm_tile_dir", required=True, help="results/.../normalized_tiles")
    ap.add_argument("--out", required=True, help="output figure path")
    ap.add_argument("--mode", default="median", choices=["median","best","worst"], help="which tile to show")
    ap.add_argument("--tile_size", type=int, default=512)
    ap.add_argument("--pmin", type=float, default=5.0)
    ap.add_argument("--pmax", type=float, default=95.0)
    args = ap.parse_args()

    bg = cv2.imread(args.preview, cv2.IMREAD_COLOR)
    if bg is None:
        raise SystemExit(f"Cannot read preview: {args.preview}")
    bg_rgb = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)

    rows = read_qc_csv(args.qc_csv)
    scores = np.array([r["score"] for r in rows], dtype=float)
    vmin, vmax = np.percentile(scores, [args.pmin, args.pmax])

    xs = np.array([r["x"] + args.tile_size//2 for r in rows])
    ys = np.array([r["y"] + args.tile_size//2 for r in rows])

    tile_path, r_pick = pick_tile(args.tile_dir, rows, mode=args.mode)
    tile = cv2.imread(tile_path, cv2.IMREAD_COLOR)
    if tile is None:
        raise SystemExit(f"Cannot read tile: {tile_path}")
    norm_tile_path = os.path.join(args.norm_tile_dir, os.path.basename(tile_path))
    norm_tile = cv2.imread(norm_tile_path, cv2.IMREAD_COLOR)
    if norm_tile is None:
        norm_tile = tile.copy()

    plt.figure(figsize=(14, 5))

    plt.subplot(1,3,1)
    plt.imshow(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB))
    plt.title("Original tile")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(cv2.cvtColor(norm_tile, cv2.COLOR_BGR2RGB))
    plt.title("Macenko normalized tile")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(bg_rgb)
    sc = plt.scatter(xs, ys, c=scores, s=8, cmap="turbo", vmin=vmin, vmax=vmax, alpha=0.9)
    plt.gca().invert_yaxis()
    plt.title("QC heatmap (tile centers)")
    plt.axis("off")
    cbar = plt.colorbar(sc, fraction=0.046, pad=0.04)
    cbar.set_label("qc_score")

    Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print("Wrote", args.out)
=======
# -*- coding: utf-8 -*-
"""
Create a tri-panel figure for a single tile:
- Original
- Macenko normalized
- ΔE (CIE-Lab) heatmap

Usage:
python -m scripts.tri_panel \
  --tile D:\wsi-qc\results\example\tiles\x000....png \
  --ref_json D:\wsi-qc\refs\stain_ref.json \
  --out_png D:\wsi-qc\results\tri_panel.png
"""
import os
import argparse
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from skimage.color import rgb2lab

from src.preprocessing.stain_norm import (
    load_ref_json, normalize_macenko
)

def deltaE_lab(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute simple ΔE*ab in CIE-Lab."""
    La = rgb2lab(a)
    Lb = rgb2lab(b)
    d = La - Lb
    return np.sqrt(np.sum(d * d, axis=-1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile", required=True, help="Path to a single RGB tile.")
    ap.add_argument("--ref_json", required=True, help="Reference JSON from calibrate_ref.")
    ap.add_argument("--out_png", required=True, help="Output PNG path for tri-panel.")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--beta", type=float, default=0.15)
    ap.add_argument("--clamp", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    args = ap.parse_args()

    ref = load_ref_json(args.ref_json)
    rgb = imageio.imread(args.tile)
    norm = normalize_macenko(rgb, ref, beta=args.beta, alpha=args.alpha, clamp=args.clamp, gamma=args.gamma)
    dE = deltaE_lab(rgb, norm)

    H, W = rgb.shape[:2]
    dpi = 200
    fig = plt.figure(figsize=(16, 5), dpi=dpi)
    gs = fig.add_gridspec(1, 3, wspace=0.03)

    ax0 = fig.add_subplot(gs[0,0])
    ax0.imshow(rgb)
    ax0.set_title("Original", fontsize=10)
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[0,1])
    ax1.imshow(norm)
    ax1.set_title("Macenko normalized", fontsize=10)
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0,2])
    im = ax2.imshow(dE, cmap="magma")
    ax2.set_title("ΔE (Lab)", fontsize=10)
    ax2.axis("off")
    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("ΔE", rotation=270, labelpad=10)

    os.makedirs(os.path.dirname(args.out_png), exist_ok=True)
    fig.savefig(args.out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] tri panel -> {args.out_png}")
>>>>>>> d44b7d6 (chore: bootstrap repo)

if __name__ == "__main__":
    main()
