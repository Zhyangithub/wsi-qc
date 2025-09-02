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

if __name__ == "__main__":
    main()
