# -*- coding: utf-8 -*-
"""
Batch Macenko stain normalization for tiles.

Usage:
python -m scripts.stain_normalize \
  --tiles_dir D:\wsi-qc\results\example\tiles \
  --out_dir   D:\wsi-qc\results\example_norm \
  --ref_json  D:\wsi-qc\refs\stain_ref.json \
  --clamp 1.0 --gamma 1.0
"""
import os
import argparse
from glob import glob
import numpy as np
import imageio.v2 as imageio

from src.preprocessing.stain_norm import (
    load_ref_json,
    normalize_macenko,
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles_dir", required=True, help="Folder with input tiles (*.png|*.jpg).")
    ap.add_argument("--out_dir", required=True, help="Output folder for normalized tiles.")
    ap.add_argument("--ref_json", required=True, help="Reference JSON from calibrate_ref.")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--beta", type=float, default=0.15)
    ap.add_argument("--clamp", type=float, default=1.0, help=">1.0 to slightly shrink extremes of concentrations.")
    ap.add_argument("--gamma", type=float, default=1.0, help="Gamma correction on final RGB (1.0 disables).")
    args = ap.parse_args()

    ref = load_ref_json(args.ref_json)
    os.makedirs(args.out_dir, exist_ok=True)

    paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
        paths.extend(glob(os.path.join(args.tiles_dir, ext)))
    if not paths:
        raise SystemExit(f"No image tiles found under: {args.tiles_dir}")

    for p in paths:
        try:
            rgb = imageio.imread(p)
            out = normalize_macenko(
                rgb, ref, beta=args.beta, alpha=args.alpha, clamp=args.clamp, gamma=args.gamma
            )
            fname = os.path.basename(p)
            imageio.imwrite(os.path.join(args.out_dir, fname), out)
        except Exception as e:
            print(f"[warn] skip {p}: {e}")

    print(f"[ok] normalized tiles -> {args.out_dir}")

if __name__ == "__main__":
    main()
