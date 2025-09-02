# -*- coding: utf-8 -*-
"""
Calibrate a reference stain basis (H_ref) and illumination (Io_ref) from a batch of tiles.

Usage:
python -m scripts.calibrate_ref \
  --tiles_dir D:\wsi-qc\results\example\tiles \
  --out_json  D:\wsi-qc\refs\stain_ref.json \
  --max_tiles 200 \
  --io_mode percentile \
  --io_percentile 95
"""
import os
import json
import argparse
import random
import numpy as np
from glob import glob

from src.preprocessing.stain_norm import (
    tissue_mask,
    estimate_io,
    rgb2od,
    macenko_basis,
    save_ref_json,
)

import imageio.v2 as imageio

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles_dir", required=True, help="Folder with input tiles (*.png|*.jpg).")
    ap.add_argument("--out_json", required=True, help="Output JSON path for reference (H_ref, Io_ref).")
    ap.add_argument("--max_tiles", type=int, default=200, help="Maximum number of tiles to use.")
    ap.add_argument("--io_mode", choices=["percentile"], default="percentile")
    ap.add_argument("--io_percentile", type=float, default=95.0, help="Percentile for Io estimation.")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--beta", type=float, default=0.15)
    args = ap.parse_args()

    paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
        paths.extend(glob(os.path.join(args.tiles_dir, ext)))
    if not paths:
        raise SystemExit(f"No image tiles found under: {args.tiles_dir}")

    random.seed(123)
    random.shuffle(paths)
    if len(paths) > args.max_tiles:
        paths = paths[:args.max_tiles]

    H_list = []
    Io_list = []
    used = 0

    for p in paths:
        try:
            rgb = imageio.imread(p)
            if rgb.ndim != 3 or rgb.shape[2] != 3:
                continue
            mask = tissue_mask(rgb)
            if not np.any(mask):
                continue
            Io = estimate_io(rgb, mask=mask, percentile=args.io_percentile)
            od = rgb2od(rgb, Io)
            H = macenko_basis(od, beta=args.beta, alpha=args.alpha)  # 3x2
            H_list.append(H)
            Io_list.append(Io)
            used += 1
        except Exception as e:
            print(f"[warn] skip {p}: {e}")

    if used == 0:
        raise SystemExit("No valid tiles for calibration.")

    # Aggregate Io by median
    Io_ref = np.median(np.stack(Io_list, axis=0), axis=0)

    # Aggregate H: we align columns using the heuristic (blue-heavy first)
    H_stack = np.stack(H_list, axis=0)  # (N, 3, 2)

    # Normalize columns and median
    H_ref = np.median(H_stack, axis=0)
    # Re-unit columns
    H_ref = H_ref / (np.linalg.norm(H_ref, axis=0, keepdims=True) + 1e-8)

    meta = {
        "alpha": args.alpha,
        "beta": args.beta,
        "io_mode": args.io_mode,
        "io_percentile": args.io_percentile,
        "used_tiles": int(used),
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    save_ref_json(args.out_json, H_ref, Io_ref, meta=meta)
    print(f"[ok] wrote reference -> {args.out_json}")
    print("H_ref =\n", H_ref)
    print("Io_ref =", Io_ref)

if __name__ == "__main__":
    main()
