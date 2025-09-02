<<<<<<< HEAD

import argparse, os, json, glob
from pathlib import Path
import cv2
import numpy as np

from src.preprocessing.stain_norm import normalize_macenko, estimate_reference_from_tiles

def load_tiles(tiles_dir, max_load=200):
    paths = sorted(glob.glob(os.path.join(tiles_dir, "*.png")))
    if len(paths) == 0:
        raise SystemExit(f"No PNG tiles found in {tiles_dir}")
    sel = paths if len(paths) <= max_load else np.random.choice(paths, max_load, replace=False)
    imgs = []
    for p in sel:
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        if im is None: 
            continue
        imgs.append(im)
    return paths, imgs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles_dir", required=True, help="Path to tiles folder (PNG)")
    ap.add_argument("--out_dir", required=True, help="Output directory for normalized tiles")
    ap.add_argument("--ref_mode", default="auto", choices=["auto", "self"], help="'auto'=estimate from sample tiles; 'self'=use canonical H_ref & unit C")
    ap.add_argument("--max_ref_tiles", type=int, default=64, help="Tiles used to estimate reference")
    args = ap.parse_args()

    paths, ref_imgs = load_tiles(args.tiles_dir, max_load=args.max_ref_tiles)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.ref_mode == "auto":
        H_ref, C_ref_max = estimate_reference_from_tiles(ref_imgs, sample_k=min(len(ref_imgs), args.max_ref_tiles))
    else:
        H_ref, C_ref_max = None, None

    meta = {"ref_mode": args.ref_mode, "H_ref": None, "C_ref_max": None}
    for p in paths:
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        if im is None: 
            continue
        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        norm, info = normalize_macenko(rgb, H_ref=H_ref, C_ref_max=C_ref_max)
        out_p = os.path.join(args.out_dir, os.path.basename(p))
        cv2.imwrite(out_p, cv2.cvtColor(norm, cv2.COLOR_RGB2BGR))
        meta["H_ref"] = info["H_ref"]
        meta["C_ref_max"] = info["C_ref_max"]
    with open(os.path.join(args.out_dir, "stain_norm_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Normalized {len(paths)} tiles → {args.out_dir}")
=======
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
>>>>>>> d44b7d6 (chore: bootstrap repo)

if __name__ == "__main__":
    main()
