
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

if __name__ == "__main__":
    main()
