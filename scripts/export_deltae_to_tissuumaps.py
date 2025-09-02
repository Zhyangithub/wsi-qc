# -*- coding: utf-8 -*-
"""
Export a TissUUmaps-compatible overlay CSV from per-tile ΔE stats.
You can choose which ΔE metric becomes the 'value' column: median/mean/p95/max.

Usage:
python -m scripts.export_deltae_to_tissuumaps \
  --deltae_csv  D:\wsi-qc\results\example_norm\deltae_tiles.csv \
  --out_csv     D:\wsi-qc\results\example_norm\overlay_deltae.csv \
  --metric      p95 \
  --tile_size   512 \
  --position    topleft   # or 'center'
"""
import os, argparse, pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deltae_csv", required=True, help="CSV from compute_deltae_tiles.py")
    ap.add_argument("--out_csv", required=True, help="Output overlay CSV for TissUUmaps")
    ap.add_argument("--metric", choices=["median","mean","p95","max"], default="p95")
    ap.add_argument("--tile_size", type=int, default=512)
    ap.add_argument("--position", choices=["topleft","center"], default="topleft",
                    help="Where to place the (x,y) marker for each tile")
    args = ap.parse_args()

    df = pd.read_csv(args.deltae_csv)
    # Ensure columns exist
    need = {"filename","x","y","deltae_mean","deltae_median","deltae_p95","deltae_max"}
    if not need.issubset(set(df.columns)):
        raise SystemExit(f"Missing required columns in {args.deltae_csv}.")

    # Select metric
    colmap = {"median":"deltae_median","mean":"deltae_mean","p95":"deltae_p95","max":"deltae_max"}
    valcol = colmap[args.metric]
    out = df[["x","y"]].copy()
    if args.position == "center":
        out["x"] = out["x"] + args.tile_size / 2.0
        out["y"] = out["y"] + args.tile_size / 2.0
    out["deltae"] = df[valcol]

    # Drop rows without coordinates
    out = out.dropna(subset=["x","y"]).reset_index(drop=True)

    # Write overlay
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"[ok] overlay -> {args.out_csv}  (metric={args.metric}, n={len(out)})")

if __name__ == "__main__":
    main()
