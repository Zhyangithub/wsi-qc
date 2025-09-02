# -*- coding: utf-8 -*-
"""
One-click batch: normalize tiles -> compute ΔE -> export overlay.
Assumes you already created a reference JSON with calibrate_ref.py

Usage:
python -m scripts.batch_norm_and_deltae ^
  --orig_tiles   D:\wsi-qc\results\example\tiles ^
  --norm_tiles   D:\wsi-qc\results\example_norm ^
  --ref_json     D:\wsi-qc\refs\stain_ref.json ^
  --tile_size    512 ^
  --tiles_index_csv D:\wsi-qc\results\example\tiles_index.csv  (optional)
"""
import os, argparse, subprocess, sys

def run(cmd):
    print(">>", " ".join(cmd))
    ret = subprocess.call(cmd, shell=False)
    if ret != 0:
        sys.exit(ret)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig_tiles", required=True)
    ap.add_argument("--norm_tiles", required=True)
    ap.add_argument("--ref_json", required=True)
    ap.add_argument("--tile_size", type=int, default=512)
    ap.add_argument("--tiles_index_csv", default=None)
    ap.add_argument("--metric", choices=["median","mean","p95","max"], default="p95")
    args = ap.parse_args()

    # 1) Normalize tiles (reuse your existing CLI if available)
    if not os.path.exists(args.norm_tiles) or not os.listdir(args.norm_tiles):
        run([sys.executable, "-m", "scripts.stain_normalize",
             "--tiles_dir", args.orig_tiles,
             "--out_dir", args.norm_tiles,
             "--ref_json", args.ref_json,
             "--alpha", "0.10", "--beta", "0.15", "--clamp", "1.0", "--gamma", "1.0"])

    # 2) Compute ΔE per tile
    deltae_csv = os.path.join(args.norm_tiles, "deltae_tiles.csv")
    run([sys.executable, "-m", "scripts.compute_deltae_tiles",
         "--orig_tiles", args.orig_tiles,
         "--norm_tiles", args.norm_tiles,
         "--out_csv", deltae_csv] + (["--tiles_index_csv", args.tiles_index_csv] if args.tiles_index_csv else []))

    # 3) Export overlay for TissUUmaps
    overlay_csv = os.path.join(args.norm_tiles, f"overlay_deltae_{args.metric}.csv")
    run([sys.executable, "-m", "scripts.export_deltae_to_tissuumaps",
         "--deltae_csv", deltae_csv,
         "--out_csv", overlay_csv,
         "--metric", args.metric,
         "--tile_size", str(args.tile_size),
         "--position", "topleft"])

    print("[DONE] Normalize -> ΔE -> Overlay pipeline completed.")
    print(" Overlay:", overlay_csv)

if __name__ == "__main__":
    main()
