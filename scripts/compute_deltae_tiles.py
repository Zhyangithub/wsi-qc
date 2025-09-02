
# -*- coding: utf-8 -*-
"""
Compute ΔE (CIE76 in Lab) for each tile: original vs. normalized.
Saves a CSV with per-tile stats and (optionally) per-tile ΔE images.

What's new:
- Adds an extra CSV column `deltae` whose values equal the existing 95th-percentile metric.
  In other words: deltae == deltae_p95.

Usage:
python -m scripts.compute_deltae_tiles \  --orig_tiles D:\wsi-qc\results\example\tiles \  --norm_tiles D:\wsi-qc\results\example_norm \  --out_csv    D:\wsi-qc\results\example_norm\deltae_tiles.csv \  --tiles_index_csv D:\wsi-qc\results\example\tiles_index.csv   (optional) \  --save_deltae_dir D:\wsi-qc\results\example_norm\deltae_tiles (optional)
"""
import os, csv, argparse, numpy as np
from glob import glob
from skimage.color import rgb2lab
import imageio.v2 as imageio

# Robust import: if your repo has src.utils.filename_parser, use it;
# otherwise fall back to a local parser.
try:
    from src.utils.filename_parser import parse_xy_from_name as _parse_xy_from_name  # type: ignore
    def parse_xy_from_name(name: str):
        return _parse_xy_from_name(name)
except Exception:
    import re
    def parse_xy_from_name(name: str):
        """Best-effort fallback: parse x/y from common patterns in filenames.
        Examples it understands:
        - tile_x123_y456.png
        - x=123_y=456.jpg
        - 123_456.png  (interprets as x_y if two ints appear at end)
        Returns (x, y) as ints, or None if not found.
        """
        base = os.path.splitext(os.path.basename(name))[0]
        # common explicit patterns
        m = re.search(r"x\s*[_=]?\s*(\d+)\D+y\s*[_=]?\s*(\d+)", base, re.IGNORECASE)
        if m:
            return (int(m.group(1)), int(m.group(2)))
        # generic "_x123_y456"
        m = re.search(r"[_-]x(\d+)[_-]y(\d+)", base, re.IGNORECASE)
        if m:
            return (int(m.group(1)), int(m.group(2)))
        # two trailing integers separated by non-digits
        m = re.findall(r"(\d+)", base)
        if len(m) >= 2:
            return (int(m[-2]), int(m[-1]))
        return None

def deltaE_lab(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """CIE76 ΔE in Lab space; inputs are uint8 RGB images of same size."""
    La = rgb2lab(a)
    Lb = rgb2lab(b)
    d = La - Lb
    return np.sqrt((d[...,0]**2)+(d[...,1]**2)+(d[...,2]**2))

def load_index_csv(path):
    """Load optional tile index CSV with columns at least: filename,x,y"""
    data = {}
    if not path or (path and not os.path.exists(path)):
        return data
    import pandas as pd
    df = pd.read_csv(path)
    # try common schemas
    fname_col = None
    for c in ["filename","tile","name","path","file"]:
        if c in df.columns:
            fname_col = c; break
    if fname_col is None:
        # fallback: try to deduce from first column
        fname_col = df.columns[0]
    # x/y columns
    xcol = next((c for c in df.columns if c.lower() in ("x","tile_x","left","x0")), None)
    ycol = next((c for c in df.columns if c.lower() in ("y","tile_y","top","y0")), None)
    if xcol is None or ycol is None:
        return data
    for _,row in df.iterrows():
        data[str(row[fname_col])] = (int(row[xcol]), int(row[ycol]))
    return data

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig_tiles", required=True, help="Folder of ORIGINAL tiles (PNG/JPG/TIF)")
    ap.add_argument("--norm_tiles", required=True, help="Folder of NORMALIZED tiles (same filenames)")
    ap.add_argument("--out_csv", required=True, help="Path to write per-tile ΔE stats CSV")
    ap.add_argument("--tiles_index_csv", default=None, help="Optional CSV with filename,x,y for coordinates")
    ap.add_argument("--save_deltae_dir", default=None, help="Optional folder to save ΔE tile images as 16-bit PNG")
    ap.add_argument("--exts", nargs="+", default=[".png",".jpg",".jpeg",".tif",".tiff"], help="Allowed extensions")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    if args.save_deltae_dir:
        os.makedirs(args.save_deltae_dir, exist_ok=True)

    # Original tile map
    paths = []
    for ext in args.exts:
        paths.extend(glob(os.path.join(args.orig_tiles, f"*{ext}")))
    paths = sorted(paths)
    if not paths:
        raise SystemExit(f"No tiles found under: {args.orig_tiles}")

    # Optional coordinates map
    idxmap = load_index_csv(args.tiles_index_csv)

    # Iterate tiles
    rows = []
    missing = 0
    for p in paths:
        name = os.path.basename(p)
        q = os.path.join(args.norm_tiles, name)
        if not os.path.exists(q):
            missing += 1
            continue
        try:
            a = imageio.imread(p)
            b = imageio.imread(q)
            if a.shape != b.shape or a.ndim != 3 or a.shape[2] != 3:
                print(f"[skip] shape mismatch: {name} {a.shape} vs {b.shape}")
                continue
            dE = deltaE_lab(a, b).astype(np.float32)
            # stats
            dE_mean = float(np.mean(dE))
            dE_median = float(np.median(dE))
            dE_p95 = float(np.percentile(dE, 95))
            dE_max = float(np.max(dE))
            # coords
            xy = idxmap.get(name)
            if xy is None:
                xy = parse_xy_from_name(name)
            x, y = (xy if xy is not None else (None, None))

            if args.save_deltae_dir:
                # save as 16-bit grayscale PNG to preserve dynamic range
                outp = os.path.join(args.save_deltae_dir, os.path.splitext(name)[0] + "_dE.png")
                # scale to 0-65535 for storage, using robust range
                lo, hi = np.percentile(dE, [5, 95])
                denom = max(hi - lo, 1e-6)
                vis = np.clip((dE - lo) / denom, 0, 1.0)
                imageio.imwrite(outp, (vis * 65535).astype(np.uint16))

            # Append row including 'deltae' that duplicates the 95th percentile value
            rows.append((name, x, y, dE_mean, dE_median, dE_p95, dE_max, dE_p95))
        except Exception as e:
            print(f"[warn] {name}: {e}")

    # Write CSV
    import pandas as pd
    df = pd.DataFrame(
        rows,
        columns=[
            "filename","x","y",
            "deltae_mean","deltae_median","deltae_p95","deltae_max",
            "deltae"  # new column equal to deltae_p95
        ],
    )
    df.to_csv(args.out_csv, index=False)
    print(f"[ok] wrote ΔE per-tile stats -> {args.out_csv}")
    if missing:
        print(f"[warn] missing normalized tiles for {missing} files")

if __name__ == "__main__":
    main()
