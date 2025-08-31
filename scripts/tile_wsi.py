import argparse, os, math
from pathlib import Path
import numpy as np
from tqdm import tqdm
import cv2

def read_wsi_region(path, level=0):
    # Prefer OpenSlide for pyramidal WSI; fallback to tifffile for simple TIFF
    img_reader = None
    try:
        import openslide
        slide = openslide.OpenSlide(path)
        w, h = slide.level_dimensions[level]
        return slide, (w, h), True
    except Exception as e:
        slide = None
    # Fallback: read as a whole image (may be huge; use only for small images)
    try:
        import tifffile as tiff
        arr = tiff.imread(path)
        h, w = arr.shape[:2]
        return arr, (w, h), False
    except Exception as e:
        raise RuntimeError(f"Failed to read {path} with OpenSlide or tifffile: {e}")

def read_region(slide_or_arr, x, y, size, level=0, use_openslide=True):
    if use_openslide:
        region = slide_or_arr.read_region((x, y), level, (size, size)).convert("RGB")
        return np.array(region)[:, :, ::-1]  # to BGR for OpenCV
    else:
        arr = slide_or_arr
        return arr[y:y+size, x:x+size].copy()

def simple_tissue_mask(rgb):
    # HSV + Otsu on saturation to remove background
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    s = hsv[:,:,1]
    _, th = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # Morph close / open to clean
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=1)
    return th

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wsi", required=True, help="Path to WSI (svs/tiff)")
    ap.add_argument("--out", required=True, help="Output dir")
    ap.add_argument("--tile", type=int, default=512, help="Tile size")
    ap.add_argument("--step", type=int, default=None, help="Stride (default=tile)")
    ap.add_argument("--level", type=int, default=0, help="WSI level for tiling")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tiles_dir = Path(args.out) / "tiles"
    os.makedirs(tiles_dir, exist_ok=True)

    slide, (W, H), use_os = read_wsi_region(args.wsi, level=args.level)
    step = args.step or args.tile

    # For tissue masking compute a downsampled preview
    preview_scale = max(1, int(max(W, H) / 4096))
    if use_os:
        import openslide
        # pick level whose width ~<=4096
        lvl = 0
        for i, (w,h) in enumerate(getattr(slide, 'level_dimensions', [(W,H)])):
            if max(w,h) <= 4096:
                lvl = i; break
        preview = np.array(slide.read_region((0,0), lvl, slide.level_dimensions[lvl]).convert("RGB"))[:,:,::-1]
        tissue = simple_tissue_mask(preview)
        # scale mask up to level 0 approx coords
        scale_x = W / tissue.shape[1]; scale_y = H / tissue.shape[0]
    else:
        preview = slide.copy()
        tissue = simple_tissue_mask(preview)
        scale_x = W / tissue.shape[1]; scale_y = H / tissue.shape[0]

    records = []
    for y in tqdm(range(0, H-args.tile+1, step)):
        for x in range(0, W-args.tile+1, step):
            # skip if tissue coverage too low
            tx0 = int(x/scale_x); ty0 = int(y/scale_y)
            tx1 = int((x+args.tile)/scale_x); ty1 = int((y+args.tile)/scale_y)
            tx1 = min(tx1, tissue.shape[1]-1); ty1 = min(ty1, tissue.shape[0]-1)
            if tx1<=tx0 or ty1<=ty0: 
                continue
            tissue_frac = (tissue[ty0:ty1, tx0:tx1]>0).mean()
            if tissue_frac < 0.2:  # skip mostly background
                continue

            tile = read_region(slide, x, y, args.tile, level=args.level, use_openslide=use_os)
            fname = f"x{x:07d}_y{y:07d}.png"
            cv2.imwrite(str(tiles_dir / fname), tile)
            records.append((x, y, fname))

    # Save an index CSV
    import csv
    with open(Path(args.out) / "tiles_index.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "filename"])
        writer.writerows(records)

    # Save a preview mask for sanity
    cv2.imwrite(str(Path(args.out) / "preview.jpg"), preview)
    cv2.imwrite(str(Path(args.out) / "preview_tissue_mask.jpg"), tissue)

if __name__ == "__main__":
    main()
