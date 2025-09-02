import argparse, os, csv, math
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm

def laplacian_var(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def brightness(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return float(hsv[:,:,2].mean())

def saturation_fraction(img, thr=25):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return float((hsv[:,:,1] > thr).mean())

def stripe_score(img):
    # crude FFT-based streak detector: energy concentration along rows/cols
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    # resize to power-of-two-ish
    H, W = 256, 256
    small = cv2.resize(gray, (W, H), interpolation=cv2.INTER_AREA)
    f = np.fft.fftshift(np.fft.fft2(small))
    mag = np.log1p(np.abs(f))
    # accumulate energy near vertical/horizontal axes (ignore center DC)
    cx, cy = W//2, H//2
    band = 4  # pixels width around axes
    vert = mag[:, cx-band:cx+band].sum()
    hori = mag[cy-band:cy+band, :].sum()
    total = mag.sum() + 1e-6
    # Higher = more stripe-like content
    return float((vert + hori) / total)

def bubble_score(img, v_thr=230, s_thr=30):
    """Fraction of bright/low-saturation pixels (air bubbles)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:,:,2]; s = hsv[:,:,1]
    mask = (v > v_thr) & (s < s_thr)
    return float(mask.mean())

def fold_score(img, edge_thr=25, dark_thr=80):
    """Fraction of dark sharp edges (tissue folds)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    diff = cv2.absdiff(gray, blur)
    mask = (diff > edge_thr) & (gray < dark_thr)
    return float(mask.mean())

def compute_score(tile):
    # Normalize components to [0,1] with robust ranges
    b = brightness(tile)          # 0..255
    lv = laplacian_var(tile)      # blur -> small
    sf = saturation_fraction(tile)
    ss = stripe_score(tile)       # higher means more streaky
    bs = bubble_score(tile)       # bright circular artifacts
    fs = fold_score(tile)         # dark fold artifacts

    b_n  = np.clip((b - 80) / (180 - 80), 0, 1)
    lv_n = np.clip((lv - 50) / (400 - 50), 0, 1)  # sharpness
    sf_n = np.clip((sf - 0.10) / (0.35 - 0.10), 0, 1)
    ss_n = np.clip((ss - 0.02) / (0.08 - 0.02), 0, 1)
    bs_n = np.clip((bs - 0.005) / (0.05 - 0.005), 0, 1)
    fs_n = np.clip((fs - 0.005) / (0.05 - 0.005), 0, 1)

    # QC-score: higher is better quality; penalize artifacts
    score = (
        0.25*lv_n + 0.15*b_n + 0.15*sf_n +
        0.15*(1-ss_n) + 0.15*(1-bs_n) + 0.15*(1-fs_n)
    )
    return float(score), {"b": b, "lv": lv, "sf": sf, "ss": ss, "bs": bs, "fs": fs}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", required=True, help="Directory with PNG tiles")
    ap.add_argument("--out", required=True, help="Output CSV")
    ap.add_argument("--tile_size", type=int, default=512)
    args = ap.parse_args()
    tiles_dir = Path(args.tiles)
    rows = []
    for fname in tqdm(sorted(os.listdir(tiles_dir))):
        if not fname.lower().endswith(".png"):
            continue
        # Parse x,y from filename
        parts = fname.replace(".png","").split("_")
        x = int(parts[0][1:])
        y = int(parts[1][1:])
        img = cv2.imread(str(tiles_dir / fname), cv2.IMREAD_COLOR)
        score, feats = compute_score(img)
        rows.append([x, y, score, feats["b"], feats["lv"], feats["sf"], feats["ss"], feats["bs"], feats["fs"]])
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "qc_score", "brightness", "laplacian_var", "sat_frac", "stripe_score", "bubble_score", "fold_score"])
        w.writerows(rows)

if __name__ == "__main__":
    main()
