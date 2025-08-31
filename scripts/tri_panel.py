
import argparse, os, csv
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def read_qc_csv(path):
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if "qc_score" in r:
                s = float(r["qc_score"])
            else:
                s = float(r.get("score", 0))
            rows.append({"x": int(float(r["x"])), "y": int(float(r["y"])), "score": s})
    return rows

def pick_tile(tile_dir, qc_rows, mode="median"):
    scores = np.array([r["score"] for r in qc_rows], dtype=float)
    if scores.size == 0:
        raise SystemExit("qc.csv contains no rows")
    if mode == "median":
        target = np.median(scores)
        idx = np.argmin(np.abs(scores - target))
    elif mode == "best":
        idx = np.argmax(scores)
    else:
        idx = np.argmin(scores)
    r = qc_rows[idx]
    fname = f"x{r['x']:07d}_y{r['y']:07d}.png"
    p = Path(tile_dir) / fname
    return str(p), r

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", required=True, help="results/.../preview.jpg from tiling")
    ap.add_argument("--qc_csv", required=True, help="results/.../qc.csv")
    ap.add_argument("--tile_dir", required=True, help="results/.../tiles")
    ap.add_argument("--norm_tile_dir", required=True, help="results/.../normalized_tiles")
    ap.add_argument("--out", required=True, help="output figure path")
    ap.add_argument("--mode", default="median", choices=["median","best","worst"], help="which tile to show")
    ap.add_argument("--tile_size", type=int, default=512)
    ap.add_argument("--pmin", type=float, default=5.0)
    ap.add_argument("--pmax", type=float, default=95.0)
    args = ap.parse_args()

    bg = cv2.imread(args.preview, cv2.IMREAD_COLOR)
    if bg is None:
        raise SystemExit(f"Cannot read preview: {args.preview}")
    bg_rgb = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)

    rows = read_qc_csv(args.qc_csv)
    scores = np.array([r["score"] for r in rows], dtype=float)
    vmin, vmax = np.percentile(scores, [args.pmin, args.pmax])

    xs = np.array([r["x"] + args.tile_size//2 for r in rows])
    ys = np.array([r["y"] + args.tile_size//2 for r in rows])

    tile_path, r_pick = pick_tile(args.tile_dir, rows, mode=args.mode)
    tile = cv2.imread(tile_path, cv2.IMREAD_COLOR)
    if tile is None:
        raise SystemExit(f"Cannot read tile: {tile_path}")
    norm_tile_path = os.path.join(args.norm_tile_dir, os.path.basename(tile_path))
    norm_tile = cv2.imread(norm_tile_path, cv2.IMREAD_COLOR)
    if norm_tile is None:
        norm_tile = tile.copy()

    plt.figure(figsize=(14, 5))

    plt.subplot(1,3,1)
    plt.imshow(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB))
    plt.title("Original tile")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(cv2.cvtColor(norm_tile, cv2.COLOR_BGR2RGB))
    plt.title("Macenko normalized tile")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(bg_rgb)
    sc = plt.scatter(xs, ys, c=scores, s=8, cmap="turbo", vmin=vmin, vmax=vmax, alpha=0.9)
    plt.gca().invert_yaxis()
    plt.title("QC heatmap (tile centers)")
    plt.axis("off")
    cbar = plt.colorbar(sc, fraction=0.046, pad=0.04)
    cbar.set_label("qc_score")

    Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print("Wrote", args.out)

if __name__ == "__main__":
    main()
