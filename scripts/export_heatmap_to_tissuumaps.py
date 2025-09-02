import argparse, csv
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile_csv", required=True, help="CSV with columns: x,y,qc_score,...")
    ap.add_argument("--out", required=True, help="Output CSV for TissUUmaps overlay")
    ap.add_argument("--metric", default="qc_score", help="Column in tile_csv to export")
    ap.add_argument("--tile", type=int, default=512, help="Tile size (to convert tile origin to center)")
    args = ap.parse_args()
    rows_out = []
    with open(args.tile_csv, "r") as f:
        reader = csv.DictReader(f)
        if args.metric not in reader.fieldnames:
            raise ValueError(f"Column '{args.metric}' not found in {args.tile_csv}")
        for r in reader:
            x = int(r["x"]); y = int(r["y"])
            cx = x + args.tile//2; cy = y + args.tile//2
            score = float(r[args.metric])
            rows_out.append([cx, cy, score])
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "score"])
        w.writerows(rows_out)
    print(f"Wrote overlay to {args.out}. Columns: x, y, score (from {args.metric})")

if __name__ == "__main__":
    main()
