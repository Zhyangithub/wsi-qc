
import argparse, csv, json
from pathlib import Path
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as colors

def to_hex(rgb):
    r,g,b = (int(255*x) for x in rgb[:3])
    return f"#{r:02x}{g:02x}{b:02x}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile_csv", required=True, help="qc.csv (x,y,qc_score,...)")
    ap.add_argument("--out_csv", required=True, help="output overlay with color")
    ap.add_argument("--tile", type=int, default=512, help="tile size (for center)")
    ap.add_argument("--cmap", default="viridis", help="matplotlib colormap name")
    ap.add_argument("--pmin", type=float, default=5.0, help="lower percentile for clipping")
    ap.add_argument("--pmax", type=float, default=95.0, help="upper percentile for clipping")
    ap.add_argument("--gamma", type=float, default=1.0, help="gamma for contrast ( <1 expand low-end )")
    ap.add_argument("--bins", type=int, default=0, help="optional: also write binned group (0=off)")
    args = ap.parse_args()

    rows, scores = [], []
    with open(args.tile_csv, "r", newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            x = int(float(r["x"])); y = int(float(r["y"]))
            s = float(r.get("qc_score", r.get("score")))
            rows.append({"x": x + args.tile//2, "y": y + args.tile//2, "score": s})
            scores.append(s)
    if len(scores)==0:
        raise SystemExit("No rows")

    vmin, vmax = np.percentile(scores, [args.pmin, args.pmax])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = np.min(scores), np.max(scores) + 1e-6

    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap = cm.get_cmap(args.cmap)

    def enhance(u, gamma):
        return np.clip(u, 0, 1) ** (1.0/gamma)

    for r in rows:
        u = norm(r["score"])
        ue = enhance(u, args.gamma)
        c = cmap(ue)
        r["color"] = to_hex(c)
        if args.bins > 0:
            bin_idx = int(np.clip(ue,0,0.999)*args.bins)
            r["group"] = f"bin{bin_idx:02d}"

    outp = Path(args.out_csv)
    outp.parent.mkdir(parents=True, exist_ok=True)
    cols = ["x","y","score","color"] + (["group"] if args.bins>0 else [])
    with open(outp, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols); wr.writeheader()
        for r in rows: wr.writerow(r)

    if args.bins > 0:
        centers = np.linspace(0.5/args.bins, 1-0.5/args.bins, args.bins)
        d = {f"bin{i:02d}": to_hex(cmap(c)) for i,c in enumerate(centers)}
        with open(outp.with_suffix(".colordict.json"), "w") as f:
            json.dump(d, f)

    print(f"Wrote {outp} (pmin={args.pmin}, pmax={args.pmax}, gamma={args.gamma})")

if __name__ == "__main__":
    main()
