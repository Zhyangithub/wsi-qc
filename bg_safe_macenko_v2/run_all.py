
import os, argparse, json, glob, csv
import numpy as np
from imageio.v2 import imread, imwrite
from skimage import io
from stain_norm.macenko_bg import macenko_normalize, tissue_mask, _parse_ref_json
from analysis.tri_panel import deltaE_lab, tri_panel, dE_stats

def load_images(path):
    if os.path.isdir(path):
        exts = ('*.png','*.jpg','*.jpeg','*.tif','*.tiff','*.bmp')
        files = []
        for e in exts:
            files.extend(glob.glob(os.path.join(path, e)))
        files.sort()
        return files
    return [path]

def main():
    ap = argparse.ArgumentParser(description="Background-safe Macenko normalization (tiles or small images).")
    ap.add_argument("-Wsi", required=True, help="Image file or directory (WSI dir supported if pre-tiled).")
    ap.add_argument("-OutRoot", required=True, help="Output root directory.")
    ap.add_argument("-RefJson", required=False, default=None, help="JSON with W_ref, conc_p, Io_ref.")
    ap.add_argument("-BgPolicy", default="keep", choices=["keep","ref","local"], help="Background handling policy.")
    ap.add_argument("-EdgeSoftPx", type=int, default=4, help="Soft edge blending width in pixels.")
    ap.add_argument("-SaveTriPanel", type=int, default=1, help="Save tri-panel PNG per image.")
    args = ap.parse_args()

    os.makedirs(args.OutRoot, exist_ok=True)
    csv_path = os.path.join(args.OutRoot, "metrics.csv")
    tri_dir = os.path.join(args.OutRoot, "tri_panel"); os.makedirs(tri_dir, exist_ok=True)
    out_dir = os.path.join(args.OutRoot, "normalized"); os.makedirs(out_dir, exist_ok=True)

    W_ref, conc_p, Io_ref = _parse_ref_json(args.RefJson)
    if W_ref is None:
        # Reasonable default H&E basis in OD (columns)
        W_ref = np.array([[0.650, 0.072],
                          [0.704, 0.990],
                          [0.286, 0.105]], dtype=np.float32)
        # Normalize columns
        W_ref = W_ref / np.linalg.norm(W_ref, axis=0, keepdims=True)

    header = ["image","deltae_mean","deltae_std","deltae_p95","deltae_p99","tissue_coverage"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for img_path in load_images(args.Wsi):
            name = os.path.splitext(os.path.basename(img_path))[0]
            img = imread(img_path)
            if img.ndim==2:
                img = np.stack([img]*3, axis=-1)
            msk, _ = tissue_mask(img)
            out, _ = macenko_normalize(
                img, W_ref=W_ref, conc_p=conc_p, Io_ref=Io_ref,
                mask=msk, bg_policy=args.BgPolicy, edge_soft_px=args.EdgeSoftPx
            )
            # ΔE (mask-only)
            dE = deltaE_lab(img, out, mask=msk)
            stats = dE_stats(dE, mask=msk)
            cov = float(msk.mean())

            # save outputs
            out_path = os.path.join(out_dir, f"{name}_norm.png")
            imwrite(out_path, out)

            if args.SaveTriPanel:
                tri_path = os.path.join(tri_dir, f"{name}_tri.png")
                tri_panel(img, out, np.nan_to_num(dE, nan=0.0), save_path=tri_path)

            row = [name, stats["mean"], stats["std"], stats["p95"], stats["p99"], cov]
            writer.writerow(row)

    print("Done. Outputs at:", args.OutRoot)

if __name__ == "__main__":
    main()
