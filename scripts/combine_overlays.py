# -*- coding: utf-8 -*-
"""
Merge QC and ΔE stats into a single overlay and compute keep/drop.
- join by (x, y) if present; otherwise fall back to basename('tile') keys.
- keep rule (exposed): qc_score >= qc_thresh AND deltae <= threshold
  where threshold is either absolute (--deltae_abs) or percentile (--deltae_thresh_pct).

Inputs:
  --qc_csv:        CSV with [x,y,qc_score] (from qc_score_tiles.py)
  --deltae_csv:    CSV with ΔE stats per tile (columns like deltae_mean, deltae_median, deltae_p95, deltae_max)
  --out_csv:       Output overlay CSV (x,y,qc_score,deltae,keep)
Optional:
  --deltae_metric: Which column/stat to use from deltae_csv: {median|mean|p95|max|deltae}. Default: p95
  --deltae_thresh_pct: Percentile for ΔE threshold (<= passes). Default: 95
  --deltae_abs:    Absolute ΔE threshold (<= passes). Overrides percentile if provided.
  --qc_thresh:     QC threshold (>= passes). Default: 0.80
"""
import argparse, os
import numpy as np
import pandas as pd

def _key_cols(df: pd.DataFrame):
    if {'x','y'}.issubset(df.columns):
        return ('x','y')
    cand = [c for c in df.columns if 'tile' in c.lower() or 'path' in c.lower() or 'name' in c.lower()]
    if cand:
        c = cand[0]
        df['_key'] = df[c].map(lambda p: os.path.splitext(os.path.basename(str(p)))[0])
        return ('_key',)
    raise ValueError("Neither (x,y) nor tile/path column found to join.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--qc_csv', required=True)
    ap.add_argument('--deltae_csv', required=True)
    ap.add_argument('--out_csv', required=True)
    ap.add_argument('--qc_thresh', type=float, default=0.80)
    ap.add_argument('--deltae_metric', default='p95', choices=['median','mean','p95','max','deltae'])
    ap.add_argument('--deltae_thresh_pct', type=int, default=95)
    ap.add_argument('--deltae_abs', type=float, default=None)
    ap.add_argument('--tile_size', type=int, default=512)
    ap.add_argument('--position', default='topleft')
    args = ap.parse_args()

    qc = pd.read_csv(args.qc_csv)
    de = pd.read_csv(args.deltae_csv)

    # normalize column names
    qc.columns = [c.strip() for c in qc.columns]
    de.columns = [c.strip() for c in de.columns]

    # qc_score column fallback
    if 'qc_score' not in qc.columns:
        if 'score' in qc.columns:
            qc = qc.rename(columns={'score':'qc_score'})
        else:
            raise ValueError(f"{args.qc_csv} must contain 'qc_score' (or 'score').")

    # choose deltae column
    colmap = {
        'median': 'deltae_median',
        'mean':   'deltae_mean',
        'p95':    'deltae_p95',
        'max':    'deltae_max',
        'deltae': 'deltae',  # already computed outside
    }
    de_col = colmap.get(args.deltae_metric, args.deltae_metric)
    if de_col not in de.columns:
        # try case-insensitive search
        cand = [c for c in de.columns if c.lower() == de_col.lower()]
        if cand:
            de_col = cand[0]
        else:
            raise ValueError(f"Column '{de_col}' not found in {args.deltae_csv}. Available: {list(de.columns)[:10]}…")

    # keys for join
    kq = _key_cols(qc)
    kd = _key_cols(de)
    merged = qc.merge(de[[*kd, de_col]], left_on=list(kq), right_on=list(kd), how='inner')
    if merged.empty:
        raise RuntimeError("Merged DataFrame is empty. Check key columns (x,y or tile basename).")

    # ensure x,y exist for overlay
    if not {'x','y'}.issubset(merged.columns):
        # bring over from either side with suffixes if present
        for side in ('_x','_y','_qc','_de'):
            cx = [c for c in merged.columns if c.endswith('x'+side) or c == 'x'+side]
            cy = [c for c in merged.columns if c.endswith('y'+side) or c == 'y'+side]
            if cx and cy:
                merged['x'] = merged[cx[0]]
                merged['y'] = merged[cy[0]]
                break
    if not {'x','y'}.issubset(merged.columns):
        raise RuntimeError("Cannot infer x,y for overlay output.")

    # assemble final columns
    merged = merged.rename(columns={de_col: 'deltae'})
    merged['qc_score'] = pd.to_numeric(merged['qc_score'], errors='coerce')
    merged['deltae']   = pd.to_numeric(merged['deltae'], errors='coerce')
    merged = merged.dropna(subset=['qc_score','deltae','x','y']).copy()

    # thresholds
    if args.deltae_abs is not None:
        thr = float(args.deltae_abs)
    else:
        thr = float(np.percentile(merged['deltae'].values, args.deltae_thresh_pct))
    keep = (merged['qc_score'] >= args.qc_thresh) & (merged['deltae'] <= thr)
    merged['keep'] = keep.astype(int)

    out = merged[['x','y','qc_score','deltae','keep']].copy()
    out.to_csv(args.out_csv, index=False)
    print(f"[ok] combined overlay -> {args.out_csv} (n={len(out)})")
    print(f"     thresholds: qc_score >= {args.qc_thresh:.3f}, deltae <= {thr:.3f} ({'abs' if args.deltae_abs is not None else f'P{args.deltae_thresh_pct}'})")

if __name__ == '__main__':
    main()
