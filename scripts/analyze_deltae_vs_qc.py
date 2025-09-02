# -*- coding: utf-8 -*-
"""
Analyze DeltaE (Lab) vs QC score, export a combined overlay for TissUUmaps,
and produce a dual-threshold keep list.

Input:
  --qc_csv:       CSV with at least columns [x, y, qc_score] (from qc_score_tiles.py)
  --deltae_csv:   CSV with at least columns [x, y, deltae]    (from compute_deltae_tiles.py)
  --out_dir:      Output directory

Optional:
  --qc_thresh:    float, default 0.80 (qc_score >= thresh -> pass)
  --deltae_pct:   int in [50..100], default 95 (deltae <= Pct(deltae_pct) -> pass)
  --deltae_abs:   float, if given use absolute threshold instead of percentile
  --fig_dpi:      int, default 180

Outputs:
  combined_overlay.csv     (x,y,qc_score,deltae,keep,tile[if available])
  keep_list.csv            (subset rows with keep==1)
  qc_vs_deltae_scatter.png (scatter/hexbin with correlation)
  distributions.png        (histograms of qc_score & deltae)
  summary.txt              (Pearson/Spearman correlation & thresholds)
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def read_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normalize column names
    cols = {c.lower(): c for c in df.columns}
    # required columns
    req_xy = []
    # find x,y
    cand_x = [c for c in df.columns if c.lower() == "x"]
    cand_y = [c for c in df.columns if c.lower() == "y"]
    if not cand_x or not cand_y:
        raise ValueError(f"{path} must contain columns 'x' and 'y'")
    # cast to numeric
    df[cand_x[0]] = pd.to_numeric(df[cand_x[0]], errors="coerce")
    df[cand_y[0]] = pd.to_numeric(df[cand_y[0]], errors="coerce")
    df = df.rename(columns={cand_x[0]: "x", cand_y[0]: "y"})
    # keep tile if exists
    if "tile" in [c.lower() for c in df.columns]:
        # find actual column name
        for c in df.columns:
            if c.lower() == "tile":
                df = df.rename(columns={c: "tile"})
                break
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qc_csv", required=True)
    ap.add_argument("--deltae_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--qc_thresh", type=float, default=0.80)
    ap.add_argument("--deltae_pct", type=int, default=95,
                   help="Use percentile threshold on deltaE (<= this percentile passes). Ignored if --deltae_abs is set.")
    ap.add_argument("--deltae_abs", type=float, default=None,
                   help="Absolute threshold on deltaE (<= passes). Overrides --deltae_pct if provided.")
    ap.add_argument("--fig_dpi", type=int, default=180)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load
    df_qc = read_df(args.qc_csv)
    if "qc_score" not in [c.lower() for c in df_qc.columns]:
        # try to find 'score'
        if "score" in [c.lower() for c in df_qc.columns]:
            for c in df_qc.columns:
                if c.lower() == "score":
                    df_qc = df_qc.rename(columns={c: "qc_score"})
                    break
        else:
            raise ValueError(f"{args.qc_csv} must contain 'qc_score' or 'score' column.")
    else:
        for c in df_qc.columns:
            if c.lower() == "qc_score":
                # rename to consistent
                if c != "qc_score":
                    df_qc = df_qc.rename(columns={c: "qc_score"})
                break

    df_de = read_df(args.deltae_csv)
    # deltae column
    if "deltae" not in [c.lower() for c in df_de.columns]:
        # try typical names
        cand = [c for c in df_de.columns if c.lower() in ("de", "d_e", "delta_e", "lab_de", "de_lab")]
        if not cand:
            raise ValueError(f"{args.deltae_csv} must contain a 'deltae' column.")
        df_de = df_de.rename(columns={cand[0]: "deltae"})
    else:
        for c in df_de.columns:
            if c.lower() == "deltae":
                if c != "deltae":
                    df_de = df_de.rename(columns={c: "deltae"})
                break

    # Merge on (x,y)
    on_cols = ["x", "y"]
    common_cols = ["x", "y"]
    if "tile" in df_qc.columns and "tile" in df_de.columns:
        # we keep both but still merge on (x,y) which is safer
        pass

    df = pd.merge(df_qc, df_de, on=on_cols, how="inner", suffixes=("_qc", "_de"))
    if df.empty:
        raise RuntimeError("Merged DataFrame is empty. Check that x,y coordinates match between QC and DeltaE CSVs.")

    # Correlation
    qc = pd.to_numeric(df["qc_score"], errors="coerce")
    de = pd.to_numeric(df["deltae"], errors="coerce")
    m = qc.notna() & de.notna()
    qc, de = qc[m], de[m]

    pearson_r, pearson_p = stats.pearsonr(qc, de) if len(qc) > 2 else (np.nan, np.nan)
    spear_r, spear_p   = stats.spearmanr(qc, de) if len(qc) > 2 else (np.nan, np.nan)

    # ΔE threshold
    if args.deltae_abs is not None:
        de_thr = float(args.deltae_abs)
        thr_desc = f"abs ≤ {de_thr:.3f}"
        pass_color = de <= de_thr
    else:
        de_thr = np.percentile(de, args.deltae_pct)
        thr_desc = f"≤ P{args.deltae_pct} ({de_thr:.3f})"
        pass_color = de <= de_thr

    # QC threshold
    pass_qc = qc >= args.qc_thresh

    # keep: both pass
    keep = (pass_qc & pass_color).astype(int)

    # Export combined overlay
    df_out = df.copy()
    df_out["keep"] = keep
    # make sure expected columns present for TissUUmaps
    cols_export = ["x", "y", "qc_score", "deltae", "keep"]
    if "tile" in df_out.columns:
        cols_export.append("tile")
    df_out_export = df_out[cols_export].copy()
    df_out_export.to_csv(out_dir / "combined_overlay.csv", index=False)

    # keep list (only passed)
    df_keep = df_out_export[df_out_export["keep"] == 1].copy()
    df_keep.to_csv(out_dir / "keep_list.csv", index=False)

    # Figures
    # 1) scatter/hexbin
    fig = plt.figure(figsize=(6,5))
    hb = plt.hexbin(qc, de, gridsize=50, cmap="magma", mincnt=1)
    plt.colorbar(hb, label="count")
    plt.xlabel("QC score (↑ better)")
    plt.ylabel("ΔE (Lab; ↑ larger color change)")
    plt.title(f"QC vs ΔE\nPearson r={pearson_r:.3f} (p={pearson_p:.1e}), Spearman ρ={spear_r:.3f} (p={spear_p:.1e})")
    plt.tight_layout()
    fig.savefig(out_dir / "qc_vs_deltae_scatter.png", dpi=args.fig_dpi)
    plt.close(fig)

    # 2) distributions
    fig = plt.figure(figsize=(10,4))
    ax1 = plt.subplot(1,2,1)
    ax1.hist(qc, bins=40, color="#4C78A8")
    ax1.axvline(args.qc_thresh, color="k", linestyle="--", label=f"QC ≥ {args.qc_thresh:.2f}")
    ax1.set_title("QC score distribution")
    ax1.set_xlabel("qc_score"); ax1.set_ylabel("count"); ax1.legend()

    ax2 = plt.subplot(1,2,2)
    ax2.hist(de, bins=40, color="#F58518")
    if args.deltae_abs is not None:
        ax2.axvline(de_thr, color="k", linestyle="--", label=f"ΔE {thr_desc}")
    else:
        ax2.axvline(de_thr, color="k", linestyle="--", label=f"ΔE {thr_desc}")
    ax2.set_title("ΔE (Lab) distribution")
    ax2.set_xlabel("ΔE"); ax2.set_ylabel("count"); ax2.legend()

    plt.tight_layout()
    fig.savefig(out_dir / "distributions.png", dpi=args.fig_dpi)
    plt.close(fig)

    # Summary
    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("DeltaE × QC analysis summary\n")
        f.write("="*40 + "\n")
        f.write(f"Rows merged: {len(df)}\n")
        f.write(f"QC threshold: qc_score ≥ {args.qc_thresh:.3f}\n")
        if args.deltae_abs is not None:
            f.write(f"ΔE threshold: deltae ≤ {de_thr:.3f} (absolute)\n")
        else:
            f.write(f"ΔE threshold: deltae ≤ P{args.deltae_pct} = {de_thr:.3f}\n")
        f.write(f"Keep (both pass): {int(df_keep.shape[0])} / {len(df)} ({100.0*df_keep.shape[0]/len(df):.2f}%)\n\n")
        f.write(f"Pearson r = {pearson_r:.4f}, p = {pearson_p:.3e}\n")
        f.write(f"Spearman ρ = {spear_r:.4f}, p = {spear_p:.3e}\n")

    print(f"[ok] combined overlay  -> {out_dir/'combined_overlay.csv'}")
    print(f"[ok] keep list         -> {out_dir/'keep_list.csv'}")
    print(f"[ok] scatter figure    -> {out_dir/'qc_vs_deltae_scatter.png'}")
    print(f"[ok] distributions     -> {out_dir/'distributions.png'}")
    print(f"[ok] summary           -> {out_dir/'summary.txt'}")


if __name__ == "__main__":
    main()
