# WSI-QC Project — 2-page Summary (v0 placeholder)

## Problem
H&E WSI quality varies due to staining differences and common artifacts (streaks, bubbles, folds). We need robust stain normalization and automated QC with interpretable heatmaps.

## Data
TCGA WSI (H&E), plus optional in-house slides later. Start with 20–50 slides across 2–3 cancer types.

## Method (MVP)
- Macenko/Vahadane normalization (pick one).
- Artifact detection: brightness/blur + FFT-based streak score.
- Tile-level QC -> heatmap; uncertainty via MC Dropout (v1).

## Metrics
Color distance histograms, ROC-AUC on synthetic artifacts, cross-cohort robustness, runtime.

## Deliverables
TissUUmaps overlay/ plugin, code, and a short paper/workshop submission.
