# WSI-QC: Stain Normalization + Artifact Quality Control (TissUUmaps Plugin)

**Goal (8 weeks MVP):**
- Implement Macenko/Vahadane stain normalization + at least one artifact detector (streaks/air-bubbles/folds).
- Produce tile-level QC heatmaps and uncertainty scores.
- Ship a minimal **TissUUmaps** plugin for interactive visualization.

This repo was bootstrapped on 2025-08-31 as a starter template.

## Layout
```
wsi-qc/
  env/environment.yml         # conda env (conda-forge + pip)
  scripts/                    # CLIs for tiling, QC, exporting overlays
  src/                        # python packages (preprocessing, qc, viz)
  plugins/tissuumaps_wsi_qc/  # plugin scaffold (to be wired later)
  data/raw/                   # put WSI here (ignored by git)
  results/                    # outputs (ignored by git)
  notebooks/                  # sanity checks, quick EDA
  docs/                       # 2-page summary etc.
```
## Quickstart
1) Create conda env:
```
mamba env create -f env/environment.yml
conda activate wsiqc
```
2) Verify OpenSlide & libraries:
```
python scripts/sanity_check.py
```
3) Tile a WSI and compute a tiny QC score (brightness+blur):
```
python scripts/tile_wsi.py --wsi data/raw/example.svs --out results/example --tile 512
python scripts/qc_score_tiles.py --tiles results/example/tiles --out results/example/qc.csv
python scripts/export_heatmap_to_tissuumaps.py --tile_csv results/example/qc.csv --out results/example/overlay.csv
```
4) Load the base image in TissUUmaps and overlay `overlay.csv` as a heatmap.
