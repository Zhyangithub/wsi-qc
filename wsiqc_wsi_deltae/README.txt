# WSI batch normalization + ΔE overlay (scripts)

This bundle gives you **tile-level batch normalization** (Macenko, via your existing `scripts/stain_normalize.py`),
**ΔE (Lab)** computation per tile, and a **TissUUmaps-ready overlay** for whole-slide visualization.

## Scripts
- `scripts/compute_deltae_tiles.py`: compute per-tile ΔE stats between **original** and **normalized** tiles.
- `scripts/export_deltae_to_tissuumaps.py`: turn the ΔE CSV into an overlay CSV with `x,y,deltae` for TissUUmaps.
- `scripts/batch_norm_and_deltae.py`: one-click orchestrator (normalize -> ΔE -> overlay).

## Quick start (PowerShell)
```powershell
# 0) You already have: stain_ref.json (from calibrate_ref) and stain_normalize.py

# 1) One-click batch for a slide
python -m scripts.batch_norm_and_deltae `
  --orig_tiles   D:\wsi-qc\results\example\tiles `
  --norm_tiles   D:\wsi-qc\results\example_norm `
  --ref_json     D:\wsi-qc\refs\stain_ref.json `
  --tile_size    512 `
  --tiles_index_csv D:\wsi-qc\results\example\tiles_index.csv

# Outputs
# - Normalized tiles -> D:\wsi-qc\results\example_norm\*.png
# - Per-tile ΔE stats -> D:\wsi-qc\results\example_norm\deltae_tiles.csv
# - TissUUmaps overlay -> D:\wsi-qc\results\example_norm\overlay_deltae_p95.csv
```

## Notes
- Coordinates: if `tiles_index.csv` is missing, the parser falls back to filenames like `x0003072_y0025600.png`.
- Metric: use `--metric median` if you prefer a conservative overlay; `p95` highlights outliers at tile edges.
- ΔE tile images: add `--save_deltae_dir` to `compute_deltae_tiles.py` to save per-tile ΔE visuals (16-bit PNG).
