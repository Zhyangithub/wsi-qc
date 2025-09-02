
# Background-safe Macenko (drop-in replacement)

**What's new**
- Background is never stained (OD threshold + tissue mask).
- Soft edge blending to avoid pink halos.
- Configurable background policy: `keep` (default), `ref`, or `local`.
- ΔE statistics computed on tissue only.

## Usage (one-line)
```bash
python run_all.py \
  -Wsi /path/to/images_or_dir \
  -OutRoot /path/to/out \
  -RefJson /path/to/macenko_ref.json \
  -BgPolicy keep \
  -EdgeSoftPx 4 \
  -SaveTriPanel 1
```

### `RefJson` example
```json
{
  "W_ref": [[0.650, 0.072],
            [0.704, 0.990],
            [0.286, 0.105]],
  "conc_p": [1, 99],
  "Io_ref": [255, 255, 255]
}
```
`W_ref` can be 3x2 (columns as H, E); if you provide 2x3 it will be transposed automatically.

## Outputs
- `normalized/NAME_norm.png` — background-safe normalized image
- `tri_panel/NAME_tri.png` — Original | Normalized | ΔE
- `metrics.csv` — `deltae_mean, deltae_std, deltae_p95, deltae_p99, tissue_coverage`

## Where to patch in your repo
If you already have a pipeline, you can **replace the Macenko implementation** with:
```python
from stain_norm.macenko_bg import macenko_normalize, tissue_mask
```
and call `macenko_normalize(..., bg_policy="keep")`. ΔE should be computed only on `mask`.

## Requirements
- numpy
- scipy
- scikit-image
- opencv-python
- imageio
- matplotlib
```
