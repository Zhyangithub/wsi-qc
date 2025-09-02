$ErrorActionPreference = "Stop"

# Prompt for inputs
$wsi = Read-Host "Path to input WSI (.svs)"
$outRoot = Read-Host "Directory to store outputs"
$refJson = Read-Host "Path to stain reference JSON (will be created if missing)"

$tileSizeInput = Read-Host "Tile size [512]"
if ([string]::IsNullOrWhiteSpace($tileSizeInput)) { $tileSize = 512 } else { $tileSize = [int]$tileSizeInput }

$qcThreshInput = Read-Host "QC threshold [0.80]"
if ([string]::IsNullOrWhiteSpace($qcThreshInput)) { $qcThresh = 0.80 } else { $qcThresh = [double]$qcThreshInput }

$deltaeMetricInput = Read-Host "DeltaE metric (p95, mean, max, median) [p95]"
if ([string]::IsNullOrWhiteSpace($deltaeMetricInput)) { $deltaeMetric = "p95" } else { $deltaeMetric = $deltaeMetricInput }

$deltaePctInput = Read-Host "DeltaE percentile [95]"
if ([string]::IsNullOrWhiteSpace($deltaePctInput)) { $deltaePct = 95 } else { $deltaePct = [int]$deltaePctInput }

$alphaInput = Read-Host "Macenko alpha [0.10]"
if ([string]::IsNullOrWhiteSpace($alphaInput)) { $alpha = 0.10 } else { $alpha = [double]$alphaInput }

$betaInput = Read-Host "Macenko beta [0.15]"
if ([string]::IsNullOrWhiteSpace($betaInput)) { $beta = 0.15 } else { $beta = [double]$betaInput }

$clampInput = Read-Host "Macenko clamp [1.0]"
if ([string]::IsNullOrWhiteSpace($clampInput)) { $clamp = 1.0 } else { $clamp = [double]$clampInput }

$gammaInput = Read-Host "Macenko gamma [1.0]"
if ([string]::IsNullOrWhiteSpace($gammaInput)) { $gamma = 1.0 } else { $gamma = [double]$gammaInput }

# Derived paths
$tilesDir = Join-Path $outRoot "tiles"
$qcCsv = Join-Path $outRoot "qc.csv"
$qcOverlayCsv = Join-Path $outRoot "overlay.csv"
$tilesIndexCsv = Join-Path $outRoot "tiles_index.csv"
$normDir = Join-Path $outRoot "example_norm"
$deltaeTilesCsv = Join-Path $normDir "deltae_tiles.csv"
$deltaeOverlayCsv = Join-Path $normDir ("overlay_deltae_{0}.csv" -f $deltaeMetric)
$analysisDir = Join-Path $outRoot "analysis"
$triPanelPng = Join-Path $outRoot "tri_panel.png"

# Ensure directories exist
$null = New-Item -ItemType Directory -Force -Path $outRoot,$tilesDir,$normDir,$analysisDir

function RunStep([string]$title, [scriptblock]$block) {
  Write-Host "`n=== $title ===" -ForegroundColor Cyan
  & $block
  if ($LASTEXITCODE -ne 0) { throw "FAILED: $title" }
}

# 1) Tile WSI
RunStep "1/9 tile_wsi" { python scripts/tile_wsi.py --wsi "$wsi" --out "$outRoot" --tile $tileSize }

# 2) QC scoring
RunStep "2/9 qc_score_tiles" { python scripts/qc_score_tiles.py --tiles "$tilesDir" --out "$qcCsv" }
RunStep "3/9 export_heatmap_to_tissuumaps" { python scripts/export_heatmap_to_tissuumaps.py --tile_csv "$qcCsv" --out "$qcOverlayCsv" }

# 3) Calibrate stain reference if needed
if (-not (Test-Path -Path "$refJson")) {
  RunStep "4/9 calibrate_ref" { python -m scripts.calibrate_ref --tiles_dir "$tilesDir" --out_json "$refJson" --max_tiles 200 --io_mode percentile --io_percentile 95 }
} else {
  Write-Host "ref_json exists => $refJson" -ForegroundColor Yellow
}

# 4) Batch normalize tiles
RunStep "5/9 stain_normalize" { python -m scripts.stain_normalize --tiles_dir "$tilesDir" --out_dir "$normDir" --ref_json "$refJson" --alpha $alpha --beta $beta --clamp $clamp --gamma $gamma }

# Tri-panel for first tile
$firstTile = (Get-ChildItem -Path $tilesDir -Filter *.png | Select-Object -First 1 -ExpandProperty FullName)
if ($null -ne $firstTile) {
  RunStep "tri_panel" { python -m scripts.tri_panel --tile "$firstTile" --ref_json "$refJson" --out_png "$triPanelPng" --alpha $alpha --beta $beta --clamp $clamp --gamma $gamma }
} else {
  Write-Host "WARN: no tile found for tri-panel." -ForegroundColor Yellow
}

# 5) Compute DeltaE and overlays
RunStep "6/9 compute_deltae_tiles" { python -m scripts.compute_deltae_tiles --orig_tiles "$tilesDir" --norm_tiles "$normDir" --tiles_index_csv "$tilesIndexCsv" --out_csv "$deltaeTilesCsv" }

RunStep "7/9 export_deltae_to_tissuumaps" { python -m scripts.export_deltae_to_tissuumaps --deltae_csv "$deltaeTilesCsv" --out_csv "$deltaeOverlayCsv" --metric $deltaeMetric --tile_size $tileSize --position topleft }

RunStep "8/9 analyze_deltae_vs_qc" { python -m scripts.analyze_deltae_vs_qc --qc_csv "$qcCsv" --deltae_csv "$deltaeTilesCsv" --out_dir "$analysisDir" --qc_thresh $qcThresh --deltae_pct $deltaePct }

Write-Host "`n[OK] All done. Outputs at $outRoot" -ForegroundColor Green
