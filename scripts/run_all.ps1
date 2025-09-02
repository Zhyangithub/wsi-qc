param(
  [Parameter(Mandatory = $true)][string]$Wsi,        # e.g. D:\wsi-qc\data\raw\test.svs
  [Parameter(Mandatory = $true)][string]$OutRoot,    # e.g. D:\wsi-qc\results\example
  [Parameter(Mandatory = $true)][string]$RefJson,    # e.g. D:\wsi-qc\refs\stain_ref.json

  [int]$TileSize = 512,
  [double]$QcThresh = 0.80,

  # ΔE 统计口径与阈值（默认 p95）
  [ValidateSet("p95","mean","max","median")][string]$DeltaeMetric = "p95",
  [int]$DeltaePct = 95,

  # Macenko 参数
  [double]$Alpha = 0.10, [double]$Beta = 0.15, [double]$Clamp = 1.0, [double]$Gamma = 1.0
)

$ErrorActionPreference = "Stop"

# ---------- 目录与文件 ----------
$tilesDir          = Join-Path $OutRoot "tiles"
$qcCsv             = Join-Path $OutRoot "qc.csv"
$qcOverlayCsv      = Join-Path $OutRoot "overlay.csv"
$tilesIndexCsv     = Join-Path $OutRoot "tiles_index.csv"

$normDir           = Join-Path $OutRoot "example_norm"
$deltaeTilesCsv    = Join-Path $normDir "deltae_tiles.csv"
$deltaeOverlayCsv  = Join-Path $normDir ("overlay_deltae_{0}.csv" -f $DeltaeMetric)

$analysisDir       = Join-Path $OutRoot "analysis"
$combinedOverlay   = Join-Path $analysisDir "combined_overlay.csv"

$triPanelPng       = Join-Path $OutRoot "tri_panel.png"

# 确保目录存在
$null = New-Item -ItemType Directory -Force -Path $OutRoot,$tilesDir,$normDir,$analysisDir

function RunStep([string]$title, [scriptblock]$block) {
  Write-Host "`n=== $title ===" -ForegroundColor Cyan
  & $block
  if ($LASTEXITCODE -ne 0) { throw "FAILED: $title" }
}

# ---------- 1) 切 tile ----------
RunStep "1/8 tile_wsi" {
  python -m scripts.tile_wsi --wsi "$Wsi" --out "$OutRoot" --tile $TileSize
}

# ---------- 2) QC 评分 + 导出 overlay ----------
RunStep "2/8 qc_score_tiles" {
  python -m scripts.qc_score_tiles --tiles "$tilesDir" --out "$qcCsv"
}
RunStep "3/8 export QC overlay" {
  python -m scripts.export_heatmap_to_tissuumaps --tile_csv "$qcCsv" --out "$qcOverlayCsv"
}

# ---------- 3) Macenko 参考（若不存在才标定） ----------
if (-not (Test-Path -Path "$RefJson")) {
  RunStep "4/8 calibrate_ref (build stain_ref.json)" {
    python -m scripts.calibrate_ref `
      --tiles_dir "$tilesDir" `
      --out_json "$RefJson" `
      --max_tiles 200 `
      --io_mode percentile `
      --io_percentile 95
  }
} else {
  Write-Host "ref_json exists => $RefJson" -ForegroundColor Yellow
}

# ---------- 4) 批量标准化 + 单张三联图 ----------
RunStep "5/8 stain_normalize (batch)" {
  python -m scripts.stain_normalize `
    --tiles_dir "$tilesDir" `
    --out_dir  "$normDir" `
    --ref_json "$RefJson" `
    --alpha $Alpha --beta $Beta --clamp $Clamp --gamma $Gamma
}

# 自动取第一张 tile 生成三联图
$firstTile = (Get-ChildItem -Path $tilesDir -Filter *.png | Select-Object -First 1 -ExpandProperty FullName)
if ($null -ne $firstTile) {
  RunStep "tri_panel (single tile)" {
    python -m scripts.tri_panel `
      --tile "$firstTile" `
      --ref_json "$RefJson" `
      --out_png "$triPanelPng" `
      --alpha $Alpha --beta $Beta --clamp $Clamp --gamma $Gamma
  }
} else {
  Write-Host "WARN: no tile found for tri-panel." -ForegroundColor Yellow
}

# ---------- 5) ΔE 明细 + ΔE overlay ----------
RunStep "6/8 compute_deltae_tiles" {
  python -m scripts.compute_deltae_tiles `
    --orig_tiles "$tilesDir" `
    --norm_tiles "$normDir" `
    --tiles_index_csv "$tilesIndexCsv" `
    --out_csv "$deltaeTilesCsv"
}

RunStep "7/8 export_deltae_to_tissuumaps" {
  python -m scripts.export_deltae_to_tissuumaps `
    --deltae_csv "$deltaeTilesCsv" `
    --out_csv   "$deltaeOverlayCsv" `
    --metric    $DeltaeMetric `
    --tile_size $TileSize `
    --position  topleft
}

# ---------- 6) 合成 keep/drop 覆盖层 ----------
RunStep "8/8 analyze_deltae_vs_qc (keep/drop overlay)" {
  python -m scripts.analyze_deltae_vs_qc `
    --qc_csv     "$qcCsv" `
    --deltae_csv "$deltaeTilesCsv" `
    --out_dir    "$analysisDir" `
    --qc_thresh  $QcThresh `
    --deltae_pct $DeltaePct
}

Write-Host "`n[OK] All done. Outputs:" -ForegroundColor Green
Write-Host (" tiles dir              : {0}" -f $tilesDir)
Write-Host (" tiles_index.csv        : {0}" -f $tilesIndexCsv)
Write-Host (" qc csv                 : {0}" -f $qcCsv)
Write-Host (" qc overlay             : {0}" -f $qcOverlayCsv)
Write-Host (" normalized dir         : {0}" -f $normDir)
Write-Host (" tri panel              : {0}" -f $triPanelPng)
Write-Host (" deltae tiles csv       : {0}" -f $deltaeTilesCsv)
Write-Host (" deltae overlay         : {0}" -f $deltaeOverlayCsv)
Write-Host (" combined overlay (keep): {0}" -f $combinedOverlay)
