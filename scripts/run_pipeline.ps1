# This wrapper script runs the full WSI QC workflow.
# It prompts the user for required paths and optional parameters,
# then invokes run_all.ps1 with those values.

# Prompt for mandatory inputs
$wsi = Read-Host "Path to input WSI (.svs)"
$outRoot = Read-Host "Directory to store outputs"
$refJson = Read-Host "Path to stain reference JSON (will be created if missing)"

# Optional parameters with defaults
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

# Call the main pipeline script with collected parameters
$runAll = Join-Path $PSScriptRoot 'run_all.ps1'
& $runAll -Wsi $wsi -OutRoot $outRoot -RefJson $refJson -TileSize $tileSize -QcThresh $qcThresh -DeltaeMetric $deltaeMetric -DeltaePct $deltaePct -Alpha $alpha -Beta $beta -Clamp $clamp -Gamma $gamma
