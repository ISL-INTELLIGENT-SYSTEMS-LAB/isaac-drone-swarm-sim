# Resumable driver for visible-count counting bins. Re-run the same command to resume.
# A frame is kept only if its n_visible_ge1px is inside its bin. Bin i (0-based, in -Bins order) owns frame indices
# [i*PerBin, (i+1)*PerBin) in <Out>\images|masks|ann; bins.json in <Out> records the mapping.
#   .\ds_tools\run_counting_bins.ps1                                   # 11 bins 1-5 ... 91-100, 200 frames each
#   .\ds_tools\run_counting_bins.ps1 -Bins "1-5,6-10" -PerBin 3 -Seed 1000 -Out datasets\smoke_bins
# Old 4-bin set (counting_bins_v1), identical layout to before:
#   .\ds_tools\run_counting_bins.ps1 -Bins "1-10,11-50,51-100,101-150" -Out datasets\counting_bins_v1 -Seed 20261106
param(
  [string]$Bins = "1-5,6-10,11-20,21-30,31-40,41-50,51-60,61-70,71-80,81-90,91-100",
  [int]$PerBin = 200,
  [int]$Seed = 20261106,
  [string]$Out = "datasets\counting_bins",
  [string]$ModelMix = "legacy",  # legacy (75% Ghost) | balanced | isaac | ghost | "type:weight,..."
  [string]$ColorMode = "none",   # none | random | palette (new colour per drone per frame)
  [double]$ColorStrength = 0.85,
  [int]$Chunk = 50,
  [int]$PoolGhost = 400,
  [int]$Rt = 6
)
$R = if ($env:DRONE_SCENE_ROOT) { $env:DRONE_SCENE_ROOT } else { Split-Path -Parent $PSScriptRoot }; Set-Location $R
$Py = if (Test-Path "$R\env\python.exe") { "$R\env\python.exe" } else { "$R\env\Scripts\python.exe" }
if (-not [IO.Path]::IsPathRooted($Out)) { $Out = Join-Path $R $Out }
$env:OMNI_KIT_ACCEPT_EULA = "YES"; $env:PYTHONUNBUFFERED = "1"
$env:OUT = $Out; $env:SEED = "$Seed"
$env:POOL_GHOST = "$PoolGhost"; $env:POOL_CF = "100"; $env:POOL_QUAD = "100"
$env:RT = "$Rt"; $env:MAX_ACCEPT = "40"
$env:MODEL_MIX = $ModelMix; $env:COLOR_MODE = $ColorMode; $env:COLOR_STRENGTH = "$ColorStrength"; $env:COLOR_WEAR = "1"; Remove-Item Env:POOL_EACH -ErrorAction SilentlyContinue
Remove-Item Env:NFORCE -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force "$Out\logs","$Out\images","$Out\masks","$Out\ann" | Out-Null

$binList = @(); $i = 0
foreach ($tok in ($Bins -split ",")) {
  $a = $tok.Trim() -split "-"; $binList += @{ Lo = [int]$a[0]; Hi = [int]$a[1]; Start = $i * $PerBin }; $i++
}
(@{ bins = @($binList | ForEach-Object { [ordered]@{ bin = "$($_.Lo)-$($_.Hi)"; first_frame = $_.Start; last_frame = $_.Start + $PerBin - 1 } }); per_bin = $PerBin; seed = $Seed; model_mix = $ModelMix; color_mode = $ColorMode; bin_rule = "n_visible_ge1px of the frame" } | ConvertTo-Json -Depth 4) | Set-Content "$Out\bins.json"

function Write-ProgressSummary {
  $summary = [ordered]@{}; $total = 0
  foreach ($b in $binList) {
    $lo = $b.Start; $hi = $b.Start + $PerBin
    $n = @(Get-ChildItem "$Out\ann\*.json" -ErrorAction SilentlyContinue | Where-Object { $j = [int]$_.BaseName; $j -ge $lo -and $j -lt $hi }).Count
    $summary["$($b.Lo)-$($b.Hi)"] = $n; $total += $n
  }
  $summary["total"] = $total; $summary["updated"] = (Get-Date -Format s)
  ($summary | ConvertTo-Json) | Set-Content "$Out\progress.json"
  "$(Get-Date -Format s) progress $($summary | ConvertTo-Json -Compress)" | Add-Content "$Out\logs\driver.log"
}

"$(Get-Date -Format s) START bins=$Bins perbin=$PerBin seed=$Seed chunk=$Chunk pool_ghost=$PoolGhost rt=$Rt" | Add-Content "$Out\logs\driver.log"
foreach ($b in $binList) {
  $binStart = $b.Start; $binEnd = $b.Start + $PerBin
  $env:BIN_LO = "$($b.Lo)"; $env:BIN_HI = "$($b.Hi)"
  $env:NMAX = "$([Math]::Max($b.Hi + 80, 220))"
  for ($s = $binStart; $s -lt $binEnd; $s += $Chunk) {
    $e = [Math]::Min($s + $Chunk, $binEnd)
    for ($try = 0; $try -lt 2; $try++) {
      $env:START = "$s"; $env:END = "$e"
      "$(Get-Date -Format s) bin $($b.Lo)-$($b.Hi) chunk $s-$e try $try" | Add-Content "$Out\logs\driver.log"
      $logBase = "$Out\logs\bin_$($b.Lo)_$($b.Hi)_chunk_$s"
      & $Py ds_tools\gen_counting_bins.py 1> "$logBase.log" 2> "$logBase.err"
      $n = @(Get-ChildItem "$Out\ann\*.json" -ErrorAction SilentlyContinue | Where-Object { $j = [int]$_.BaseName; $j -ge $s -and $j -lt $e }).Count
      "$(Get-Date -Format s) bin $($b.Lo)-$($b.Hi) chunk $s-$e exit $LASTEXITCODE have $n / $($e-$s)" | Add-Content "$Out\logs\driver.log"
      if ($n -ge ($e - $s)) { break }
    }
    Write-ProgressSummary
    $free = (Get-Item $R).PSDrive.Free / 1GB
    if ($free -lt 6) { "$(Get-Date -Format s) STOP: free $([Math]::Round($free,1)) GB" | Add-Content "$Out\logs\driver.log"; exit 2 }
  }
}
Write-ProgressSummary
"$(Get-Date -Format s) DONE" | Add-Content "$Out\logs\driver.log"
