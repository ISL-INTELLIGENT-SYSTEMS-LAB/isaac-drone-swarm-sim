# Resumable chunk driver for the counting dataset. Re-run to resume (finished frames are skipped).
param([int]$Total = 4000, [int]$Chunk = 400, [int]$NMax = 1000, [int]$PoolGhost = 900)
$R = if ($env:DRONE_SCENE_ROOT) { $env:DRONE_SCENE_ROOT } else { Split-Path -Parent $PSScriptRoot }; Set-Location $R
$Py = if (Test-Path "$R\env\python.exe") { "$R\env\python.exe" } else { "$R\env\Scripts\python.exe" }
$env:OMNI_KIT_ACCEPT_EULA = "YES"; $env:PYTHONUNBUFFERED = "1"; $env:OUT = "$R\datasets\counting_v1"
$env:MODEL_MIX = "legacy"; $env:COLOR_MODE = "none"   # legacy datasets: original 3-model mix, no recolouring
$env:NMAX = "$NMax"; $env:POOL_GHOST = "$PoolGhost"; Remove-Item Env:NFORCE -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force "$R\datasets\counting_v1\logs" | Out-Null
for ($s = 0; $s -lt $Total; $s += $Chunk) {
  $e = [Math]::Min($s + $Chunk, $Total)
  for ($try = 0; $try -lt 2; $try++) {
    $env:START = "$s"; $env:END = "$e"
    "$(Get-Date -Format s) chunk $s-$e try $try" | Add-Content "$R\datasets\counting_v1\logs\driver.log"
    & $Py ds_tools\gen_counting.py 1> "$R\datasets\counting_v1\logs\chunk_$s.log" 2> "$R\datasets\counting_v1\logs\chunk_$s.err"
    $n = (Get-ChildItem "$R\datasets\counting_v1\ann\*.json" | Where-Object { $i = [int]$_.BaseName; $i -ge $s -and $i -lt $e }).Count
    "$(Get-Date -Format s) chunk $s-$e exit $LASTEXITCODE have $n" | Add-Content "$R\datasets\counting_v1\logs\driver.log"
    if ($n -ge ($e - $s)) { break }
  }
  $free = (Get-Item $R).PSDrive.Free / 1GB
  if ($free -lt 6) { "$(Get-Date -Format s) STOP: free $([Math]::Round($free,1)) GB" | Add-Content "$R\datasets\counting_v1\logs\driver.log"; break }
}
"$(Get-Date -Format s) DONE" | Add-Content "$R\datasets\counting_v1\logs\driver.log"
