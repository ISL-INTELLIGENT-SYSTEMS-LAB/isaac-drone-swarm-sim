# Breakaway top-up for tracking_v1: renders seq_036..seq_047 with FORCE_BRK=1 (every sequence gets 1-3 breakaways; cruise extended if short).
# Resumable: finished sequences have meta.json. Writes DONE_BRK to logs\driver_brk.log at the end.
param([int]$Start = 36, [int]$End = 48, [int]$PerProc = 4, [int]$NMaxT = 600)
$R = if ($env:DRONE_SCENE_ROOT) { $env:DRONE_SCENE_ROOT } else { Split-Path -Parent $PSScriptRoot }; Set-Location $R
$Py = if (Test-Path "$R\env\python.exe") { "$R\env\python.exe" } else { "$R\env\Scripts\python.exe" }
$env:OMNI_KIT_ACCEPT_EULA = "YES"; $env:PYTHONUNBUFFERED = "1"; $env:OUT = "$R\datasets\tracking_v1"; $env:NMAX_T = "$NMaxT"; $env:POOL_GHOST = "600"; $env:FORCE_BRK = "1"
$env:MODEL_MIX = "legacy"; $env:COLOR_MODE = "none"   # legacy datasets: original 3-model mix, no recolouring
$L = "$R\datasets\tracking_v1\logs"; New-Item -ItemType Directory -Force $L | Out-Null
for ($s = $Start; $s -lt $End; $s += $PerProc) {
  $e = [Math]::Min($s + $PerProc, $End)
  for ($try = 0; $try -lt 3; $try++) {
    $n = 0; for ($q = $s; $q -lt $e; $q++) { if (Test-Path ("$R\datasets\tracking_v1\seq_{0:D3}\meta.json" -f $q)) { $n++ } }
    if ($n -ge ($e - $s)) { break }
    $env:SEQ_START = "$s"; $env:SEQ_END = "$e"
    "$(Get-Date -Format s) seqs $s-$e try $try" | Add-Content "$L\driver_brk.log"
    & $Py ds_tools\gen_tracking.py 1> "$L\seqs_$s.log" 2> "$L\seqs_$s.err"
  }
  $free = (Get-Item $R).PSDrive.Free / 1GB
  if ($free -lt 6) { "$(Get-Date -Format s) STOP: free $([Math]::Round($free,1)) GB" | Add-Content "$L\driver_brk.log"; break }
}
"$(Get-Date -Format s) DONE_BRK" | Add-Content "$L\driver_brk.log"
