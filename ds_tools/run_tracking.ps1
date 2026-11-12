# Tracking / re-ID sequence driver. Resumable: finished sequences have meta.json; just re-run the same command.
#
# BINNED mode (default; for scene re-ID datasets): sequences are kept only if the MEDIAN n_visible_ge1px over the
# rendered frames falls inside the bin.  Output: <Out>\bin_LLL-HHH\seq_000 ... (one folder per bin).
#   .\ds_tools\run_tracking.ps1 -Seed 1000 -PerBin 10 -Out datasets\reid_membera_1-20
#   .\ds_tools\run_tracking.ps1 -Bins "1-5,6-10" -PerBin 2 -ClipFrames 16       # short clips, quick test
# Binned defaults: -ModelMix balanced (4 drone models) -ColorMode random (each drone its own colour for the whole sequence).
# LEGACY mode (old tracking_v1 behaviour, random swarm sizes, no bins): pass -Total
#   .\ds_tools\run_tracking.ps1 -Total 36
# SWARM options (both modes; defaults = old behaviour, same designs for the same seed):
#   -Swarms 2 -SwarmSep 1.0      two swarms flying side by side, gap = 1 swarm radius
#   -BrkProb 1 -BrkCount 1 -BrkGroup 5   every sequence gets 1 breakaway event of 5 drones leaving together
#   -Total 1 -NDrones 80 -NMaxT 100 -ModelMix balanced -ColorMode random -CamFraming chase -ClipFrames 32 -Swarms 2 -BrkProb 1 -BrkCount 1
#                                quick test: one 32-frame clip, 2 swarms of ~40, one breakaway
#   -Spread 0.6                  tighter formations (1.0 = original spacing)
#   -ReplicatorWriter            also write Replicator BasicWriter output (boxes, instance ids, 'swarm' semantic masks) to seq\replicator
#   Swarm annotations (seq\swarm\..., seq\swarm_masks\...) are written after every sequence unless -NoSwarmAnn.
param(
  [string]$Bins = "1-5,6-10,11-20,21-30,31-40,41-50,51-60,61-70,71-80,81-90,91-100",
  [int]$PerBin = 10,            # sequences per bin
  [int]$Seed = 77001,           # use a different seed per team member (e.g. 1000 * member number)
  [string]$Out = "",            # default datasets\tracking_bins (binned) / datasets\tracking_v1 (legacy)
  [int]$ClipFrames = 0,         # 0 = full takeoff->cruise sequence (140-200 frames); e.g. 16 = 16-frame clip from the cruise part
  [string]$CamFraming = "",     # chase (default in binned mode): camera follows the swarm close enough for re-ID crops; wide = original tracking framing
  [string]$ModelMix = "",       # balanced (default in binned mode) | isaac | ghost | legacy | "ghost:1,cf2x:1,quad:1,arl:1"
  [string]$ColorMode = "",      # random (default in binned mode) | palette | none
  [double]$ColorStrength = 0.85,# 0 = original colours, 1 = full paint
  [switch]$NoWear,              # keep original roughness / metallic
  [double]$CamDist = 1.0,       # camera distance multiplier; <1 = closer camera (bigger drones), >1 = farther
  [int]$MaxAccept = 30,         # designs tried per sequence before giving up (re-run to retry)
  [int]$Rt = 4,
  [int]$PerProc = 4,            # sequences per Isaac Sim process
  [int]$Total = 0,              # LEGACY: number of unbinned sequences
  [int]$NMaxT = 600,            # LEGACY: max swarm size
  [int]$NDrones = 0,            # >0: fixed number of drones per sequence (otherwise random / chosen for the bin)
  [int]$Swarms = 1,             # number of swarms (the formation is split into this many groups flying side by side)
  [double]$SwarmSep = 1.0,      # gap between neighbouring swarms, in swarm radii
  [double]$Spread = 1.0,        # formation spacing multiplier (<1 tighter, >1 looser)
  [double]$BrkProb = 0.5,       # probability a sequence has breakaways
  [int]$BrkCount = 0,           # breakaway events per sequence (0 = original random 1-3)
  [int]$BrkGroup = 1,           # drones per breakaway event (>1 = subgroup leaves together)
  [switch]$ReplicatorWriter,    # also attach Replicator BasicWriter -> seq\replicator
  [switch]$NoSwarmAnn           # skip swarm annotations (run ds_tools\swarm_annotate.py later)
)
$R = if ($env:DRONE_SCENE_ROOT) { $env:DRONE_SCENE_ROOT } else { Split-Path -Parent $PSScriptRoot }; Set-Location $R
$Py = if (Test-Path "$R\env\python.exe") { "$R\env\python.exe" } else { "$R\env\Scripts\python.exe" }
$env:OMNI_KIT_ACCEPT_EULA = "YES"; $env:PYTHONUNBUFFERED = "1"; $env:RT = "$Rt"; $env:SEED = "$Seed"
$env:COLOR_STRENGTH = "$ColorStrength"; $env:COLOR_WEAR = if ($NoWear) { "0" } else { "1" }; Remove-Item Env:POOL_EACH -ErrorAction SilentlyContinue
$env:CLIP_FRAMES = "$ClipFrames"; $env:CAM_DIST = "$CamDist"; $env:MAX_ACCEPT = "$MaxAccept"
Remove-Item Env:FORCE_BRK -ErrorAction SilentlyContinue
$env:N_DRONES = "$NDrones"; $env:SWARMS = "$Swarms"; $env:SWARM_SEP = "$SwarmSep"; $env:SPREAD = "$Spread"; $env:BRK_PROB = "$BrkProb"; $env:BRK_COUNT = "$BrkCount"; $env:BRK_GROUP = "$BrkGroup"
$env:REP_WRITER = if ($ReplicatorWriter) { "1" } else { "0" }; $env:SWARM_ANN = if ($NoSwarmAnn) { "0" } else { "1" }

function Run-Seqs([string]$O, [int]$N, [string]$Tag) {
  $env:OUT = $O; $L = "$O\logs"; New-Item -ItemType Directory -Force $L | Out-Null
  for ($s = 0; $s -lt $N; $s += $PerProc) {
    $e = [Math]::Min($s + $PerProc, $N)
    for ($try = 0; $try -lt 2; $try++) {
      $env:SEQ_START = "$s"; $env:SEQ_END = "$e"
      "$(Get-Date -Format s) $Tag seqs $s-$e try $try" | Add-Content "$L\driver.log"
      & $Py ds_tools\gen_tracking.py 1> "$L\seqs_$s.log" 2> "$L\seqs_$s.err"
      $n = 0; for ($q = $s; $q -lt $e; $q++) { if (Test-Path ("$O\seq_{0:D3}\meta.json" -f $q)) { $n++ } }
      "$(Get-Date -Format s) $Tag seqs $s-$e have $n" | Add-Content "$L\driver.log"
      if ($n -ge ($e - $s)) { break }
    }
    $free = (Get-Item $R).PSDrive.Free / 1GB
    if ($free -lt 6) { "$(Get-Date -Format s) STOP: free $([Math]::Round($free,1)) GB" | Add-Content "$L\driver.log"; exit 2 }
  }
}

if ($Total -gt 0) {
  if (-not $Out) { $Out = "$R\datasets\tracking_v1" }
  if (-not [IO.Path]::IsPathRooted($Out)) { $Out = Join-Path $R $Out }
  Remove-Item Env:BIN_LO, Env:BIN_HI -ErrorAction SilentlyContinue
  $env:NMAX_T = "$NMaxT"; $env:POOL_GHOST = "600"; $env:CAM_FRAMING = if ($CamFraming) { $CamFraming } else { "wide" }
  $env:MODEL_MIX = if ($ModelMix) { $ModelMix } else { "legacy" }; $env:COLOR_MODE = if ($ColorMode) { $ColorMode } else { "none" }
  Run-Seqs $Out $Total "legacy"
  "$(Get-Date -Format s) DONE" | Add-Content "$Out\logs\driver.log"; exit 0
}

$env:CAM_FRAMING = if ($CamFraming) { $CamFraming } else { "chase" }
$env:MODEL_MIX = if ($ModelMix) { $ModelMix } else { "balanced" }; $env:COLOR_MODE = if ($ColorMode) { $ColorMode } else { "random" }
if (-not $Out) { $Out = "$R\datasets\tracking_bins" }
if (-not [IO.Path]::IsPathRooted($Out)) { $Out = Join-Path $R $Out }
New-Item -ItemType Directory -Force $Out | Out-Null
$binList = @($Bins -split "," | ForEach-Object { $a = $_.Trim() -split "-"; @{ Lo = [int]$a[0]; Hi = [int]$a[1] } })
@{ bins = $Bins; per_bin = $PerBin; seed = $Seed; clip_frames = $ClipFrames; cam_dist = $CamDist; cam_framing = $env:CAM_FRAMING; model_mix = $env:MODEL_MIX; color_mode = $env:COLOR_MODE; color_strength = $ColorStrength; swarms = $Swarms; swarm_sep = $SwarmSep; spread = $Spread; brk_prob = $BrkProb; brk_count = $BrkCount; brk_group = $BrkGroup; replicator_writer = [bool]$ReplicatorWriter; bin_rule = "median n_visible_ge1px over rendered frames" } | ConvertTo-Json | Set-Content "$Out\run_config.json"
foreach ($b in $binList) {
  $env:BIN_LO = "$($b.Lo)"; $env:BIN_HI = "$($b.Hi)"
  $env:POOL_GHOST = "$([Math]::Max(20, [Math]::Ceiling($b.Hi * 1.3) + 10))"
  $tag = "bin_{0:D3}-{1:D3}" -f $b.Lo, $b.Hi
  Run-Seqs "$Out\$tag" $PerBin $tag
}
$summary = [ordered]@{}
foreach ($b in $binList) { $tag = "bin_{0:D3}-{1:D3}" -f $b.Lo, $b.Hi; $summary[$tag] = @(Get-ChildItem "$Out\$tag\seq_*\meta.json" -ErrorAction SilentlyContinue).Count }
$summary["updated"] = (Get-Date -Format s); ($summary | ConvertTo-Json) | Set-Content "$Out\progress.json"
"$(Get-Date -Format s) DONE $($summary | ConvertTo-Json -Compress)" | Add-Content "$Out\driver.log"
