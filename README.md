# drone-scene-sim generator kit

Generate synthetic drone datasets (swarm boxes, counting, tracking, scene re-ID) with **Isaac Sim 6.2** on Windows + an NVIDIA GPU.
The desert scene and the drone models stream from NVIDIA's Omniverse S3 bucket at run time, so the kit itself is only scripts.

Contents: [Requirements](#requirements) · [Setup](#one-time-setup) · [Swarm annotations](#swarm-annotations) · [Re-ID datasets](#scene-re-id-datasets-tracking-sequences-per-visible-count-bin) ·
[Team assignment](#team-assignment-re-id-data-collection) · [Counting bins](#counting-bins-single-frames) · [Other generators](#other-generators) ·
[Output formats](#output-formats) · [Explore OpenUSD](#explore-openusd) · [Speed](#speed-rtx-3060) · [License](#license--assets)

## Requirements

- Windows 10/11 x64, NVIDIA GPU (RTX 3060 12 GB works; more VRAM helps big swarms), recent NVIDIA driver
- Python **3.12** (64-bit)
- Disk: ~25 GB for the Isaac Sim env, plus datasets (roughly 0.3-0.5 MB per frame incl. masks)
- Internet on the first run (scene + drone assets are downloaded into the local cache)

## One-time setup

In PowerShell, in the unzipped kit folder:

```powershell
py -3.12 -m venv env
.\env\Scripts\python.exe -m pip install --upgrade pip
$env:OMNI_KIT_ACCEPT_EULA = "YES"
.\env\Scripts\python.exe -m pip install "isaacsim[all,extscache]==6.2.0" --extra-index-url https://pypi.nvidia.com
```

- The `.ps1` drivers find Python automatically (`env\Scripts\python.exe` for a venv, `env\python.exe` for a conda env) and set `OMNI_KIT_ACCEPT_EULA=YES`.
  When you call a script by hand, use `.\env\Scripts\python.exe` (venv) and set `$env:OMNI_KIT_ACCEPT_EULA = "YES"` first.
- If PowerShell refuses to run scripts: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` (once).
- The first launch takes several minutes (shader + asset caches). Later runs start in ~25 s.
- Never zip or share `env\` (~22 GB); every machine makes its own.

## Layout

```
ds_tools/                generators (gen_*.py), drivers (run_*.ps1), exporters and checks
datasets/stages/heightmap.npz   terrain height map (required, do not delete)
stages/                  sample USD layers to open in the GUI (see Explore OpenUSD)
viz_scripts/             optional visualisation helpers (viz_swarm.py draws swarm boxes)
```

Paths are relative to the kit folder (or set `DRONE_SCENE_ROOT`). Everything is written under `datasets\`.
All drivers are **resumable**: if a run stops, run the exact same command again and it continues.

## Swarm annotations

**Goal:** one box per **swarm** (1 drone up to thousands), its coordinates every frame, and breakaways flagged. We keep the
per-drone labels too, because the plan is *detect drones (or a drone mask / heatmap) -> cluster -> swarm boxes -> track*.

Every tracking sequence now also gets swarm labels (written automatically after each sequence):

```
seq_000\swarm\swarm_annotations.json   everything: swarms, breakaways, events, per-frame boxes, 2D-cluster tracks
seq_000\swarm\labels_yolo\000001.txt   YOLO: "class cx cy w h" (normalised), class 0 = swarm, 1 = breakaway
seq_000\swarm\coco_swarm.json          COCO: category 1 = swarm, 2 = breakaway, each box has a track_id
seq_000\swarm\swarm_tracks.csv         frame,id,class,x1,y1,x2,y2,cx,cy,n_visible,n_total
seq_000\swarm_masks\000001.png         binary swarm mask (255 = any drone pixel)
```

How the labels are made (ground truth):
- Every drone has a `swarm_id` from the scenario (`frames.jsonl` -> `drones[].swarm_id`). Swarm box = tight box around the
  visible drones of that swarm (from the drone masks). Also stored: `centroid_px`, `centroid_m` (3D), `n_visible`, `n_total`. Swarm IDs (1, 2, ...) never change.
- A scripted breakaway (one drone or a subgroup) stays part of its swarm until it is more than `max(3 m, 1.5 x swarm radius)` from the
  swarm centre and has moved at least 3 m farther out than where it started (`--brk-k`, `--brk-min-m` in `swarm_annotate.py`). From that frame on it is its own box: id 101, 102, ..., class `breakaway`. The JSON logs both the frame where
  it starts to leave (`breakaway_scripted_start`) and the frame where it counts as detached (`breakaway`).
- `clusters_2d` in the JSON is what a detector-side post-process would see: visible drones are clustered per frame and tracked
  without drone IDs; it logs `split`, `merge` and `breakaway` events. `eval` compares those events with the ground truth.

Scenario options (`run_tracking.ps1`; defaults reproduce the old sequences exactly):

| option | default | meaning |
|---|---|---|
| `-Swarms N` | 1 | split the formation into N swarms flying side by side |
| `-SwarmSep x` | 1.0 | gap between neighbouring swarms, in swarm radii |
| `-Spread x` | 1.0 | drone spacing multiplier (0.5 = twice as dense) |
| `-BrkProb p` | 0.5 | chance that a sequence has breakaways |
| `-BrkCount n` | 0 | breakaway events per sequence (0 = random 1-3) |
| `-BrkGroup n` | 1 | drones per breakaway event (>1 = a subgroup leaves together) |
| `-NDrones n` | 0 | fixed number of drones (0 = random, or picked for the bin) |
| `-ReplicatorWriter` | off | also save Replicator BasicWriter output in `seq\replicator\` |
| `-NoSwarmAnn` | off | skip swarm labels (make them later with `swarm_annotate.py`) |

```powershell
# quick test: one 32-frame clip, 2 swarms of 40, one breakaway of 3 drones, then draw it
.\ds_tools\run_tracking.ps1 -Total 1 -NDrones 80 -NMaxT 100 -ModelMix balanced -ColorMode random -CamFraming chase `
    -ClipFrames 32 -Swarms 2 -BrkProb 1 -BrkCount 1 -BrkGroup 3 -Out datasets\swarm_test
.\env\Scripts\python.exe viz_scripts\viz_swarm.py datasets\swarm_test\seq_000 --clusters --sheet

# binned swarm data (sizes 1-100 per frame), 1-3 swarms is up to you
.\ds_tools\run_tracking.ps1 -Seed 1000 -PerBin 10 -Bins "1-5,6-10,11-20" -Swarms 2 -BrkProb 0.7 -Out datasets\swarm_membera

# (re)make swarm labels for any old tracking dataset (no Isaac Sim needed; old data = 1 swarm)
.\env\Scripts\python.exe ds_tools\swarm_annotate.py datasets\tracking_v1
```

`viz_swarm.py` colours: green = swarm, red = breakaway, line = centroid trail, yellow (`--clusters`) = 2D clusters.

**What uses Replicator.** All renders go through Replicator (`omni.replicator.core`): render product, `orchestrator.step`, and the
annotators `rgb`, `instance_id_segmentation_fast`, `bounding_box_2d_tight_fast` and `camera_params`. The drone masks, and so the
swarm boxes and masks, come from those annotators. Swarm grouping, breakaway detection, the YOLO/COCO/JSON export and the
viz are our own Python (`swarm_annotate.py`), because Replicator has no swarm concept. With `-ReplicatorWriter`, the stock
Replicator **BasicWriter** also writes tight 2D boxes, instance-id masks and a semantic mask with the label `swarm_<id>`
(`replicator\*_NNNN.*`, file index = frame - 1). On the test clip that mask matched our swarm mask exactly (IoU 1.0, 32/32 frames).
Its labels are set once per sequence, so breakaway drones keep their parent swarm's label there.
Scene randomisation (layout, lights, sky, colours) is still our own numpy code, not Replicator randomizers.

**Big swarms (1,000-100,000).** The drone pool above makes one USD prim per drone (fine up to a few hundred). For huge swarms use one
`UsdGeom.PointInstancer`: `ds_tools\scale_test.py` renders 1k / 10k / 100k drones this way on the RTX 3060 (results in
[Speed](#speed-rtx-3060)). When drones are around one pixel or smaller, per-drone boxes stop making sense; label the swarm mask
(Replicator semantic mask) or a density map (drone count per pixel from projected 3D positions, which `scale_test.py` also saves).

## Scene re-ID datasets (tracking sequences per visible-count bin)

Re-ID data comes from the **tracking generator**: a swarm takes off, forms up and cruises over the desert (some sequences
have 1-3 drones breaking away). Every drone keeps the **same track ID for the whole sequence**, so the same physical
drone can be matched across frames.

```powershell
# full set: 11 bins (1-5, 6-10, 11-20, ..., 91-100), 10 sequences per bin
.\ds_tools\run_tracking.ps1 -Seed 1000 -PerBin 10 -Out datasets\reid_membera_1-30 -Bins "1-5,6-10,11-20,21-30"

# quick test: 1 short 16-frame clip in each of two bins
.\ds_tools\run_tracking.ps1 -Bins "1-5,6-10" -PerBin 1 -ClipFrames 16 -Out datasets\reid_test

# check IDs, then export re-ID crops
.\env\Scripts\python.exe ds_tools\check_track_ids.py datasets\reid_test
.\env\Scripts\python.exe ds_tools\export_reid_crops.py datasets\reid_test
```

`run_tracking.ps1` options:

| option | default | meaning |
|---|---|---|
| `-Bins` | `1-5,6-10,11-20,21-30,31-40,41-50,51-60,61-70,71-80,81-90,91-100` | visible-drone bins |
| `-PerBin` | 10 | sequences per bin |
| `-Seed` | 77001 | random seed. **Every team member uses a different seed** |
| `-Out` | `datasets\tracking_bins` | output folder; one sub-folder per bin: `bin_001-005\seq_000 ...` |
| `-ClipFrames` | 0 | 0 = full sequence (140-200 frames at 10-15 fps, takeoff to cruise). N = only N consecutive frames, taken from the cruise part |
| `-CamFraming` | `chase` | `chase`: the camera follows the swarm at close range and slowly orbits, so drones are big enough for re-ID crops. `wide`: the original tracking camera (whole flight in view, drones often < 16 px) |
| `-CamDist` | 1.0 | camera distance multiplier (< 1 = closer, bigger drones) |
| `-ModelMix` | `balanced` | drone models: `balanced` (all 4 models, 25 % each), `isaac` (the 3 NVIDIA models), `ghost`, `legacy` (old mix: 50 % all-Ghost sequences, else 80/10/10 Ghost/Crazyflie/Quadcopter), or weights like `"ghost:2,arl:1,cf2x:1"`. See *Drone models and colours* |
| `-ColorMode` | `random` | `random` (random HSV body colour per drone), `palette` (16 named colours), `none` (original materials). Each drone keeps its colour for the whole sequence |
| `-ColorStrength` | 0.85 | 0 = original look, 1 = fully painted |
| `-NoWear` | off | by default roughness / metallic are also randomised per drone (worn / glossy look); `-NoWear` keeps the originals |
| `-MaxAccept` | 30 | swarm designs tried per sequence before giving up (re-run to retry) |
| `-Total` | - | legacy mode: old unbinned `tracking_v1` set (`.\ds_tools\run_tracking.ps1 -Total 36`) |

**How binning works:** a sequence is kept only if the **median, over all its rendered frames, of `n_visible_ge1px`**
(drones with at least 1 visible pixel) is inside the bin. The first-frame, min and max visible counts are saved too
(`meta.json`: `visible_median`, `visible_first_frame`, `visible_min`, `visible_max`). A quick 5-frame pre-check rejects
most bad designs before the full render.

**IDs:** in every frame, `drones[].track_id` (same value as `drones[].id`) is that drone's ID for the whole sequence, and
its pixels in `masks/NNNNNN.png` have exactly that value. IDs are unique **within a sequence**; across sequences use
(sequence, track_id), which is what the crop exporter's global ID does. `check_track_ids.py` verifies this for every sequence
(mask pixel counts match the annotations, model and colour never change, no ID jumps). Each drone's `model`, `type` and
`appearance` (`color_name`, `rgb`, `hsv`, `strength`, `roughness`, `metallic`) are in every frame's `drones[]` and in `meta.json` `track_ids`.

> The **counting** datasets (`run_counting_bins.ps1`, `run_counting.ps1`) are independent single frames. Their mask values /
> `drones[].k` are per-frame indices and are **NOT** persistent IDs. Do not use them for re-ID.

### Re-ID crops (Market-1501 style)

```powershell
.\env\Scripts\python.exe ds_tools\export_reid_crops.py datasets\reid_membera_1-30 --min-px 16 --stride 1
```

- Searches the folder recursively for sequences, so you can point it at a folder holding several members' outputs.
- `crops/<gid>/<seq>_<frame>.jpg`: `gid` (5 digits) = one drone in one sequence; `<seq>` = sequence path, e.g. `bin_001-005_seq_000`. Each sequence is one camera.
- Crops whose tight box is smaller than `--min-px` (default **16 px**) on its short side are skipped. `--stride 5` keeps every 5th frame (smaller dataset for long sequences). `--pad 0.15` adds context around the box.
- `ids.csv` (one row per identity), `crops.csv` (every crop with bbox, mask_px, visibility, `split` and `role`), `pairs.csv`
  (query vs. same-ID positives and vs. other drones of the same sequence as hard negatives), `summary.json`, `crop_sheet.jpg` (preview).
- Split: `--test-frac 0.5` of the **sequences** are test (each test ID: 1 query crop, all other crops gallery); the rest are train. No identity is in both.

## Drone models and colours

| key | model | source URL (under `https://omniverse-content-production.s3-us-west-2.amazonaws.com/`) | terms | pool prims per drone |
|---|---|---|---|---|
| `ghost` | SK_DroneGhost (Squad), quadcopter, ~0.7 m | `Assets/Machinima/Squad/Vehicles/Drone/SK_DroneGhost.usd` | **CC BY-NC-ND 4.0** | instanced unless coloured |
| `cf2x` | Bitcraze Crazyflie 2.x micro quad | `Assets/Isaac/6.2/Isaac/Robots/Bitcraze/Crazyflie/cf2x.usd` | NVIDIA Isaac Sim asset terms | |
| `quad` | Isaac "Quadcopter" (simple disc + 4 rotors) | `Assets/Isaac/6.2/Isaac/Robots/IsaacSim/Quadcopter/quadcopter.usd` | NVIDIA Isaac Sim asset terms | |
| `arl` | NTNU ARL-Robot-1 quadrotor (Aerial Gym) | `Assets/Isaac/6.2/Isaac/Robots/NTNU/ARL-Robot-1/arl_robot_1.usd` | NVIDIA Isaac Sim asset terms | **new**, decimated (see below) |

(Exact URLs: `ASSETS` in `ds_tools\common.py`.) All models are rescaled to the sampled span (0.3-1.0 m) and are Z-up. These are all the
drone USDs on NVIDIA's content server (checked Nov 2026); the Squad MQ-9 / Pchela are fixed-wing and also CC BY-NC-ND, so they are not used.
The ARL body mesh has 369k triangles; the generators build a decimated copy (49k triangles, same materials) once into
`datasets\stages\arl_lite_*.usdc` (a small local layer that references the original URL and overrides only that mesh; ~0.5 MB,
a second on first use; not shipped, delete it any time). Without it, 50 coloured ARL drones need ~3 GB extra VRAM. `LITE_ASSETS=0`
uses the original mesh. Look at all models and colour variants with:

```powershell
$env:OMNI_KIT_ACCEPT_EULA = "YES"; $env:COLOR_MODE = "random"; .\env\Scripts\python.exe ds_tools\preview_models.py   # writes datasets\model_preview.jpg (+ .json)
```

**Model mix** (`-ModelMix`, env `MODEL_MIX`): presets `balanced` (25 % each of ghost / cf2x / quad / arl), `isaac` (1/3 each of cf2x / quad / arl), `ghost`, `legacy`,
or explicit weights `"ghost:1,cf2x:1,quad:0.5,arl:1"`. With a mix, the pool holds enough prims of each model for the largest bin.

**Colours** (`-ColorMode`, `-ColorStrength`, `-NoWear`; env `COLOR_MODE`, `COLOR_STRENGTH`, `COLOR_WEAR`): each drone gets its own body colour,
drawn with a separate seeded random stream (same `-Seed` = same colours; colours never change the scene / flight design). `random` = uniform hue,
saturation 0.25-1, value 0.2-1; `palette` = 16 named colours (red, orange, yellow, olive, green, teal, cyan, blue, navy, purple, magenta, pink,
brown, white, grey, black). The colour is blended into every paint material of that drone (`strength`); very dark parts (rotors, motors, lenses)
keep their colour. Ghost's material multiplies a dark texture, so it gets a boosted tint and looks more muted than the others. Roughness (0.15-0.85)
and metallic (0 for 2/3 of drones, else 0.2-0.8) are also randomised unless `-NoWear`. In tracking/re-ID the colour is **fixed per track ID for the whole sequence**.
Colouring makes each coloured drone a separate (non-instanced) copy, which costs some GPU memory and startup time.

Counting: `run_counting_bins.ps1 -ModelMix balanced -ColorMode random` also works (a new colour per drone per frame); the default there stays
`legacy` / `none`, so earlier counting sets are reproducible.

> **Known issue in datasets made before this version:** the Isaac `quad` model has wrong authored extents, so the auto-scaler made it ~9x too small:
> in older sets (`counting_v1`, `counting_bins_v1`, `tracking_v1`) the drones labelled `quad` are only a few pixels. Fixed in `common.mesh_bounds`.

## Team assignment (re-ID data collection)

Each member renders **different bins with their own seed**, then we merge the folders. Seed = 1000 × member number.
Name outputs `datasets\reid_<yourname>_<bins>` (lower case, no spaces), e.g. `datasets\reid_membera_1-30`.

Default plan for **4 members**, bins split so the work is about even (bigger swarms render slower), **10 sequences per bin** (full length):

| member | seed | bins (`-Bins`) | output name |
|---|---|---|---|
| Member A | 1000 | `1-5,6-10,11-20,21-30` | `datasets\reid_<name>_1-30` |
| Member B | 2000 | `31-40,41-50,51-60` | `datasets\reid_<name>_31-60` |
| Member C | 3000 | `61-70,71-80` | `datasets\reid_<name>_61-80` |
| Member D | 4000 | `81-90,91-100` | `datasets\reid_<name>_81-100` |

```powershell
# Member B example
.\ds_tools\run_tracking.ps1 -Seed 2000 -PerBin 10 -Bins "31-40,41-50,51-60" -Out datasets\reid_<name>_31-60
.\env\Scripts\python.exe ds_tools\check_track_ids.py datasets\reid_<name>_31-60
```

- **2 members:** A = `1-5 ... 41-50` (6 bins), B = `51-60 ... 91-100` (5 bins). **3 members:** A = `1-5 ... 31-40`, B = `41-50,51-60,61-70`, C = `71-80,81-90,91-100`.
- **Everyone uses the defaults** `-ModelMix balanced -ColorMode random` (all 4 drone models, 25 % each; each drone a random colour that stays
  fixed for its track). Don't change them, so the merged dataset is consistent. `run_config.json` in your output records what was used.
- **More than 4 members:** extra members repeat an existing member's bins with their **own seed** (doubles the data for those bins).
- Before the full run, do the quick test above (2 clips, ~3 min) to check your install.
- Hand in the whole output folder (zip it); run `check_track_ids.py` first and include its output.
- Single-frame counting data per bin is also possible: same bins, `run_counting_bins.ps1 -Seed <seed> -Bins "..." -PerBin 200 -Out datasets\counting_<name>_<bins>`.

## Counting bins (single frames)

Frames are kept only if their `n_visible_ge1px` falls in the bin. Bin *i* (in `-Bins` order) owns frame numbers
`i*PerBin ... (i+1)*PerBin-1`; `bins.json` in the output folder lists the mapping.

```powershell
.\ds_tools\run_counting_bins.ps1                                         # 11 bins 1-5 ... 91-100, 200 frames each -> datasets\counting_bins
.\ds_tools\run_counting_bins.ps1 -Bins "1-5,6-10" -PerBin 3 -Seed 1000 -Out datasets\counting_test   # quick test
# the old 4-bin test set (same layout as counting_bins_v1):
.\ds_tools\run_counting_bins.ps1 -Bins "1-10,11-50,51-100,101-150" -Seed 20261106 -Out datasets\counting_bins_v1
```

Options: `-Bins`, `-PerBin` (200), `-Seed` (20261106), `-Out` (`datasets\counting_bins`), `-Chunk` (frames per Isaac process, 50), `-PoolGhost`, `-Rt`.

## Other generators

```powershell
.\ds_tools\run_counting.ps1 -Total 4000 -Chunk 400 -NMax 1000      # counting_v1: up to NMAX drones per frame
.\ds_tools\run_tracking.ps1 -Total 36                              # tracking_v1 (legacy, unbinned)
.\ds_tools\run_tracking_brk.ps1                                    # extra breakaway sequences for tracking_v1
.\env\Scripts\python.exe ds_tools\export_counting.py datasets\counting_v1   # counting -> COCO + YOLO (needs opencv-python-headless)
python ds_tools\overlay_tracking.py <seq_dir> <out_dir>            # draw boxes + IDs on a tracking sequence (plain Python + Pillow)
```

## Output formats

**Tracking / re-ID** (`<out>\bin_LLL-HHH\seq_NNN\`): `img/000001.jpg` (1280×720), `masks/000001.png` (uint16, value = track_id, 0 = background),
`ann/frames.jsonl` (one JSON line per frame: camera, `n_visible_ge1px`, and per drone `track_id`, `type`, `model`, `pos_m`, `quat_wxyz`, `yaw_deg`, `pitch_deg`,
`vel_mps`, `bbox_tight`, `center_px`, `mask_px`, `occlusion_ratio`, `visible_fraction`, `depth_m`, `span_px`, `phase`, `is_breakaway`),
`gt/gt.txt` (MOT format: frame,id,x,y,w,h,conf,class,visibility), `seqinfo.ini`, `meta.json` (sequence settings, bin, visible stats, `track_ids` list with type/model).
With `-ClipFrames`, frames are renumbered from 1; `source_frame` / `source_frame_offset` give the position in the full flight, and a
`breakaway_start_frame` ≤ 0 means the breakaway started before the clip.

`occlusion_ratio` / `visible_fraction` can be null in tracking output (the fast box annotator does not always report occlusion); `mask_px` is always filled.

**Counting** (`images/NNNNNN.jpg`, `masks/NNNNNN.png` = per-frame index + 1, `ann/NNNNNN.json`): `n_visible_ge1px`, `n_visible_ge10px`, `n_total`, `visible_bin`,
per-drone `mask_px`, boxes, pose, type, occlusion.

## Explore OpenUSD

Everything here is plain OpenUSD: a scene is a stack of **layers** (`.usd`/`.usda` files); stronger layers add or override
**prims** (scene objects) and their attributes without changing the files underneath.

**Open the GUI** (from the kit folder):

```powershell
$env:OMNI_KIT_ACCEPT_EULA = "YES"
.\env\Scripts\isaacsim.exe          # full Isaac Sim app (same as running `isaacsim` inside the activated env)
```

(`python -m isaacsim` is NOT the GUI, it only writes VS Code settings.) The first start can take 5-10 minutes.

**Open the desert stage:** *File > Open*, paste the URL the generators use:

```
https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Machinima/Squad/Stages/Desert/DesertStage.usd
```

Or open a kit layer: *File > Open* > `stages\desert_drones.usda`:

| file | what it is |
|---|---|
| `stages\desert_drones.usda` | desert stage (as a sublayer) + 10 Ghost drones under `/DroneSwarm` + cameras `/RenderCams/wide_40m`, `/RenderCams/close` |
| `stages\drone_gallery.usda` | the drone models side by side on a grey ground (Ghost, Crazyflie, Quadcopter) - a quick look at models; `preview_models.py` shows all four including ARL-Robot-1 |
| `stages\all_outputs.usda` | `desert_drones.usda` + `over` opinions that add semantic class labels (vehicle, wall, ...) to existing desert props without editing the desert files |
| `stages\tracking.usda`, `stages\dr_v2.usda` | one-line layers that just sublayer the files above (an example of layer composition) |

Open the `.usda` files in a text editor too: they are readable text and show exactly how sublayers (`subLayers = [@...@]`) and references (`references = @...@`) are written.

**Inspect:** *Window > Stage* shows the prim tree; click a prim and *Window > Property* shows its attributes, references and transforms.
*Window > Layer* shows the layer stack (the strongest layer is on top; the one with the edit-target icon receives your edits).
Without the GUI: `.\env\Scripts\python.exe ds_tools\inspect_stage.py [file-or-url] [--prims 200]` prints layers, units, lights, cameras and the top-level prims.
The desert stage has `defaultPrim /Root`, ~3200 prims, **centimetres** (`metersPerUnit = 0.01`) and **Z up**; the scripts work in metres and convert.

**Light / time of day:** the sun is the desert's `DistantLight` `/Root/Lighting_Objects/Global_Lighting/BPA_Standard_Lighting_2/DirectionalLight`. Rotate it (or edit `xformOp:orient`) to move the sun,
lower `intensity` and `colorTemperature` (~3000-4000 K) for dusk. The generators add a `DomeLight` `/Root/DS_Sky` with an HDR sky. In the
generators this is randomised per frame/sequence: `set_sun(el, az, inten, ct)` with the ranges in `sample_frame()` (counting) / `design()` (tracking,
`sunp = dict(el=..., az=..., ...)`); sky HDRs are the `SKIES` list in `ds_tools\common.py`. Narrow those ranges to make e.g. a dusk-only dataset.

**Swap drone models:** models are referenced by URL. In the GUI: select e.g. `/DroneSwarm/drone_00/asset` (desert_drones) or a model in drone_gallery,
and change its **reference** in the Property panel to another USD (for example the Crazyflie URL from `drone_gallery.usda`, or any other `.usd` file or URL). In the generators: `ASSETS` in `ds_tools\common.py` maps the model keys to URLs. Pointing a key at a
different USD swaps that model everywhere; a new key (URL, license note, instanceable flag) is picked up by `-ModelMix "newkey:1,ghost:1"`. Models are auto-scaled to the sampled size (0.3-1.0 m span).

**How the generators compose the scene:** each run creates a tiny root layer `datasets\stages\<name>_<pid>.usda` whose only content is the desert
stage as a **sublayer** (`common.open_layer`). Everything else is authored in memory on top, so the desert files are never changed:
- `/Root/DS/d_0000 ...`: a pool of drone Xforms (translate / orient / scale) that each **reference** a model URL (Ghost is `instanceable` to save memory),
  labelled `class=drone` for the segmentation annotators. Unused drones are parked far below the terrain.
- `/Root/DSCam`: the render camera; `/Root/DS_Sky`: dome light; the desert sun gets new orient / intensity / colour-temperature values.
- Each frame only changes attribute values (positions, camera transform, lights), then Replicator renders RGB, instance segmentation, boxes and camera params.
These root layers are not saved with the in-memory edits, so they are safe to delete.

## Speed (RTX 3060)

- Isaac startup ~25 s per process (drivers start one process per chunk / 4 sequences).
- Re-ID / tracking, chase camera, old 3-model mix without colours (`-ModelMix legacy -ColorMode none`): ~1 s per rendered frame, also for big swarms (measured 16-frame clips: bins 1-5 / 6-10 15-16 s each,
  91-100 with 97 drones ~15 s), plus ~3-4 s per design rejected by the pre-check (usually 0-4 per sequence). A full 140-200-frame sequence is ~3-4 min, so
  10 full sequences per bin is ~35-45 min per bin, ~2 h for a 3-bin share (estimate).
- With the default 4-model mix + random colours (this version), measured 16-frame clip in bin 41-50: 19.2 s (1.20 s/frame), 50 drones, VRAM peak 10247 MiB (one rejected design). Earlier 16-frame clip timings from the previous kit version: bin 6-10 22 s, 41-50 19 s, 91-100 (96 drones)
  29 s (each incl. ~0-1 rejected design), i.e. ~1.2-1.8 s/frame, VRAM peak 9.2-10.1 GB. The coloured pool also adds ~5 s per process start.
- Counting bins: ~2 s/frame for small swarms; large bins are slower and reject more frames that miss the bin.
- Swarm test clips (2 swarms x 40 drones, 4 models + colours, chase camera, 32 frames, measured Nov 11): 25-50 s per clip
  (~0.8-1.6 s/frame incl. setup), GPU memory peak 9.9-10.3 GB (whole GPU, ~2.2 GB of that is the Windows desktop).
- Big swarms with one PointInstancer (`scale_test.py`, Crazyflie model, 0.5 m drones, 3 m spacing, sky background, measured Nov 11):

  | drones | view | s/frame | GPU peak | drone size | Replicator swarm mask / box |
  |---|---|---|---|---|---|
  | 1,000 | fill / far | 0.82 / 0.80 | 8.8 / 9.0 GB | 8.8 / 2.2 px | works |
  | 10,000 | fill / far | 1.20 / 1.18 | 9.2 / 9.2 GB | 4.5 / 1.1 px | works |
  | 100,000 (`--light`) | fill / far | 1.30 / 1.18 | 9.2 / 9.5 GB | 2.2 / 0.5 px | works (also when every drone is < 1 px) |

  The swarm box from the Replicator semantic mask was within 6 px of the box from the projected 3D positions in every case.
  At 100,000 the per-instance annotators (`instance_id_segmentation`, `bounding_box_2d_tight`) stalled for >9 min on the first frame,
  so use `--light` (rgb + semantic mask + camera only). Instance IDs from the instancer are per sub-mesh, not per drone (1,000 drones gave
  ~6,500 IDs), so per-drone labels at this scale should come from the projected positions (density map), not the instance mask.

## License / assets

- **Squad Desert Stage** and **SK_DroneGhost** (Offworld Industries / Squad): **CC BY-NC-ND 4.0**. Internal, non-commercial research use only.
  Do not redistribute the assets, and check the license before publishing imagery rendered with them.
- Crazyflie, Quadcopter, ARL-Robot-1 and Isaac Sim itself: hosted by NVIDIA under the NVIDIA Isaac Sim asset terms (ARL-Robot-1 comes from
  NTNU's Autonomous Robots Lab / Aerial Gym). `-ModelMix isaac` avoids the
  CC BY-NC-ND Ghost model; the desert stage is still CC BY-NC-ND.
- This kit ships only scripts, a precomputed heightmap and small USDA layers that reference the assets by URL.
