# Team data plan: manifest, render, check

Goal for Dr. B's request: one box per swarm (1 drone up to 100,000), box coordinates over time, breakaways flagged, a model fast
enough for a laptop / edge device. To make sure the team's data is diverse and nobody renders the same thing twice, every sequence
is planned up front in one shared file, `manifests/team_v1/manifest.csv` (same content as `manifest.json`). Do not pick settings by hand.

## 1. Your slice

Mosa gives everyone a person number `N` (1-4). Your rows are the ones with `assignee_slot = PN` (the manifest is dealt for 4 people).
If the team size changes, use `--person N --of M`: rows are re-dealt in the same stratified order, so every person still gets the
same mix of tiers, splits, lighting, cameras and breakaways.

```powershell
# list your rows and the Isaac Sim processes that would run (nothing is rendered)
.\env\Scripts\python.exe ds_tools\run_manifest.py --slot P2 --name alex --dry-run
# write your slice as its own CSV (optional, for reading)
.\env\Scripts\python.exe ds_tools\make_manifest.py --out manifests\team_v1 --slice --person 2 --of 4
```

## 2. Render

```powershell
.\env\Scripts\python.exe ds_tools\run_manifest.py --slot P2 --name alex
```

- Output: `datasets\alex_tier1\t1_007\...`, `datasets\alex_tier2\...`, `datasets\alex_tier3\...`. Every sequence folder has
  `img\`, `masks\`, `gt\`, `ann\frames.jsonl`, `meta.json` (includes the manifest row) and `swarm\` (swarm boxes: YOLO, COCO, CSV).
- It is resumable: if it stops, run the same command again and finished rows are skipped.
- Tier 4 rows (1,001-100,000 drones per swarm) are planned but SKIPPED for now: the flight generator does not have the big-swarm
  mode yet (`generator = needs_bigswarm_mode`). Do not try to render them another way.
- Quick test first (2 rows, 10 frames each):
  `.\env\Scripts\python.exe ds_tools\run_manifest.py --slot P2 --name alex_test --limit 2 --frames 10 --out-root datasets\_test`

### How long it takes (measured on Mosa's RTX 3060, Nov 16)

- Smoke test, 8 rows of 10 frames (tiers 1-3, 6-175 drones): 0.67-0.86 s per frame, plus about 25-35 s to start Isaac Sim per process.
- One full 50-frame row (73 drones, 20-drone breakaway): 0.71 s per frame, 70 s wall time including startup.
- GPU memory peak 8.6-9.4 GB; the Isaac process used about 10 GB of RAM. One run crashed at startup when Windows was almost out of
  memory (Edge, Discord etc. open); close big apps before a long run. The runner retries once and kills a crashed / hung process.
- Not measured yet: tier 3 rows near 1,000 drones (the largest tested was 175).
- Estimate from those numbers (not measured): one 4-person slice is ~45 renderable rows x 50 frames, about 30 min of rendering plus
  ~20 Isaac Sim starts (one per tier / model mix / colour group), so roughly 40-60 min per person.

## 2b. Still images (second run, after tracking finishes)

Target: **1,000 stills per person** (plan: `manifests/team_v1/stills_plan.json`). Do not run at the same time as tracking.

```powershell
# a, tier 1 (2 bins x 500)
.\ds_tools\run_counting_bins.ps1 -Bins "1-5,6-10" -PerBin 500 -Seed 3001 -ModelMix balanced -ColorMode random -Out datasets\a_stills_tier1
# b, tier 2 (10 bins x 100)
.\ds_tools\run_counting_bins.ps1 -Bins "11-19,20-28,29-37,38-46,47-55,56-64,65-73,74-82,83-91,92-100" -PerBin 100 -Seed 3002 -ModelMix balanced -ColorMode random -Out datasets\b_stills_tier2
# c, tier 3 (5 bins x 200) - smoke test first with -PerBin 3 -Out datasets\_test\c_stills
.\ds_tools\run_counting_bins.ps1 -Bins "101-150,151-200,201-300,301-500,501-1000" -PerBin 200 -Seed 3003 -ModelMix ghost -PoolGhost 1100 -ColorMode random -Out datasets\c_stills_tier3
```

Resumable: re-run the same command. `progress.json` in the output folder shows frames per bin; done = every bin full (1,000 total).

## 3. Check before you upload

```powershell
.\env\Scripts\python.exe ds_tools\contact_sheet.py "datasets\alex_tier*"          # contact_sheet.jpg in each folder
.\env\Scripts\python.exe ds_tools\check_coverage.py "datasets\alex_tier*" --tiers 1,2,3 --out datasets\alex_coverage
```

Open the contact sheet: first frame of every sequence with its settings and the swarm boxes (green = swarm, red = breakaway). Then read
`coverage_report.md`: it lists missing / incomplete rows, rows rendered twice, duplicate seeds and near-duplicate first frames.
Upload the `<name>_tier*` folders plus your coverage report to the shared folder. Mosa runs `check_coverage.py` on everyone's folders together.

## What the manifest varies (and what it cannot)

| column | levels |
|---|---|
| tier | 1: 1-10, 2: 11-100, 3: 101-1,000, 4: 1,001-100,000 drones per swarm (log-uniform inside the tier) |
| n_swarms | 1 (80 %), 2-3 (20 %) flying side by side, gap `swarm_sep` 0.3-2 swarm radii |
| breakaway | on for 40 %: `brk_size` drones (10-30 % of the swarm, 1 for tiny swarms) leave together, 1-2 events, start 15-40 % into the clip |
| time_of_day | mid_sun (sun 20-45 deg), high_sun (45-80 deg); low_sun (10-20 deg, warm light) is TEST-ONLY |
| weather | clear or partly_cloudy = sky HDR (2 each) |
| camera_setup | chase_near / chase_mid / chase_far (camera follows the swarm, whole formation in view); wide (original far tracking cameras) is TEST-ONLY |
| cam_angle | camera elevation low 8-18, mid 18-30, high 30-40 deg (chase cameras) |
| model_mix | balanced, isaac (no Ghost), ghost, ghost_heavy |
| color_mode | none / random (per-drone paint) |
| wind | calm / breezy / gusty = 0.5x / 1x / 2x per-drone position jitter |
| density | tight / normal / loose formation spacing (0.6x / 1x / 1.5x) |
| split | train 70 % / val 15 % / test 15 %, by SEQUENCE (frames of one clip never end up in two splits) |

Test rows have a `test_kind`: `in_dist`, `ood_light` (low sun) or `ood_cam` (wide camera), so we can show on Friday whether the
detector holds up on light and camera setups it never saw.

Not in the kit (do not promise these on slides): rain / fog / dust (weather is the sky only), physical wind (it is a jitter proxy),
crossing swarms (they fly side by side), scenes other than the desert, tier 4 in the flight generator. fps is picked by the generator (10-15).

## Rules

- Always render from the manifest; never change seeds or settings by hand. Every row has its own seed.
- Use your own `--name`; never write into someone else's folder or into the old datasets (`counting_v1`, `tracking_v1`, `counting_bins_v1`).
- No military base, rocket or missile scenes; only the desert stage this kit uses.
