# Team data plan: manifest, render, check

Goal for Dr. B's request: one box per swarm (1 drone up to 100,000), box coordinates over time, breakaways flagged, a model fast
enough for a laptop / edge device. To make sure the team's data is diverse and nobody renders the same thing twice, every sequence
is planned up front in one shared file, `manifests/team_v1/manifest.csv` (same content as `manifest.json`). Do not pick settings by hand.

**Python:** use the Isaac Sim env python your install guide set up. On the reference PC that is `.\env\python.exe`; on other installs it
may be `.\env\Scripts\python.exe`. The commands below use `.\env\python.exe`; swap in your path if that one is not found.

## 1. Your rows (size bins)

The manifest has 8 swarm-size bins (drones per swarm). Each bin belongs to one person; the `tier` column is the person group, so
`--tiers N` selects one person's rows. `size_bin` says which bin a row is in, `assignee_slot` who owns it.

| Bin (`size_bin`) | Drones per swarm | Person | `assignee_slot` / `tier` | Sequences | train / val / test |
|---|---|---|---|---|---|
| S1_1-10 | 1-10 | a | PA / 1 | 30 | 21 / 5 / 4 |
| S2_11-20 | 11-20 | a | PA / 1 | 30 | 21 / 4 / 5 |
| S3_21-50 | 21-50 | b | PB / 2 | 30 | 21 / 5 / 4 |
| S4_51-100 | 51-100 | b | PB / 2 | 30 | 21 / 4 / 5 |
| S5_101-200 | 101-200 | c | PC / 3 | 20 | 14 / 3 / 3 |
| S6_201-500 | 201-500 | c | PC / 3 | 20 | 14 / 3 / 3 |
| S7_501-1000 | 501-1,000 | c | PC / 3 | 20 | 14 / 3 / 3 |
| S8_1001-100000 | 1,001-100,000 | lead | PL / 4 | 40 | 28 / 6 / 6 |

Per person: a, b, c each 60 sequences (42 / 9 / 9), lead 40 (28 / 6 / 6). Total 220 sequences x 50 frames = 11,000 frames,
154 / 33 / 33. Each of a, b, c has 3 `in_dist`, 3 `ood_light` and 3 `ood_cam` test rows, spread over their bins.

```powershell
# list your rows and the Isaac Sim processes that would run (nothing is rendered); b shown, a = --tiers 1 --name a, c = --tiers 3 --name c
.\env\python.exe ds_tools\run_manifest.py --person 1 --of 1 --tiers 2 --name b --dry-run
# same rows by slot, or only one of your bins
.\env\python.exe ds_tools\run_manifest.py --slot PB --tiers 2 --name b --dry-run
.\env\python.exe ds_tools\run_manifest.py --slot PB --tiers 2 --bins S3 --name b --dry-run
# write your rows as their own CSV (optional, for reading)
.\env\python.exe ds_tools\make_manifest.py --out manifests\team_v1 --slice --slot PB
```

The manifest files in the repo are the plan of record. Only run `make_manifest.py` with `--slice` (it rewrites manifest.csv/.json
with the same content, because same `--plan-seed` + args = same manifest); never commit a regenerated manifest with other args.

## 2. Render

```powershell
.\env\python.exe ds_tools\run_manifest.py --person 1 --of 1 --tiers 2 --name b
```

- Output: `datasets\b_tier2\s3_007\...` (all of a person's bins go into `<name>_tier<N>`). Every sequence folder has
  `img\`, `masks\`, `gt\`, `ann\frames.jsonl`, `meta.json` (includes the manifest row) and `swarm\` (swarm boxes: YOLO, COCO, CSV).
- It is resumable: if it stops, run the same command again and finished rows are skipped.
- S8 rows (1,001-100,000 drones per swarm, tier 4, lead) are planned but SKIPPED for now: the flight generator does not have the
  big-swarm mode yet (`generator = needs_bigswarm_mode`). Do not try to render them another way.
- Quick test first (2 rows, 10 frames each):
  `.\env\python.exe ds_tools\run_manifest.py --person 1 --of 1 --tiers 2 --name b_test --limit 2 --frames 10 --out-root datasets\_test`

### How long it takes (measured on an RTX 3060, Nov 16)

- Smoke test, 8 rows of 10 frames (6-175 drones): 0.67-0.86 s per frame, plus about 25-35 s to start Isaac Sim per process.
- One full 50-frame row (73 drones, 20-drone breakaway): 0.71 s per frame, 70 s wall time including startup.
- GPU memory peak 8.6-9.4 GB; the Isaac process used about 10 GB of RAM. One run crashed at startup when Windows was almost out of
  memory (Edge, Discord etc. open); close big apps before a long run. The runner retries once and kills a crashed / hung process.
- Not measured yet: flight-generator rows above 175 drones (S6, S7 go up to 1,000).
- Estimate from those numbers (not measured): 60 rows x 50 frames = 3,000 frames, about 34-43 min of rendering, plus one
  Isaac Sim start (25-35 s) per chunk of up to 4 rows with the same model mix / colour, so roughly 45-60 min of tracking per person.

## GPU memory per bin (`gpu_class` / `gpu_note` columns in the manifest)

| Bins | Drones per swarm | Who | gpu_class | What we measured |
|---|---|---|---|---|
| S1, S2 | 1-20 | a | mid (10 GB+ card) | 8.6-9.4 GB peak |
| S3, S4 | 21-100 | b | mid (10 GB+ card) | 8.6-9.4 GB peak |
| S5, S6 | 101-500 | c | mid (10 GB+ card) | up to 9.4 GB at 175 drones |
| S7 | 501-1,000 | c | mid (10 GB+ card) | 9.0 GB at 1,000 in the scale test; not measured with the flight generator above 175 drones |
| S8 | 1,001-100,000 | lead | high (12 GB+ card) | 9.2-9.5 GB in the scale test; flight mode not measured |

Memory is dominated by Isaac Sim itself (about 7.5 GB with just the scene loaded), so it barely changes with split or drone count.
Train, val and test rows of a bin need the same card. Cards under 10 GB are not expected to work for any bin (no "low" class).
Measured on an RTX 3060 12 GB.

## 2b. Still images (second run, after tracking finishes)

Target: **1,000 stills per person** (plan: `manifests/team_v1/stills_plan.json`); the still bins cover exactly your tracking range.
Do not run at the same time as tracking.

```powershell
# a, 1-20 drones (4 bins x 250)
.\ds_tools\run_counting_bins.ps1 -Bins "1-5,6-10,11-15,16-20" -PerBin 250 -Seed 3001 -ModelMix balanced -ColorMode random -Out datasets\a_stills_tier1
# b, 21-100 drones (8 bins x 125)
.\ds_tools\run_counting_bins.ps1 -Bins "21-30,31-40,41-50,51-60,61-70,71-80,81-90,91-100" -PerBin 125 -Seed 3002 -ModelMix balanced -ColorMode random -Out datasets\b_stills_tier2
# c, 101-1,000 drones (5 bins x 200) - smoke test first with -PerBin 3 -Out datasets\_test\c_stills
.\ds_tools\run_counting_bins.ps1 -Bins "101-150,151-200,201-300,301-500,501-1000" -PerBin 200 -Seed 3003 -ModelMix ghost -PoolGhost 1100 -ColorMode random -Out datasets\c_stills_tier3
```

Resumable: re-run the same command. `progress.json` in the output folder shows frames per bin; done = every bin full (1,000 total).
`run_counting_bins.ps1` is only tested to about 150 drones, so c should report any bin that will not fill.

## 3. Check before you upload

```powershell
.\env\python.exe ds_tools\contact_sheet.py "datasets\b_tier*"          # contact_sheet.jpg in each folder
.\env\python.exe ds_tools\check_coverage.py "datasets\b_tier*" --tiers 2 --out datasets\b_coverage
```

Open the contact sheet: first frame of every sequence with its size bin, settings and the swarm boxes (green = swarm, red = breakaway).
Then read `coverage_report.md`: it lists missing / incomplete rows, rows rendered twice, duplicate seeds and near-duplicate first frames,
and done / planned per `size_bin` and every other dimension. `--bins S3` limits the check to one bin.
Upload `<name>_tier<N>`, `<name>_stills_tier<N>` and `<name>_coverage` to the shared folder (not `datasets\_test`).
The lead runs `check_coverage.py` on everyone's folders together.

## What the manifest varies (and what it cannot)

| column | levels |
|---|---|
| size_bin | S1 1-10, S2 11-20, S3 21-50, S4 51-100, S5 101-200, S6 201-500, S7 501-1,000, S8 1,001-100,000 drones per swarm (log-uniform inside the bin) |
| tier | person group: 1 = a (S1-S2), 2 = b (S3-S4), 3 = c (S5-S7), 4 = lead (S8) |
| n_swarms | 1 (80 %), 2-3 (20 %) flying side by side, gap `swarm_sep` 0.3-2 swarm radii; S7 is single-swarm only (two 501+ swarms would exceed the 1,000-drone total), its share moved to S5/S6 |
| breakaway | on for 40 % of every bin: `brk_size` drones (10-30 % of the swarm, 1 for tiny swarms) leave together, 1-2 events, start 15-40 % into the clip |
| time_of_day | mid_sun (sun 20-45 deg), high_sun (45-80 deg); low_sun (10-20 deg, warm light) is TEST-ONLY |
| weather | clear or partly_cloudy = sky HDR (2 each) |
| camera_setup | chase_near / chase_mid / chase_far (camera follows the swarm, whole formation in view); wide (original far tracking cameras) is TEST-ONLY |
| cam_angle | camera elevation low 8-18, mid 18-30, high 30-40 deg (chase cameras) |
| model_mix | balanced, isaac (no Ghost), ghost, ghost_heavy |
| color_mode | none / random (per-drone paint) |
| wind | calm / breezy / gusty = 0.5x / 1x / 2x per-drone position jitter |
| density | tight / normal / loose formation spacing (0.6x / 1x / 1.5x) |
| split | train 70 % / val 15 % / test 15 %, by SEQUENCE (frames of one clip never end up in two splits); fixed counts per bin (table above) |

Categorical levels are dealt evenly per person and per bin (every level equally often, +-1). Test rows have a `test_kind`:
`in_dist`, `ood_light` (low sun) or `ood_cam` (wide camera), so we can show on Friday whether the detector holds up on light and
camera setups it never saw.

Not in the kit (do not promise these on slides): rain / fog / dust (weather is the sky only), physical wind (it is a jitter proxy),
crossing swarms (they fly side by side), scenes other than the desert, S8 (1,001+) in the flight generator. fps is picked by the generator (10-15).

## Rules

- Always render from the manifest; never change seeds or settings by hand. Every row has its own seed (910000-910219).
- Use your own `--name`; never write into someone else's folder or into the old datasets (`counting_v1`, `tracking_v1`, `counting_bins_v1`).
- No military base, rocket or missile scenes; only the desert stage this kit uses.
