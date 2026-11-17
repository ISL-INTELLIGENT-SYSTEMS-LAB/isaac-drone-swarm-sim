# Team walkthrough - meeting Nov 17 (read aloud)

## 1. What we're making and why

Dr. B asked for one box per swarm (from 1 drone up to 100,000), the box coordinates over time, breakaways flagged, and a model
fast enough for a laptop / edge device. This week each of us renders swarm tracking clips and still images from one shared plan,
so Friday we can show box accuracy per swarm size, breakaway catches and speed.

## 2. The manifest (`manifests/team_v1/manifest.csv`, same content as `manifest.json`)

Every sequence is planned in advance with its own seed (910000-910219). 220 sequences x 50 frames = 11,000 frames, split by
sequence into **154 train / 33 val / 33 test**. Eight swarm-size bins, each owned by one person:

| Bin | Drones per swarm | Who | Sequences | train / val / test | gpu_class |
|---|---|---|---|---|---|
| S1 | 1-10 | A | 30 | 21 / 5 / 4 | mid |
| S2 | 11-20 | A | 30 | 21 / 4 / 5 | mid |
| S3 | 21-50 | B | 30 | 21 / 5 / 4 | mid |
| S4 | 51-100 | B | 30 | 21 / 4 / 5 | mid |
| S5 | 101-200 | C | 20 | 14 / 3 / 3 | mid |
| S6 | 201-500 | C | 20 | 14 / 3 / 3 | mid |
| S7 | 501-1,000 | C | 20 | 14 / 3 / 3 | mid |
| S8 | 1,001-100,000 | Mosa | 40 | 28 / 6 / 6 | high |

- **A = 1-20 drones, B = 21-100, C = 101-1,000**: 60 sequences each, 42 / 9 / 9. **Mosa = 1,001+**: 40 sequences, 28 / 6 / 6.
- Columns that tell you what is yours: `assignee_slot` (PA, PB, PC, PM), `tier` (person group: 1 = A, 2 = B, 3 = C, 4 = Mosa),
  `size_bin` (e.g. `S3_21-50`).
- **Test hold-outs:** low-sun lighting and the wide camera appear only in test rows (`test_kind` = `ood_light` / `ood_cam`;
  normal test rows are `in_dist`). A, B and C each have 3 of each kind, spread over their bins.
- 40 % of each bin has a breakaway; about 20 % of each person's rows have 2-3 swarms (S7 is single-swarm only, because two
  swarms of 501+ would go over the 1,000 drones per clip the flight generator is used for; that share moved to S5 / S6).
- **GPU columns** (`gpu_class`, `gpu_note`), measured on Mosa's RTX 3060 12 GB:
  - A, B, C = **mid, 10 GB+ card**: 8.6-9.4 GB peak in the Nov 16 smoke test. S7 note: 9.0 GB at 1,000 drones in the scale test,
    not measured with the flight generator above 175 drones.
  - Mosa = **high, 12 GB+ card**: 9.2-9.5 GB in the scale test at 10k-100k drones; flight mode not measured yet.
  - Isaac Sim alone uses about 7.5 GB, so cards under 10 GB are not expected to work for any bin.

## 3. Get the code

```powershell
git fetch
git checkout team-manifest
# if you already checked it out earlier:
git pull
```

## 4. Python path

Use the Isaac Sim python your install guide set up, and run the commands from the folder that has your `env\`:

```powershell
.\env\python.exe          # conda-style env (this is the one on Mosa's PC)
.\env\Scripts\python.exe  # venv, as in the README install steps
```

The commands below use `.\env\python.exe`; if it is not found, use `.\env\Scripts\python.exe`. (`run_counting_bins.ps1` finds
either one by itself.) Before a long run close Edge, Discord and other big apps: a run needs about 9 GB of GPU memory and about
10 GB of RAM.

## 5. Exact commands per person

Never change seeds or settings. Use your letter exactly as the name.

### Person A - S1 + S2 (1-20 drones), tier 1

```powershell
# 1. dry run: lists your 60 sequences, renders nothing
.\env\python.exe ds_tools\run_manifest.py --person 1 --of 1 --tiers 1 --name a --dry-run
# 2. quick test: 2 sequences x 10 frames into a test folder
.\env\python.exe ds_tools\run_manifest.py --person 1 --of 1 --tiers 1 --name a_test --limit 2 --frames 10 --out-root datasets\_test
# 3. full tracking render: 60 sequences x 50 frames -> datasets\a_tier1\<seq_id>\ (re-run the same command to resume)
.\env\python.exe ds_tools\run_manifest.py --person 1 --of 1 --tiers 1 --name a
# 4. stills, AFTER tracking finishes: 4 bins x 250 = 1,000 stills
.\ds_tools\run_counting_bins.ps1 -Bins "1-5,6-10,11-15,16-20" -PerBin 250 -Seed 3001 -ModelMix balanced -ColorMode random -Out datasets\a_stills_tier1
# 5. contact sheet + coverage check
.\env\python.exe ds_tools\contact_sheet.py "datasets\a_tier*"
.\env\python.exe ds_tools\check_coverage.py "datasets\a_tier*" --tiers 1 --out datasets\a_coverage
```

Stills bins: 1-5 and 6-10 cover S1, 11-15 and 16-20 cover S2. Upload: `a_tier1`, `a_stills_tier1`, `a_coverage`.

### Person B - S3 + S4 (21-100 drones), tier 2

```powershell
.\env\python.exe ds_tools\run_manifest.py --person 1 --of 1 --tiers 2 --name b --dry-run
.\env\python.exe ds_tools\run_manifest.py --person 1 --of 1 --tiers 2 --name b_test --limit 2 --frames 10 --out-root datasets\_test
.\env\python.exe ds_tools\run_manifest.py --person 1 --of 1 --tiers 2 --name b
# stills, AFTER tracking: 8 bins x 125 = 1,000 stills
.\ds_tools\run_counting_bins.ps1 -Bins "21-30,31-40,41-50,51-60,61-70,71-80,81-90,91-100" -PerBin 125 -Seed 3002 -ModelMix balanced -ColorMode random -Out datasets\b_stills_tier2
.\env\python.exe ds_tools\contact_sheet.py "datasets\b_tier*"
.\env\python.exe ds_tools\check_coverage.py "datasets\b_tier*" --tiers 2 --out datasets\b_coverage
```

Stills bins: 21-30, 31-40, 41-50 cover S3; 51-60 ... 91-100 cover S4. Upload: `b_tier2`, `b_stills_tier2`, `b_coverage`.

### Person C - S5 + S6 + S7 (101-1,000 drones), tier 3

```powershell
.\env\python.exe ds_tools\run_manifest.py --person 1 --of 1 --tiers 3 --name c --dry-run
.\env\python.exe ds_tools\run_manifest.py --person 1 --of 1 --tiers 3 --name c_test --limit 2 --frames 10 --out-root datasets\_test
.\env\python.exe ds_tools\run_manifest.py --person 1 --of 1 --tiers 3 --name c
# stills smoke test first (the stills script is only tested to about 150 drones)
.\ds_tools\run_counting_bins.ps1 -Bins "101-150,151-200,201-300,301-500,501-1000" -PerBin 3 -Seed 3003 -ModelMix ghost -PoolGhost 1100 -ColorMode random -Out datasets\_test\c_stills
# stills, AFTER tracking: 5 bins x 200 = 1,000 stills
.\ds_tools\run_counting_bins.ps1 -Bins "101-150,151-200,201-300,301-500,501-1000" -PerBin 200 -Seed 3003 -ModelMix ghost -PoolGhost 1100 -ColorMode random -Out datasets\c_stills_tier3
.\env\python.exe ds_tools\contact_sheet.py "datasets\c_tier*"
.\env\python.exe ds_tools\check_coverage.py "datasets\c_tier*" --tiers 3 --out datasets\c_coverage
```

Stills bins: 101-150 and 151-200 cover S5, 201-300 and 301-500 cover S6, 501-1000 covers S7. Tell Mosa if a bin will not fill.
Upload: `c_tier3`, `c_stills_tier3`, `c_coverage`.

**For everyone:** tracking output per sequence is `img\`, `masks\`, `gt\`, `ann\frames.jsonl`, `meta.json` and `swarm\` (swarm
boxes). Stills progress is in `progress.json` in the stills folder. Do not upload `datasets\_test`. Run tracking and stills one
after the other, never at the same time.

## 6. Flag reference (from the scripts' argparse)

**`ds_tools/make_manifest.py`** - builds the plan (the repo copy is the plan of record; you normally only use `--slice`)

| flag | default | meaning |
|---|---|---|
| `--out` | `datasets\manifest` | output folder for manifest.csv / .json (team copy: `manifests\team_v1`) |
| `--frames` | 50 | frames per sequence |
| `--brk-frac` | 0.4 | share of sequences with a breakaway |
| `--multi-frac` | 0.2 | share with 2-3 swarms |
| `--base-seed` | 910000 | first row seed |
| `--plan-seed` | 2026 | seed of the plan (same seed + args = same manifest) |
| `--slice` | off | also write a slice CSV |
| `--slot` | - | with `--slice`: PA, PB, PC or PM -> `manifest_<slot>.csv` |
| `--person`, `--of` | 1, 1 | with `--slice` and no `--slot`: deal_rank slice -> `manifest_P<person>of<of>.csv` |

**`ds_tools/run_manifest.py`** - renders manifest rows with `gen_tracking.py` (one of `--person/--of`, `--slot`, `--rows` is required)

| flag | default | meaning |
|---|---|---|
| `--name` | required | your folder prefix (output `<out-root>\<name>_tier<N>`) |
| `--person`, `--of` | - | rows with `deal_rank % of == person-1`; `--person 1 --of 1` = all rows |
| `--slot` | - | rows of one `assignee_slot` (PA / PB / PC / PM) |
| `--rows` | - | comma-separated seq_ids (ignores `--tiers`) |
| `--tiers` | `1,2,3` | person groups to keep (1 = A, 2 = B, 3 = C, 4 = Mosa); also applies with `--slot`, so PM needs `--tiers 4` |
| `--bins` | all | only these size bins, e.g. `S3` or `3,4` |
| `--limit` | 0 (all) | only the first N rows |
| `--frames` | 0 (manifest) | override frames per sequence (quick tests) |
| `--out-root` | `datasets` | where the output folders go |
| `--dry-run` | off | list rows and processes, render nothing |
| `--per-proc` | 4 | rows per Isaac Sim process |
| `--rt` | 4 | value passed to the generator as `RT` |
| `--py` | this python | interpreter used to start `gen_tracking.py` |
| `--manifest` | `manifests\team_v1\manifest.json` | plan file |
| `--min-free-gb` | 6.0 | stop if free disk drops below this |
| `--close-grace` | 45 | seconds to wait for Isaac Sim to exit after all rows are written, then kill it |
| `--timeout-per-row` | 900 | kill a process taking longer than this per row (+300 s) |

**`ds_tools/check_coverage.py`** - checks rendered folders against the manifest (tracking data only)

| flag | default | meaning |
|---|---|---|
| `folders` | required | dataset or sequence folders, wildcards ok (`"datasets\b_tier*"`) |
| `--tiers` | all | only plan these person groups, e.g. `2` |
| `--bins` | all | only plan these size bins, e.g. `S3,S4` |
| `--rows` | all | only these seq_ids |
| `--out` | `<common folder>\_coverage` | report folder |
| `--min-frac` | 0.5 | flag a level with done < this share of planned |
| `--min-seqs` | 3 | flag a level with fewer done sequences than this |
| `--hash-dist` | 4 | first frames within this many dHash bits count as near-duplicates |
| `--manifest` | `manifests\team_v1\manifest.json` | plan file |

## 7. Done criteria and rules

**You're done when:**
- `coverage_report.md` shows all 60 of your planned rows done: 0 missing, 0 incomplete, 0 duplicate seeds, 0 near-duplicate first frames.
- `contact_sheet.jpg` looks varied (green boxes = swarms, red = breakaways).
- Your stills folder's `progress.json` shows every bin full, 1,000 stills in total.
- Expect at least an hour of GPU time. Estimate, not measured: tracking alone is about 45-60 minutes (3,000 frames at the measured
  0.67-0.86 s per frame, plus 25-35 s per Isaac Sim start); stills come on top.

**Rules:**
- Always render from the manifest; never change seeds or settings by hand.
- Only write into your own name folders (`a_*`, `b_*`, `c_*`).
- Never touch the old datasets: `counting_v1`, `tracking_v1`, `counting_bins_v1`.
- Desert scene only. Weather means the sky image only, wind is drone jitter, multiple swarms fly side by side (they don't cross).

## 8. Mosa's part - S8, 1,001 to 100,000 drones

The 40 S8 rows (28 / 6 / 6) are planned and marked `generator = needs_bigswarm_mode`; the runner skips and lists them
(`run_manifest.py --person 1 --of 1 --tiers 4 --name m --dry-run` shows 40 skipped). Next step: wire the instanced big-swarm
renderer from `ds_tools/scale_test.py` into the flight generator (launch, formation flight, breakaways), then render those rows
with the same runner and `--tiers 4`. The scale test already ran at about 0.8 s per frame with 1,000 drones and about 1.2-1.3 s
per frame with 10,000-100,000, at 9.2-9.5 GB GPU memory; the flight mode at that size is not measured yet. S8 stills are not planned yet.
