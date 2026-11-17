# team_v1 splits and commands (per person)

Each person owns a swarm-size range. Run all commands in PowerShell from the repo root on the `team-manifest` branch (`git fetch; git checkout team-manifest; git pull`). Never push to `main`.

Python: commands below use `.\env\Scripts\python.exe`, the venv the README install creates inside the repo. Check it with `.\env\Scripts\python.exe -m pip show isaacsim`; it should print `Name: isaacsim`.

If Isaac Sim is installed somewhere else (a conda env or another folder), link that environment into the repo as `env` once: `New-Item -ItemType Junction -Path env -Target <that env folder>`. A conda env has `python.exe` directly inside it, so use `.\env\python.exe` in place of `.\env\Scripts\python.exe` in every command. The `.ps1` stills script finds either one on its own.

| Person | Bins (drones per swarm) | Sequences | Train / val / test | Stills | Manifest slot / tier |
|---|---|---|---|---|---|
| A | S1 1-10, S2 11-20 | 60 | 42 / 9 / 9 | 1,000 | PA / 1 |
| B | S3 21-50, S4 51-100 | 60 | 42 / 9 / 9 | 1,000 | PB / 2 |
| C | S5 101-200, S6 201-500, S7 501-1,000 | 60 | 42 / 9 / 9 | 1,000 | PC / 3 |
| D | S8 1,001-100,000 | 40 | 28 / 6 / 6 | - | PL / 4 (not ready) |

Run tracking first, then stills. Never run both at once. You're done when your coverage report shows nothing missing and you have 1,000 stills.

## A: 1-20 drones

- Dry run: `.\env\Scripts\python.exe ds_tools\run_manifest.py --person 1 --of 1 --tiers 1 --name a --dry-run`
- Quick test: `.\env\Scripts\python.exe ds_tools\run_manifest.py --person 1 --of 1 --tiers 1 --name a_test --limit 2 --frames 10 --out-root datasets\_test`
- Tracking: `.\env\Scripts\python.exe ds_tools\run_manifest.py --person 1 --of 1 --tiers 1 --name a`
- Stills: `.\ds_tools\run_counting_bins.ps1 -Bins "1-5,6-10,11-15,16-20" -PerBin 250 -Seed 3001 -ModelMix balanced -ColorMode random -Out datasets\a_stills_tier1`
- Contact sheet: `.\env\Scripts\python.exe ds_tools\contact_sheet.py "datasets\a_tier*"`
- Coverage: `.\env\Scripts\python.exe ds_tools\check_coverage.py "datasets\a_tier*" --tiers 1 --out datasets\a_coverage`
- Upload: `a_tier1`, `a_stills_tier1`, `a_coverage`

## B: 21-100 drones

- Dry run: `.\env\Scripts\python.exe ds_tools\run_manifest.py --person 1 --of 1 --tiers 2 --name b --dry-run`
- Quick test: `.\env\Scripts\python.exe ds_tools\run_manifest.py --person 1 --of 1 --tiers 2 --name b_test --limit 2 --frames 10 --out-root datasets\_test`
- Tracking: `.\env\Scripts\python.exe ds_tools\run_manifest.py --person 1 --of 1 --tiers 2 --name b`
- Stills: `.\ds_tools\run_counting_bins.ps1 -Bins "21-30,31-40,41-50,51-60,61-70,71-80,81-90,91-100" -PerBin 125 -Seed 3002 -ModelMix balanced -ColorMode random -Out datasets\b_stills_tier2`
- Contact sheet: `.\env\Scripts\python.exe ds_tools\contact_sheet.py "datasets\b_tier*"`
- Coverage: `.\env\Scripts\python.exe ds_tools\check_coverage.py "datasets\b_tier*" --tiers 2 --out datasets\b_coverage`
- Upload: `b_tier2`, `b_stills_tier2`, `b_coverage`

## C: 101-1,000 drones

- Dry run: `.\env\Scripts\python.exe ds_tools\run_manifest.py --person 1 --of 1 --tiers 3 --name c --dry-run`
- Quick test: `.\env\Scripts\python.exe ds_tools\run_manifest.py --person 1 --of 1 --tiers 3 --name c_test --limit 2 --frames 10 --out-root datasets\_test`
- Tracking: `.\env\Scripts\python.exe ds_tools\run_manifest.py --person 1 --of 1 --tiers 3 --name c`
- Stills test first: `.\ds_tools\run_counting_bins.ps1 -Bins "101-150,151-200,201-300,301-500,501-1000" -PerBin 3 -Seed 3003 -ModelMix ghost -PoolGhost 1100 -ColorMode random -Out datasets\_test\c_stills`
- Stills, if every bin fills: same command with `-PerBin 200 -Out datasets\c_stills_tier3`
- Contact sheet: `.\env\Scripts\python.exe ds_tools\contact_sheet.py "datasets\c_tier*"`
- Coverage: `.\env\Scripts\python.exe ds_tools\check_coverage.py "datasets\c_tier*" --tiers 3 --out datasets\c_coverage`
- Upload: `c_tier3`, `c_stills_tier3`, `c_coverage`

## D: 1,001-100,000 drones (not ready)

The big-swarm launch scripts aren't built yet, so `--tiers 4` currently lists these 40 rows as skipped. Once they're built, the command will be:

`.\env\Scripts\python.exe ds_tools\run_manifest.py --person 1 --of 1 --tiers 4 --name d`