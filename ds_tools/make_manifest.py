"""Team data manifest: every sequence is planned (and given its own seed) BEFORE anyone renders, so nobody has to pick
settings by hand and no two people can render the same scene. No Isaac Sim needed (numpy only).

    python ds_tools/make_manifest.py                                  # default plan -> datasets/manifest/manifest.csv + .json
    python ds_tools/make_manifest.py --people 5 --frames 50 --out datasets/manifest
    python ds_tools/make_manifest.py --slice --person 2 --of 5        # also write datasets/manifest/manifest_P2of5.csv

Plan (defaults): tier 1 = 1-10 drones per swarm, tier 2 = 11-100, tier 3 = 101-1000, tier 4 = 1001-100000 (60/60/60/40 sequences).
~40 % of sequences get a scripted breakaway (a subgroup of 10-30 % of the swarm, 1 drone for tiny swarms), starting 15-40 % into the
clip so the breakaway is visible for most of the clip; ~20 % have 2-3 swarms. Splits are by SEQUENCE (70/15/15 per tier, stratified
by breakaway / multi-swarm). Hold-outs: the low-sun lighting condition and the wide (original tracking-style) camera appear ONLY in
test rows (test_kind = ood_light / ood_cam); train and val never see them.
Categorical settings are dealt evenly per tier (every level appears equally often, then shuffled), continuous settings use a
Latin-hypercube draw per tier, and rows are dealt round-robin to people in a stratified order, so every person gets the same mix of
tiers, splits, breakaways, lighting and cameras.

Only settings the generator (ds_tools/gen_tracking.py, manifest mode) actually supports are written. See UNSUPPORTED below for
what does not exist in the kit (and how it is mapped, if at all)."""
import os, sys, csv, json, math, argparse
import numpy as np

ROOT = os.environ.get("DRONE_SCENE_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TIERS = {1: (1, 10), 2: (11, 100), 3: (101, 1000), 4: (1001, 100000)}
GEN_TRACKING_MAX_TOTAL = 1000   # gen_tracking.py makes one USD prim per drone; tier 1-3 rows stay <= 1000 drones in total
SKIES = ["noon_grass_4k", "qwantani_4k", "kloofendal_48d_partly_cloudy_4k", "champagne_castle_1_4k"]   # common.SKIES order
# lighting: sun elevation (deg), sun intensity multiplier, colour temperature (K); all inside gen_tracking's own random ranges
# (el 10-80, intensity 0.5-1.6, 4200-7500 K)
TIME_OF_DAY = {"low_sun": dict(el=(10, 20), inten=(0.5, 0.9), ct=(4200, 5200)),    # golden hour / dusk  -> TEST-ONLY hold-out
               "mid_sun": dict(el=(20, 45), inten=(0.8, 1.3), ct=(5000, 6500)),
               "high_sun": dict(el=(45, 80), inten=(1.0, 1.6), ct=(6000, 7500))}
WEATHER = {"clear": [0, 1], "partly_cloudy": [2, 3]}   # = sky HDR (dome light); the sun is still on under the cloudy skies
CAMERA = {"chase_near": dict(framing="chase", dist=1.0), "chase_mid": dict(framing="chase", dist=1.6), "chase_far": dict(framing="chase", dist=3.0),
          "wide": dict(framing="wide", dist=1.0)}                     # wide = original tracking cameras -> TEST-ONLY hold-out
CAM_ANGLE = {"low": (8, 18), "mid": (18, 30), "high": (30, 40)}      # chase-camera elevation (gen_tracking range 8-40 deg)
WIND = {"calm": 0.5, "breezy": 1.0, "gusty": 2.0}                   # multiplier on per-drone position jitter (proxy, no wind physics)
MODEL_MIX = {"balanced": "balanced", "isaac": "isaac", "ghost": "ghost", "ghost_heavy": "ghost:2,cf2x:1,quad:1,arl:1"}
COLOR = ["none", "random"]
SPREAD = {"tight": 0.6, "normal": 1.0, "loose": 1.5}
HOLDOUT = {"time_of_day": "low_sun", "camera_setup": "wide"}
UNSUPPORTED = {
    "weather (rain, fog, dust, snow)": "not in the kit; 'weather' = sky HDR only: clear (2 HDRs) or partly_cloudy (2 HDRs)",
    "wind (physical drift / gusts)": "not in the kit; 'wind' = turbulence multiplier on the per-drone position jitter (calm 0.5x, breezy 1x, gusty 2x)",
    "crossing swarms": "not in the kit; multi-swarm = 2-3 groups flying the same path side by side, gap swarm_sep (0.3-2 swarm radii)",
    "scenes other than the desert": "not in the kit; every sequence uses the Omniverse Desert stage",
    "1,000-100,000 drones in the flight generator": "tier 4 needs the PointInstancer big-swarm mode (scale_test.py has it, gen_tracking.py does not); rows are planned but marked needs_bigswarm_mode",
    "exact breakaway start frame": "requested start frame is honoured unless the scripted breakaway is too close to the end of the source sequence; the actual frame is in meta.json",
    "fps": "chosen by the generator from the row seed (10/12/15)",
}
COLS = ["seq_id", "assignee_slot", "deal_rank", "tier", "n_per_swarm", "n_swarms", "n_drones_total", "seed", "split", "test_kind",
        "time_of_day", "sun_el", "sun_az", "sun_int", "sun_ct", "weather", "sky_tex", "sky_name", "sky_int", "wind", "jitter",
        "camera_setup", "cam_framing", "cam_dist", "cam_angle", "cam_el_deg", "chase_fit", "model_mix_name", "model_mix", "color_mode",
        "density", "spread", "swarm_sep", "breakaway", "brk_count", "brk_size", "brk_start_frame", "frames", "generator"]

def lhs(rng, n):
    """1-D Latin hypercube in [0,1): one point per stratum, shuffled."""
    return (rng.permutation(n) + rng.random(n)) / n

def deal(rng, levels, n):
    """Every level equally often (+-1), shuffled."""
    out = [levels[i % len(levels)] for i in range(n)]
    rng.shuffle(out); return out

def split_pattern(n, rng):
    """70/15/15 by sequence; rows are pre-sorted by stratum, so a systematic pattern stratifies the split."""
    pat = list("TTTVTTTXTTTVTTXTTVTX")   # 14 T, 3 V, 3 X per 20
    off = int(rng.integers(len(pat)))
    return [{"T": "train", "V": "val", "X": "test"}[pat[(i + off) % len(pat)]] for i in range(n)]

def build(args):
    rng = np.random.default_rng(args.plan_seed)
    rows = []; seed_next = args.base_seed
    for tier, nseq in zip((1, 2, 3, 4), (args.t1, args.t2, args.t3, args.t4)):
        if nseq <= 0: continue
        lo, hi = TIERS[tier]; R = []
        n_brk = int(round(args.brk_frac * nseq)); n_multi = int(round(args.multi_frac * nseq))
        brk = np.array([1] * n_brk + [0] * (nseq - n_brk)); rng.shuffle(brk)
        multi = np.array([1] * n_multi + [0] * (nseq - n_multi)); rng.shuffle(multi)
        # stratified split: sort by (breakaway, multi) then systematic pattern
        order = np.lexsort((rng.random(nseq), multi, brk)); split = [None] * nseq
        for i, sp in zip(order, split_pattern(nseq, rng)): split[i] = sp
        test_idx = [i for i in range(nseq) if split[i] == "test"]; rng.shuffle(test_idx)
        test_kind = [""] * nseq
        for j, i in enumerate(test_idx): test_kind[i] = ["in_dist", "ood_light", "ood_cam"][j % 3]
        # continuous: Latin hypercube per tier
        u_n, u_az, u_el, u_int, u_ct, u_cam, u_sep, u_bs, u_bf, u_sky = (lhs(rng, nseq) for _ in range(10))
        # categorical: dealt evenly among the rows that may use them (hold-outs only in test rows)
        idl = [i for i in range(nseq) if test_kind[i] != "ood_light"]; idc = [i for i in range(nseq) if test_kind[i] != "ood_cam"]
        tod = ["low_sun"] * nseq; cams = ["wide"] * nseq
        for i, v in zip(idl, deal(rng, ["mid_sun", "high_sun"], len(idl))): tod[i] = v
        for i, v in zip(idc, deal(rng, ["chase_near", "chase_mid", "chase_far"], len(idc))): cams[i] = v
        wea = deal(rng, list(WEATHER), nseq); ang = deal(rng, list(CAM_ANGLE), nseq); wind = deal(rng, list(WIND), nseq)
        mix = deal(rng, list(MODEL_MIX), nseq); col = deal(rng, COLOR, nseq); dens = deal(rng, list(SPREAD), nseq)
        for i in range(nseq):
            ns = int(rng.choice([2, 3], p=[0.6, 0.4])) if multi[i] else 1
            plo, phi = lo, hi
            if tier <= 3: phi = min(hi, GEN_TRACKING_MAX_TOTAL // ns)
            if tier == 1 and (multi[i] or brk[i]): plo = max(plo, 3)   # need >= 2 drones per swarm to split, >= 3 for a breakaway
            nps = int(round(math.exp(math.log(plo) + u_n[i] * (math.log(phi + 1) - math.log(plo))))); nps = int(np.clip(nps, plo, phi))
            t = TIME_OF_DAY[tod[i]]; c = CAMERA[cams[i]]
            r = dict(tier=tier, n_per_swarm=nps, n_swarms=ns, n_drones_total=nps * ns, seed=seed_next, split=split[i], test_kind=test_kind[i],
                     time_of_day=tod[i], sun_el=round(t["el"][0] + u_el[i] * (t["el"][1] - t["el"][0]), 2), sun_az=round(360 * u_az[i], 1),
                     sun_int=round(t["inten"][0] + u_int[i] * (t["inten"][1] - t["inten"][0]), 3), sun_ct=int(t["ct"][0] + u_ct[i] * (t["ct"][1] - t["ct"][0])),
                     weather=wea[i], wind=wind[i], jitter=WIND[wind[i]], camera_setup=cams[i], cam_framing=c["framing"], cam_dist=c["dist"],
                     model_mix_name=mix[i], model_mix=MODEL_MIX[mix[i]], color_mode=col[i], density=dens[i], spread=SPREAD[dens[i]],
                     swarm_sep=round(0.3 + 1.7 * u_sep[i], 2) if ns > 1 else "", frames=args.frames, chase_fit=1)
            sk = WEATHER[wea[i]][int(u_sky[i] * 2) % 2]; r.update(sky_tex=sk, sky_name=SKIES[sk], sky_int=int(400 + 900 * ((u_sky[i] * 2) % 1)))
            if c["framing"] == "chase":
                a = CAM_ANGLE[ang[i]]; r.update(cam_angle=ang[i], cam_el_deg=round(a[0] + u_cam[i] * (a[1] - a[0]), 2))
            else: r.update(cam_angle="auto", cam_el_deg="")
            if brk[i]:
                bs = 1 if nps < 10 and u_bs[i] < 0.5 else max(1, int(round(nps * (0.10 + 0.20 * u_bs[i]))))
                bs = min(bs, max(1, nps // 2 - 1)) if nps >= 4 else 1
                r.update(breakaway="on", brk_count=2 if (u_bs[i] * 7) % 1 < 0.3 and nps >= 2 * (bs + 1) else 1, brk_size=bs,
                         brk_start_frame=int(round(args.frames * (0.15 + 0.25 * u_bf[i]))) + 1)
            else: r.update(breakaway="off", brk_count=0, brk_size=0, brk_start_frame="")
            r["generator"] = "gen_tracking" if tier <= 3 else "needs_bigswarm_mode"
            seed_next += 1; R.append(r)
        # stratified deal to people: order by split, breakaway, multi, lighting, camera, then size
        key = sorted(range(nseq), key=lambda i: (R[i]["split"], R[i]["breakaway"], R[i]["n_swarms"] > 1, R[i]["time_of_day"], R[i]["camera_setup"], R[i]["n_per_swarm"]))
        for rank, i in enumerate(key): R[i]["deal_rank"] = rank
        for k, r in enumerate(R): r["seq_id"] = f"t{tier}_{k:03d}"
        rows += R
    # tier offsets so that every person gets rows from every tier in turn
    for r in rows: r["assignee_slot"] = f"P{r['deal_rank'] % args.people + 1}"
    seeds = [r["seed"] for r in rows]; assert len(set(seeds)) == len(seeds), "duplicate seeds"
    return rows

def person_slice(rows, person, of):
    return [r for r in rows if r["deal_rank"] % of == person - 1]

def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, COLS); w.writeheader()
        for r in rows: w.writerow({k: r.get(k, "") for k in COLS})

def summary(rows):
    s = {}
    for dim in ("tier", "split", "test_kind", "time_of_day", "weather", "camera_setup", "cam_angle", "wind", "model_mix_name", "color_mode", "density", "breakaway", "n_swarms", "assignee_slot", "generator"):
        c = {}
        for r in rows: c[str(r[dim]) or "-"] = c.get(str(r[dim]) or "-", 0) + 1
        s[dim] = dict(sorted(c.items()))
    tx = {}
    for r in rows: tx.setdefault(f"tier{r['tier']}", {}).setdefault(r["split"], 0); tx[f"tier{r['tier']}"][r["split"]] += 1
    s["tier_x_split"] = tx; s["sequences"] = len(rows); s["frames"] = int(sum(int(r["frames"]) for r in rows))
    return s

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(ROOT, "datasets", "manifest")); ap.add_argument("--people", type=int, default=4)
    ap.add_argument("--t1", type=int, default=60); ap.add_argument("--t2", type=int, default=60); ap.add_argument("--t3", type=int, default=60); ap.add_argument("--t4", type=int, default=40)
    ap.add_argument("--frames", type=int, default=50); ap.add_argument("--brk-frac", type=float, default=0.4); ap.add_argument("--multi-frac", type=float, default=0.2)
    ap.add_argument("--base-seed", type=int, default=910000, help="first row seed (rows get base, base+1, ...); old datasets used 77001 / 1000*member")
    ap.add_argument("--plan-seed", type=int, default=2026, help="seed of the plan itself (same plan-seed + args = same manifest)")
    ap.add_argument("--slice", action="store_true", help="also write manifest_P<person>of<of>.csv"); ap.add_argument("--person", type=int, default=1); ap.add_argument("--of", type=int, default=0)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    rows = build(a)
    write_csv(os.path.join(a.out, "manifest.csv"), rows)
    meta = dict(created_by="ds_tools/make_manifest.py", args=vars(a), tiers={k: list(v) for k, v in TIERS.items()}, holdout_test_only=HOLDOUT,
                levels=dict(time_of_day=TIME_OF_DAY, weather={k: [SKIES[i] for i in v] for k, v in WEATHER.items()}, camera_setup=CAMERA, cam_angle=CAM_ANGLE,
                            wind_jitter_multiplier=WIND, model_mix=MODEL_MIX, color_mode=COLOR, density_spread=SPREAD),
                gen_tracking_max_total_drones=GEN_TRACKING_MAX_TOTAL, unsupported=UNSUPPORTED, summary=summary(rows), rows=rows)
    json.dump(meta, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    s = meta["summary"]; print(f"{s['sequences']} sequences, {s['frames']} frames -> {a.out}")
    for k in ("tier_x_split", "assignee_slot", "time_of_day", "camera_setup", "breakaway", "n_swarms", "generator"): print(f"  {k}: {s[k]}")
    if a.slice:
        of = a.of or a.people; sl = person_slice(rows, a.person, of); p = os.path.join(a.out, f"manifest_P{a.person}of{of}.csv"); write_csv(p, sl)
        print(f"slice P{a.person}/{of}: {len(sl)} rows -> {p}")
