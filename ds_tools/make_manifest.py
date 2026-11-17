"""Team data manifest: every sequence is planned (and given its own seed) BEFORE anyone renders, so nobody has to pick
settings by hand and no two people can render the same scene. No Isaac Sim needed (numpy only).

    python ds_tools/make_manifest.py --out manifests/team_v1                 # default plan (8 size bins, 220 sequences)
    python ds_tools/make_manifest.py --out manifests/team_v1 --slice --slot PB   # also write manifest_PB.csv
    python ds_tools/make_manifest.py --out manifests/team_v1 --slice --person 2 --of 4   # deal_rank slice (manifest_P2of4.csv)

Plan (defaults, table BINS below = the one place the size bins live): 8 swarm-size bins (drones per swarm), each owned by one
person (assignee_slot) and one person group (tier column: 1 = A, 2 = B, 3 = C, 4 = Mosa):
  S1 1-10, S2 11-20 -> A (PA, tier 1), 30 sequences each      S3 21-50, S4 51-100 -> B (PB, tier 2), 30 each
  S5 101-200, S6 201-500, S7 501-1000 -> C (PC, tier 3), 20 each   S8 1001-100000 -> Mosa (PM, tier 4), 40, needs_bigswarm_mode
Splits are by SEQUENCE with fixed counts per bin (30 -> 21/5/4 or 21/4/5, 20 -> 14/3/3, 40 -> 28/6/6), so A/B/C get 42/9/9 and
Mosa 28/6/6 (154/33/33 overall), stratified by breakaway / multi-swarm inside each bin. Hold-outs: the low-sun lighting condition
and the wide (original tracking-style) camera appear ONLY in test rows (test_kind = ood_light / ood_cam), dealt in turn over each
person's test rows bin by bin, so every bin gets them as evenly as possible.
~40 % of sequences per bin get a scripted breakaway (a subgroup of 10-30 % of the swarm, 1 drone for tiny swarms), starting 15-40 %
into the clip; ~20 % have 2-3 swarms (none in S7: two swarms of 501+ would exceed the 1,000-drone total gen_tracking.py is used for;
those rows go to S5/S6). Drones per swarm are log-uniform inside the bin (Latin hypercube). Categorical settings are dealt evenly per
person AND per bin (every level equally often +-1, order shuffled per dimension), continuous settings use a Latin-hypercube draw per
person. deal_rank is a stratified rank inside each person, so --person k --of M re-splits one person's rows evenly.

Only settings the generator (ds_tools/gen_tracking.py, manifest mode) actually supports are written. See UNSUPPORTED below."""
import os, sys, csv, json, math, argparse
import numpy as np

ROOT = os.environ.get("DRONE_SCENE_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# ---- the bin table (single source of truth) -------------------------------------------------------------------------------
# name, lo, hi (drones per swarm), slot, tier (= person group), sequences, (train, val, test), allow multi-swarm
BINS = [("S1", 1, 10, "PA", 1, 30, (21, 5, 4), True), ("S2", 11, 20, "PA", 1, 30, (21, 4, 5), True),
        ("S3", 21, 50, "PB", 2, 30, (21, 5, 4), True), ("S4", 51, 100, "PB", 2, 30, (21, 4, 5), True),
        ("S5", 101, 200, "PC", 3, 20, (14, 3, 3), True), ("S6", 201, 500, "PC", 3, 20, (14, 3, 3), True),
        ("S7", 501, 1000, "PC", 3, 20, (14, 3, 3), False), ("S8", 1001, 100000, "PM", 4, 40, (28, 6, 6), True)]
PEOPLE = {"PA": dict(name="a", tier=1), "PB": dict(name="b", tier=2), "PC": dict(name="c", tier=3), "PM": dict(name="Mosa", tier=4)}
GPU = {  # gpu_class / gpu_note per bin (measured on Mosa's RTX 3060 12 GB)
    "mid": "10 GB+ card; measured 8.6-9.4 GB peak (RTX 3060 12 GB, Nov 16 smoke test)",
    "mid_S7": "10 GB+ card; 1,000-drone scale test 9.0 GB peak; not measured with the flight generator above 175 drones",
    "high": "12 GB+ card; scale test 9.2-9.5 GB at 10k-100k drones; flight mode not measured yet (Mosa)"}
def gpu_for(b): return ("high", GPU["high"]) if b[4] == 4 else ("mid", GPU["mid_S7"] if b[0] == "S7" else GPU["mid"])
def bin_label(b): return f"{b[0]}_{b[1]}-{b[2]}"
TIERS = {t: [min(b[1] for b in BINS if b[4] == t), max(b[2] for b in BINS if b[4] == t)] for t in sorted({b[4] for b in BINS})}
# ------------------------------------------------------------------------------------------------------------------------------
GEN_TRACKING_MAX_TOTAL = 1000   # gen_tracking.py makes one USD prim per drone; tier 1-3 rows stay <= 1000 drones in total
SKIES = ["noon_grass_4k", "qwantani_4k", "kloofendal_48d_partly_cloudy_4k", "champagne_castle_1_4k"]   # common.SKIES order
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
    "1,000-100,000 drones in the flight generator": "S8 needs the PointInstancer big-swarm mode (scale_test.py has it, gen_tracking.py does not); rows are planned but marked needs_bigswarm_mode",
    "multi-swarm in S7 (501-1000)": "two or more swarms of 501+ drones exceed the 1,000-drone total used with gen_tracking.py; S7 rows are single-swarm",
    "exact breakaway start frame": "requested start frame is honoured unless the scripted breakaway is too close to the end of the source sequence; the actual frame is in meta.json",
    "fps": "chosen by the generator from the row seed (10/12/15)",
}
COLS = ["seq_id", "assignee_slot", "deal_rank", "tier", "size_bin", "n_per_swarm", "n_swarms", "n_drones_total", "seed", "split", "gpu_class", "gpu_note",
        "test_kind", "time_of_day", "sun_el", "sun_az", "sun_int", "sun_ct", "weather", "sky_tex", "sky_name", "sky_int", "wind", "jitter",
        "camera_setup", "cam_framing", "cam_dist", "cam_angle", "cam_el_deg", "chase_fit", "model_mix_name", "model_mix", "color_mode",
        "density", "spread", "swarm_sep", "breakaway", "brk_count", "brk_size", "brk_start_frame", "frames", "generator"]

def lhs(rng, n):
    """1-D Latin hypercube in [0,1): one point per stratum, shuffled."""
    return (rng.permutation(n) + rng.random(n)) / n

def interleave(counts, labels):
    """Systematic sequence with the given count per label, each label spread evenly (e.g. 21/5/4 -> T T T T V T ...)."""
    keys = sorted((((j + 0.5) / c, k), lab) for k, (c, lab) in enumerate(zip(counts, labels)) for j in range(c))
    return [lab for _, lab in keys]

def deal_balanced(rng, levels, groups):
    """groups = list of index lists (one per bin, in bin order). Each dimension shuffles the rows inside every bin, then deals the
    levels round-robin over the concatenation: every level equally often (+-1) per person AND per bin."""
    out = {}; seq = []
    for g in groups: g = list(g); rng.shuffle(g); seq += g
    off = int(rng.integers(len(levels)))
    for k, i in enumerate(seq): out[i] = levels[(k + off) % len(levels)]
    return out

def flags(rng, n, frac):
    k = int(round(frac * n)); v = np.array([1] * k + [0] * (n - k)); rng.shuffle(v); return v

def build(args):
    rng = np.random.default_rng(args.plan_seed); rows = []; seed_next = args.base_seed
    for slot in PEOPLE:
        PB = [b for b in BINS if b[3] == slot]
        if not PB: continue
        R = []; groups = []
        for b in PB:
            name, lo, hi, _, tier, nseq, (nt, nv, nx), allow_multi = b
            brk = flags(rng, nseq, args.brk_frac)
            multi = flags(rng, nseq, args.multi_frac) if allow_multi else np.zeros(nseq, int)
            if not allow_multi:   # move this bin's multi-swarm share to the other bins of the same person (keeps ~20 % per person)
                args._carry = getattr(args, "_carry", {}); args._carry[slot] = args._carry.get(slot, 0) + int(round(args.multi_frac * nseq))
            order = np.lexsort((rng.random(nseq), multi, brk)); pat = interleave((nt, nv, nx), ("train", "val", "test"))
            off = int(rng.integers(nseq)); split = [None] * nseq
            for j, i in enumerate(order): split[i] = pat[(j + off) % nseq]
            u_n = lhs(rng, nseq)
            g0 = len(R)
            for i in range(nseq):
                R.append(dict(_bin=b, tier=tier, size_bin=bin_label(b), split=split[i], _brk=int(brk[i]), _multi=int(multi[i]), _un=float(u_n[i]), frames=args.frames))
            groups.append(list(range(g0, len(R))))
        # carry multi-swarm rows from a single-swarm-only bin to the person's other multi-capable bins (single, non-test-specific rows)
        carry = getattr(args, "_carry", {}).get(slot, 0)
        cand = [i for g, b in zip(groups, PB) if b[7] for i in g if not R[i]["_multi"]]; rng.shuffle(cand)
        # round-robin over the bins so the extra multi rows spread evenly
        byb = {}
        for i in cand: byb.setdefault(R[i]["size_bin"], []).append(i)
        keys = sorted(byb); k = 0
        while carry > 0 and any(byb.values()):
            lst = byb[keys[k % len(keys)]]
            if lst: R[lst.pop()]["_multi"] = 1; carry -= 1
            k += 1
        # test hold-outs dealt in turn over the person's test rows, bin by bin
        kinds = ["in_dist", "ood_light", "ood_cam"]; t = 0; koff = int(rng.integers(3))
        for g in groups:
            tl = [i for i in g if R[i]["split"] == "test"]; rng.shuffle(tl)
            for i in tl: R[i]["test_kind"] = kinds[(t + koff) % 3]; t += 1
        for r in R: r.setdefault("test_kind", "")
        n = len(R)
        u_az, u_el, u_int, u_ct, u_cam, u_sep, u_bs, u_bf, u_sky = (lhs(rng, n) for _ in range(9))
        gl = [[i for i in g if R[i]["test_kind"] != "ood_light"] for g in groups]; gc = [[i for i in g if R[i]["test_kind"] != "ood_cam"] for g in groups]
        tod = deal_balanced(rng, ["mid_sun", "high_sun"], gl); cams = deal_balanced(rng, ["chase_near", "chase_mid", "chase_far"], gc)
        wea = deal_balanced(rng, list(WEATHER), groups); ang = deal_balanced(rng, list(CAM_ANGLE), groups); wind = deal_balanced(rng, list(WIND), groups)
        mix = deal_balanced(rng, list(MODEL_MIX), groups); col = deal_balanced(rng, COLOR, groups); dens = deal_balanced(rng, list(SPREAD), groups)
        for i, r in enumerate(R):
            b = r["_bin"]; lo, hi, tier = b[1], b[2], b[4]
            ns = int(rng.choice([2, 3], p=[0.6, 0.4])) if r["_multi"] else 1
            plo, phi = lo, hi
            if tier <= 3:
                if ns == 3 and GEN_TRACKING_MAX_TOTAL // 3 < lo: ns = 2
                phi = min(hi, GEN_TRACKING_MAX_TOTAL // ns)
            if lo < 3 and (r["_multi"] or r["_brk"]): plo = max(plo, 3)   # >= 2 drones per swarm to split, >= 3 for a breakaway
            nps = int(round(math.exp(math.log(plo) + r["_un"] * (math.log(phi + 1) - math.log(plo))))); nps = int(np.clip(nps, plo, phi))
            td = "low_sun" if r["test_kind"] == "ood_light" else tod[i]; cm = "wide" if r["test_kind"] == "ood_cam" else cams[i]
            tt = TIME_OF_DAY[td]; c = CAMERA[cm]; gcl, gnote = gpu_for(b)
            r.update(n_per_swarm=nps, n_swarms=ns, n_drones_total=nps * ns, seed=seed_next, gpu_class=gcl, gpu_note=gnote,
                     time_of_day=td, sun_el=round(tt["el"][0] + u_el[i] * (tt["el"][1] - tt["el"][0]), 2), sun_az=round(360 * u_az[i], 1),
                     sun_int=round(tt["inten"][0] + u_int[i] * (tt["inten"][1] - tt["inten"][0]), 3), sun_ct=int(tt["ct"][0] + u_ct[i] * (tt["ct"][1] - tt["ct"][0])),
                     weather=wea[i], wind=wind[i], jitter=WIND[wind[i]], camera_setup=cm, cam_framing=c["framing"], cam_dist=c["dist"],
                     model_mix_name=mix[i], model_mix=MODEL_MIX[mix[i]], color_mode=col[i], density=dens[i], spread=SPREAD[dens[i]],
                     swarm_sep=round(0.3 + 1.7 * u_sep[i], 2) if ns > 1 else "", chase_fit=1)
            sk = WEATHER[wea[i]][int(u_sky[i] * 2) % 2]; r.update(sky_tex=sk, sky_name=SKIES[sk], sky_int=int(400 + 900 * ((u_sky[i] * 2) % 1)))
            if c["framing"] == "chase":
                a = CAM_ANGLE[ang[i]]; r.update(cam_angle=ang[i], cam_el_deg=round(a[0] + u_cam[i] * (a[1] - a[0]), 2))
            else: r.update(cam_angle="auto", cam_el_deg="")
            if r["_brk"]:
                bs = 1 if nps < 10 and u_bs[i] < 0.5 else max(1, int(round(nps * (0.10 + 0.20 * u_bs[i]))))
                bs = min(bs, max(1, nps // 2 - 1)) if nps >= 4 else 1
                r.update(breakaway="on", brk_count=2 if (u_bs[i] * 7) % 1 < 0.3 and nps >= 2 * (bs + 1) else 1, brk_size=bs,
                         brk_start_frame=int(round(args.frames * (0.15 + 0.25 * u_bf[i]))) + 1)
            else: r.update(breakaway="off", brk_count=0, brk_size=0, brk_start_frame="")
            r["generator"] = "gen_tracking" if tier <= 3 else "needs_bigswarm_mode"
            r["assignee_slot"] = slot; seed_next += 1
        # stratified rank inside the person: by split, bin, breakaway, multi, lighting, camera, size
        key = sorted(range(n), key=lambda i: (R[i]["split"], R[i]["size_bin"], R[i]["breakaway"], R[i]["n_swarms"] > 1, R[i]["time_of_day"], R[i]["camera_setup"], R[i]["n_per_swarm"]))
        for rank, i in enumerate(key): R[i]["deal_rank"] = rank
        cnt = {}
        for r in R:
            k = r["_bin"][0]; r["seq_id"] = f"{k.lower()}_{cnt.get(k, 0):03d}"; cnt[k] = cnt.get(k, 0) + 1
        for r in R:
            for k in [k for k in r if k.startswith("_")]: del r[k]
        rows += R
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
    for dim in ("tier", "size_bin", "split", "test_kind", "time_of_day", "weather", "camera_setup", "cam_angle", "wind", "model_mix_name", "color_mode", "density",
                "breakaway", "n_swarms", "assignee_slot", "gpu_class", "generator"):
        c = {}
        for r in rows: c[str(r[dim]) or "-"] = c.get(str(r[dim]) or "-", 0) + 1
        s[dim] = dict(sorted(c.items()))
    def cross(a, b):
        x = {}
        for r in rows: x.setdefault(str(r[a]), {}).setdefault(str(r[b]) or "-", 0); x[str(r[a])][str(r[b]) or "-"] += 1
        return dict(sorted(x.items()))
    s["tier_x_split"] = {f"tier{k}": v for k, v in cross("tier", "split").items()}
    s["size_bin_x_split"] = cross("size_bin", "split"); s["slot_x_split"] = cross("assignee_slot", "split")
    s["size_bin_x_test_kind"] = cross("size_bin", "test_kind"); s["size_bin_x_breakaway"] = cross("size_bin", "breakaway")
    s["size_bin_x_n_swarms"] = cross("size_bin", "n_swarms")
    s["sequences"] = len(rows); s["frames"] = int(sum(int(r["frames"]) for r in rows))
    return s

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(ROOT, "datasets", "manifest"))
    ap.add_argument("--frames", type=int, default=50); ap.add_argument("--brk-frac", type=float, default=0.4); ap.add_argument("--multi-frac", type=float, default=0.2)
    ap.add_argument("--base-seed", type=int, default=910000, help="first row seed (rows get base, base+1, ...); old datasets used 77001 / 1000*member")
    ap.add_argument("--plan-seed", type=int, default=2026, help="seed of the plan itself (same plan-seed + args = same manifest)")
    ap.add_argument("--slice", action="store_true", help="also write a slice CSV: manifest_<slot>.csv with --slot, or manifest_P<person>of<of>.csv")
    ap.add_argument("--slot", default="", help="with --slice: PA, PB, PC or PM"); ap.add_argument("--person", type=int, default=1); ap.add_argument("--of", type=int, default=1)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    rows = build(a); args_out = {k: v for k, v in vars(a).items() if not k.startswith("_")}
    write_csv(os.path.join(a.out, "manifest.csv"), rows)
    meta = dict(created_by="ds_tools/make_manifest.py", args=args_out,
                size_bins=[dict(size_bin=bin_label(b), name=b[0], lo=b[1], hi=b[2], slot=b[3], person=PEOPLE[b[3]]["name"], tier=b[4], sequences=b[5],
                                train=b[6][0], val=b[6][1], test=b[6][2], multi_swarm=b[7], gpu_class=gpu_for(b)[0], gpu_note=gpu_for(b)[1]) for b in BINS],
                tiers={str(k): v for k, v in TIERS.items()}, tier_meaning="tier = person group: 1 = a (PA), 2 = b (PB), 3 = c (PC), 4 = Mosa (PM)",
                people=PEOPLE, holdout_test_only=HOLDOUT,
                levels=dict(time_of_day=TIME_OF_DAY, weather={k: [SKIES[i] for i in v] for k, v in WEATHER.items()}, camera_setup=CAMERA, cam_angle=CAM_ANGLE,
                            wind_jitter_multiplier=WIND, model_mix=MODEL_MIX, color_mode=COLOR, density_spread=SPREAD),
                gen_tracking_max_total_drones=GEN_TRACKING_MAX_TOTAL, unsupported=UNSUPPORTED, summary=summary(rows), rows=rows)
    json.dump(meta, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    s = meta["summary"]; print(f"{s['sequences']} sequences, {s['frames']} frames -> {a.out}")
    for k in ("size_bin_x_split", "slot_x_split", "size_bin_x_test_kind", "size_bin_x_breakaway", "size_bin_x_n_swarms", "time_of_day", "camera_setup", "gpu_class", "generator"): print(f"  {k}: {s[k]}")
    if a.slice:
        if a.slot: sl = [r for r in rows if r["assignee_slot"] == a.slot]; p = os.path.join(a.out, f"manifest_{a.slot}.csv")
        else: sl = person_slice(rows, a.person, a.of); p = os.path.join(a.out, f"manifest_P{a.person}of{a.of}.csv")
        write_csv(p, sl); print(f"slice: {len(sl)} rows -> {p}")
