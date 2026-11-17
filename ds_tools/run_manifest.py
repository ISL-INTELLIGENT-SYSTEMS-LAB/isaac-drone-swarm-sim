"""Render manifest rows with the existing generator (ds_tools/gen_tracking.py in manifest mode). Resumable: a row whose
<out>/<seq_id>/meta.json exists is skipped, so just re-run the same command after a crash.

Run it with the kit's Isaac Sim python (it starts gen_tracking.py with the same interpreter):
    .\\env\\Scripts\\python.exe ds_tools\\run_manifest.py --person 2 --of 4 --name alex          # Alex's slice -> datasets\\alex_tier1 ... tier3
    .\\env\\Scripts\\python.exe ds_tools\\run_manifest.py --person 2 --of 4 --name alex --dry-run   # list the rows and the commands only
    .\\env\\Scripts\\python.exe ds_tools\\run_manifest.py --rows t1_003,t2_010 --frames 10 --name smoke --out-root datasets\\_manifest_smoke

Output: <out-root>/<name>_tier<k>/<seq_id>/ (img, masks, gt, ann/frames.jsonl, meta.json with the manifest row, swarm/ labels),
logs in <out-root>/<name>_tier<k>/logs/. Rows are grouped by tier, model mix and colour mode (the drone pool is built once per
Isaac Sim process) and rendered --per-proc rows per process. Tier 4 rows (generator = needs_bigswarm_mode) are skipped and listed."""
import os, sys, csv, json, time, shutil, argparse, subprocess

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.environ.get("DRONE_SCENE_ROOT") or os.path.abspath(os.path.join(HERE, ".."))
INT = ("tier", "n_per_swarm", "n_swarms", "n_drones_total", "seed", "deal_rank", "sky_tex", "brk_count", "brk_size", "frames", "chase_fit")

def default_manifest(root=ROOT):
    for p in (os.path.join(root, "manifests", "team_v1", "manifest.json"), os.path.join(root, "datasets", "manifest", "manifest.json")):
        if os.path.exists(p): return p
    return os.path.join(root, "manifests", "team_v1", "manifest.json")

def load_rows(path):
    if path.endswith(".json"): return json.load(open(path))["rows"]
    rows = list(csv.DictReader(open(path, newline="")))
    for r in rows:
        for k in INT:
            if r.get(k) not in (None, ""): r[k] = int(float(r[k]))
    return rows

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=default_manifest(), help="default: manifests/team_v1/manifest.json (the shared team plan)")
    ap.add_argument("--person", type=int, default=0); ap.add_argument("--of", type=int, default=0, help="number of people (slice = deal_rank % of == person-1)")
    ap.add_argument("--slot", default="", help="use the assignee_slot column instead, e.g. P2"); ap.add_argument("--rows", default="", help="comma-separated seq_ids")
    ap.add_argument("--tiers", default="1,2,3"); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--frames", type=int, default=0, help="override frames per sequence (smoke tests)")
    ap.add_argument("--name", required=True, help="your name (folder prefix)"); ap.add_argument("--out-root", default=os.path.join(ROOT, "datasets"))
    ap.add_argument("--per-proc", type=int, default=4); ap.add_argument("--rt", type=int, default=4); ap.add_argument("--py", default=sys.executable)
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--min-free-gb", type=float, default=6.0)
    ap.add_argument("--close-grace", type=float, default=45, help="seconds to wait for Isaac Sim to exit after all rows are written, then kill it")
    ap.add_argument("--timeout-per-row", type=float, default=900, help="kill a process that takes longer than this per row (+300 s)")
    a = ap.parse_args()
    rows = load_rows(a.manifest)
    if a.rows: want = set(a.rows.split(",")); rows = [r for r in rows if r["seq_id"] in want]
    elif a.slot: rows = [r for r in rows if r["assignee_slot"] == a.slot]
    elif a.person and a.of: rows = [r for r in rows if int(r["deal_rank"]) % a.of == a.person - 1]
    else: sys.exit("pick rows: --person N --of M, --slot PN or --rows id,id")
    tiers = {int(t) for t in a.tiers.split(",")} if a.tiers else None
    if tiers and not a.rows: rows = [r for r in rows if int(r["tier"]) in tiers]
    skip = [r for r in rows if r.get("generator") != "gen_tracking"]; rows = [r for r in rows if r.get("generator") == "gen_tracking"]
    if a.limit: rows = rows[:a.limit]
    if a.frames:
        for r in rows:   # scale the breakaway start with the clip length
            if r.get("brk_start_frame") not in (None, ""): r["brk_start_frame"] = max(2, int(round(int(r["brk_start_frame"]) * a.frames / int(r["frames"]))))
            r["frames"] = a.frames
    if skip: print(f"SKIPPED {len(skip)} rows that need the big-swarm mode (not in gen_tracking.py yet): {','.join(r['seq_id'] for r in skip)}")
    out_root = a.out_root if os.path.isabs(a.out_root) else os.path.join(ROOT, a.out_root)
    groups = {}
    for r in rows: groups.setdefault((int(r["tier"]), r["model_mix"], r["color_mode"]), []).append(r)
    print(f"{len(rows)} rows in {len(groups)} groups (tier, model mix, colour)")
    t_all = time.time(); results = []
    for (tier, mix, col), G in sorted(groups.items()):
        out = os.path.join(out_root, f"{a.name}_tier{tier}"); L = os.path.join(out, "logs"); os.makedirs(L, exist_ok=True)
        for c0 in range(0, len(G), a.per_proc):
            chunk = G[c0:c0 + a.per_proc]
            todo = [r for r in chunk if not os.path.exists(os.path.join(out, r["seq_id"], "meta.json"))]
            if not todo: continue
            tag = f"{todo[0]['seq_id']}_{len(todo)}"; rp = os.path.join(L, f"rows_{tag}.json"); json.dump(todo, open(rp, "w"), indent=1)
            nmax = max(int(r["n_drones_total"]) for r in todo)
            env = dict(os.environ, OMNI_KIT_ACCEPT_EULA="YES", PYTHONUNBUFFERED="1", DRONE_SCENE_ROOT=ROOT, MANIFEST_ROWS=rp, OUT=out, SEQ_START="0", SEQ_END=str(len(todo)),
                       RT=str(a.rt), NMAX_T=str(max(nmax, 4)), MODEL_MIX=mix, COLOR_MODE=col, COLOR_STRENGTH="0.85", COLOR_WEAR="1", SWARM_ANN="1", REP_WRITER="0")
            for k in ("BIN_LO", "BIN_HI", "POOL_EACH", "FORCE_BRK", "N_DRONES", "CLIP_FRAMES", "CAM_FRAMING", "CAM_DIST"): env.pop(k, None)
            cmd = [a.py, os.path.join(HERE, "gen_tracking.py")]
            print(f"[{time.strftime('%H:%M:%S')}] tier {tier} mix={mix} color={col} rows {','.join(r['seq_id'] for r in todo)} (max {nmax} drones) -> {out}", flush=True)
            if a.dry_run: continue
            for attempt in range(2):
                t0 = time.time()
                killed = None
                with open(os.path.join(L, f"{tag}_try{attempt}.log"), "w") as lo, open(os.path.join(L, f"{tag}_try{attempt}.err"), "w") as le:
                    pr = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=lo, stderr=le); t_all_done = None; t_crash = None
                    limit = a.timeout_per_row * len(todo) + 300
                    while pr.poll() is None:   # watchdog: Isaac Sim sometimes hangs in app.close() after all sequences are written
                        time.sleep(5)
                        if all(os.path.exists(os.path.join(out, r["seq_id"], "swarm", "swarm_tracks.csv")) for r in todo):   # written last (swarm labels)
                            t_all_done = t_all_done or time.time()
                            if time.time() - t_all_done > a.close_grace: killed = "close_hang"; pr.kill(); break
                        elif time.time() - t0 > limit: killed = "timeout"; pr.kill(); break
                        elif "A crash has occurred" in open(le.name, errors="ignore").read()[-20000:]:   # Kit crashed but the process does not exit
                            t_crash = t_crash or time.time()
                            if time.time() - t_crash > 30: killed = "crash"; pr.kill(); break
                    rc = pr.wait()
                done = [r["seq_id"] for r in todo if os.path.exists(os.path.join(out, r["seq_id"], "meta.json"))]
                dt = time.time() - t0; print(f"    exit {rc}{' (killed: ' + killed + ')' if killed else ''}, {len(done)}/{len(todo)} done in {dt:.0f}s", flush=True)
                results.append(dict(tag=tag, attempt=attempt, rc=rc, killed=killed, done=len(done), todo=len(todo), seconds=round(dt, 1)))
                if len(done) == len(todo): break
            free = shutil.disk_usage(out).free / 2 ** 30
            if free < a.min_free_gb: print(f"STOP: only {free:.1f} GB free"); sys.exit(2)
    if not a.dry_run:
        print(f"finished in {time.time() - t_all:.0f}s"); sp = os.path.join(out_root, f"run_manifest_{a.name}.json"); hist = json.load(open(sp)).get("runs", []) if os.path.exists(sp) else []
        hist.append(dict(started=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t_all)), args=sys.argv[1:], results=results, skipped=[r["seq_id"] for r in skip]))
        json.dump(dict(runs=hist), open(sp, "w"), indent=1)

if __name__ == "__main__": main()
