"""Coverage check for uploaded team data against the manifest (no Isaac Sim needed: numpy + Pillow + matplotlib).

    python ds_tools/check_coverage.py datasets\\alex_tier1 datasets\\alex_tier2 datasets\\sam_tier1 ...
    python ds_tools/check_coverage.py "datasets\\*_tier*" --manifest datasets\\manifest\\manifest.json --out datasets\\coverage

Each folder is a dataset folder (contains <seq_id>\\meta.json) or a single sequence folder. Reports:
  - which manifest rows are done / missing / incomplete (wrong frame count, no swarm labels), rows rendered twice, unknown folders
  - done vs planned counts per dimension (tier, split, lighting, weather, camera, model mix, colour, wind, density, breakaway,
    multi-swarm, person) and flags under-filled levels (done < --min-frac of planned, or fewer than --min-seqs sequences)
  - planned-vs-actual checks: breakaway really present, number of swarms really split
  - histograms: drones per swarm, swarm box size (sqrt of box area, px), breakaway box size; swarm boxes cut by the image border
  - near-duplicates: duplicate seeds, and first frames whose 64-bit difference hash is within --hash-dist bits
Writes <out>/coverage_report.md, coverage_report.json, hist_swarm_size.png, hist_box_size.png, coverage_by_dim.png."""
import os, sys, csv, json, glob, argparse
import numpy as np
from PIL import Image

DIMS = ["tier", "split", "test_kind", "time_of_day", "weather", "camera_setup", "cam_angle", "model_mix_name", "color_mode", "wind", "density", "breakaway", "multi_swarm", "assignee_slot"]

def default_manifest():
    root = os.environ.get("DRONE_SCENE_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    for p in (os.path.join(root, "manifests", "team_v1", "manifest.json"), os.path.join(root, "datasets", "manifest", "manifest.json")):
        if os.path.exists(p): return p
    return os.path.join(root, "manifests", "team_v1", "manifest.json")

def seq_dirs(paths):
    out = []
    for p in paths:
        for q in (glob.glob(p) or [p]):
            if os.path.exists(os.path.join(q, "meta.json")): out.append(q)
            else: out += sorted(os.path.dirname(m) for m in glob.glob(os.path.join(q, "*", "meta.json")))
    return out

def dhash(path, n=8):
    g = np.asarray(Image.open(path).convert("L").resize((n + 1, n), Image.BILINEAR), np.int16)
    return int("".join("1" if b else "0" for b in (g[:, 1:] > g[:, :-1]).ravel()), 2)

def level(r, d):
    if d == "multi_swarm": return "multi" if int(r["n_swarms"]) > 1 else "single"
    v = r.get(d, ""); return str(v) if str(v) != "" else "-"

def read_tracks(sd):
    p = os.path.join(sd, "swarm", "swarm_tracks.csv")
    if not os.path.exists(p): return None
    rows = list(csv.reader(open(p)))
    if rows and not rows[0][0].strip().lstrip("-").isdigit(): rows = rows[1:]
    return [r for r in rows if len(r) >= 7]

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+"); ap.add_argument("--manifest", default=default_manifest(), help="default: manifests/team_v1/manifest.json (the shared team plan)")
    ap.add_argument("--out", default=""); ap.add_argument("--min-frac", type=float, default=0.5); ap.add_argument("--min-seqs", type=int, default=3)
    ap.add_argument("--hash-dist", type=int, default=4); ap.add_argument("--rows", default="", help="only check these seq_ids (e.g. a smoke test)")
    ap.add_argument("--tiers", default="", help="only plan these tiers, e.g. 1,2,3 while tier 4 is not wired up")
    a = ap.parse_args()
    M = json.load(open(a.manifest)); plan = {r["seq_id"]: r for r in M["rows"]}
    if a.rows: plan = {k: v for k, v in plan.items() if k in set(a.rows.split(","))}
    if a.tiers: plan = {k: v for k, v in plan.items() if int(v["tier"]) in {int(t) for t in a.tiers.split(",")}}
    out = a.out or os.path.join(os.path.commonpath([os.path.abspath(f) for f in seq_dirs(a.folders)] or ["."]), "_coverage")
    os.makedirs(out, exist_ok=True)
    found = {}; unknown = []; dup_render = []; seq_info = {}
    for sd in seq_dirs(a.folders):
        meta = json.load(open(os.path.join(sd, "meta.json"))); row = meta.get("manifest_row") or {}
        sid = row.get("seq_id") or os.path.basename(sd)
        if sid not in plan and sid not in {r["seq_id"] for r in M["rows"]}: unknown.append(sd); continue
        if sid in found: dup_render.append([found[sid], sd]); continue
        found[sid] = sd
        tr = read_tracks(sd); nimg = len(glob.glob(os.path.join(sd, "img", "*.jpg")))
        sw = [b for b in (tr or []) if b[2].strip() in ("0", "swarm")]; bk = [b for b in (tr or []) if b[2].strip() in ("1", "breakaway")]
        def side(b): x1, y1, x2, y2 = map(float, b[3:7]); return float(np.sqrt(max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)))
        def cut(b): x1, y1, x2, y2 = map(float, b[3:7]); return x1 <= 0 or y1 <= 0 or x2 >= 1279 or y2 >= 719
        sa = {}
        sp = os.path.join(sd, "swarm", "swarm_annotations.json")
        if os.path.exists(sp):
            try:
                J = json.load(open(sp)); sa = dict(n_breakaway_entities=sum(1 for e in J.get("entities", []) if e.get("cls") == "breakaway"))
            except Exception as e: sa = dict(error=repr(e))
        first = os.path.join(sd, "img", "000001.jpg")
        seq_info[sid] = dict(dir=sd, frames_img=nimg, frames_meta=meta.get("frames"), seed=meta.get("seed"), fps=meta.get("fps"), render_s=meta.get("render_seconds"),
                             swarms=[s["n_drones"] for s in meta.get("swarms", [])], has_breakaway=bool(meta.get("has_breakaway")),
                             brk_start=[b["start_frame"] for b in meta.get("breakaways", [])], swarm_labels=tr is not None,
                             swarm_box_px=[side(b) for b in sw], brk_box_px=[side(b) for b in bk], swarm_box_cut=sum(cut(b) for b in sw), n_swarm_boxes=len(sw), n_brk_boxes=len(bk),
                             dhash=dhash(first) if os.path.exists(first) else None, **sa)
    # status per row
    status = {}
    for sid, r in plan.items():
        if sid not in seq_info: status[sid] = "needs_bigswarm_mode" if r.get("generator") != "gen_tracking" else "missing"; continue
        s = seq_info[sid]; prob = []
        if s["frames_img"] != s["frames_meta"]: prob.append(f"{s['frames_img']} images vs meta frames {s['frames_meta']}")
        if not s["swarm_labels"]: prob.append("no swarm labels (run ds_tools/swarm_annotate.py)")
        if str(r.get("breakaway")) == "on" and not s["has_breakaway"]: prob.append("breakaway planned but none rendered")
        if int(r["n_swarms"]) > 1 and len(s["swarms"]) < int(r["n_swarms"]): prob.append(f"{len(s['swarms'])} swarms rendered, {r['n_swarms']} planned")
        status[sid] = "done" if not prob else "incomplete: " + "; ".join(prob)
    done = [k for k, v in status.items() if v == "done" or v.startswith("incomplete")]
    # duplicates
    seeds = {}
    for sid, s in seq_info.items(): seeds.setdefault(s["seed"], []).append(sid)
    dup_seed = {str(k): v for k, v in seeds.items() if len(v) > 1}
    near = []; ids = [k for k in seq_info if seq_info[k]["dhash"] is not None]
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            d = bin(seq_info[ids[i]]["dhash"] ^ seq_info[ids[j]]["dhash"]).count("1")
            if d <= a.hash_dist: near.append([ids[i], ids[j], d])
    # coverage per dimension
    cov = {}; flags = []
    for d in DIMS:
        c = {}
        for sid, r in plan.items():
            if r.get("generator") != "gen_tracking" and d != "tier": pass
            lv = level(r, d); e = c.setdefault(lv, dict(planned=0, done=0)); e["planned"] += 1; e["done"] += sid in done
        cov[d] = dict(sorted(c.items()))
        for lv, e in c.items():
            if lv == "-": continue
            if e["done"] < a.min_frac * e["planned"] or e["done"] < min(a.min_seqs, e["planned"]): flags.append(f"{d}={lv}: {e['done']}/{e['planned']} done")
    combo = {}
    for sid in done:
        r = plan[sid]; k = f"tier{r['tier']}|{r['time_of_day']}|{r['weather']}|{r['camera_setup']}|brk_{r['breakaway']}|{level(r, 'multi_swarm')}"; combo[k] = combo.get(k, 0) + 1
    # histograms
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    sizes = [n for s in seq_info.values() for n in s["swarms"]]; boxes = [x for s in seq_info.values() for x in s["swarm_box_px"]]; bboxes = [x for s in seq_info.values() for x in s["brk_box_px"]]
    fig, ax = plt.subplots(figsize=(7, 4))
    if sizes: ax.hist(sizes, bins=np.logspace(0, np.log10(max(sizes) + 1), 25), color="#3a7")
    ax.set_xscale("log"); ax.set_xlabel("drones per swarm (rendered)"); ax.set_ylabel("swarms"); ax.set_title(f"Swarm size, {len(sizes)} swarms in {len(seq_info)} sequences")
    for t, (lo, hi) in M["tiers"].items(): ax.axvline(lo, color="#999", ls=":", lw=1)
    fig.tight_layout(); fig.savefig(os.path.join(out, "hist_swarm_size.png"), dpi=110); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    if boxes: ax.hist(boxes, bins=np.logspace(0, np.log10(1300), 30), color="#38c", alpha=0.85, label=f"swarm boxes ({len(boxes)})")
    if bboxes: ax.hist(bboxes, bins=np.logspace(0, np.log10(1300), 30), color="#d33", alpha=0.7, label=f"breakaway boxes ({len(bboxes)})")
    ax.set_xscale("log"); ax.set_xlabel("box size, sqrt(w*h) px (1280x720 image)"); ax.set_ylabel("boxes (all frames)"); ax.legend(); ax.set_title("Swarm / breakaway box size")
    fig.tight_layout(); fig.savefig(os.path.join(out, "hist_box_size.png"), dpi=110); plt.close(fig)
    show = [d for d in ("tier", "split", "time_of_day", "weather", "camera_setup", "model_mix_name", "wind", "breakaway", "multi_swarm", "assignee_slot")]
    fig, axs = plt.subplots(2, 5, figsize=(16, 6))
    for axx, d in zip(axs.ravel(), show):
        lv = list(cov[d]); p = [cov[d][k]["planned"] for k in lv]; q = [cov[d][k]["done"] for k in lv]
        axx.bar(range(len(lv)), p, color="#ddd", label="planned"); axx.bar(range(len(lv)), q, color="#3a7", label="done")
        axx.set_xticks(range(len(lv))); axx.set_xticklabels(lv, rotation=35, ha="right", fontsize=8); axx.set_title(d, fontsize=10)
    axs[0, 0].legend(fontsize=8); fig.suptitle(f"Coverage vs manifest: {len(done)}/{len(plan)} planned sequences rendered"); fig.tight_layout(); fig.savefig(os.path.join(out, "coverage_by_dim.png"), dpi=100); plt.close(fig)
    # report
    nfr = sum(s["frames_img"] for s in seq_info.values()); rs = [s["render_s"] for s in seq_info.values() if s["render_s"]]
    rep = dict(manifest=os.path.abspath(a.manifest), folders=a.folders, planned=len(plan), done=sum(v == "done" for v in status.values()),
               incomplete={k: v for k, v in status.items() if v.startswith("incomplete")}, missing=[k for k, v in status.items() if v == "missing"],
               needs_bigswarm_mode=[k for k, v in status.items() if v == "needs_bigswarm_mode"], unknown_folders=unknown, rendered_twice=dup_render,
               duplicate_seeds=dup_seed, near_duplicate_first_frames=near, underfilled=flags, coverage=cov, combos_done=dict(sorted(combo.items())),
               frames_rendered=nfr, sequences_found=len(seq_info), swarm_boxes=len(boxes), breakaway_boxes=len(bboxes),
               swarm_boxes_cut_by_border=sum(s["swarm_box_cut"] for s in seq_info.values()),
               render_s_per_frame=[round(s["render_s"] / s["frames_img"], 2) for s in seq_info.values() if s["render_s"] and s["frames_img"]],
               sequences={k: {kk: vv for kk, vv in v.items() if kk not in ("swarm_box_px", "brk_box_px")} for k, v in seq_info.items()})
    json.dump(rep, open(os.path.join(out, "coverage_report.json"), "w"), indent=1)
    L = [f"# Coverage report", "", f"Manifest: `{a.manifest}`  ", f"Folders: {', '.join('`'+f+'`' for f in a.folders)}", "",
         f"- Planned rows checked: **{len(plan)}**; done: **{rep['done']}**; incomplete: **{len(rep['incomplete'])}**; missing: **{len(rep['missing'])}**; need big-swarm mode (tier 4): **{len(rep['needs_bigswarm_mode'])}**",
         f"- Sequences found: {len(seq_info)}, frames: {nfr}, swarm boxes: {len(boxes)}, breakaway boxes: {len(bboxes)}, swarm boxes touching the image border: {rep['swarm_boxes_cut_by_border']}",
         f"- Rendered twice: {len(dup_render)}; duplicate seeds: {len(dup_seed)}; near-duplicate first frames (dHash <= {a.hash_dist} bits): {len(near)}; folders not in the manifest: {len(unknown)}", ""]
    if rep["incomplete"]: L += ["## Incomplete", ""] + [f"- `{k}`: {v}" for k, v in rep["incomplete"].items()] + [""]
    if near: L += ["## Near-duplicate first frames", ""] + [f"- `{x}` ~ `{y}` ({d} bits)" for x, y, d in near] + [""]
    if dup_seed: L += ["## Duplicate seeds", ""] + [f"- seed {k}: {v}" for k, v in dup_seed.items()] + [""]
    L += ["## Under-filled levels", ""] + ([f"- {f}" for f in flags] or ["- none"]) + [""]
    L += ["## Done / planned per dimension", ""]
    for d in DIMS: L.append(f"- **{d}**: " + ", ".join(f"{k} {v['done']}/{v['planned']}" for k, v in cov[d].items()))
    L += ["", "## Rendered sequences", "", "| seq | frames | seed | fps | swarms (drones) | breakaway start | swarm boxes | brk boxes | cut by border | s/frame | status |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, s in sorted(seq_info.items()):
        spf = f"{s['render_s'] / s['frames_img']:.2f}" if s["render_s"] and s["frames_img"] else "-"
        L.append(f"| {k} | {s['frames_img']} | {s['seed']} | {s['fps']} | {s['swarms']} | {s['brk_start'] or '-'} | {s['n_swarm_boxes']} | {s['n_brk_boxes']} | {s['swarm_box_cut']} | {spf} | {status.get(k, '-')} |")
    L += ["", "s/frame = meta.render_seconds / frames (includes scene setup and swarm labelling, not Isaac Sim startup).", "",
          "Images: `hist_swarm_size.png`, `hist_box_size.png`, `coverage_by_dim.png`."]
    open(os.path.join(out, "coverage_report.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L[:8])); print("report ->", out)

if __name__ == "__main__": main()
