"""Summary of tracking_v1: python summarize_tracking.py [dir] -> dir/stats/tracking_v1_summary.png + summary.json"""
import os, sys, json, glob, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets", "tracking_v1")
out = os.path.join(D, "stats"); os.makedirs(out, exist_ok=True)
seqs = []
for m in sorted(glob.glob(os.path.join(D, "seq_*", "meta.json"))):
    meta = json.load(open(m)); sd = os.path.dirname(m)
    L = [json.loads(l) for l in open(os.path.join(sd, "ann", "frames.jsonl"))]
    vis = np.array([r["n_visible_ge1px"] / r["n_total"] for r in L])
    ph = np.array([[sum(d["phase"] == p for d in r["drones"]) for p in ("ground", "takeoff", "cruise")] for r in L])
    dist = np.array([[d["dist_to_swarm_centroid_m"] for d in r["drones"]] for r in L])
    brk_ids = [b["id"] for b in meta["breakaways"]]
    gt_lines = sum(1 for _ in open(os.path.join(sd, "gt", "gt.txt")) if _.strip())
    ids_per_frame = [sorted(d["id"] for d in r["drones"]) for r in L]
    stable = all(x == ids_per_frame[0] for x in ids_per_frame)
    seqs.append(dict(seq=meta["seq"], N=meta["n_drones"], F=meta["frames"], fps=meta["fps"], launch=meta["launch"], cam=meta["camera_mode"], ground_cam=meta["ground_camera"],
                     formation=meta["formation"], brk=meta["breakaways"], vis_mean=float(vis.mean()), gt_boxes=gt_lines, ids_stable=stable, render_s=meta["render_seconds"],
                     first_takeoff_frame=int(np.argmax(ph[:, 1] + ph[:, 2] > 0)) + 1, all_cruise_frame=int(np.argmax(ph[:, 2] == meta["n_drones"])) + 1 if (ph[:, 2] == meta["n_drones"]).any() else None,
                     _dist=dist, _brk_ids=brk_ids, _ph=ph))
fig, ax = plt.subplots(2, 2, figsize=(15, 9))
a = ax[0, 0]; x = np.arange(len(seqs)); a.bar(x, [s["N"] for s in seqs], color=["#d83b3b" if s["brk"] else "#3b7dd8" for s in seqs])
a.set_yscale("log"); a.set_xticks(x); a.set_xticklabels([s["seq"] for s in seqs], fontsize=7); a.set_title("Drones per sequence (red = has breakaway, blue = negative)"); a.set_xlabel("sequence")
a = ax[0, 1]; ex = next((s for s in seqs if s["N"] >= 50), seqs[0]); t = np.arange(ex["F"]) / ex["fps"]
for j, lab in enumerate(("ground", "takeoff", "cruise")): a.plot(t, ex["_ph"][:, j], label=lab)
a.set_title(f"Phase labels over time, seq {ex['seq']} (N={ex['N']})"); a.set_xlabel("s"); a.set_ylabel("drones"); a.legend()
a = ax[1, 0]; shown = 0
for s in seqs:
    if not s["brk"] or shown >= 4: continue
    t = np.arange(s["F"]) / s["fps"]; sw = [i for i in range(s["N"]) if i + 1 not in s["_brk_ids"]]
    l, = a.plot(t, np.percentile(s["_dist"][:, sw], 95, axis=1), "--", lw=1)
    for b in s["brk"]: a.plot(t, s["_dist"][:, b["id"] - 1], color=l.get_color(), lw=2); a.axvline(b["start_time_s"], color=l.get_color(), lw=0.5)
    shown += 1
a.set_title("Distance to swarm centroid: breakaway drones (solid) vs swarm 95th pct (dashed)"); a.set_xlabel("s"); a.set_ylabel("m"); a.set_yscale("symlog")
a = ax[1, 1]; a.axis("off")
nb = sum(1 for s in seqs if s["brk"]); nbd = sum(len(s["brk"]) for s in seqs)
from collections import Counter
kinds = Counter(b["type"] for s in seqs for b in s["brk"])
txt = (f"sequences: {len(seqs)}  ({nb} with breakaway, {len(seqs)-nb} negative)\nbreakaway drones: {nbd}  {dict(kinds)}\n"
       f"frames: {sum(s['F'] for s in seqs)}  (per seq {min(s['F'] for s in seqs)}-{max(s['F'] for s in seqs)}, fps {sorted(set(s['fps'] for s in seqs))})\n"
       f"drones per seq: {min(s['N'] for s in seqs)}-{max(s['N'] for s in seqs)}\nMOT gt boxes: {sum(s['gt_boxes'] for s in seqs)}\n"
       f"launch: {dict(Counter(s['launch'] for s in seqs))}\ncamera: {dict(Counter(s['cam'] for s in seqs))}, ground-level cams: {sum(s['ground_cam'] for s in seqs)}\n"
       f"track IDs identical in every frame of every seq: {all(s['ids_stable'] for s in seqs)}\nmean visible fraction: {np.mean([s['vis_mean'] for s in seqs]):.2f}\n"
       f"render time per seq: {np.mean([s['render_s'] for s in seqs]):.0f} s mean")
a.text(0, 1, txt, va="top", family="monospace", fontsize=10)
fig.suptitle("tracking_v1 - synthetic swarm takeoff / cruise / breakaway sequences (Isaac Sim 6.2, Squad DesertStage)", fontsize=13); fig.tight_layout()
fig.savefig(os.path.join(out, "tracking_v1_summary.png"), dpi=110)
json.dump([{k: v for k, v in s.items() if not k.startswith("_")} for s in seqs], open(os.path.join(out, "summary.json"), "w"), indent=1)
print(txt)
