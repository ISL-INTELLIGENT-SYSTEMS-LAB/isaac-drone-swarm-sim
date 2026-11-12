"""Breakaway example plot for one sequence: python plot_breakaway.py <seq_dir> <out.png>"""
import sys, json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sd, out = sys.argv[1], sys.argv[2]
meta = json.load(open(sd + "/meta.json")); L = [json.loads(l) for l in open(sd + "/ann/frames.jsonl")]
ids = [d["id"] for d in L[0]["drones"]]; P = np.array([[d["pos_m"] for d in r["drones"]] for r in L]); Dc = np.array([[d["dist_to_swarm_centroid_m"] for d in r["drones"]] for r in L])
t = np.array([r["time_s"] for r in L]); bk = {b["id"]: b for b in meta["breakaways"]}; C = np.array([r["swarm_centroid_m"] for r in L])
fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))
for j, i in enumerate(ids):
    b = i in bk; kw = dict(color="#d83b3b", lw=2, zorder=3) if b else dict(color="#3b7dd8", lw=0.5, alpha=0.35)
    ax[0].plot(P[:, j, 0], P[:, j, 1], **kw); ax[1].plot(t, P[:, j, 2] - P[0, j, 2], **kw); ax[2].plot(t, Dc[:, j], **kw)
for i, b in bk.items():
    for a in ax[1:]: a.axvline(b["start_time_s"], color="#d83b3b", ls="--", lw=0.8)
    ax[2].text(b["start_time_s"], ax[2].get_ylim()[1] * 0.95, f" id{i} {b['type']} f{b['start_frame']}", color="#d83b3b", fontsize=8, va="top")
ax[0].plot(C[:, 0], C[:, 1], "k-", lw=1.5, label="swarm centroid"); ax[0].set_aspect("equal"); ax[0].set_title("Top-down paths (red = breakaway)"); ax[0].set_xlabel("x m"); ax[0].set_ylabel("y m"); ax[0].legend()
ax[1].set_title("Height above launch point (ground -> takeoff -> cruise)"); ax[1].set_xlabel("s"); ax[1].set_ylabel("m")
ax[2].set_title("Distance to swarm centroid"); ax[2].set_xlabel("s"); ax[2].set_ylabel("m")
fig.suptitle(f"tracking_v1 seq {meta['seq']:03d}: N={meta['n_drones']}, {meta['launch']} launch, {meta['formation']}, breakaways: " + ", ".join(f"id{i} {b['type']}@f{b['start_frame']}" for i, b in bk.items()))
fig.tight_layout(); fig.savefig(out, dpi=110); print("saved", out)
