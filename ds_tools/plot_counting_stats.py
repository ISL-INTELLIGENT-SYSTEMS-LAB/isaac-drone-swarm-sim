"""Stats figures for counting_v1 (run after export_counting.py). python plot_counting_stats.py [dataset_dir]"""
import os, sys, json, numpy as np, collections
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets", "counting_v1")
F = json.load(open(os.path.join(D, "stats_frames.json"))); I = np.load(os.path.join(D, "stats_instances.npz"))
out = os.path.join(D, "stats"); os.makedirs(out, exist_ok=True)
def b4(n): return "1-10" if n <= 10 else "11-50" if n <= 50 else "51-100" if n <= 100 else "101+"
B4 = ["1-10", "11-50", "51-100", "101+"]
n = np.array([f["n_total"] for f in F]); v1 = np.array([f["n_visible_ge1px"] for f in F]); v10 = np.array([f["n_visible_ge10px"] for f in F])
fig, ax = plt.subplots(2, 3, figsize=(17, 9))
a = ax[0, 0]; bins = np.logspace(0, np.log10(n.max() + 1), 40); a.hist(n, bins, color="#3b7dd8", label="true N (placed)"); a.hist(v1, bins, histtype="step", color="k", label="visible >=1 px"); a.hist(v10, bins, histtype="step", color="#d83b3b", label="visible >=10 px")
a.set_xscale("log"); a.set_title("Drones per frame"); a.set_xlabel("count"); a.set_ylabel("frames"); a.legend()
a = ax[0, 1]; sp = I["span_px"]; sp = sp[sp > 0]; a.hist(sp, np.logspace(np.log10(max(sp.min(), 0.3)), np.log10(sp.max()), 50), color="#3bb36b"); a.set_xscale("log"); a.set_title("Drone size in image (projected span, px), visible drones"); a.set_xlabel("px")
a = ax[0, 2]; mp = I["mask_px"]; a.hist(mp, np.logspace(0, np.log10(mp.max()), 50), color="#8a5cd8"); a.set_xscale("log"); a.set_yscale("log"); a.set_title("Visible mask area per drone (px)"); a.set_xlabel("px")
a = ax[1, 0]; oc = I["occl"]; oc = oc[oc >= 0]; a.hist(oc, 40, color="#d8963b"); a.set_title(f"Occlusion ratio, visible drones (n={len(oc)}; {int((I['occl']<0).sum())} without value)"); a.set_xlabel("0 = fully visible, 1 = fully hidden")
a = ax[1, 1]; rat = v1 / np.maximum(n, 1); a.scatter(n, rat, s=3, alpha=0.4); a.set_xscale("log"); a.set_title("Fraction of placed drones with >=1 visible px"); a.set_xlabel("true N"); a.set_ylabel("visible / true")
a = ax[1, 2]; a.axis("off")
rows = []; tot = collections.Counter(); per = collections.defaultdict(collections.Counter)
for f in F: k = b4(f["n_total"]); per[k][f["split"]] += 1
for k in B4:
    m = np.array([b4(x) == k for x in n]); c = per[k]
    rows.append([k, int(m.sum()), c["train"], c["val"], c["test"], f"{np.median(v1[m]/np.maximum(n[m],1)):.2f}" if m.any() else "-"])
rows.append(["total", len(n), sum(per[k]["train"] for k in B4), sum(per[k]["val"] for k in B4), sum(per[k]["test"] for k in B4), f"{np.median(rat):.2f}"])
t = a.table(cellText=rows, colLabels=["bucket (true N)", "frames", "train", "val", "test", "median vis/true"], loc="center"); t.scale(1, 2); t.auto_set_font_size(False); t.set_fontsize(11)
a.set_title(f"Swarm-size buckets  (max N = {n.max()}, {len(sp)} visible drone instances)")
fig.suptitle("counting_v1 - synthetic drone counting dataset (Isaac Sim 6.2, Squad DesertStage)", fontsize=14); fig.tight_layout()
fig.savefig(os.path.join(out, "counting_v1_stats.png"), dpi=110)
json.dump({"bucket_table": rows, "max_n": int(n.max()), "frames": len(n), "visible_instances": int(len(sp)),
           "span_px_pct": {p: float(np.percentile(sp, p)) for p in (5, 25, 50, 75, 95)}, "occl_mean": float(oc.mean()) if len(oc) else None}, open(os.path.join(out, "summary.json"), "w"), indent=1)
print(rows)
