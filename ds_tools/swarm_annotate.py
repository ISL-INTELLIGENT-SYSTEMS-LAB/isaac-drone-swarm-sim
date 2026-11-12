"""Swarm-level annotations for a rendered tracking sequence (no Isaac Sim needed: numpy + Pillow).

    python ds_tools/swarm_annotate.py <seq_dir | dataset_dir> [--brk-k 1.5 --brk-min-m 3 --gap-k 3 ...]

Reads  seq/meta.json, seq/ann/frames.jsonl, seq/masks/*.png   (written by gen_tracking.py)
Writes seq/swarm/swarm_annotations.json   per-sequence swarm JSON (entities, events, per-frame boxes, 2D-cluster tracks)
       seq/swarm/labels_yolo/000001.txt   YOLO boxes per frame: class 0 = swarm, class 1 = breakaway  (cx cy w h, normalised)
       seq/swarm/coco_swarm.json          COCO detection json (category 1 = swarm, 2 = breakaway, + track_id per box)
       seq/swarm/swarm_tracks.csv         frame,id,class,x1,y1,x2,y2,cx,cy,n_visible,n_total  (MOT-style, one row per box)
       seq/swarm_masks/000001.png         binary swarm mask (255 = any drone pixel) = union of the drone masks

Two kinds of swarm labels (both in the JSON):
 1. Ground truth from the scenario ("entities"; this is what the YOLO / COCO files contain).
    Each drone has a swarm_id from the generator. A scripted breakaway group (meta.breakaways[].members) counts as part of its
    parent swarm until it is DETACHED: the distance (3D, metres) between the group centroid and the parent swarm's core centroid
    exceeds max(brk_min_m, brk_k * R90) AND has grown by more than brk_min_m since the scripted start (so a drone that already
    sits at the edge is not counted at once); R90 = 90th-percentile distance of core drones to their centroid. From that frame on it is
    its own entity (id 100 + group number, class breakaway). Event log: scripted start frame + detached frame.
    Box = union of the visible member drones' tight mask boxes (= tight box of the union mask); centroid_px = mean of the
    member mask pixels; centroid_m = mean 3D position of all members (visible or not).
 2. "clusters_2d": what a detector-side post-process would see. Visible drone box centres are clustered per frame (single
    linkage, gap = max(gap_k * median nearest-neighbour distance, size_k * median drone box size, 8 px)) and tracked frame to frame
    without drone IDs: every point votes for the cluster of its nearest point in the previous frame (shifted by the track motion).
    One track -> several clusters = split (largest keeps the id); a child with <= brk_frac of the parent's drones = breakaway.
    Several tracks -> one cluster = merge. Splits that merge back within `confirm` frames are dropped as flicker.
    Drone IDs are only used afterwards to score these events against the ground truth ("eval").
Boxes are inclusive pixel coords [x1, y1, x2, y2] in the 1280x720 image unless noted.
"""
import os, sys, json, glob, argparse
import numpy as np
from PIL import Image

DEF = dict(brk_k=1.5, brk_min_m=3.0, gap_k=3.0, size_k=2.0, gap_min_px=8.0, brk_frac=0.25, confirm=3, max_age=5)

def _box(b):
    return [int(b[0]), int(b[1]), int(b[2]), int(b[3])]

def _union(bs):
    bs = np.asarray(bs); return [int(bs[:, 0].min()), int(bs[:, 1].min()), int(bs[:, 2].max()), int(bs[:, 3].max())]

def cluster_points(P, sizes, gap_k, size_k, gap_min):
    """Single-linkage clustering of 2D points P (n,2). Returns labels (n,) and the gap used."""
    n = len(P)
    if n == 0: return np.zeros(0, int), 0.0
    if n == 1: return np.zeros(1, int), float(gap_min)
    lab = np.arange(n)
    def find(i):
        while lab[i] != i: lab[i] = lab[lab[i]]; i = lab[i]
        return i
    # nearest-neighbour distances (chunked)
    nn = np.empty(n)
    for a in range(0, n, 2048):
        d = np.sqrt(((P[a:a + 2048, None] - P[None]) ** 2).sum(2)); d[np.arange(d.shape[0]), np.arange(a, a + d.shape[0])] = np.inf; nn[a:a + 2048] = d.min(1)
    gap = float(max(gap_k * np.median(nn), size_k * np.median(sizes), gap_min))
    # grid hashing so this also works for many thousands of points
    cell = gap; key = np.floor(P / cell).astype(np.int64); grid = {}
    for i, k in enumerate(map(tuple, key)): grid.setdefault(k, []).append(i)
    for (gx, gy), idx in grid.items():
        cand = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1): cand += grid.get((gx + dx, gy + dy), [])
        cand = np.array(cand); idx = np.array(idx)
        d = np.sqrt(((P[idx, None] - P[None, cand]) ** 2).sum(2))
        for ii, jj in zip(*np.nonzero(d <= gap)):
            a, b = find(int(idx[ii])), find(int(cand[jj]))
            if a != b: lab[max(a, b)] = min(a, b)
    roots = np.array([find(i) for i in range(n)]); _, out = np.unique(roots, return_inverse=True)
    return out, gap

class ClusterTracker:
    """ID-free swarm tracker: nearest-point voting between consecutive frames."""
    def __init__(self, p):
        self.p = p; self.next_id = 1; self.tracks = {}; self.prev = None; self.events = []; self.frames = []
    def _new(self, f, cls="swarm", parent=None):
        i = self.next_id; self.next_id += 1
        self.tracks[i] = dict(id=i, cls=cls, parent=parent, born=f, last=f, vel=np.zeros(2), cen=None, size=0, ended=None, end_reason=None); return i
    def step(self, f, P, sizes, tids):
        p = self.p; lab, gap = cluster_points(P, sizes, p["gap_k"], p["size_k"], p["gap_min_px"])
        K = int(lab.max()) + 1 if len(lab) else 0
        votes = np.zeros((K, 0)); tlist = []
        if self.prev is not None and len(P) and len(self.prev["P"]):
            tlist = sorted(set(self.prev["tid"].tolist())); col = {t: j for j, t in enumerate(tlist)}
            Pp = self.prev["P"] + np.array([self.tracks[t]["vel"] for t in self.prev["tid"]])
            thr = 1.5 * max(gap, self.prev["gap"])
            votes = np.zeros((K, len(tlist)))
            for a in range(0, len(P), 2048):
                d = np.sqrt(((P[a:a + 2048, None] - Pp[None]) ** 2).sum(2)); j = d.argmin(1); ok = d[np.arange(len(j)), j] <= thr
                for i in np.nonzero(ok)[0]: votes[lab[a + i], col[int(self.prev["tid"][j[i]])]] += 1
        prev_size = {t: int((self.prev["tid"] == t).sum()) for t in tlist} if tlist else {}
        csize = np.bincount(lab, minlength=K) if K else np.zeros(0, int)
        prim = [tlist[int(votes[c].argmax())] if votes.shape[1] and votes[c].sum() > 0 else None for c in range(K)]
        cid = [None] * K
        # merges: a cluster that took the majority of two or more previous tracks
        for c in range(K):
            if not votes.shape[1]: break
            got = [tlist[j] for j in np.nonzero(votes[c] >= 0.5 * np.array([prev_size[t] for t in tlist]))[0]]
            if prim[c] is not None and prim[c] not in got: got.append(prim[c])
            if len(got) >= 2:
                keep = max(got, key=lambda t: prev_size[t]); prim[c] = keep
                for t in got:
                    if t != keep and self.tracks[t]["ended"] is None:
                        self.tracks[t]["ended"] = f; self.tracks[t]["end_reason"] = f"merged_into_{keep}"
                self.events.append(dict(type="merge", frame=f, into=keep, merged=[t for t in got if t != keep]))
        # splits: one previous track is the main parent of several clusters
        for t in set(x for x in prim if x is not None):
            if self.tracks[t]["ended"] is not None: continue   # merged away this frame: its other clusters start new tracks
            cs = [c for c in range(K) if prim[c] == t]
            cs.sort(key=lambda c: -csize[c]); cid[cs[0]] = t
            for c in cs[1:]:
                small = csize[c] <= self.p["brk_frac"] * prev_size[t]
                n = self._new(f, "breakaway" if small else "swarm", parent=t); cid[c] = n
                self.events.append(dict(type="breakaway" if small else "split", frame=f, parent=t, child=n, child_size=int(csize[c]), parent_size_before=prev_size[t]))
        for c in range(K):
            if cid[c] is None: cid[c] = self._new(f); self.events.append(dict(type="appear", frame=f, id=cid[c], size=int(csize[c])))
        out = []; tid_pt = np.zeros(len(P), int)
        for c in range(K):
            m = lab == c; t = self.tracks[cid[c]]; tid_pt[m] = cid[c]
            cen = P[m].mean(0)
            if t["cen"] is not None and t["last"] == f - 1: t["vel"] = 0.5 * t["vel"] + 0.5 * (cen - t["cen"])
            t["cen"] = cen; t["last"] = f; t["size"] = int(m.sum())
            bx = np.array(self._boxes)[m]
            out.append(dict(id=cid[c], cls=t["cls"], box=_union(bx), centroid_px=[round(float(cen[0]), 1), round(float(cen[1]), 1)], n=int(m.sum()), members=sorted(int(x) for x in tids[m])))
        for t in self.tracks.values():
            if t["ended"] is None and f - t["last"] > self.p["max_age"]: t["ended"] = t["last"] + 1; t["end_reason"] = "lost"
        self.prev = dict(P=P, tid=tid_pt, gap=gap); self.frames.append(dict(frame=f, gap_px=round(gap, 1), clusters=out))
    def finish(self):
        """Drop flicker: a split/breakaway child that merges back into its parent within `confirm` frames."""
        drop = set()
        for e in self.events:
            if e["type"] in ("split", "breakaway"):
                t = self.tracks[e["child"]]
                if t["end_reason"] == f"merged_into_{e['parent']}" and t["ended"] - t["born"] < self.p["confirm"]: drop.add(e["child"])
        ev = []
        for e in self.events:
            if e["type"] in ("split", "breakaway") and e["child"] in drop: continue
            if e["type"] == "merge" and set(e["merged"]) <= drop: continue
            ev.append(e)
        return ev, drop

def annotate(sd, **kw):
    p = dict(DEF); p.update({k: v for k, v in kw.items() if v is not None})
    meta = json.load(open(os.path.join(sd, "meta.json")))
    recs = [json.loads(l) for l in open(os.path.join(sd, "ann", "frames.jsonl"))]
    W, H = 1280, 720
    m0 = os.path.join(sd, "masks", f"{recs[0]['frame']:06d}.png")
    if os.path.exists(m0): H, W = np.asarray(Image.open(m0)).shape[:2]
    N = recs[0]["n_total"]
    sid = np.ones(N + 1, int)
    for d in recs[0]["drones"]: sid[d["track_id"]] = d.get("swarm_id", 1)
    swarms = sorted(set(sid[1:].tolist()))
    groups = []
    for g, b in enumerate(meta.get("breakaways", []), 1):
        mem = b.get("members", [b["id"]])
        groups.append(dict(id=100 + g, group=g, members=mem, parent=int(sid[mem[0]]), kind=b.get("type"), scripted_start_frame=int(b["start_frame"]), detached_frame=None))
    in_group = {m: gr for gr in groups for m in gr["members"]}
    core = {s: np.array([k for k in range(1, N + 1) if sid[k] == s and k not in in_group]) for s in swarms}
    # --- ground-truth detachment (3D)
    for rec in recs:
        pos = np.zeros((N + 1, 3))
        for d in rec["drones"]: pos[d["track_id"]] = d["pos_m"]
        rec["_pos"] = pos
        for gr in groups:
            if gr["detached_frame"] is not None or rec["frame"] < gr["scripted_start_frame"]: continue
            c = core[gr["parent"]]
            if not len(c): gr["detached_frame"] = rec["frame"]; continue
            cen = pos[c].mean(0); R = float(np.percentile(np.linalg.norm(pos[c] - cen, axis=1), 90))
            dist = float(np.linalg.norm(pos[gr["members"]].mean(0) - cen)); gr.setdefault("dist0", dist)
            if dist > max(p["brk_min_m"], p["brk_k"] * R) and dist - gr["dist0"] > p["brk_min_m"]: gr["detached_frame"] = rec["frame"]; gr["detach_dist_m"] = round(dist, 2); gr["detach_R90_m"] = round(R, 2)
    ents = [dict(id=s, cls="swarm", parent=None, members=[int(k) for k in range(1, N + 1) if sid[k] == s]) for s in swarms] + \
           [dict(id=g["id"], cls="breakaway", parent=g["parent"], members=g["members"], kind=g["kind"], scripted_start_frame=g["scripted_start_frame"], detached_frame=g["detached_frame"],
                 **({"detach_dist_m": g["detach_dist_m"], "detach_R90_m": g["detach_R90_m"]} if "detach_dist_m" in g else {})) for g in groups]
    events = []
    for g in groups:
        events.append(dict(type="breakaway_scripted_start", frame=g["scripted_start_frame"], id=g["id"], parent=g["parent"], members=g["members"]))
        if g["detached_frame"] is not None: events.append(dict(type="breakaway", frame=g["detached_frame"], id=g["id"], parent=g["parent"], members=g["members"]))
    events.sort(key=lambda e: e["frame"])
    od = os.path.join(sd, "swarm"); yd = os.path.join(od, "labels_yolo"); md = os.path.join(sd, "swarm_masks")
    for d in (yd, md): os.makedirs(d, exist_ok=True)
    tr = ClusterTracker(p); frames = []; coco_im = []; coco_an = []; csv = ["frame,id,class,x1,y1,x2,y2,cx,cy,n_visible,n_total"]; aid = 1
    for rec in recs:
        f = rec["frame"]; dr = {d["track_id"]: d for d in rec["drones"]}
        mk = os.path.join(sd, rec.get("mask", f"masks/{f:06d}.png"))
        mask = np.asarray(Image.open(mk)).astype(np.int64) if os.path.exists(mk) else None
        if mask is not None: Image.fromarray(((mask > 0) * 255).astype(np.uint8)).save(os.path.join(md, f"{f:06d}.png"), optimize=True)
        owner = np.zeros(N + 1, int)   # entity id per track_id at this frame
        for k in range(1, N + 1):
            g = in_group.get(k); owner[k] = g["id"] if g and g["detached_frame"] is not None and f >= g["detached_frame"] else sid[k]
        fe = []; ylines = []
        if mask is not None:
            ys, xs = np.nonzero(mask); ow = owner[mask[ys, xs]]
        for e in ents:
            mem = [k for k in e["members"] if owner[k] == e["id"]]
            if not mem: continue
            vis = [k for k in mem if dr[k]["mask_px"] > 0 and dr[k]["bbox_tight"] is not None]
            cm = rec["_pos"][mem].mean(0)
            o = dict(id=e["id"], cls=e["cls"], n_visible=len(vis), n_total=len(mem), centroid_m=[round(float(x), 3) for x in cm], box=None, centroid_px=None, mask_px=0)
            if vis:
                o["box"] = _union([dr[k]["bbox_tight"] for k in vis])
                if mask is not None:
                    sel = ow == e["id"]; o["mask_px"] = int(sel.sum())
                    if sel.any(): o["centroid_px"] = [round(float(xs[sel].mean()), 1), round(float(ys[sel].mean()), 1)]
                x1, y1, x2, y2 = o["box"]; c = 0 if e["cls"] == "swarm" else 1
                ylines.append(f"{c} {(x1 + x2 + 1) / 2 / W:.6f} {(y1 + y2 + 1) / 2 / H:.6f} {(x2 - x1 + 1) / W:.6f} {(y2 - y1 + 1) / H:.6f}")
                coco_an.append(dict(id=aid, image_id=f, category_id=c + 1, bbox=[x1, y1, x2 - x1 + 1, y2 - y1 + 1], area=(x2 - x1 + 1) * (y2 - y1 + 1), iscrowd=0,
                                    track_id=e["id"], n_visible=len(vis), n_total=len(mem), mask_px=o["mask_px"])); aid += 1
                cx, cy = o["centroid_px"] or [(x1 + x2) / 2, (y1 + y2) / 2]
                csv.append(f"{f},{e['id']},{c},{x1},{y1},{x2},{y2},{cx},{cy},{len(vis)},{len(mem)}")
            fe.append(o)
        open(os.path.join(yd, f"{f:06d}.txt"), "w").write("\n".join(ylines) + ("\n" if ylines else ""))
        coco_im.append(dict(id=f, file_name=rec["image"], width=W, height=H, frame=f, seq=meta.get("seq")))
        frames.append(dict(frame=f, image=rec["image"], swarm_mask=f"swarm_masks/{f:06d}.png", n_visible_drones=rec["n_visible_ge1px"], entities=fe))
        # 2D clusters from visible drone boxes (what a detector would give)
        vis = [k for k in range(1, N + 1) if dr[k]["mask_px"] > 0 and dr[k]["bbox_tight"] is not None]
        bx = np.array([dr[k]["bbox_tight"] for k in vis], float).reshape(-1, 4)
        P = np.stack([(bx[:, 0] + bx[:, 2]) / 2, (bx[:, 1] + bx[:, 3]) / 2], 1) if len(bx) else np.zeros((0, 2))
        sz = np.maximum(bx[:, 2] - bx[:, 0] + 1, bx[:, 3] - bx[:, 1] + 1) if len(bx) else np.zeros(0)
        tr._boxes = bx.astype(int).tolist(); tr.step(f, P, sz, np.array(vis, int))
    cev, drop = tr.finish()
    # score 2D-cluster breakaway events against ground truth
    ev_eval = []
    for g in groups:
        hit = None
        for e in cev:
            if e["type"] not in ("breakaway", "split") or e["frame"] < g["scripted_start_frame"]: continue
            fr = next(x for x in tr.frames if x["frame"] == e["frame"]); ch = next((c for c in fr["clusters"] if c["id"] == e["child"]), None)
            if ch and set(ch["members"]) & set(g["members"]) and len(set(ch["members"]) - set(g["members"])) <= max(1, len(ch["members"]) // 4): hit = e; break
        ev_eval.append(dict(gt_id=g["id"], members=g["members"], scripted_start_frame=g["scripted_start_frame"], gt_detached_frame=g["detached_frame"],
                            cluster_event=hit["type"] if hit else None, cluster_event_frame=hit["frame"] if hit else None,
                            delay_vs_detached_frames=(hit["frame"] - g["detached_frame"]) if hit and g["detached_frame"] is not None else None))
    out = dict(format="drone-swarm-annotations v1", seq=meta.get("seq"), image_size=[W, H], fps=meta.get("fps"), n_frames=len(recs), n_drones=N, params=p,
               box_format="inclusive pixel coords [x1, y1, x2, y2]", classes={"0": "swarm", "1": "breakaway"},
               entities=ents, events=events, frames=frames,
               clusters_2d=dict(note="detector-side post-process on visible drone boxes, no drone IDs used for tracking; members listed only for evaluation",
                                events=cev, flicker_tracks_dropped=sorted(drop), frames=tr.frames,
                                tracks=[dict(id=t["id"], cls=t["cls"], parent=t["parent"], born=t["born"], last=t["last"], end_reason=t["end_reason"]) for t in tr.tracks.values() if t["id"] not in drop]),
               eval=dict(breakaways=ev_eval))
    json.dump(out, open(os.path.join(od, "swarm_annotations.json"), "w"), separators=(",", ":"))
    json.dump(dict(info=dict(description=f"swarm boxes seq {meta.get('seq')}", box="xywh, top-left, pixels"), images=coco_im, annotations=coco_an,
                   categories=[dict(id=1, name="swarm"), dict(id=2, name="breakaway")]), open(os.path.join(od, "coco_swarm.json"), "w"))
    open(os.path.join(od, "swarm_tracks.csv"), "w").write("\n".join(csv) + "\n")
    cnt = {}
    for e in cev: cnt[e["type"]] = cnt.get(e["type"], 0) + 1
    return dict(frames=len(recs), swarms=len(swarms), gt_breakaways=[(g["id"], g["scripted_start_frame"], g["detached_frame"]) for g in groups],
                cluster_events=cnt, eval=[(x["gt_id"], x["cluster_event"], x["cluster_event_frame"], x["delay_vs_detached_frames"]) for x in ev_eval])

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="sequence dir (has meta.json) or a dataset dir (all seq_* below it)")
    for k, v in DEF.items(): ap.add_argument("--" + k.replace("_", "-"), type=type(v), default=None, help=f"default {v}")
    a = ap.parse_args(); kw = {k: getattr(a, k) for k in DEF}
    seqs = [a.path] if os.path.exists(os.path.join(a.path, "meta.json")) else sorted(os.path.dirname(m) for m in glob.glob(os.path.join(a.path, "**", "meta.json"), recursive=True))
    for sd in seqs: print(sd, annotate(sd, **kw), flush=True)
