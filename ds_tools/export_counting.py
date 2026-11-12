"""Export counting_v1 -> COCO (instance masks as compressed RLE), YOLO boxes, stratified splits, stats + per-bucket examples.
Plain python (numpy + PIL). Usage: python export_counting.py [dataset_dir] [yolo_min_px]"""
import os, sys, json, glob, random, math
import numpy as np
from PIL import Image, ImageDraw
D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets", "counting_v1")
YOLO_MIN = int(sys.argv[2]) if len(sys.argv) > 2 else 1
import cv2
BUCKETS = [(1, 10), (11, 50), (51, 100), (101, 300), (301, 10 ** 6)]
def bucket(n):
    for a, b in BUCKETS:
        if a <= n <= b: return f"{a}-{b}" if b < 10 ** 6 else f"{a}+"
def rle_string(cnts):
    out = []
    for i in range(len(cnts)):
        x = int(cnts[i])
        if i > 2: x -= int(cnts[i - 2])
        more = 1
        while more:
            c = x & 0x1f; x >>= 5
            more = (x != -1) if (c & 0x10) else (x != 0)
            if more: c |= 0x20
            out.append(chr(c + 48))
    return "".join(out)
def rles_for_mask(mask, N):
    H, W = mask.shape
    ys, xs = np.nonzero(mask); lab = mask[ys, xs].astype(np.int64) - 1
    f = xs.astype(np.int64) * H + ys
    o = np.lexsort((f, lab)); lab, f = lab[o], f[o]
    res = {}
    if not len(f): return res
    starts = np.r_[0, np.nonzero(np.diff(lab))[0] + 1, len(lab)]
    for a, b in zip(starts[:-1], starts[1:]):
        p = f[a:b]; br = np.nonzero(np.diff(p) != 1)[0]
        rs = np.r_[p[0], p[br + 1]]; re_ = np.r_[p[br], p[-1]] + 1
        cnts = []; prev = 0
        for s, e in zip(rs, re_): cnts += [int(s - prev), int(e - s)]; prev = e
        cnts.append(int(H * W - prev))
        res[int(lab[a])] = {"size": [H, W], "counts": rle_string(cnts)}
    return res
def poly_for(mask, k, box):
    """YOLO-seg polygon (continuous pixel coords) for drone k (mask value k+1).
    Every connected part becomes a polygon: its outer contour (shifted to pixel centres +0.5), or, for parts too thin
    for a contour (1-2 px, lines), the pixel-edge rectangle of that part. Several parts are merged into ONE polygon with
    zero-width bridges between nearest points (same idea as Ultralytics merge_multi_segment). Returns (poly, source, n_parts)."""
    x1, y1, x2, y2 = box
    m = (mask[y1:y2 + 1, x1:x2 + 1] == k + 1).astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs: return np.array([[x1, y1], [x2 + 1, y1], [x2 + 1, y2 + 1], [x1, y2 + 1]], float), "bbox_fallback", 0
    parts = []; rect_used = False
    for c in sorted(cs, key=cv2.contourArea, reverse=True):
        c = c.reshape(-1, 2)
        if len(c) >= 3 and cv2.contourArea(c.reshape(-1, 1, 2)) >= 1.0:
            ap = cv2.approxPolyDP(c.reshape(-1, 1, 2), 0.5, True).reshape(-1, 2)
            parts.append((ap if len(ap) >= 3 else c).astype(float) + 0.5)
        else:
            u0, v0 = c.min(0); u1, v1 = c.max(0); rect_used = True
            parts.append(np.array([[u0, v0], [u1 + 1, v0], [u1 + 1, v1 + 1], [u0, v1 + 1]], float))
    poly = parts[0]
    for q in parts[1:]:
        d = ((poly[:, None, :] - q[None, :, :]) ** 2).sum(-1); i, j = np.unravel_index(np.argmin(d), d.shape)
        qr = np.roll(q, -j, 0)
        poly = np.concatenate([poly[:i + 1], qr, qr[:1], poly[i:]], 0)
    poly = poly + [x1, y1]
    return poly, ("contour" if not rect_used else ("pixel_rect" if len(parts) == 1 else "contour+pixel_rect")), len(parts)
anns = sorted(glob.glob(os.path.join(D, "ann", "*.json")))
recs = [json.load(open(p)) for p in anns]
print("frames", len(recs))
# stratified split by true swarm size bucket
rng = random.Random(1234); split = {}
for bk in sorted({bucket(r["n_total"]) for r in recs}):
    fr = [r["frame"] for r in recs if bucket(r["n_total"]) == bk]; rng.shuffle(fr)
    nt = max(1, round(0.12 * len(fr))); nv = max(1, round(0.08 * len(fr)))
    for j, f in enumerate(fr): split[f] = "test" if j < nt else ("val" if j < nt + nv else "train")
os.makedirs(os.path.join(D, "coco"), exist_ok=True); os.makedirs(os.path.join(D, "labels"), exist_ok=True); os.makedirs(os.path.join(D, "yolo"), exist_ok=True)
coco = {s: {"info": {"description": "ISL synthetic drone counting v1 (Isaac Sim, DesertStage)", "version": "1.0"}, "licenses": [{"id": 1, "name": "Research only - see README (Squad assets CC BY-NC-ND 4.0; Isaac assets NVIDIA terms)"}],
            "categories": [{"id": 1, "name": "drone", "supercategory": "uav"}], "images": [], "annotations": []} for s in ("train", "val", "test")}
lists = {s: [] for s in coco}; aid = 1
ystat = {"contour": 0, "pixel_rect": 0, "contour+pixel_rect": 0, "bbox_fallback": 0, "multipart": 0, "tiny_le2px": 0}; yflags = {}
stats = {"frames": [], "span_px": [], "mask_px": [], "occl": [], "vis_frac": [], "type": [], "ann_vs_mask_mismatch": 0}
for r in recs:
    s = split[r["frame"]]; iid = r["frame"] + 1
    mask = np.asarray(Image.open(os.path.join(D, r["mask"])))
    rl = rles_for_mask(mask, r["n_total"])
    coco[s]["images"].append({"id": iid, "file_name": r["image"], "width": r["width"], "height": r["height"], "n_total": r["n_total"], "n_in_frustum": r["n_in_frustum"],
                              "n_visible_ge1px": r["n_visible_ge1px"], "n_visible_ge10px": r["n_visible_ge10px"], "n_visible_ge50px": r["n_visible_ge50px"], "bucket": bucket(r["n_total"]),
                              "cam_mode": r["cam_mode"], "formation": r["formation"], "license": 1})
    ylines = []
    for d in r["drones"]:
        if d["mask_px"] > 0 and d["bbox_tight"]:
            x1, y1, x2, y2 = d["bbox_tight"]; w, h = x2 - x1 + 1, y2 - y1 + 1
            coco[s]["annotations"].append({"id": aid, "image_id": iid, "category_id": 1, "bbox": [x1, y1, w, h], "area": d["mask_px"], "iscrowd": 0, "segmentation": rl[d["k"]],
                                           "drone_type": d["type"], "drone_k": d["k"], "span_px": d["span_px"], "depth_m": d["depth_m"], "occlusion_ratio": d["occlusion_ratio"],
                                           "visible_fraction": d["visible_fraction"], "bbox_amodal_proj": d["bbox_amodal_proj"]}); aid += 1
            if d["mask_px"] >= YOLO_MIN:
                pg, srcp, nparts = poly_for(mask, d["k"], d["bbox_tight"])
                pg[:, 0] = np.clip(pg[:, 0] / r["width"], 0, 1); pg[:, 1] = np.clip(pg[:, 1] / r["height"], 0, 1)
                ylines.append("0 " + " ".join(f"{v:.6f}" for v in pg.ravel()))
                ystat[srcp] += 1; ystat["multipart"] += int(nparts > 1); ystat["tiny_le2px"] += int(d["mask_px"] <= 2)
                if srcp != "contour" or nparts > 1: yflags.setdefault(r["frame"], []).append({"line": len(ylines) - 1, "drone_k": d["k"], "mask_px": d["mask_px"], "poly": srcp, "parts": nparts})
            stats["span_px"].append(d["span_px"]); stats["mask_px"].append(d["mask_px"]); stats["occl"].append(d["occlusion_ratio"] if d["occlusion_ratio"] is not None else -1)
            stats["vis_frac"].append(d["visible_fraction"] if d["visible_fraction"] is not None else -1); stats["type"].append(d["type"])
    with open(os.path.join(D, "labels", f"{r['frame']:06d}.txt"), "w") as f: f.write("\n".join(ylines) + ("\n" if ylines else ""))
    lists[s].append(os.path.join(D, r["image"].replace("/", os.sep)))  # absolute: Ultralytics resolves "./" relative to the txt folder
    stats["frames"].append({k: r[k] for k in ("frame", "n_total", "n_in_frustum", "n_visible_ge1px", "n_visible_ge10px", "n_visible_ge50px", "cam_mode", "formation", "near_ground", "alt_agl_m", "ann_bbox_tight_count")} | {"split": s, "bucket": bucket(r["n_total"]), "eye_agl_m": r["camera"]["eye_agl_m"], "focal_mm": r["camera"]["focal_mm"]})
for s in coco:
    json.dump(coco[s], open(os.path.join(D, "coco", f"instances_{s}.json"), "w"), separators=(",", ":"))
    open(os.path.join(D, "yolo", f"{s}.txt"), "w").write("\n".join(lists[s]) + "\n")
    print(s, len(coco[s]["images"]), "images", len(coco[s]["annotations"]), "instances")
json.dump({"stats": ystat, "flagged": {str(k): v for k, v in yflags.items()}}, open(os.path.join(D, "yolo", "seg_polygon_flags.json"), "w")); print("yolo-seg", ystat)
open(os.path.join(D, "yolo", "data.yaml"), "w").write(f"# Ultralytics-style (YOLO-seg polygons; also usable by detect models); labels/ holds one txt per image (drones with >= {YOLO_MIN} visible mask px)\npath: {D}\ntrain: yolo/train.txt\nval: yolo/val.txt\ntest: yolo/test.txt\nnames:\n  0: drone\n")
json.dump({"yolo_min_px": YOLO_MIN, "split": {str(k): v for k, v in split.items()}, "buckets": [bucket(a) for a, b in BUCKETS]}, open(os.path.join(D, "split.json"), "w"))
np.savez_compressed(os.path.join(D, "stats_instances.npz"), span_px=np.array(stats["span_px"], np.float32), mask_px=np.array(stats["mask_px"], np.int32),
                    occl=np.array(stats["occl"], np.float32), vis_frac=np.array(stats["vis_frac"], np.float32), type=np.array(stats["type"]))
json.dump(stats["frames"], open(os.path.join(D, "stats_frames.json"), "w"))
# examples with boxes: 3 per bucket (test split preferred)
ex = os.path.join(D, "examples"); os.makedirs(ex, exist_ok=True)
for bk in sorted({bucket(r["n_total"]) for r in recs}, key=lambda b: int(b.split("-")[0].rstrip("+"))):
    cand = [r for r in recs if bucket(r["n_total"]) == bk]; rng.shuffle(cand)
    cand.sort(key=lambda r: (r["cam_mode"] != "aerial", -r["n_visible_ge1px"] / max(1, r["n_total"])))
    pick = cand[:2] + [r for r in cand if r["cam_mode"] == "ground_up"][:1]
    for r in pick:
        im = Image.open(os.path.join(D, r["image"])).convert("RGB"); dr = ImageDraw.Draw(im)
        for d in r["drones"]:
            if d["bbox_tight"] and d["mask_px"] > 0:
                x1, y1, x2, y2 = d["bbox_tight"]; col = (0, 255, 0) if d["mask_px"] >= 50 else ((255, 220, 0) if d["mask_px"] >= 10 else (255, 60, 60))
                dr.rectangle([x1 - 1, y1 - 1, x2 + 1, y2 + 1], outline=col, width=1)
        dr.rectangle([0, 0, 760, 22], fill=(0, 0, 0))
        dr.text((5, 5), f"frame {r['frame']} bucket {bk}  true N={r['n_total']} in-frustum={r['n_in_frustum']} visible>=1px={r['n_visible_ge1px']} >=10px={r['n_visible_ge10px']} >=50px={r['n_visible_ge50px']}  {r['cam_mode']} {r['formation']}", fill=(255, 255, 255))
        im.save(os.path.join(ex, f"bucket_{bk}_f{r['frame']:06d}.jpg"), quality=90)
print("done")
