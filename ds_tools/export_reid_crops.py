"""Export re-ID crops from TRACKING output (run_tracking.ps1) in a Market-1501-like layout. Plain python (numpy + Pillow).

  python ds_tools/export_reid_crops.py <tracking_root> [--out DIR] [--min-px 16] [--stride 1] [--pad 0.15] [--test-frac 0.5] [--sheet 8]

<tracking_root> is searched recursively for sequence folders (seq_*/meta.json), e.g. datasets\\reid_membera_1-20.
Writes to --out (default <tracking_root>\\reid_crops):
  crops/<gid>/<seqkey>_<frame>.jpg   gid = global identity (5 digits) = one drone (track_id) in one sequence
  ids.csv      gid, seq, track_id, type, model, visible_bin, n_crops, split
  crops.csv    one row per crop: path, gid, seq, track_id, frame, bbox, size, mask_px, visibility, split, role
               split: train/test by SEQUENCE (no identity is in both); role: train | query | gallery
               (test ids: 1 query crop per id, every other crop of that id is gallery)
  pairs.csv    verification pairs for test ids: query vs same id (label 1) and vs another drone of the SAME sequence
               (label 0, hard negatives; neg_kind same_type / diff_type)
  summary.json, crop_sheet.jpg (if --sheet > 0)
Crops whose tight mask box is smaller than --min-px on its SHORT side are skipped. Each sequence is one camera (camid = sequence)."""
import os, sys, json, csv, argparse, random
from PIL import Image, ImageDraw
ap = argparse.ArgumentParser()
ap.add_argument("root"); ap.add_argument("--out", default=None)
ap.add_argument("--min-px", type=int, default=16, help="skip crops whose tight bbox short side < this (default 16)")
ap.add_argument("--stride", type=int, default=1, help="use every Nth frame (default 1 = all frames)")
ap.add_argument("--pad", type=float, default=0.15, help="context padding, fraction of box size on each side")
ap.add_argument("--test-frac", type=float, default=0.5, help="fraction of sequences used as test (query/gallery)")
ap.add_argument("--pairs", type=int, default=5, help="positive and negative pairs per test id")
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--sheet", type=int, default=8, help="ids in crop_sheet.jpg (0 = none)")
a = ap.parse_args()
root = os.path.abspath(a.root); out = os.path.abspath(a.out or os.path.join(root, "reid_crops"))
seqs = sorted(os.path.dirname(os.path.join(dp, "meta.json")) for dp, dn, fn in os.walk(root) if "meta.json" in fn and os.path.exists(os.path.join(dp, "ann", "frames.jsonl")) and not os.path.abspath(dp).startswith(out))
if not seqs: sys.exit(f"no sequences (seq_*/meta.json) under {root}")
rnd = random.Random(a.seed); order = seqs[:]; rnd.shuffle(order)
n_test = max(1, int(round(len(seqs) * a.test_frac))) if len(seqs) > 1 else 0
test_seqs = set(order[:n_test])
os.makedirs(os.path.join(out, "crops"), exist_ok=True)
gid_of = {}; ids = []; rows = []; skipped = 0
for sd in seqs:
    meta = json.load(open(os.path.join(sd, "meta.json")))
    key = os.path.relpath(sd, root).replace("\\", "/").replace("/", "_") if sd != root else os.path.basename(sd)
    split = "test" if sd in test_seqs else "train"
    for fl in open(os.path.join(sd, "ann", "frames.jsonl")):
        r = json.loads(fl)
        if (r["frame"] - 1) % a.stride: continue
        im = None
        for d in r["drones"]:
            bb = d.get("bbox_tight")
            if not bb: continue
            x1, y1, x2, y2 = bb; w, h = x2 - x1 + 1, y2 - y1 + 1
            if min(w, h) < a.min_px: skipped += 1; continue
            tid = d.get("track_id", d["id"])
            g = gid_of.get((sd, tid))
            if g is None:
                g = len(gid_of) + 1; gid_of[(sd, tid)] = g
                ap = d.get("appearance") or {}
                ids.append({"gid": f"{g:05d}", "seq": key, "track_id": tid, "type": d["type"], "model": d.get("model", ""), "color_name": ap.get("color_name") or "",
                            "color_rgb": "#%02x%02x%02x" % tuple(int(255 * min(1, x)) for x in ap["rgb"]) if ap.get("rgb") else "", "roughness": ap.get("roughness", ""), "metallic": ap.get("metallic", ""), "visible_bin": meta.get("visible_bin"), "n_crops": 0, "split": split})
            if im is None: im = Image.open(os.path.join(sd, r["image"])).convert("RGB")
            px, py = int(round(w * a.pad)), int(round(h * a.pad))
            c = im.crop((max(0, x1 - px), max(0, y1 - py), min(im.width, x2 + 1 + px), min(im.height, y2 + 1 + py)))
            rel = f"crops/{g:05d}/{key}_{r['frame']:06d}.jpg"; os.makedirs(os.path.join(out, "crops", f"{g:05d}"), exist_ok=True)
            c.save(os.path.join(out, rel), quality=95); ids[g - 1]["n_crops"] += 1
            rows.append({"path": rel, "gid": f"{g:05d}", "seq": key, "track_id": tid, "frame": r["frame"], "type": d["type"], "x1": x1, "y1": y1, "x2": x2, "y2": y2, "w": w, "h": h,
                         "mask_px": d["mask_px"], "visible_fraction": d.get("visible_fraction"), "occlusion_ratio": d.get("occlusion_ratio"), "split": split, "role": "train" if split == "train" else "gallery"})
by_gid = {}
for i, r in enumerate(rows): by_gid.setdefault(r["gid"], []).append(i)
pairs = []
for g, ix in by_gid.items():
    if rows[ix[0]]["split"] != "test" or len(ix) < 2: continue
    q = rnd.choice(ix); rows[q]["role"] = "query"
    pos = [i for i in ix if i != q]
    for i in rnd.sample(pos, min(a.pairs, len(pos))): pairs.append((rows[q]["path"], rows[i]["path"], 1, ""))
    neg = [i for i in range(len(rows)) if rows[i]["seq"] == rows[q]["seq"] and rows[i]["gid"] != g]
    for i in rnd.sample(neg, min(a.pairs, len(neg))): pairs.append((rows[q]["path"], rows[i]["path"], 0, "same_type" if rows[i]["type"] == rows[q]["type"] else "diff_type"))
def wcsv(name, recs, cols):
    with open(os.path.join(out, name), "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(cols); [wr.writerow([r[c] for c in cols] if isinstance(r, dict) else r) for r in recs]
wcsv("ids.csv", ids, ["gid", "seq", "track_id", "type", "model", "color_name", "color_rgb", "roughness", "metallic", "visible_bin", "n_crops", "split"])
wcsv("crops.csv", rows, ["path", "gid", "seq", "track_id", "frame", "type", "x1", "y1", "x2", "y2", "w", "h", "mask_px", "visible_fraction", "occlusion_ratio", "split", "role"])
wcsv("pairs.csv", pairs, ["a", "b", "same_id", "neg_kind"])
summ = {"sequences": len(seqs), "test_sequences": len(test_seqs), "ids": len(ids), "crops": len(rows), "skipped_below_min_px": skipped, "min_px": a.min_px, "stride": a.stride,
        "ids_per_type": {t: sum(1 for x in ids if x["type"] == t) for t in sorted({x["type"] for x in ids})}, "query": sum(r["role"] == "query" for r in rows), "gallery": sum(r["role"] == "gallery" for r in rows), "train": sum(r["role"] == "train" for r in rows),
        "pairs_pos": sum(p[2] == 1 for p in pairs), "pairs_neg": sum(p[2] == 0 for p in pairs)}
json.dump(summ, open(os.path.join(out, "summary.json"), "w"), indent=1)
if a.sheet > 0 and rows:
    srt = sorted(by_gid, key=lambda g: -len(by_gid[g])); tq = {}
    for g in srt: tq.setdefault(ids[int(g) - 1]["type"], []).append(g)
    top = []   # round-robin over drone types so every model shows up
    while len(top) < min(a.sheet, len(srt)):
        for t in sorted(tq):
            if tq[t] and len(top) < a.sheet: top.append(tq[t].pop(0))
    T = 72; ncol = 10
    sheet = Image.new("RGB", (70 + ncol * (T + 4), len(top) * (T + 4) + 4), (25, 25, 25)); dr = ImageDraw.Draw(sheet)
    for ri, g in enumerate(top):
        ix = by_gid[g]; pick = [ix[int(j * (len(ix) - 1) / max(1, ncol - 1))] for j in range(min(ncol, len(ix)))]
        idr = ids[int(g) - 1]
        dr.text((4, ri * (T + 4) + 6), f"id {g}\n{idr['type']}\n{idr['color_rgb'] or 'orig'}\nn={len(ix)}", fill=(255, 255, 255))
        if idr["color_rgb"]: dr.rectangle([4, ri * (T + 4) + 58, 60, ri * (T + 4) + 66], fill=idr["color_rgb"])
        for ci, i in enumerate(pick):
            c = Image.open(os.path.join(out, rows[i]["path"])); c.thumbnail((T, T))
            x0 = 70 + ci * (T + 4) + (T - c.width) // 2; y0 = ri * (T + 4) + 4 + (T - c.height) // 2; sheet.paste(c, (x0, y0))
            dr.text((70 + ci * (T + 4) + 2, ri * (T + 4) + T - 8), f"f{rows[i]['frame']}", fill=(255, 220, 0))
    sheet.save(os.path.join(out, "crop_sheet.jpg"), quality=90)
print(json.dumps(summ))
