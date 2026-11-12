"""Draw swarm annotations on the frames of a sequence.

    python viz_scripts/viz_swarm.py <seq_dir> [--frames 1,16,32] [--clusters] [--drones] [--trail 40] [--sheet]

green = swarm box (ground truth), red = breakaway box, dot + line = centroid trail (last --trail frames),
--clusters adds the detector-side 2D-cluster tracks in yellow (thin, "C<id>"), --drones adds faint per-drone boxes.
Writes <seq>/swarm/viz/000001.jpg ... and, with --sheet, <seq>/swarm/viz/sheet.jpg (up to 6 frames).
Needs seq/swarm/swarm_annotations.json (ds_tools/swarm_annotate.py)."""
import os, sys, json, argparse
from PIL import Image, ImageDraw, ImageFont

def font(sz):
    for f in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try: return ImageFont.truetype(f, sz)
        except Exception: pass
    return ImageFont.load_default()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("seq"); ap.add_argument("--frames", default="")
    ap.add_argument("--clusters", action="store_true"); ap.add_argument("--drones", action="store_true"); ap.add_argument("--trail", type=int, default=40)
    ap.add_argument("--sheet", action="store_true"); a = ap.parse_args()
    J = json.load(open(os.path.join(a.seq, "swarm", "swarm_annotations.json")))
    od = os.path.join(a.seq, "swarm", "viz"); os.makedirs(od, exist_ok=True)
    want = set(int(x) for x in a.frames.split(",") if x.strip()) if a.frames else None
    drones = {}
    if a.drones:
        for l in open(os.path.join(a.seq, "ann", "frames.jsonl")):
            r = json.loads(l); drones[r["frame"]] = [d["bbox_tight"] for d in r["drones"] if d["bbox_tight"]]
    cl = {x["frame"]: x for x in J["clusters_2d"]["frames"]}
    ev_by_f = {}
    for e in J["events"]:
        if e["type"] == "breakaway": ev_by_f.setdefault(e["frame"], []).append(f"BREAKAWAY B{e['id']} left swarm S{e['parent']}")
    for e in J["clusters_2d"]["events"]:
        if e["type"] in ("breakaway", "split"): ev_by_f.setdefault(e["frame"], []).append(f"2D {e['type']}: C{e['child']} from C{e['parent']}")
        if e["type"] == "merge": ev_by_f.setdefault(e["frame"], []).append(f"2D merge: C{e['merged']} -> C{e['into']}")
    F1, F2 = font(18), font(15); trails = {}; saved = []
    for fr in J["frames"]:
        f = fr["frame"]
        for e in fr["entities"]:
            if e["centroid_px"]: trails.setdefault(e["id"], []).append((f, tuple(e["centroid_px"])))
        if want is not None and f not in want: continue
        im = Image.open(os.path.join(a.seq, fr["image"])).convert("RGB"); d = ImageDraw.Draw(im)
        for b in drones.get(f, []): d.rectangle(b, outline=(255, 255, 255), width=1)
        if a.clusters and f in cl:
            for c in cl[f]["clusters"]:
                x1, y1, x2, y2 = c["box"]; d.rectangle([x1 - 4, y1 - 4, x2 + 4, y2 + 4], outline=(255, 220, 0), width=1)
                d.text((x1 - 4, y2 + 6), f"C{c['id']} n={c['n']}", fill=(255, 220, 0), font=F2)
        for e in fr["entities"]:
            col = (40, 230, 60) if e["cls"] == "swarm" else (255, 40, 40)
            pts = [p for (ff, p) in trails.get(e["id"], []) if f - a.trail < ff <= f]
            if len(pts) > 1: d.line(pts, fill=col, width=2)
            if e["box"] is None: continue
            d.rectangle(e["box"], outline=col, width=3)
            if e["centroid_px"]: x, y = e["centroid_px"]; d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=col)
            lab = f"{'S' if e['cls'] == 'swarm' else 'B'}{e['id']}  {e['n_visible']}/{e['n_total']}"
            x1, y1 = e["box"][:2]; tw = d.textlength(lab, font=F1); ty = max(0, y1 - 24)
            d.rectangle([x1, ty, x1 + tw + 8, ty + 22], fill=(0, 0, 0)); d.text((x1 + 4, ty + 1), lab, fill=col, font=F1)
        hdr = f"seq {J['seq']}  frame {f}/{J['n_frames']}  drones visible {fr['n_visible_drones']}/{J['n_drones']}"
        d.rectangle([0, 0, 12 + d.textlength(hdr, font=F1), 26], fill=(0, 0, 0)); d.text((6, 3), hdr, fill=(255, 255, 255), font=F1)
        for i, t in enumerate(ev_by_f.get(f, [])):
            d.rectangle([0, 30 + 24 * i, 12 + d.textlength(t, font=F1), 52 + 24 * i], fill=(120, 0, 0)); d.text((6, 32 + 24 * i), t, fill=(255, 255, 255), font=F1)
        p = os.path.join(od, f"{f:06d}.jpg"); im.save(p, quality=90); saved.append(p)
    if a.sheet and saved:
        pick = saved if len(saved) <= 6 else [saved[round(i * (len(saved) - 1) / 5)] for i in range(6)]
        ims = [Image.open(p) for p in pick]; w, h = ims[0].size[0] // 2, ims[0].size[1] // 2; cols = 3 if len(ims) > 4 else 2
        rows = (len(ims) + cols - 1) // cols; sh = Image.new("RGB", (w * cols, h * rows))
        for i, im in enumerate(ims): sh.paste(im.resize((w, h)), ((i % cols) * w, (i // cols) * h))
        sh.save(os.path.join(od, "sheet.jpg"), quality=88); saved.append(os.path.join(od, "sheet.jpg"))
    print("wrote", len(saved), "images to", od)

if __name__ == "__main__": main()
