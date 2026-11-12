"""Draw tracking overlays (plain python): python overlay_tracking.py <seq_dir> <out_dir>"""
import os, sys, json, colorsys
from PIL import Image, ImageDraw
sd, od = sys.argv[1], sys.argv[2]; os.makedirs(od, exist_ok=True)
meta = json.load(open(os.path.join(sd, "meta.json")))
S = 0.75; trail = {}
def col(i):
    r, g, b = colorsys.hsv_to_rgb((i * 0.618034) % 1, 0.65, 1.0); return (int(r * 255), int(g * 255), int(b * 255))
for line in open(os.path.join(sd, "ann", "frames.jsonl")):
    r = json.loads(line)
    im = Image.open(os.path.join(sd, r["image"])).convert("RGB"); im = im.resize((int(im.width * S), int(im.height * S)), Image.BILINEAR); dr = ImageDraw.Draw(im)
    ph = {"ground": 0, "takeoff": 0, "cruise": 0}
    for d in r["drones"]:
        ph[d["phase"]] += 1
        if d["mask_px"] > 0: trail.setdefault(d["id"], []).append((d["center_px"][0] * S, d["center_px"][1] * S))
        t = trail.get(d["id"], [])[-(40 if d["is_breakaway"] else 8):]
        if d["is_breakaway"] and len(t) > 1: dr.line(t, fill=(255, 40, 40), width=2)
        elif len(t) > 1 and len(r["drones"]) <= 150: dr.line(t, fill=col(d["id"]), width=1)
        if d["bbox_tight"]:
            x1, y1, x2, y2 = [v * S for v in d["bbox_tight"]]
            if d["is_breakaway"]:
                dr.rectangle([x1 - 4, y1 - 4, x2 + 4, y2 + 4], outline=(255, 30, 30), width=3); dr.text((x2 + 6, y1 - 4), f"BREAKAWAY id{d['id']}", fill=(255, 60, 60))
            else:
                dr.rectangle([x1 - 1, y1 - 1, x2 + 1, y2 + 1], outline=col(d["id"]), width=1)
                if len(r["drones"]) <= 40: dr.text((x2 + 2, y1 - 2), str(d["id"]), fill=col(d["id"]))
    dr.rectangle([0, 0, im.width, 34], fill=(0, 0, 0))
    bk = ", ".join(f"id{b['id']} {b['type']}@f{b['start_frame']}" for b in meta["breakaways"]) or "none (negative sequence)"
    dr.text((6, 3), f"seq {meta['seq']:03d}  frame {r['frame']}/{meta['frames']}  t={r['time_s']:.1f}s  N={r['n_total']} visible={r['n_visible_ge1px']}  ground={ph['ground']} takeoff={ph['takeoff']} cruise={ph['cruise']}", fill=(255, 255, 255))
    dr.text((6, 18), f"{meta['launch']} launch, {meta['formation']}, cam {meta['camera_mode']}{' (ground)' if meta['ground_camera'] else ''}   breakaways: {bk}   active now: {r['n_breakaway_active']}", fill=(255, 200, 120))
    im.save(os.path.join(od, f"{r['frame']:06d}.jpg"), quality=85)
print("overlay done", sd)
