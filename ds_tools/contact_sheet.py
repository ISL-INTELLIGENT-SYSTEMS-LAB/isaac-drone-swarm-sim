"""Contact sheet: first frame of every sequence in a folder, one grid image per folder (person / tier), labelled with the
seq_id and the key manifest settings, swarm boxes drawn (green = swarm, red = breakaway) when swarm labels exist.
A glance shows whether a share is all the same light, sky or camera. No Isaac Sim needed (Pillow only).

    python ds_tools/contact_sheet.py datasets\\alex_tier1                       # -> datasets\\alex_tier1\\contact_sheet.jpg
    python ds_tools/contact_sheet.py "datasets\\*_tier*" --cols 6 --frame 1 --out datasets\\coverage
"""
import os, glob, json, csv, argparse
from PIL import Image, ImageDraw, ImageFont

def font(sz):
    for f in ("arial.ttf", "DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try: return ImageFont.truetype(f, sz)
        except Exception: pass
    return ImageFont.load_default()

def seqs(folder):
    if os.path.exists(os.path.join(folder, "meta.json")): return [folder]
    return sorted(os.path.dirname(m) for m in glob.glob(os.path.join(folder, "*", "meta.json")))

def label(sd, meta):
    r = meta.get("manifest_row") or {}; sid = r.get("seq_id") or os.path.basename(sd)
    sw = [s["n_drones"] for s in meta.get("swarms", [])] or [meta.get("n_drones")]
    l1 = f"{sid}  {r.get('size_bin') or 'T' + str(r.get('tier', '?'))} {r.get('split', '')}{('/' + r['test_kind']) if r.get('test_kind') else ''}  swarms {'+'.join(map(str, sw))}"
    brk = f"brk {r.get('brk_size')}x{r.get('brk_count')}@{r.get('brk_start_frame')}" if meta.get("has_breakaway") else "no brk"
    l2 = f"{r.get('time_of_day', '?')} {r.get('weather', '')} | {r.get('camera_setup', meta.get('camera_framing', ''))} {r.get('cam_angle', '')}"
    l3 = f"{r.get('model_mix_name', '')} {r.get('color_mode', '')} | {r.get('wind', '')} {r.get('density', '')} | {brk}"
    return [l1, l2, l3]

def boxes(sd, frame):
    p = os.path.join(sd, "swarm", "swarm_tracks.csv"); out = []
    if not os.path.exists(p): return out
    for r in csv.reader(open(p)):
        if r and r[0].isdigit() and int(r[0]) == frame: out.append((int(r[2]), [float(v) for v in r[3:7]]))
    return out

def sheet(folder, out_path, cols=5, tw=384, frame=1, draw_boxes=True, S=None):
    S = S if S is not None else seqs(folder)
    if not S: return None
    th = tw * 720 // 1280; lh = 54; rows = (len(S) + cols - 1) // cols
    im = Image.new("RGB", (cols * tw, rows * (th + lh) + 30), (24, 24, 24)); dr = ImageDraw.Draw(im); f = font(13); ft = font(16)
    dr.text((8, 6), f"{folder}  ({len(S)} sequences, frame {frame})", fill=(255, 255, 255), font=ft)
    for i, sd in enumerate(S):
        x0, y0 = (i % cols) * tw, 30 + (i // cols) * (th + lh)
        meta = json.load(open(os.path.join(sd, "meta.json"))); p = os.path.join(sd, "img", f"{frame:06d}.jpg")
        if os.path.exists(p):
            t = Image.open(p).convert("RGB"); W, H = t.size; t = t.resize((tw, th)); td = ImageDraw.Draw(t); sx, sy = tw / W, th / H
            if draw_boxes:
                for c, (x1, y1, x2, y2) in boxes(sd, frame): td.rectangle([x1 * sx, y1 * sy, x2 * sx, y2 * sy], outline=(60, 230, 90) if c == 0 else (240, 60, 60), width=2)
            im.paste(t, (x0, y0))
        for j, s in enumerate(label(sd, meta)): dr.text((x0 + 4, y0 + th + 2 + j * 17), s, fill=(235, 235, 235) if j == 0 else (180, 180, 180), font=f)
    im.save(out_path, quality=90); return out_path

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+"); ap.add_argument("--out", default="", help="folder for the sheets (default: inside each folder)")
    ap.add_argument("--cols", type=int, default=5); ap.add_argument("--width", type=int, default=384); ap.add_argument("--frame", type=int, default=1); ap.add_argument("--no-boxes", action="store_true")
    ap.add_argument("--combined", default="", help="also write ONE sheet with every sequence of all folders to this file")
    a = ap.parse_args(); allS = []
    for pat in a.folders:
        for fo in (glob.glob(pat) or [pat]):
            if not os.path.isdir(fo): continue
            allS += seqs(fo)
            dst = os.path.join(a.out, f"contact_sheet_{os.path.basename(os.path.normpath(fo))}.jpg") if a.out else os.path.join(fo, "contact_sheet.jpg")
            if a.out: os.makedirs(a.out, exist_ok=True)
            r = sheet(fo, dst, a.cols, a.width, a.frame, not a.no_boxes); print(fo, "->", r or "no sequences")
    if a.combined and allS: print("combined ->", sheet(", ".join(a.folders), a.combined, a.cols, a.width, a.frame, not a.no_boxes, S=allS))
