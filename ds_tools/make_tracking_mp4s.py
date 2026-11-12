r"""Make H.264 MP4s (raw + labeled overlay) for every tracking_v1 sequence. CPU only.
usage: train_env\python.exe ds_tools\make_tracking_mp4s.py [dataset_dir]
Writes <dataset>/videos_mp4/seq_XXX.mp4 and seq_XXX_overlay.mp4 (h264, yuv420p, native fps, 1280x720).
Overlay style matches overlay_tracking.py: ID boxes, short trails, phase counts, breakaways in red."""
import os, sys, json, glob, colorsys, subprocess, re
from PIL import Image, ImageDraw
import imageio_ffmpeg
FF = imageio_ffmpeg.get_ffmpeg_exe()
ds = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets", "tracking_v1")
od = os.path.join(ds, "videos_mp4"); os.makedirs(od, exist_ok=True)
ENC = ["-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
def col(i):
    r, g, b = colorsys.hsv_to_rgb((i * 0.618034) % 1, 0.65, 1.0); return (int(r * 255), int(g * 255), int(b * 255))
def overlay_frames(sd, meta):
    trail = {}
    for line in open(os.path.join(sd, "ann", "frames.jsonl")):
        r = json.loads(line)
        im = Image.open(os.path.join(sd, r["image"])).convert("RGB"); dr = ImageDraw.Draw(im)
        ph = {"ground": 0, "takeoff": 0, "cruise": 0}; n = len(r["drones"])
        for d in r["drones"]:
            ph[d["phase"]] = ph.get(d["phase"], 0) + 1
            if d["mask_px"] > 0: trail.setdefault(d["id"], []).append(tuple(d["center_px"]))
            t = trail.get(d["id"], [])[-(40 if d["is_breakaway"] else 8):]
            if d["is_breakaway"] and len(t) > 1: dr.line(t, fill=(255, 40, 40), width=3)
            elif len(t) > 1 and n <= 150: dr.line(t, fill=col(d["id"]), width=1)
            if d["bbox_tight"]:
                x1, y1, x2, y2 = d["bbox_tight"]
                if d["is_breakaway"]:
                    dr.rectangle([x1 - 5, y1 - 5, x2 + 5, y2 + 5], outline=(255, 30, 30), width=3); dr.text((x2 + 8, y1 - 5), f"BREAKAWAY id{d['id']}", fill=(255, 60, 60))
                else:
                    dr.rectangle([x1 - 1, y1 - 1, x2 + 1, y2 + 1], outline=col(d["id"]), width=1)
                    if n <= 40: dr.text((x2 + 2, y1 - 2), str(d["id"]), fill=col(d["id"]))
        dr.rectangle([0, 0, im.width, 34], fill=(0, 0, 0))
        bk = ", ".join(f"id{b['id']} {b['type']}@f{b['start_frame']}" for b in meta.get("breakaways", [])) or "none (negative sequence)"
        dr.text((6, 3), f"seq {meta['seq']:03d}  frame {r['frame']}/{meta['frames']}  t={r['time_s']:.1f}s  N={r['n_total']} visible={r['n_visible_ge1px']}  ground={ph.get('ground',0)} takeoff={ph.get('takeoff',0)} cruise={ph.get('cruise',0)}", fill=(255, 255, 255))
        dr.text((6, 18), f"{meta['launch']} launch, {meta['formation']}, cam {meta['camera_mode']}{' (ground)' if meta.get('ground_camera') else ''}   breakaways: {bk}   active now: {r['n_breakaway_active']}", fill=(255, 200, 120))
        yield im
for sd in sorted(glob.glob(os.path.join(ds, "seq_*"))):
    if not os.path.isdir(sd): continue
    name = os.path.basename(sd); meta = json.load(open(os.path.join(sd, "meta.json"))); fps = meta["fps"]
    raw = os.path.join(od, name + ".mp4"); ovl = os.path.join(od, name + "_overlay.mp4")
    subprocess.run([FF, "-y", "-loglevel", "error", "-framerate", str(fps), "-start_number", "1", "-i", os.path.join(sd, "img", "%06d.jpg"), "-vf", "scale=in_range=pc:out_range=tv,format=yuv420p", *ENC, "-r", str(fps), raw], check=True)
    p = subprocess.Popen([FF, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1280x720", "-framerate", str(fps), "-i", "-", *ENC, "-r", str(fps), ovl], stdin=subprocess.PIPE)
    for im in overlay_frames(sd, meta):
        if im.size != (1280, 720): im = im.resize((1280, 720))
        p.stdin.write(im.tobytes())
    p.stdin.close(); assert p.wait() == 0, name
    print("done", name, flush=True)
# verify
bad = 0
for sd in sorted(glob.glob(os.path.join(ds, "seq_*"))):
    name = os.path.basename(sd); nimg = len(glob.glob(os.path.join(sd, "img", "*.jpg")))
    for f in (name + ".mp4", name + "_overlay.mp4"):
        fp = os.path.join(od, f)
        info = subprocess.run([FF, "-hide_banner", "-i", fp], capture_output=True, text=True).stderr
        nf, _ = imageio_ffmpeg.count_frames_and_secs(fp)
        ok = ("h264" in info) and ("yuv420p(" in info or "yuv420p," in info) and ("1280x720" in info) and nf == nimg
        if not ok: bad += 1; print("BAD", f, nf, nimg)
print("VERIFY bad =", bad)

