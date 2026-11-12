"""Swarm tracking + breakaway sequence on DesertStage. New layer stages/tracking.usda sublayers stages/desert_drones.usda.
Per frame: RGB, instance-ID mask, 2D tight boxes keyed by stable track id, 3D pos/vel, visibility, camera params, breakaway labels."""
import os, sys, time, json, threading, subprocess, math, re, csv
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "samples", "tracking"); FR = os.path.join(OUT, "frames"); MK = os.path.join(OUT, "masks")
for d in (OUT, FR, MK): os.makedirs(d, exist_ok=True)
BASE = os.path.join(ROOT, "stages", "desert_drones.usda")
LAYER = os.path.join(ROOT, "stages", "tracking.usda")
W, H = int(os.environ.get("TRK_W", "1280")), int(os.environ.get("TRK_H", "720"))
NF = int(os.environ.get("TRK_FRAMES", "90")); FPS = 10.0; DT = 1.0 / FPS
BF = int(os.environ.get("TRK_BREAK_FRAME", str(NF // 2)))
BREAK_IDS = [3, 7]
SUB = int(os.environ.get("TRK_SUBFRAMES", "8"))
CAMERA = os.environ.get("TRK_CAM", "/Root/RenderCams/wide_40m")
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
peak = {"mb": 0}
def vram_poll():
    while True:
        try:
            o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
            peak["mb"] = max(peak["mb"], int(o))
        except Exception: pass
        time.sleep(2)
threading.Thread(target=vram_poll, daemon=True).start()

from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "width": W, "height": H})
import omni.usd
import numpy as np
from pxr import Usd, UsdGeom, Gf, Sdf
import omni.replicator.core as rep
from isaacsim.core.utils.stage import is_stage_loading
from isaacsim.core.utils.semantics import add_labels
from PIL import Image

lay = Sdf.Layer.CreateNew(LAYER) if not os.path.exists(LAYER) else Sdf.Layer.FindOrOpen(LAYER)
lay.Clear(); lay.subLayerPaths.append(BASE); lay.Save()
ctx = omni.usd.get_context(); ctx.open_stage(LAYER)
n = 0
while True:
    app.update(); n += 1
    if not is_stage_loading() and n > 60: break
for _ in range(60): app.update()
stage = ctx.get_stage(); mpu = UsdGeom.GetStageMetersPerUnit(stage); M = 1.0 / mpu
drones = sorted(stage.GetPrimAtPath("/Root/DroneSwarm").GetChildren(), key=lambda p: p.GetName())
ND = len(drones); ids = [p.GetName() for p in drones]
for p in drones: add_labels(p, labels=["drone"], instance_name="class")
ops = [UsdGeom.Xformable(p).GetOrderedXformOps() for p in drones]
t_ops = [o[0] for o in ops]; r_ops = [o[1] for o in ops]
base = np.array([list(o.Get()) for o in t_ops]) * mpu        # metres, world (Z up)
rot0 = np.array([list(o.Get()) for o in r_ops])
GROUND_M = -4420.31 * mpu
EYE = np.array([29347.28, 72271.82, -420.31]) * mpu; TGT = np.array([32203.57, 67324.57, -4420.31]) * mpu
fwd = (TGT - EYE) / np.linalg.norm(TGT - EYE); up = np.array([0, 0, 1.0])
right = np.cross(fwd, up); right /= np.linalg.norm(right)
fwd_h = np.array([fwd[0], fwd[1], 0]); fwd_h /= np.linalg.norm(fwd_h)
log("drones", ids, "camera", CAMERA, "frames", NF, "break frame", BF, "breakaway ids", [ids[i] for i in BREAK_IDS])

# ---------- trajectories (metres) ----------
rng = np.random.default_rng(11)
c0 = base.mean(0); offs = base - c0
t = np.arange(NF) * DT
cen = c0[None] + right[None] * (-5.0 + 1.3 * t)[:, None] + fwd_h[None] * (1.2 * np.sin(0.45 * t))[:, None] + up[None] * (0.4 * np.sin(0.6 * t))[:, None]
P = np.zeros((NF, ND, 3))
for i in range(ND):
    ph = rng.uniform(0, 2 * np.pi, 3); fr = rng.uniform(0.25, 0.7, 3); amp = rng.uniform(0.12, 0.3, 3)
    jit = np.stack([amp[k] * np.sin(2 * np.pi * fr[k] * t + ph[k]) for k in range(3)], 1)
    P[:, i] = cen + offs[i] + jit
tb = BF * DT
dirs = {3: np.array(0.55 * right + 0.6 * fwd_h - 0.58 * up), 7: np.array(-0.85 * right - 0.25 * fwd_h + 0.45 * up)}
for i, d in dirs.items():
    d = d / np.linalg.norm(d); tau = np.clip(t - tb, 0, None)
    s = np.where(tau < 1.5, 0.5 * 1.6 * tau ** 2, 0.5 * 1.6 * 1.5 ** 2 + 1.6 * 1.5 * (tau - 1.5))   # accelerate 1.5 s then constant (2.4 m/s rel.)
    P[:, i] += d[None] * s[:, None]
V = np.gradient(P, DT, axis=0)
is_break = np.zeros((NF, ND), int)
for i in BREAK_IDS: is_break[BF:, i] = 1
core = [i for i in range(ND) if i not in BREAK_IDS]
swarm_c = P[:, core].mean(1); all_c = P.mean(1)
dist_core = np.linalg.norm(P - swarm_c[:, None], axis=2); dist_all = np.linalg.norm(P - all_c[:, None], axis=2)

def set_frame(f):
    for i in range(ND):
        t_ops[i].Set(Gf.Vec3d(*(P[f, i] * M)))
        v = V[f, i]; sp = np.linalg.norm(v[:2]); tilt = min(12.0, 4.0 * sp)
        hd = v[:2] / (sp + 1e-6)
        r_ops[i].Set(Gf.Vec3f(float(-tilt * hd[1]), float(tilt * hd[0]), float(rot0[i][2])))

# ---------- render ----------
rp = rep.create.render_product(CAMERA, (W, H))
a_rgb = rep.AnnotatorRegistry.get_annotator("rgb"); a_rgb.attach(rp)
a_seg = rep.AnnotatorRegistry.get_annotator("instance_id_segmentation_fast", init_params={"colorize": False}); a_seg.attach(rp)
a_bb = rep.AnnotatorRegistry.get_annotator("bounding_box_2d_tight"); a_bb.attach(rp)
a_cp = rep.AnnotatorRegistry.get_annotator("camera_params"); a_cp.attach(rp)
a_occ = rep.AnnotatorRegistry.get_annotator("occlusion"); a_occ.attach(rp)
set_frame(0)
for _ in range(60): app.update()
rep.orchestrator.step(rt_subframes=32)
rx = re.compile(r"/Root/DroneSwarm/(drone_\d\d)")
rows = []; mot = []; jl = open(os.path.join(OUT, "frames.jsonl"), "w"); cam_first = None
for f in range(NF):
    tf = time.time()
    set_frame(f); app.update()
    rep.orchestrator.step(rt_subframes=SUB)
    img = np.asarray(a_rgb.get_data())[..., :3].astype(np.uint8)
    Image.fromarray(img).save(os.path.join(FR, f"{f:04d}.jpg"), quality=92)
    sd = a_seg.get_data(); seg = np.asarray(sd["data"]); id2 = sd["info"]["idToLabels"]
    track_mask = np.zeros(seg.shape[:2], np.uint8); mpx = np.zeros(ND, int); inst2t = {}
    for k, pth in id2.items():
        if "DroneSwarm" not in str(pth): continue
        m = rx.search(str(pth))
        if m:
            ti = ids.index(m.group(1)); inst2t[int(k)] = ti; sel = seg == int(k); track_mask[sel] = ti + 1; mpx[ti] += int(sel.sum())
    occ_t = {}
    od = a_occ.get_data(); od = od["data"] if isinstance(od, dict) else od
    for o in np.asarray(od):
        ti = inst2t.get(int(o["instanceId"]))
        if ti is not None: occ_t[ti] = max(occ_t.get(ti, -1.0), float(o["occlusionRatio"]))
    Image.fromarray(track_mask).save(os.path.join(MK, f"{f:04d}.png"))        # pixel value = track index + 1 (0 = background)
    bd = a_bb.get_data(); paths = bd["info"].get("primPaths", [])
    box = {}
    for b, pth in zip(bd["data"], paths):
        m = rx.search(str(pth))
        if m:
            ti = ids.index(m.group(1))
            box[ti] = (int(b["x_min"]), int(b["y_min"]), int(b["x_max"]), int(b["y_max"]), occ_t.get(ti, -1.0))
    cp = a_cp.get_data(); cpj = {k: (np.asarray(v).tolist() if hasattr(v, "__len__") and not isinstance(v, str) else (float(v) if isinstance(v, (np.floating, float)) else v)) for k, v in cp.items()}
    if cam_first is None: cam_first = cpj
    view = np.array(cp["cameraViewTransform"]).reshape(4, 4); proj = np.array(cp["cameraProjection"]).reshape(4, 4)
    fr = {"frame": f, "t": round(f * DT, 3), "camera_params": cpj, "drones": []}
    for i in range(ND):
        ph = np.append(P[f, i] * M, 1.0) @ view @ proj; ndc = ph[:3] / ph[3]
        u, v = (ndc[0] + 1) / 2 * W, (1 - ndc[1]) / 2 * H
        bx = box.get(i); vis = int(bx is not None and mpx[i] >= 4 and (bx[4] < 0.95))  # occlusion from the occlusion annotator (-1 = not reported)
        e = {"track_id": i, "id": ids[i], "position_m": P[f, i].round(4).tolist(), "velocity_mps": V[f, i].round(4).tolist(),
             "speed_mps": round(float(np.linalg.norm(V[f, i])), 3), "altitude_agl_m": round(float(P[f, i, 2] - GROUND_M), 3),
             "projected_center_px": [round(float(u), 1), round(float(v), 1)], "in_frame": int(0 <= u < W and 0 <= v < H and ph[3] > 0),
             "bbox_xywh": [bx[0], bx[1], bx[2] - bx[0], bx[3] - bx[1]] if bx else None, "occlusion_ratio": round(bx[4], 3) if bx else None,
             "mask_px": int(mpx[i]), "visible": vis, "is_breakaway": int(is_break[f, i]),
             "breakaway_start_frame": BF if i in BREAK_IDS else -1,
             "dist_to_swarm_centroid_m": round(float(dist_core[f, i]), 3), "dist_to_all_centroid_m": round(float(dist_all[f, i]), 3)}
        fr["drones"].append(e)
        if vis: mot.append([f + 1, i + 1, bx[0], bx[1], bx[2] - bx[0], bx[3] - bx[1], 1, -1, -1, -1])
        rows.append([f, round(f * DT, 3), i, ids[i], *P[f, i].round(4), *V[f, i].round(4), e["speed_mps"], e["altitude_agl_m"],
                     e["is_breakaway"], e["breakaway_start_frame"], e["dist_to_swarm_centroid_m"], e["dist_to_all_centroid_m"],
                     vis, e["in_frame"], e["occlusion_ratio"] if bx else "", *(e["bbox_xywh"] if bx else ["", "", "", ""]), int(mpx[i]),
                     round(float(u), 1), round(float(v), 1)])
    fr["swarm_centroid_m"] = swarm_c[f].round(4).tolist(); fr["n_visible"] = int(sum(d["visible"] for d in fr["drones"]))
    jl.write(json.dumps(fr) + "\n")
    if f % 10 == 0 or f == NF - 1:
        log(f"frame {f} visible {fr['n_visible']}/{ND} boxes {len(box)} {time.time()-tf:.1f}s vram {peak['mb']}")
jl.close()
with open(os.path.join(OUT, "gt_mot.txt"), "w") as fh:
    for r in mot: fh.write(",".join(str(x) for x in r) + "\n")
with open(os.path.join(OUT, "trajectories_3d.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["frame", "t_s", "track_id", "id", "x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps", "speed_mps", "altitude_agl_m",
                                    "is_breakaway", "breakaway_start_frame", "dist_to_swarm_centroid_m", "dist_to_all_centroid_m",
                                    "visible", "in_frame", "occlusion_ratio", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "mask_px", "proj_u", "proj_v"])
    w.writerows(rows)
json.dump({"layer": LAYER, "camera": CAMERA, "resolution": [W, H], "fps": FPS, "n_frames": NF, "rt_subframes": SUB,
           "track_ids": {i: ids[i] for i in range(ND)}, "breakaway_ids": [ids[i] for i in BREAK_IDS], "breakaway_start_frame": BF,
           "breakaway_dirs_world": {ids[i]: (d / np.linalg.norm(d)).round(3).tolist() for i, d in dirs.items()},
           "swarm_centroid_m": swarm_c.round(4).tolist(), "camera_params_frame0": cam_first,
           "ground_ref_z_m": GROUND_M, "mot_format": "frame(1-based),id(track_id+1),x,y,w,h,conf,-1,-1,-1",
           "mask_format": "masks/NNNN.png uint8, value = track_id+1, 0 = background",
           "peak_vram_mb": peak["mb"], "total_seconds": round(time.time() - T0, 1)},
          open(os.path.join(OUT, "sequence_meta.json"), "w"), indent=1)
log("DONE frames", NF, "mot rows", len(mot), "peak VRAM", peak["mb"])
app.close()
