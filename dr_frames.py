"""Domain-randomisation frames v2 (fixes stale boxes + over-bright sun of v1). Frame 'ref' = no randomisation, then 3 randomised frames."""
import os, time, json, threading, subprocess, math, re
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
ROOT = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(ROOT, "samples", "all_outputs", "dr_v2"); os.makedirs(OUT, exist_ok=True)
BASE = os.path.join(ROOT, "stages", "all_outputs.usda"); LAYER = os.path.join(ROOT, "stages", "dr_v2.usda")
W, H = 1280, 720; T0 = time.time()
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
import omni.usd, numpy as np
from pxr import Usd, UsdGeom, UsdLux, Gf, Sdf
import omni.replicator.core as rep
from isaacsim.core.utils.stage import is_stage_loading
from PIL import Image, ImageDraw
lay = Sdf.Layer.CreateNew(LAYER) if not os.path.exists(LAYER) else Sdf.Layer.FindOrOpen(LAYER)
lay.Clear(); lay.subLayerPaths.append(BASE); lay.Save()
ctx = omni.usd.get_context(); ctx.open_stage(LAYER)
n = 0
while True:
    app.update(); n += 1
    if not is_stage_loading() and n > 60: break
for _ in range(60): app.update()
stage = ctx.get_stage(); mpu = UsdGeom.GetStageMetersPerUnit(stage); M = 1.0 / mpu
drones = sorted(stage.GetPrimAtPath("/Root/DroneSwarm").GetChildren(), key=lambda p: p.GetName()); ids = [p.GetName() for p in drones]
t_ops = [UsdGeom.Xformable(p).GetOrderedXformOps()[0] for p in drones]; base_pos = [o.Get() for o in t_ops]
CAM = "/Root/RenderCams/close"; cam = stage.GetPrimAtPath(CAM)
cam_attr = UsdGeom.Xformable(cam).GetOrderedXformOps()[0].GetAttr(); cam0 = cam_attr.Get()
sun = [p for p in stage.Traverse() if p.IsA(UsdLux.DistantLight)][0]
sun_xf = UsdGeom.Xformable(sun); sun_m0 = sun_xf.GetLocalTransformation(); sun_ops0 = sun_xf.GetXformOpOrderAttr().Get()
sun_i0 = UsdLux.DistantLight(sun).GetIntensityAttr().Get()
sun_dir0 = Gf.Vec3d(*sun_xf.ComputeLocalToWorldTransform(0).TransformDir(Gf.Vec3d(0, 0, -1))).GetNormalized()
log("sun attrs", {a.GetName(): str(a.Get())[:60] for a in sun.GetAttributes() if a.HasAuthoredValue()})
log("sun", sun.GetPath(), "intensity0", sun_i0, "dir0(world)", [round(x, 3) for x in sun_dir0], "ops", sun_ops0)
INT_ATTR = sun.GetAttribute("intensity") if sun.GetAttribute("intensity").HasAuthoredValue() else UsdLux.DistantLight(sun).GetIntensityAttr()
INT0 = INT_ATTR.Get(); log("intensity attr used", INT_ATTR.GetName(), INT0)
rp = rep.create.render_product(CAM, (W, H))
a_rgb = rep.AnnotatorRegistry.get_annotator("rgb"); a_rgb.attach(rp)
a_seg = rep.AnnotatorRegistry.get_annotator("instance_id_segmentation_fast", init_params={"colorize": False}); a_seg.attach(rp)
a_bb = rep.AnnotatorRegistry.get_annotator("bounding_box_2d_tight"); a_bb.attach(rp)
a_cp = rep.AnnotatorRegistry.get_annotator("camera_params"); a_cp.attach(rp)
for _ in range(120): app.update()
rep.orchestrator.step(rt_subframes=32)
rx = re.compile(r"/Root/DroneSwarm/(drone_\d\d)")
meta = []
for f in ["ref", 0, 1, 2]:
    ent = {"frame": f}
    if f == "ref":
        for o, p0 in zip(t_ops, base_pos): o.Set(p0)
        pass  # ref: nothing touched
        ent.update(sun_azimuth_rot_deg=0, sun_tilt_deg=0, sun_intensity_scale=1.0, camera_jitter_m=[0, 0, 0], camera_rot_deg=0)
    else:
        rr = np.random.default_rng(200 + f)
        offs = rr.uniform([-3, -3, -2], [3, 3, 2], (len(ids), 3))
        for o, p0, d in zip(t_ops, base_pos, offs): o.Set(p0 + Gf.Vec3d(*(d * M)))
        az = rr.uniform(-120, 120); tilt = rr.uniform(-15, 15); sc = rr.uniform(0.6, 1.2)
        R = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), tilt)) * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), az))
        sun_xf.ClearXformOpOrder(); sun_xf.AddTransformOp().Set(sun_m0 * R)
        INT_ATTR.Set(float(INT0) * sc)
        jt = rr.uniform(-1.5, 1.5, 3); ax = Gf.Vec3d(*rr.normal(0, 1, 3)).GetNormalized(); ang = rr.uniform(-3, 3)
        cam_attr.Set(Gf.Matrix4d().SetRotate(Gf.Rotation(ax, ang)) * cam0 * Gf.Matrix4d().SetTranslate(Gf.Vec3d(*(jt * M))))
        ent.update(sun_azimuth_rot_deg=round(az, 1), sun_tilt_deg=round(tilt, 1), sun_intensity_scale=round(sc, 3), camera_jitter_m=np.round(jt, 2).tolist(), camera_rot_deg=round(ang, 2),
                   drone_offsets_m={i: np.round(d, 2).tolist() for i, d in zip(ids, offs)})
    sd = Gf.Vec3d(*sun_xf.ComputeLocalToWorldTransform(0).TransformDir(Gf.Vec3d(0, 0, -1))).GetNormalized()
    ent["sun_dir_world"] = [round(x, 3) for x in sd]; ent["sun_elevation_deg"] = round(math.degrees(math.asin(-sd[2])), 1)
    for _ in range(40): app.update()
    rep.orchestrator.step(rt_subframes=32)
    img = np.asarray(a_rgb.get_data())[..., :3].astype(np.uint8)
    sdat = a_seg.get_data(); seg = np.asarray(sdat["data"]); id2 = sdat["info"]["idToLabels"]
    tm = np.zeros(seg.shape[:2], np.uint8)
    for k, pth in id2.items():
        m = rx.search(str(pth))
        if m: tm[seg == int(k)] = ids.index(m.group(1)) + 1
    bd = a_bb.get_data(); boxes = {}
    for b, pth in zip(bd["data"], bd["info"].get("primPaths", [])):
        m = rx.search(str(pth))
        if m: boxes[m.group(1)] = [int(b["x_min"]), int(b["y_min"]), int(b["x_max"]), int(b["y_max"])]
    # cross-check: box vs mask extent
    chk = {}
    for i, nme in enumerate(ids):
        ys, xs = np.nonzero(tm == i + 1)
        if len(xs): chk[nme] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    ent["drone_boxes_xyxy"] = boxes; ent["drone_mask_extent_xyxy"] = chk; ent["mean_rgb"] = float(img.mean())
    cp = a_cp.get_data(); ent["camera_view_transform"] = np.asarray(cp["cameraViewTransform"]).tolist()
    Image.fromarray(img).save(os.path.join(OUT, f"dr_{f}_rgb.png")); Image.fromarray(tm).save(os.path.join(OUT, f"dr_{f}_drone_mask.png"))
    ov = Image.fromarray(img.copy()); d = ImageDraw.Draw(ov)
    for nme, b in boxes.items(): d.rectangle(b, outline=(255, 40, 40), width=2); d.text((b[0], b[1] - 12), nme, fill=(255, 60, 60))
    ov.save(os.path.join(OUT, f"dr_{f}_boxes.png"))
    meta.append(ent); log("frame", f, "boxes", len(boxes), "mean rgb", round(ent["mean_rgb"], 1), "sun el", ent["sun_elevation_deg"], {k: v for k, v in ent.items() if k in ("sun_azimuth_rot_deg", "sun_intensity_scale", "camera_jitter_m")})
json.dump(meta, open(os.path.join(OUT, "domain_randomization_v2.json"), "w"), indent=1)
log("DONE peak VRAM", peak["mb"])
app.close()
