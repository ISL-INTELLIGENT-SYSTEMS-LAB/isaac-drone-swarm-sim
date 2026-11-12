"""Single-frame sample of every Replicator output we can get in Isaac Sim 6.2, on DesertStage + 10 drones.
All edits go into a NEW layer stages/all_outputs.usda that sublayers stages/desert_drones.usda (source untouched)."""
import os, sys, time, json, threading, subprocess, math, random, re, traceback
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "samples", "all_outputs"); os.makedirs(OUT, exist_ok=True)
BASE = os.path.join(ROOT, "stages", "desert_drones.usda")
LAYER = os.path.join(ROOT, "stages", "all_outputs.usda")
W, H = 1280, 720
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
import omni.usd, omni.kit.commands
import numpy as np
from pxr import Usd, UsdGeom, UsdLux, Gf, Sdf
import omni.replicator.core as rep
from isaacsim.core.utils.stage import is_stage_loading
from isaacsim.core.utils.semantics import add_labels
from PIL import Image, ImageDraw

status = {}
def ok(k, v="ok"): status[k] = v; log("STATUS", k, "->", v)

lay = Sdf.Layer.CreateNew(LAYER) if not os.path.exists(LAYER) else Sdf.Layer.FindOrOpen(LAYER)
lay.Clear(); lay.subLayerPaths.append(BASE); lay.Save()
ctx = omni.usd.get_context(); ctx.open_stage(LAYER)
def settle(k=60):
    n = 0
    while True:
        app.update(); n += 1
        if not is_stage_loading() and n > k: break
settle(); [app.update() for _ in range(60)]
stage = ctx.get_stage(); mpu = UsdGeom.GetStageMetersPerUnit(stage); M = 1.0 / mpu
log("opened", stage.GetRootLayer().identifier, "mpu", mpu)

# ---------- semantic labels (scene categories by prim-name heuristics) ----------
RULES = [("vehicle", r"Car|Truck|Van|Technical|Sedan|Wreck"),
         ("building", r"Bld|Building|UrbRes|Garrage|Garage|UrbCnt|silo|Market|House|WallStd|villagecluster|singlerow|doublerow|Anglecompound|nonlinear"),
         ("wall_fence", r"Wall|Fence|fence|Barrier|Hesco|Gate|Corigated|Corrugated"),
         ("road", r"Pavement|Paveing|Curb|bridge|Flooring"),
         ("vegetation", r"Foliage|Tree|Palm|Bush|Plant|HierarchicalInstanced")]
cands = []
for parent in ["/Root", "/Root/Statics", "/Root/Statics/Blueprints", "/Root/Landscape/LandscapeStreamingProxy_0"]:
    pp = stage.GetPrimAtPath(parent)
    if pp: cands += list(pp.GetChildren())
counts = {}
for p in cands:
    nm = p.GetName()
    if nm in ("Statics", "Blueprints", "Landscape", "DroneSwarm", "RenderCams", "Looks", "Lighting_Objects", "LandscapeStreamingProxy_0"): continue
    cls = None
    for c, rx in RULES:
        if re.search(rx, nm): cls = c; break
    if cls is None and str(p.GetPath()).startswith("/Root/Landscape/"): cls = "ground"
    if cls: add_labels(p, labels=[cls], instance_name="class"); counts[cls] = counts.get(cls, 0) + 1
drones = sorted([p for p in stage.GetPrimAtPath("/Root/DroneSwarm").GetChildren()], key=lambda p: p.GetName())
for p in drones: add_labels(p, labels=["drone"], instance_name="class")
counts["drone"] = len(drones)
log("labels", counts); ok("semantic_labels", counts)

# ---------- cameras ----------
cam_path = "/Root/RenderCams/close"
cam = UsdGeom.Camera(stage.GetPrimAtPath(cam_path))
for c in ["/Root/RenderCams/close", "/Root/RenderCams/wide_40m"]:
    UsdGeom.Camera(stage.GetPrimAtPath(c)).GetVerticalApertureAttr().Set(20.955 * H / W)
cam_xf_attr = UsdGeom.Xformable(cam).GetOrderedXformOps()[0].GetAttr(); cam_xf0 = cam_xf_attr.Get()
ctx.save_stage(); settle(10)

def drone_poses():
    xc = UsdGeom.XformCache(Usd.TimeCode.Default()); out = []
    for p in drones:
        m = xc.GetLocalToWorldTransform(p); t = m.ExtractTranslation()
        tr = Gf.Transform(m); q = tr.GetRotation().GetQuat(); s = tr.GetScale()
        out.append({"id": p.GetName(), "path": str(p.GetPath()), "position_m": [round(v * mpu, 4) for v in t],
                    "orientation_quat_wxyz": [round(q.GetReal(), 6)] + [round(v, 6) for v in q.GetImaginary()],
                    "scale": [round(v, 5) for v in s]})
    return out
tops = [UsdGeom.Xformable(p).GetOrderedXformOps()[0].GetAttr() for p in drones]   # translate ops
base_pos = [a.Get() for a in tops]

def jsonable(o):
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, bytes): return o.decode(errors="ignore")
    return str(o)
def dump(name, d):
    files = []
    if isinstance(d, dict):
        data = d.get("data"); info = {k: v for k, v in d.items() if k != "data"}
    else:
        data, info = d, {}
    big = {}
    def strip(x, path):
        if isinstance(x, dict): return {k: strip(v, path + "." + str(k)) for k, v in x.items()}
        if isinstance(x, np.ndarray) and x.size > 512 and not x.dtype.names: big[path.strip(".")] = x; return f"<array {x.shape} {x.dtype} in npz>"
        if isinstance(x, np.ndarray) and x.dtype.names: return [dict(zip(x.dtype.names, [jsonable(v) for v in r])) for r in x]
        return x
    info = strip(info, "info")
    if data is not None:
        a = np.asarray(data) if not isinstance(data, dict) else None
        if a is None: info["data"] = strip(data, "data")
        elif a.dtype.names: info["data"] = strip(a, "data")
        elif a.size > 512: big["data"] = a
        else: info["data"] = a.tolist()
    if big:
        f = os.path.join(OUT, f"{name}.npz"); np.savez_compressed(f, **big); files.append(f)
    f = os.path.join(OUT, f"{name}.json"); json.dump(info, open(f, "w"), indent=1, default=jsonable); files.append(f)
    return files

# ---------- annotators ----------
WANT = [("rgb", "rgb", None), ("distance_to_camera", "distance_to_camera", None),
        ("distance_to_image_plane", "distance_to_image_plane", None), ("normals", "normals", None),
        ("motion_vectors", "motion_vectors", None),
        ("semantic_segmentation", "semantic_segmentation", {"colorize": False}),
        ("instance_segmentation", "instance_segmentation", {"colorize": False}),
        ("instance_id_segmentation", "instance_id_segmentation", {"colorize": False}),
        ("bounding_box_2d_tight", "bounding_box_2d_tight", None), ("bounding_box_2d_loose", "bounding_box_2d_loose", None),
        ("bounding_box_3d", "bounding_box_3d", None), ("occlusion", "occlusion", None), ("pointcloud", "pointcloud", None),
        ("camera_params", "camera_params", None), ("skeleton_data", "skeleton_data", None)]
SKIP = [x for x in os.environ.get("SKIP_ANN", "").split(",") if x]
rp = rep.create.render_product(cam_path, (W, H))
anns = {}
for key, reg, ip in WANT:
    if key in SKIP: ok("attach:" + key, "SKIPPED (env)"); continue
    tried = [reg] + ([reg + "_fast"] if reg == "instance_id_segmentation" else [])
    for r in tried:
        try:
            a = rep.AnnotatorRegistry.get_annotator(r, init_params=ip) if ip else rep.AnnotatorRegistry.get_annotator(r)
            a.attach(rp); anns[key] = a; ok("attach:" + key, r); break
        except Exception as e:
            ok("attach:" + key, f"FAILED {r}: {str(e)[:200]}")
for _ in range(60): app.update()
log("attached", list(anns), "vram", peak["mb"])

# frame A (drones at base), then move drones for motion vectors
rng = np.random.default_rng(3)
vel = []   # m/s per drone: shared swarm velocity + per-drone variation
fwd = Gf.Vec3d(0.5, -0.86, 0)
for i in range(len(drones)):
    v = np.array([3.0, 1.0, 0.0]) + rng.normal(0, 1.5, 3) * np.array([1, 1, 0.5]); vel.append(v)
DT = 0.1
t1 = time.time()
rep.orchestrator.step(rt_subframes=16)
poses_A = drone_poses()
for a, p0, v in zip(tops, base_pos, vel):
    a.Set(p0 + Gf.Vec3d(*(v * DT * M)))
rep.orchestrator.step(rt_subframes=1)
mv_rt1 = None
if "motion_vectors" in anns:
    mv_rt1 = np.asarray(anns["motion_vectors"].get_data()).copy()
    log("motion rt1 stats", mv_rt1.shape, mv_rt1.dtype, float(np.abs(mv_rt1[..., :2]).max()), float(np.abs(mv_rt1[..., :2]).mean()))
rep.orchestrator.step(rt_subframes=32)
log("main capture rendered", round(time.time() - t1, 1), "s vram", peak["mb"])
poses_B = drone_poses()
saved = {}
for key, a in anns.items():
    try:
        d = a.get_data()
        if key == "rgb":
            img = np.asarray(d)[..., :3].astype(np.uint8); Image.fromarray(img).save(os.path.join(OUT, "rgb.png")); saved[key] = ["rgb.png"]
        else:
            fs = dump(key, d); saved[key] = [os.path.basename(f) for f in fs]
        ok("data:" + key, saved[key])
    except Exception as e:
        ok("data:" + key, "FAILED " + str(e)[:300])
if mv_rt1 is not None:
    np.savez_compressed(os.path.join(OUT, "motion_vectors_rt1.npz"), data=mv_rt1); saved["motion_vectors_rt1"] = ["motion_vectors_rt1.npz"]
mv_final = np.asarray(anns["motion_vectors"].get_data()) if "motion_vectors" in anns else None
if mv_final is not None: log("motion final stats", float(np.abs(mv_final[..., :2]).max()), float(np.abs(mv_final[..., :2]).mean()))
json.dump({"frame_A_poses": poses_A, "frame_B_poses": poses_B, "dt_s": DT, "velocity_mps": [v.tolist() for v in vel],
           "note": "frame B is the captured frame; motion vectors describe A->B"}, open(os.path.join(OUT, "drone_world_poses.json"), "w"), indent=1)
ok("drone_world_poses", "drone_world_poses.json")

# ---------- domain randomization (rgb + instance seg + 2D boxes) ----------
for k in ["distance_to_camera", "distance_to_image_plane", "normals", "motion_vectors", "semantic_segmentation", "instance_segmentation",
          "bounding_box_2d_loose", "bounding_box_3d", "occlusion", "pointcloud", "skeleton_data"]:
    if k in anns:
        try: anns[k].detach()
        except Exception: pass
lights = [p for p in stage.Traverse() if p.IsA(UsdLux.DistantLight)]
domes = [p for p in stage.Traverse() if p.IsA(UsdLux.DomeLight)]
log("lights", [str(p.GetPath()) for p in lights], "domes", [str(p.GetPath()) for p in domes])
dr_meta = []
try:
    sun = lights[0] if lights else None
    sun_int0 = UsdLux.DistantLight(sun).GetIntensityAttr().Get() if sun else None
    sun_xf = UsdGeom.Xformable(sun) if sun else None
    for f in range(3):
        rr = np.random.default_rng(100 + f)
        for a, p0 in zip(tops, base_pos):
            a.Set(p0 + Gf.Vec3d(*(rr.uniform([-3, -3, -2], [3, 3, 2]) * M)))
        el = rr.uniform(15, 70); az = rr.uniform(0, 360); inten = None
        if sun is not None:
            d = Gf.Vec3d(-math.cos(math.radians(el)) * math.cos(math.radians(az)), -math.cos(math.radians(el)) * math.sin(math.radians(az)), -math.sin(math.radians(el)))
            mat = Gf.Matrix4d().SetLookAt(Gf.Vec3d(0, 0, 0), d, Gf.Vec3d(0, 0, 1) if abs(d[2]) < 0.99 else Gf.Vec3d(0, 1, 0)).GetInverse()
            sun_xf.ClearXformOpOrder(); sun_xf.AddTransformOp().Set(mat)
            inten = float(sun_int0 or 1.0) * rr.uniform(0.4, 1.3); UsdLux.DistantLight(sun).GetIntensityAttr().Set(inten)
        jit = Gf.Matrix4d().SetTranslate(Gf.Vec3d(*(rr.uniform(-1.5, 1.5, 3) * M)))
        rot = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(*rr.normal(0, 1, 3)).GetNormalized(), rr.uniform(-3, 3)))
        cam_xf_attr.Set(rot * cam_xf0 * jit)
        for _ in range(5): app.update()
        rep.orchestrator.step(rt_subframes=32)
        img = np.asarray(anns["rgb"].get_data())[..., :3].astype(np.uint8)
        bb = anns["bounding_box_2d_tight"].get_data()
        ov = Image.fromarray(img.copy()); dr = ImageDraw.Draw(ov); nb = 0
        for b in bb["data"]:
            dr.rectangle([int(b["x_min"]), int(b["y_min"]), int(b["x_max"]), int(b["y_max"])], outline=(255, 40, 40), width=2); nb += 1
        Image.fromarray(img).save(os.path.join(OUT, f"dr_{f}_rgb.png")); ov.save(os.path.join(OUT, f"dr_{f}_boxes.png"))
        seg = anns["instance_id_segmentation"].get_data()
        np.savez_compressed(os.path.join(OUT, f"dr_{f}_instance_id.npz"), data=np.asarray(seg["data"]))
        dump(f"dr_{f}_bounding_box_2d_tight", bb)
        ent = {"frame": f, "sun_elevation_deg": round(el, 1), "sun_azimuth_deg": round(az, 1), "sun_intensity": inten,
               "camera_jitter": "translate +-1.5 m, rotate +-3 deg", "drone_offsets": "uniform +-3 m xy, +-2 m z", "n_boxes": nb,
               "poses": drone_poses()}
        dr_meta.append(ent); log("DR frame", f, {k: v for k, v in ent.items() if k != "poses"})
    json.dump(dr_meta, open(os.path.join(OUT, "domain_randomization.json"), "w"), indent=1)
    ok("domain_randomization", f"{len(dr_meta)} frames")
except Exception as e:
    ok("domain_randomization", "FAILED " + traceback.format_exc()[-600:])
# restore
for a, p0 in zip(tops, base_pos): a.Set(p0)
cam_xf_attr.Set(cam_xf0)
for a in list(anns.values()):
    try: a.detach()
    except Exception: pass
try: rp.destroy()
except Exception: pass
log("vram after camera passes", peak["mb"])

# ---------- RTX LiDAR / radar quick single-scan attempt ----------
def sensor_try(kind, cmd, configs, ann_names, frames=30):
    eye = cam_xf0.ExtractTranslation()
    for cfg in configs:
        try:
            kw = dict(path=f"/Root/Sensors/{kind}", parent=None, translation=Gf.Vec3d(*eye), orientation=Gf.Quatd(1, 0, 0, 0))
            if cfg: kw["config"] = cfg
            res = omni.kit.commands.execute(cmd, **kw)
            sensor = res[1] if isinstance(res, tuple) else res
            spath = sensor.GetPath() if hasattr(sensor, "GetPath") else f"/Root/Sensors/{kind}"
            log(kind, "created", spath, "config", cfg, "type", stage.GetPrimAtPath(spath).GetTypeName())
            srp = rep.create.render_product(str(spath), [1, 1], name=f"{kind}_rp")
            got = None
            for an in ann_names:
                try:
                    a = rep.AnnotatorRegistry.get_annotator(an); a.attach([srp])
                    import omni.timeline
                    tl = omni.timeline.get_timeline_interface(); tl.play()
                    best = None
                    for _ in range(frames):
                        app.update()
                        d = a.get_data()
                        arr = np.asarray(d["data"]) if isinstance(d, dict) and "data" in d else np.asarray(d)
                        if arr.size and (best is None or arr.size > np.asarray(best["data"]).size): best = d
                    tl.stop()
                    if best is not None and np.asarray(best["data"]).size:
                        dump(f"{kind}_{an}", best); got = (an, np.asarray(best["data"]).shape); break
                    a.detach()
                except Exception as e:
                    log(kind, an, "annot fail", str(e)[:200])
            ok(kind, f"config={cfg} annotator={got}" if got else f"created (config={cfg}) but no points in {frames} frames")
            return
        except Exception as e:
            log(kind, "config", cfg, "fail", str(e)[:300])
    ok(kind, "FAILED to create sensor")
try:
    import omni.replicator.core as _r
    try: rep.orchestrator.set_capture_on_play(False)
    except Exception: pass
    sensor_try("lidar", "IsaacSensorCreateRtxLidar", ["Example_Rotary", "OS1_REV6", "Example_Solid_State", None],
               ["IsaacExtractRTXSensorPointCloudNoAccumulator", "IsaacCreateRTXLidarScanBuffer"])
except Exception as e: ok("lidar", "FAILED " + str(e)[:300])
try:
    sensor_try("radar", "IsaacSensorCreateRtxRadar", ["Example", None], ["IsaacExtractRTXSensorPointCloudNoAccumulator", "IsaacCreateRTXRadarPointCloud"])
except Exception as e: ok("radar", "FAILED " + str(e)[:300])

meta = {"layer": LAYER, "camera": cam_path, "resolution": [W, H], "status": status, "files": saved,
        "peak_vram_mb": peak["mb"], "total_seconds": round(time.time() - T0, 1)}
json.dump(meta, open(os.path.join(OUT, "sample_meta.json"), "w"), indent=1, default=jsonable)
log("DONE peak VRAM", peak["mb"])
app.close()
