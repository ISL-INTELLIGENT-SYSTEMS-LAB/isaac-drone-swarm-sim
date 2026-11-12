"""Drones in DesertStage, authored in a NEW layer (source untouched). Headless Isaac Sim.
env: DRONE_ASSET (url), DRONE_SPAN_M (default 0.6), N_DRONES (default 10)
"""
import os, sys, time, json, threading, subprocess, math, random
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "renders", "drones")
os.makedirs(OUT, exist_ok=True); os.makedirs(os.path.join(ROOT, "stages"), exist_ok=True)
LAYER = os.path.join(ROOT, "stages", "desert_drones.usda")
DESERT = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Machinima/Squad/Stages/Desert/DesertStage.usd"
ASSET = os.environ.get("DRONE_ASSET", "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Machinima/Squad/Vehicles/Drone/SK_DroneGhost.usd")
SPAN_M = float(os.environ.get("DRONE_SPAN_M", "0.6"))
N = int(os.environ.get("N_DRONES", "10"))
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

# --- author the new layer: it only sublayers DesertStage; drones are added below ---


from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "width": 1920, "height": 1080})
import omni.usd
import numpy as np
from pxr import Usd, UsdGeom, Gf, Sdf
lay = Sdf.Layer.CreateNew(LAYER) if not os.path.exists(LAYER) else Sdf.Layer.FindOrOpen(LAYER)
lay.Clear(); lay.subLayerPaths.append(DESERT); lay.Save()
import omni.replicator.core as rep
try:
    from isaacsim.core.utils.stage import is_stage_loading
except Exception:
    from omni.isaac.core.utils.stage import is_stage_loading
from PIL import Image, ImageDraw

def label(prim, name):
    try:
        from isaacsim.core.utils.semantics import add_labels
        add_labels(prim, labels=[name], instance_name="class"); return "add_labels"
    except Exception as e1:
        try:
            from isaacsim.core.utils.semantics import add_update_semantics
            add_update_semantics(prim, name); return "add_update_semantics"
        except Exception as e2:
            from pxr import Semantics
            s = Semantics.SemanticsAPI.Apply(prim, "Semantics"); s.CreateSemanticTypeAttr().Set("class"); s.CreateSemanticDataAttr().Set(name)
            return "SemanticsAPI"

ctx = omni.usd.get_context()
log("open", LAYER); ctx.open_stage(LAYER)
n = 0
while True:
    app.update(); n += 1
    if not is_stage_loading() and n > 60: break
for _ in range(120): app.update()
stage = ctx.get_stage()
log("loaded; root layer =", stage.GetRootLayer().identifier, "edit target =", stage.GetEditTarget().GetLayer().identifier)
mpu = UsdGeom.GetStageMetersPerUnit(stage); M = 1.0 / mpu
log("scene metersPerUnit", mpu)

# camera from the earlier 40 m drone view (stage units = cm)
EYE = Gf.Vec3d(29347.28, 72271.82, -420.31); TGT = Gf.Vec3d(32203.57, 67324.57, -4420.31); GROUND = -4420.31
fwd = (TGT - EYE).GetNormalized(); upv = Gf.Vec3d(0, 0, 1)
right = Gf.Cross(fwd, upv).GetNormalized(); camup = Gf.Cross(right, fwd).GetNormalized()

UsdGeom.Xform.Define(stage, "/Root/DroneSwarm")
cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
# one probe ref to measure size
probe = UsdGeom.Xform.Define(stage, "/Root/DroneSwarm/_probe"); UsdGeom.Xform.Define(stage, "/Root/DroneSwarm/_probe/asset").GetPrim().GetReferences().AddReference(ASSET)
for _ in range(60): app.update()
while is_stage_loading(): app.update()
r = cache.ComputeWorldBound(probe.GetPrim()).ComputeAlignedRange()
sz = np.array(r.GetMax()) - np.array(r.GetMin()); span = max(sz[0], sz[1])
scale = (SPAN_M * M) / span
log("asset raw size (stage units)", sz.round(2).tolist(), "-> scale", scale)
stage.RemovePrim("/Root/DroneSwarm/_probe")

random.seed(7)
group_c = EYE + fwd * (24 * M)
pos = []
while len(pos) < N:
    p = group_c + right * random.uniform(-4, 4) * M + camup * random.uniform(-2.5, 2.5) * M + fwd * random.uniform(-4, 4) * M
    if all((p - q).GetLength() > 1.5 * M for q in pos): pos.append(p)
drones = []
for i, p in enumerate(pos):
    path = f"/Root/DroneSwarm/drone_{i:02d}"
    x = UsdGeom.Xform.Define(stage, path); pr = x.GetPrim(); UsdGeom.Xform.Define(stage, path + "/asset").GetPrim().GetReferences().AddReference(ASSET)
    xf = UsdGeom.Xformable(x); xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(p); xf.AddRotateXYZOp().Set(Gf.Vec3f(random.uniform(-6, 6), random.uniform(-6, 6), random.uniform(0, 360)))
    xf.AddScaleOp().Set(Gf.Vec3f(scale, scale, scale))
    how = label(pr, "drone")
    alt = (p[2] - GROUND) * mpu; dist = (p - EYE).GetLength() * mpu
    drones.append({"path": path, "alt_m": round(alt, 1), "dist_from_wide_cam_m": round(dist, 1)})
log("labels via", how); log("drones:", drones)
for _ in range(60): app.update()
while is_stage_loading(): app.update()
ctx.save_stage(); log("saved layer", LAYER)
ctx.open_stage(LAYER); n = 0
while True:
    app.update(); n += 1
    if not is_stage_loading() and n > 60: break
for _ in range(120): app.update()
stage = ctx.get_stage(); log("reopened layer")

def make_cam(name, eye, tgt, focal):
    cam = UsdGeom.Camera.Define(stage, f"/Root/RenderCams/{name}")
    cam.CreateFocalLengthAttr(focal); cam.CreateHorizontalApertureAttr(20.955); cam.CreateVerticalApertureAttr(20.955 * 1080 / 1920)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.1 * M, 50000 * M))
    v = Gf.Matrix4d().SetLookAt(eye, tgt, upv); xf = UsdGeom.Xformable(cam); xf.ClearXformOpOrder(); xf.AddTransformOp().Set(v.GetInverse())
    return str(cam.GetPath())
gc = Gf.Vec3d(*np.mean([list(p) for p in pos], axis=0))
views = [("wide_40m", make_cam("wide_40m", EYE, TGT, 18.0)),
         ("close", make_cam("close", gc - fwd * (10 * M) + camup * (0.5 * M), gc, 18.0))]
ctx.save_stage()
meta = {"layer": LAYER, "asset": ASSET, "span_m": SPAN_M, "n_drones": N, "drones": drones, "views": []}
for name, cp in views:
    t1 = time.time()
    rp = rep.create.render_product(cp, (1920, 1080))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb"); rgb.attach(rp)
    seg = rep.AnnotatorRegistry.get_annotator("instance_segmentation_fast", init_params={"colorize": True}); seg.attach(rp)
    bb = rep.AnnotatorRegistry.get_annotator("bounding_box_2d_tight_fast"); bb.attach(rp)
    for _ in range(60): app.update()
    rep.orchestrator.step(rt_subframes=64); rep.orchestrator.step(rt_subframes=64)
    img = np.asarray(rgb.get_data())[..., :3].astype(np.uint8)
    Image.fromarray(img).save(os.path.join(OUT, f"scene_{name}_rgb.png"))
    sd = seg.get_data(); s = np.asarray(sd["data"])
    Image.fromarray(s[..., :3].astype(np.uint8)).save(os.path.join(OUT, f"scene_{name}_instseg.png"))
    bd = bb.get_data(); boxes = bd["data"]
    ov = Image.fromarray(img.copy()); dr = ImageDraw.Draw(ov)
    for b in boxes:
        dr.rectangle([int(b["x_min"]), int(b["y_min"]), int(b["x_max"]), int(b["y_max"])], outline=(255, 40, 40), width=3)
    ov.save(os.path.join(OUT, f"scene_{name}_bbox.png"))
    sizes = [int(b["x_max"] - b["x_min"]) for b in boxes]
    ent = {"name": name, "seg_instances": len(sd["info"].get("idToLabels", {})), "bboxes": len(boxes),
           "bbox_widths_px": sizes, "seconds": round(time.time() - t1, 1)}
    meta["views"].append(ent); log("view", ent, "vram_peak", peak["mb"])
    rgb.detach(); seg.detach(); bb.detach(); rp.destroy()
rep.orchestrator.wait_until_complete()
meta["peak_vram_mb"] = peak["mb"]; meta["total_seconds"] = round(time.time() - T0, 1)
json.dump(meta, open(os.path.join(OUT, "scene_meta.json"), "w"), indent=2, default=str)
log("DONE peak VRAM", peak["mb"]); app.close()
