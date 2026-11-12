"""Render a close-up gallery of drone USD assets on a neutral ground. Headless Isaac Sim."""
import os, sys, time, json, threading, subprocess, math
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
ROOT = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(ROOT, "renders", "drones")
os.makedirs(OUT, exist_ok=True)
S3 = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
ASSETS = [
    ("Squad_SK_DroneGhost", S3 + "Assets/Machinima/Squad/Vehicles/Drone/SK_DroneGhost.usd", "CC BY-NC-ND 4.0 (Offworld Industries / Squad, via NVIDIA Omniverse content)"),
    ("Isaac_Crazyflie_cf2x", S3 + "Assets/Isaac/6.2/Isaac/Robots/Bitcraze/Crazyflie/cf2x.usd", "NVIDIA Isaac Sim asset terms"),
    ("Isaac_Quadcopter", S3 + "Assets/Isaac/6.2/Isaac/Robots/IsaacSim/Quadcopter/quadcopter.usd", "NVIDIA Isaac Sim asset terms"),
]
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
app = SimulationApp({"headless": True, "width": 1280, "height": 720})
import omni.usd
import numpy as np
from pxr import Usd, UsdGeom, UsdLux, UsdShade, Gf, Sdf, Vt
import omni.replicator.core as rep
try:
    from isaacsim.core.utils.stage import is_stage_loading
except Exception:
    from omni.isaac.core.utils.stage import is_stage_loading
from PIL import Image

ctx = omni.usd.get_context(); ctx.new_stage()
for _ in range(10): app.update()
stage = ctx.get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.Xform.Define(stage, "/World")
# lights
dome = UsdLux.DomeLight.Define(stage, "/World/Dome"); dome.CreateIntensityAttr(450); dome.CreateColorAttr(Gf.Vec3f(0.85, 0.9, 1.0))
sun = UsdLux.DistantLight.Define(stage, "/World/Sun"); sun.CreateIntensityAttr(1500); sun.CreateAngleAttr(1.0)
UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-50, 20, 35))
# neutral ground: big quad with grey preview material
g = UsdGeom.Mesh.Define(stage, "/World/Ground"); S = 200
g.CreatePointsAttr([(-S,-S,0),(S,-S,0),(S,S,0),(-S,S,0)]); g.CreateFaceVertexCountsAttr([4]); g.CreateFaceVertexIndicesAttr([0,1,2,3])
g.CreateNormalsAttr([(0,0,1)]*4); g.SetNormalsInterpolation("vertex")
mat = UsdShade.Material.Define(stage, "/World/Looks/Grey"); sh = UsdShade.Shader.Define(stage, "/World/Looks/Grey/S")
sh.CreateIdAttr("UsdPreviewSurface"); sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.22, 0.22, 0.21))
sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(mat)
UsdGeom.Xformable(g).AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))

cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
info = []
prims = []
for name, url, lic in ASSETS:
    path = f"/World/{name}"
    x = UsdGeom.Xform.Define(stage, path)
    UsdGeom.Xform.Define(stage, path + "/asset").GetPrim().GetReferences().AddReference(url)
    try:
        src = Usd.Stage.Open(url, Usd.Stage.LoadNone)
        mpu = UsdGeom.GetStageMetersPerUnit(src); upa = UsdGeom.GetStageUpAxis(src)
    except Exception as e:
        mpu, upa = 1.0, "Z"; log("meta read failed", name, e)
    prims.append((name, url, lic, x, mpu, upa))
log("waiting for references to load")
n = 0
while True:
    app.update(); n += 1
    if not is_stage_loading() and n > 60: break
    if n > 20000: break
for _ in range(60): app.update()
log("loaded after", n, "updates")

def mesh_bounds(root, max_ext_m=5.0):
    c = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    lo = np.array([1e18]*3); hi = -lo; nm = 0
    for p in Usd.PrimRange(root):
        if not p.IsA(UsdGeom.Gprim): continue
        if UsdGeom.Imageable(p).ComputeVisibility() == UsdGeom.Tokens.invisible: continue
        r = c.ComputeWorldBound(p).ComputeAlignedRange()
        if r.IsEmpty(): continue
        a, b = np.array(r.GetMin()), np.array(r.GetMax())
        if (b - a).max() > max_ext_m: continue
        lo = np.minimum(lo, a); hi = np.maximum(hi, b); nm += 1
    return lo, hi, nm

cache.Clear()
for name, url, lic, x, mpu, upa in prims:
    xf = UsdGeom.Xformable(x); xf.ClearXformOpOrder()
    xf.AddTranslateOp(); 
    if upa == "Y": xf.AddRotateXOp().Set(90.0)
    xf.AddScaleOp().Set(Gf.Vec3f(mpu, mpu, mpu))
    x.GetPrim().GetAttribute("xformOp:translate").Set(Gf.Vec3d(0, 0, 0))
    UsdGeom.Imageable(x).MakeVisible()
    lo, hi, nm = mesh_bounds(x.GetPrim())
    # sit on ground at origin
    off = Gf.Vec3d(-(lo[0]+hi[0])/2, -(lo[1]+hi[1])/2, -lo[2])
    x.GetPrim().GetAttribute("xformOp:translate").Set(Gf.Vec3d(0, 0, 0))  # assets stay put; ground moves to each asset
    size = (hi - lo)
    log(f"{name}: gprims={nm} source metersPerUnit={mpu} up={upa} size(m)={size.round(3).tolist()}")
    info.append({"name": name, "usd": url, "license": lic, "metersPerUnit": mpu, "upAxis": upa, "size_m": size.round(3).tolist()})

_unused = 0  # per-asset cameras below
if False: cam = None; cam.CreateFocalLengthAttr(35.0); cam.CreateHorizontalApertureAttr(20.955)

# ---- save and reopen so Fabric/physics pick up the authored transforms ----
GAL = os.path.join(ROOT, "stages", "drone_gallery.usda"); os.makedirs(os.path.dirname(GAL), exist_ok=True)
from pxr import UsdPhysics
nphys = 0
for p in stage.Traverse():
    if p.HasAPI(UsdPhysics.RigidBodyAPI):
        p.CreateAttribute("physics:rigidBodyEnabled", Sdf.ValueTypeNames.Bool).Set(False); nphys += 1
    if p.HasAPI(UsdPhysics.ArticulationRootAPI):
        p.CreateAttribute("physxArticulation:articulationEnabled", Sdf.ValueTypeNames.Bool).Set(False); nphys += 1
    if p.IsA(UsdPhysics.Scene):
        p.SetActive(False); nphys += 1
log("physics disabled on", nphys, "prims")
ctx.save_as_stage(GAL); log("saved", GAL)
ctx.open_stage(GAL); n = 0
while True:
    app.update(); n += 1
    if not is_stage_loading() and n > 60: break
for _ in range(60): app.update()
stage = ctx.get_stage()
prims = [(nm, u, l, UsdGeom.Xform(stage.GetPrimAtPath(f"/World/{nm}")), mp, ua) for (nm, u, l, _x, mp, ua) in prims]
log("reopened gallery stage")
# ---- autoframe: render, find the object in the instance mask, re-aim and move in ----
def label(prim, nm):
    try:
        from isaacsim.core.utils.semantics import add_labels; add_labels(prim, labels=[nm], instance_name="class")
    except Exception:
        from isaacsim.core.utils.semantics import add_update_semantics; add_update_semantics(prim, nm)
W, H = 1280, 720; HA = 20.955; VA = HA * H / W
def shoot(i, k, eye, tgt, focal, subframes):
    path = f"/World/Cam_{i}_{k}"
    cam = UsdGeom.Camera.Define(stage, path); cam.CreateFocalLengthAttr(focal); cam.CreateHorizontalApertureAttr(HA); cam.CreateVerticalApertureAttr(VA)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.005, 10000))
    m = Gf.Matrix4d().SetLookAt(eye, tgt, Gf.Vec3d(0, 0, 1)).GetInverse()
    UsdGeom.Xformable(cam).AddTransformOp().Set(m)
    for _ in range(5): app.update()
    rp = rep.create.render_product(path, (W, H))
    a_rgb = rep.AnnotatorRegistry.get_annotator("rgb"); a_rgb.attach(rp)
    a_seg = rep.AnnotatorRegistry.get_annotator("instance_id_segmentation_fast", init_params={"colorize": False}); a_seg.attach(rp)
    a_d = rep.AnnotatorRegistry.get_annotator("distance_to_camera"); a_d.attach(rp)
    for _ in range(30): app.update()
    rep.orchestrator.step(rt_subframes=subframes); rep.orchestrator.step(rt_subframes=subframes)
    img = np.asarray(a_rgb.get_data())[..., :3].astype(np.uint8)
    sd = a_seg.get_data(); seg = np.asarray(sd["data"]).reshape(H, W)
    ids = [int(k2) for k2, v in sd["info"]["idToLabels"].items() if str(v).startswith(f"/World/{prims[i][0]}/")]
    global LAST; LAST = sd["info"]["idToLabels"]
    mask = np.isin(seg, ids) if ids else np.zeros_like(seg, bool)
    dep = np.asarray(a_d.get_data()).reshape(H, W)
    a_rgb.detach(); a_seg.detach(); a_d.detach(); rp.destroy()
    return img, mask, dep, m
for i, (name, url, lic, x, mpu, upa) in enumerate(prims):
    for j, pp in enumerate(prims):
        UsdGeom.Imageable(pp[3]).MakeVisible() if j == i else UsdGeom.Imageable(pp[3]).MakeInvisible()
    label(x.GetPrim(), "drone")
    lo2, hi2, _ = mesh_bounds(x.GetPrim()); cc = (lo2 + hi2) / 2
    stage.GetPrimAtPath("/World/Ground").GetAttribute("xformOp:translate").Set(Gf.Vec3d(float(cc[0]), float(cc[1]), float(lo2[2]) - 0.002))
    log(name, "bounds after reopen", lo2.round(3).tolist(), hi2.round(3).tolist())
    tgt = Gf.Vec3d(*[float(v) for v in cc]); D = max(float(np.linalg.norm(hi2 - lo2)), 0.1)
    az, el = math.radians(35), math.radians(22)
    vdir = Gf.Vec3d(math.cos(az)*math.cos(el), math.sin(az)*math.cos(el), math.sin(el))
    focal = 10.0; dist0 = min(max(D * 8, 4.0), 8.0)
    def pts3d(mask, dep, m, focal):
        ys, xs = np.nonzero(mask)
        if len(xs) > 4000:
            sel = np.random.default_rng(0).choice(len(xs), 4000, replace=False); xs, ys = xs[sel], ys[sel]
        dx = (xs - W/2) / (W/2) * (HA/2) / focal; dy = -(ys - H/2) / (H/2) * (VA/2) / focal
        dirs = np.stack([dx, dy, -np.ones_like(dx)], 1); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        rot = np.array(m.ExtractRotationMatrix()); o = np.array(m.ExtractTranslation())
        d = dep[ys, xs]; ok = np.isfinite(d) & (d > 0) & (d < 1e4)
        return o + (dirs[ok] @ rot) * d[ok, None]
    found = None
    for k in range(4):
        eye = tgt + vdir * dist0
        img, mask, dep, m = shoot(i, k, eye, tgt, focal, 8)
        log(name, "search", k, "dist", round(dist0, 2), "mask px", int(mask.sum()))
        if mask.sum() > 3:
            P = pts3d(mask, dep, m, focal); found = (P.min(0), P.max(0)); break
        dist0 *= 3
    if found is not None:
        plo, phi = found; c3 = (plo + phi) / 2; size = float(np.linalg.norm(phi - plo))
        log(name, "rendered bounds", plo.round(3).tolist(), phi.round(3).tolist())
        stage.GetPrimAtPath("/World/Ground").GetAttribute("xformOp:translate").Set(Gf.Vec3d(float(c3[0]), float(c3[1]), float(plo[2]) - 0.002))
        tgt = Gf.Vec3d(*[float(v) for v in c3]); focal = 35.0
        fill = 0.25
        for r in range(4):
            eye = tgt + vdir * max(size / fill * focal / HA, 0.15)
            img, mask, dep, m = shoot(i, 10 + r, eye, tgt, focal, 8)
            log(name, "refine", r, "mask px", int(mask.sum()), "size", round(size, 3))
            if mask.sum() < 20:
                break
            P = pts3d(mask, dep, m, focal); plo, phi = P.min(0), P.max(0)
            c3 = (plo + phi) / 2; size = max(float(np.linalg.norm(phi - plo)), 0.03)
            tgt = Gf.Vec3d(*[float(v) for v in c3]); fill = min(fill * 1.6, 0.6)
        stage.GetPrimAtPath("/World/Ground").GetAttribute("xformOp:translate").Set(Gf.Vec3d(float(c3[0]), float(c3[1]), float(plo[2]) - 0.002))
        eye = tgt + vdir * max(size / 0.6 * focal / HA, 0.15)
    img, mask, dep, m = shoot(i, 9, eye, tgt, focal, 48)
    fn = os.path.join(OUT, f"asset_{i}_{name}.png"); Image.fromarray(img).save(fn)
    info[i]["render"] = fn; info[i]["mean_rgb"] = float(img.mean()); info[i]["object_px_frac"] = float(mask.mean())
    log(f"saved {fn} obj_px={mask.mean():.3f} vram_peak={peak['mb']}MB")
rep.orchestrator.wait_until_complete()
json.dump({"assets": info, "peak_vram_mb": peak["mb"], "seconds": round(time.time()-T0, 1)}, open(os.path.join(OUT, "assets_meta.json"), "w"), indent=2)
log("DONE peak VRAM", peak["mb"])
app.close()
