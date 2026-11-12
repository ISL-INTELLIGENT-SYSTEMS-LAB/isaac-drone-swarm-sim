"""Headless Isaac Sim render of an OpenUSD scene from S3.
Usage: python render_scene.py [scene_url] [out_dir]
"""
import os, sys, time, json, threading, subprocess, math
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

DESERT = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Machinima/Squad/Stages/Desert/DesertStage.usd"
URL = sys.argv[1] if len(sys.argv) > 1 else DESERT
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "renders", "desert")
os.makedirs(OUT, exist_ok=True)
T0 = time.time()
def log(*a):
    print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)

# ---- peak VRAM poller ----
peak = {"mb": 0}
def vram_poll():
    while True:
        try:
            o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
            peak["mb"] = max(peak["mb"], int(o))
        except Exception:
            pass
        time.sleep(2)
threading.Thread(target=vram_poll, daemon=True).start()

from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "width": 1920, "height": 1080, "renderer": "RayTracedLighting"})
log("SimulationApp up")

import omni.usd, omni.kit.app
import numpy as np
from pxr import Usd, UsdGeom, Gf, Sdf
import omni.replicator.core as rep
try:
    from isaacsim.core.utils.stage import is_stage_loading
except Exception:
    from omni.isaac.core.utils.stage import is_stage_loading

ctx = omni.usd.get_context()
log("Opening", URL)
ok = ctx.open_stage(URL)
log("open_stage returned", ok)
t_load = time.time()
n = 0
while True:
    app.update(); n += 1
    loading = is_stage_loading()
    if n % 200 == 0:
        log(f"update {n}, loading={loading}, vram_peak={peak['mb']}MB")
    if not loading and n > 60:
        break
    if time.time() - t_load > float(os.environ.get("LOAD_TIMEOUT_MIN", "45")) * 60:
        log("LOAD TIMEOUT"); app.close(); sys.exit(3)
for _ in range(120): app.update()
log(f"Stage loaded in {time.time()-t_load:.1f}s")
stage = ctx.get_stage()
mpu = UsdGeom.GetStageMetersPerUnit(stage) or 1.0
M = 1.0 / mpu  # stage units per meter
up = UsdGeom.GetStageUpAxis(stage)
log(f"metersPerUnit={mpu} upAxis={up}")
ui = 2 if up == "Z" else 1
upv = Gf.Vec3d(0, 0, 1) if up == "Z" else Gf.Vec3d(0, 1, 0)

# ---- bounds from top-level children, skipping huge sky/dome prims ----
cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
lo = np.array([1e18]*3); hi = np.array([-1e18]*3)
existing_cams = []
for p in stage.Traverse():
    if p.IsA(UsdGeom.Camera) and not str(p.GetPath()).startswith("/OmniverseKit"):
        existing_cams.append(str(p.GetPath()))
root = stage.GetPseudoRoot()
cands = []
for c in root.GetChildren():
    cands += list(c.GetChildren()) or [c]
for c in cands:
    try:
        r = cache.ComputeWorldBound(c).ComputeAlignedRange()
    except Exception:
        continue
    if r.IsEmpty(): continue
    a = np.array(r.GetMin()); b = np.array(r.GetMax())
    ext = (b - a).max() * mpu
    log(f"  bound {c.GetPath()} extent={ext:.0f}m")
    if ext > 20000:  # sky domes etc.
        continue
    lo = np.minimum(lo, a); hi = np.maximum(hi, b)
log("scene bounds (stage units)", lo.tolist(), hi.tolist(), "size m:", ((hi-lo)*mpu).round(1).tolist())
log("existing cameras:", existing_cams[:20])
ctr = (lo + hi) / 2
horiz = [i for i in range(3) if i != ui]

def vec(h0, h1, u):
    v = [0.0, 0.0, 0.0]; v[horiz[0]] = h0; v[horiz[1]] = h1; v[ui] = u
    return Gf.Vec3d(*v)

cam_root = "/World/RenderCams" if stage.GetPrimAtPath("/World") else "/RenderCams"
def make_cam(name, eye, target, focal=18.0):
    path = f"{cam_root}/{name}"
    cam = UsdGeom.Camera.Define(stage, path)
    cam.CreateFocalLengthAttr(focal)
    cam.CreateHorizontalApertureAttr(20.955)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.1 * M, 50000 * M))
    view = Gf.Matrix4d().SetLookAt(eye, target, upv)
    xf = UsdGeom.Xformable(cam)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(view.GetInverse())
    return path

# ---- ground height from terrain mesh vertices (Landscape / terrain prims) ----
gpts = []
for p in stage.Traverse():
    sp = str(p.GetPath()).lower()
    if p.IsA(UsdGeom.Mesh) and ("landscape" in sp or "terrain" in sp or "ground" in sp):
        pts = UsdGeom.Mesh(p).GetPointsAttr().Get()
        if not pts: continue
        pts = np.asarray(pts, dtype=np.float64)
        if len(pts) > 20000: pts = pts[:: len(pts) // 20000]
        m = np.array(UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        w = np.c_[pts, np.ones(len(pts))] @ m
        gpts.append(w[:, :3])
gpts = np.concatenate(gpts) if gpts else None
log("terrain sample points:", 0 if gpts is None else len(gpts))
def ground_at(h0, h1):
    if gpts is None: return None
    d = (gpts[:, horiz[0]] - h0) ** 2 + (gpts[:, horiz[1]] - h1) ** 2
    idx = np.argsort(d)[:8]
    return float(np.max(gpts[idx, ui]))
# focus on the built-up area (Statics) if present
st = stage.GetPrimAtPath("/Root/Statics")
if st and st.IsValid():
    r = cache.ComputeWorldBound(st).ComputeAlignedRange()
    if not r.IsEmpty():
        sc = (np.array(r.GetMin()) + np.array(r.GetMax())) / 2
        ctr[horiz[0]] = sc[horiz[0]]; ctr[horiz[1]] = sc[horiz[1]]
        log("focus = Statics center", sc.tolist())
for _ in range(30): app.update()
ext_h = [(hi[i] - lo[i]) for i in horiz]
g0 = ground_at(ctr[horiz[0]], ctr[horiz[1]])
log("ground at center (units):", g0)
if g0 is None: g0 = lo[ui]
R = min(min(ext_h) / 2 * 0.8, 250 * M)
views = []
# 2 ground-level oblique views from opposite sides, ~2 m eye height, looking across center
# obstacle boxes (buildings, walls, vehicles) so ground cameras don't spawn inside them
boxes = []
def add_boxes(prim, depth=0):
    for ch in prim.GetChildren():
        r = cache.ComputeWorldBound(ch).ComputeAlignedRange()
        if r.IsEmpty(): continue
        e = (np.array(r.GetMax()) - np.array(r.GetMin())).max() * mpu
        if e > 150 and depth < 3: add_boxes(ch, depth + 1)
        elif e <= 150: boxes.append((np.array(r.GetMin()), np.array(r.GetMax())))
for ch in stage.GetPrimAtPath("/Root").GetChildren() if stage.GetPrimAtPath("/Root") else []:
    nm = ch.GetName().lower()
    if "landscape" in nm or "foliage" in nm or "lighting" in nm: continue
    add_boxes(ch) if ch.GetName() == "Statics" else boxes.append(tuple(np.array(x) for x in (lambda r: (r.GetMin(), r.GetMax()))(cache.ComputeWorldBound(ch).ComputeAlignedRange())))
log("obstacle boxes:", len(boxes))
def clear(p, margin=3.0 * M):
    for a, b in boxes:
        if np.all(p >= a - margin) and np.all(p <= b + margin): return False
    return True
k = 0
for ang0 in [210, 30, 120, 300, 0, 90, 180, 270]:
    if k >= 2: break
    for rf in [0.6, 0.45, 0.75, 0.3, 0.9]:
        a = math.radians(ang0)
        h0 = ctr[horiz[0]] + math.cos(a) * R * rf; h1 = ctr[horiz[1]] + math.sin(a) * R * rf
        g = ground_at(h0, h1); g = g if g is not None else g0
        eye = vec(h0, h1, g + 2.0 * M)
        if clear(np.array(eye)):
            tgt = vec(ctr[horiz[0]], ctr[horiz[1]], g0 + 2.0 * M)
            views.append((f"ground_{k}", eye, tgt)); k += 1
            log(f"ground_{k-1}: angle {ang0} r {rf}")
            break# 2 drone-altitude views: 40 m @ ~35 deg down, 75 m @ ~42 deg down
for k, (alt, pitch, ang) in enumerate([(40, 35, 120), (75, 42, 300)]):
    dist = alt / math.tan(math.radians(pitch))
    a = math.radians(ang)
    h0 = ctr[horiz[0]] + math.cos(a) * dist * M; h1 = ctr[horiz[1]] + math.sin(a) * dist * M
    eye = vec(h0, h1, g0 + alt * M)
    tgt = vec(ctr[horiz[0]], ctr[horiz[1]], g0)
    views.append((f"drone_{k}_{alt}m", eye, tgt))
for v in views: log("view", v[0], "eye", list(v[1]), "target", list(v[2]))

from PIL import Image
import omni.syntheticdata  # noqa
meta = {"url": URL, "metersPerUnit": mpu, "upAxis": up, "bounds": [lo.tolist(), hi.tolist()], "views": []}
for i, (name, eye, tgt) in enumerate(views):
    t1 = time.time()
    cp = make_cam(name, eye, tgt)
    rp = rep.create.render_product(cp, (1920, 1080))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb"); rgb.attach(rp)
    seg = None
    if name.startswith("drone_0"):
        seg = rep.AnnotatorRegistry.get_annotator("instance_id_segmentation_fast", init_params={"colorize": True})
        seg.attach(rp)
    for _ in range(60): app.update()
    rep.orchestrator.step(rt_subframes=64)
    rep.orchestrator.step(rt_subframes=64)
    img = np.asarray(rgb.get_data())
    Image.fromarray(img[..., :3].astype(np.uint8)).save(os.path.join(OUT, f"{i}_{name}_rgb.png"))
    entry = {"name": name, "eye": list(eye), "target": list(tgt), "mean_rgb": float(img[..., :3].mean())}
    if seg is not None:
        sd = seg.get_data()
        s = np.asarray(sd["data"])
        if s.ndim == 3:
            Image.fromarray(s[..., :3].astype(np.uint8)).save(os.path.join(OUT, f"{i}_{name}_instseg.png"))
        else:
            Image.fromarray((s.astype(np.uint32) * 2654435761 % 2**24).view(np.uint8).reshape(*s.shape, 4)[..., :3]).save(
                os.path.join(OUT, f"{i}_{name}_instseg.png"))
        entry["n_instances"] = len(sd.get("info", {}).get("idToLabels", {}))
        seg.detach()
    rgb.detach(); rp.destroy()
    entry["seconds"] = round(time.time() - t1, 1)
    meta["views"].append(entry)
    log(f"saved {name} in {entry['seconds']}s mean={entry['mean_rgb']:.1f} vram_peak={peak['mb']}MB")

rep.orchestrator.wait_until_complete()
meta["total_seconds"] = round(time.time() - T0, 1)
meta["peak_vram_mb"] = peak["mb"]
json.dump(meta, open(os.path.join(OUT, "render_meta.json"), "w"), indent=2)
log("DONE total", meta["total_seconds"], "s, peak VRAM", peak["mb"], "MB")
app.close()
