"""Shared helpers for the drone dataset generators (import AFTER SimulationApp is created)."""
import os, time, json, math, subprocess, threading
import numpy as np
ROOT = os.environ.get("DRONE_SCENE_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DS = os.path.join(ROOT, "datasets")
DESERT = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Machinima/Squad/Stages/Desert/DesertStage.usd"
S3 = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
ASSETS = {  # type: (url, license / terms, instanceable when appearance variation is off)
    "ghost": (S3 + "Assets/Machinima/Squad/Vehicles/Drone/SK_DroneGhost.usd", "CC BY-NC-ND 4.0 (Offworld Industries / Squad)", True),
    "cf2x": (S3 + "Assets/Isaac/6.2/Isaac/Robots/Bitcraze/Crazyflie/cf2x.usd", "NVIDIA Isaac Sim asset terms", False),
    "quad": (S3 + "Assets/Isaac/6.2/Isaac/Robots/IsaacSim/Quadcopter/quadcopter.usd", "NVIDIA Isaac Sim asset terms", False),
    "arl": (S3 + "Assets/Isaac/6.2/Isaac/Robots/NTNU/ARL-Robot-1/arl_robot_1.usd", "NVIDIA Isaac Sim asset terms (NTNU Autonomous Robots Lab quadrotor)", False),
}
LEGACY_TYPES = ("ghost", "cf2x", "quad")
HEAVY_FACES = int(os.environ.get("PAINT_HEAVY_FACES", "50000"))   # instances above this many faces are not un-instanced for colouring
MIXES = {  # MODEL_MIX presets (type: probability); "legacy" = the generators' original hard-coded mixes
    "balanced": {"ghost": 0.25, "cf2x": 0.25, "quad": 0.25, "arl": 0.25},
    "isaac": {"cf2x": 1/3, "quad": 1/3, "arl": 1/3},   # no CC BY-NC-ND Ghost
    "ghost": {"ghost": 1.0},
}
def parse_mix(s):
    """MODEL_MIX: legacy | balanced | isaac | ghost | 'ghost:2,cf2x:1,arl:1' (weights). Returns dict or None (= legacy)."""
    s = (s or "legacy").strip()
    if s == "legacy": return None
    if s in MIXES: return dict(MIXES[s])
    w = {}
    for t in s.split(","):
        k, v = t.split(":"); k = k.strip()
        if k not in ASSETS: raise ValueError(f"unknown drone type {k}; known {list(ASSETS)}")
        w[k] = float(v)
    tot = sum(w.values()); return {k: v / tot for k, v in w.items() if v > 0}

def fit_types(tsel, by_type):
    """Re-assign drones whose type has run out of pool prims to the type with the most spare prims."""
    tsel = np.array(tsel, dtype=object)
    for k in list(by_type):
        ex = int((tsel == k).sum()) - len(by_type[k])
        for j in np.nonzero(tsel == k)[0][:max(0, ex)]:
            spare = {t: len(by_type[t]) - int((tsel == t).sum()) for t in by_type}
            tsel[j] = max(spare, key=spare.get)
    return tsel

# ---- appearance (per-drone colour / material variation) ----
PALETTE = [("red", (0.80, 0.08, 0.06)), ("orange", (0.95, 0.42, 0.05)), ("yellow", (0.95, 0.82, 0.10)), ("lime", (0.45, 0.85, 0.10)),
           ("green", (0.08, 0.50, 0.15)), ("teal", (0.05, 0.55, 0.55)), ("sky", (0.30, 0.65, 0.95)), ("blue", (0.08, 0.20, 0.80)),
           ("purple", (0.45, 0.12, 0.70)), ("pink", (0.95, 0.40, 0.65)), ("white", (0.92, 0.92, 0.90)), ("grey", (0.45, 0.45, 0.45)),
           ("black", (0.04, 0.04, 0.04)), ("tan", (0.70, 0.58, 0.40)), ("olive", (0.40, 0.42, 0.15)), ("brown", (0.35, 0.20, 0.08))]
def sample_appearance(rng, mode, strength=0.85, wear=True):
    """mode: none | random (any HSV colour) | palette (16 named colours with small jitter). Returns a JSON-able dict."""
    import colorsys
    if mode in (None, "", "none"): return {"mode": "none"}
    if mode == "palette":
        name, c = PALETTE[int(rng.integers(len(PALETTE)))]; c = np.clip(np.array(c) * rng.uniform(0.9, 1.1, 3), 0, 1)
        h, s_, v = colorsys.rgb_to_hsv(*c)
    else:
        h, s_, v = float(rng.uniform(0, 1)), float(rng.uniform(0.25, 1.0)), float(rng.uniform(0.2, 1.0)); c = np.array(colorsys.hsv_to_rgb(h, s_, v)); name = None
    a = {"mode": mode, "color_name": name, "rgb": [round(float(x), 4) for x in c], "hsv": [round(float(h), 4), round(float(s_), 4), round(float(v), 4)], "strength": round(float(strength), 3)}
    if wear: a.update(roughness=round(float(rng.uniform(0.15, 0.85)), 3), metallic=round(float(rng.choice([0.0, 0.0, rng.uniform(0.2, 0.8)])), 3))
    return a

def _lum(c): return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]

def collect_paint_slots(stage, root):
    """Find colour / roughness / metallic inputs of every shader under root (works for OmniPBR, UsdPreviewSurface (+textures), Squad MDL)."""
    from pxr import Usd, UsdShade, Sdf, Gf
    slots = []
    for p in Usd.PrimRange(stage.GetPrimAtPath(root)):
        if not p.IsA(UsdShade.Shader): continue
        sh = UsdShade.Shader(p); sid = sh.GetIdAttr().Get(); src = str(p.GetAttribute("info:mdl:sourceAsset").Get() or "")
        def inp(n, t):
            i = sh.GetInput(n)
            return i if i else sh.CreateInput(n, t)
        if "OmniPBR" in src:
            i = sh.GetInput("diffuse_texture"); tex = bool(i and i.Get() and str(i.Get().path))
            if tex: slots.append(("tint", inp("diffuse_tint", Sdf.ValueTypeNames.Color3f).GetAttr(), (1.0, 1.0, 1.0)))
            else:
                a = inp("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).GetAttr(); v = a.Get(); slots.append(("const", a, tuple(v) if v is not None else (0.2, 0.2, 0.2)))
            for n, d, kind in (("reflection_roughness_constant", 0.5, "rough"), ("metallic_constant", 0.0, "metal")):
                a = inp(n, Sdf.ValueTypeNames.Float).GetAttr(); v = a.Get(); slots.append((kind, a, float(v) if v is not None else d))
        elif sid == "UsdPreviewSurface":
            dc = sh.GetInput("diffuseColor")
            if dc and dc.HasConnectedSource():
                srcs = dc.GetConnectedSources()[0]
                for s_ in srcs:
                    tp = s_.source.GetPrim()
                    if UsdShade.Shader(tp).GetIdAttr().Get() == "UsdUVTexture":
                        a = UsdShade.Shader(tp).GetInput("scale"); a = a.GetAttr() if a else UsdShade.Shader(tp).CreateInput("scale", Sdf.ValueTypeNames.Float4).GetAttr()
                        v = a.Get(); slots.append(("texscale", a, tuple(v) if v is not None else (1.0, 1.0, 1.0, 1.0)))
            else:
                a = (dc or sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)).GetAttr(); v = a.Get(); slots.append(("const", a, tuple(v) if v is not None else (0.18, 0.18, 0.18)))
            for n, d, kind in (("roughness", 0.5, "rough"), ("metallic", 0.0, "metal")):
                i = sh.GetInput(n)
                if i and i.HasConnectedSource(): continue
                a = (i or sh.CreateInput(n, Sdf.ValueTypeNames.Float)).GetAttr(); v = a.Get(); slots.append((kind, a, float(v) if v is not None else d))
        else:
            i = sh.GetInput("Albedo_colour")   # Squad / Unreal-style MDL (Ghost): float4 albedo multiplier
            if i: v = i.Get(); slots.append(("tint4", i.GetAttr(), tuple(v) if v is not None else (1.0, 1.0, 1.0, 0.0)))
    return slots

def apply_appearance(slots, ap):
    """Set (or reset when ap is None / mode none) every slot. Dark parts (rotors, motors, luminance < 0.05) keep their colour."""
    from pxr import Gf
    on = ap is not None and ap.get("mode", "none") != "none"
    c = np.array(ap["rgb"]) if on else None; st = ap.get("strength", 0.85) if on else 0.0
    for kind, a, o in slots:
        if kind == "const":
            v = np.array(o, float)
            if on and _lum(v) >= 0.05: v = (1 - st) * v + st * c
            a.Set(Gf.Vec3f(*[float(x) for x in v]))
        elif kind in ("tint", "texscale", "tint4"):
            v = np.array(o[:3], float)
            if on: v = (1 - st) * v + st * c * (2.5 if kind == "tint4" else 1.6)   # multiplies a (dark) texture: boost so the paint shows
            v = np.clip(v, 0, 2)
            if kind == "tint": a.Set(Gf.Vec3f(*[float(x) for x in v]))
            else: a.Set(Gf.Vec4f(float(v[0]), float(v[1]), float(v[2]), float(o[3])))
        elif kind == "rough": a.Set(float(ap["roughness"]) if on and "roughness" in ap else float(o))
        elif kind == "metal": a.Set(float(ap["metallic"]) if on and "metallic" in ap else float(o))

SKIES = [S3 + "Assets/Skies/Clear/noon_grass_4k.hdr", S3 + "Assets/Skies/Clear/qwantani_4k.hdr",
         S3 + "Assets/Skies/Cloudy/kloofendal_48d_partly_cloudy_4k.hdr", S3 + "Assets/Skies/Cloudy/champagne_castle_1_4k.hdr"]
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:8.1f}s]", *a, flush=True)
peak = {"mb": 0, "cur": 0}
def _vram_poll():
    while True:
        try:
            o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
            peak["cur"] = int(o); peak["mb"] = max(peak["mb"], int(o))
        except Exception: pass
        time.sleep(2)
threading.Thread(target=_vram_poll, daemon=True).start()

def settle(app, k=60):
    from isaacsim.core.utils.stage import is_stage_loading
    n = 0
    while True:
        app.update(); n += 1
        if not is_stage_loading() and n > k: break

def open_layer(app, path, sublayers):
    import omni.usd
    from pxr import Sdf
    lay = Sdf.Layer.CreateNew(path) if not os.path.exists(path) else Sdf.Layer.FindOrOpen(path)
    lay.Clear()
    for s in sublayers: lay.subLayerPaths.append(s)
    lay.Save()
    ctx = omni.usd.get_context(); ctx.open_stage(path); settle(app); [app.update() for _ in range(30)]
    return ctx.get_stage()

def mesh_bounds(prim, max_ext=None):
    """World AABB of the renderable geometry under prim. Implicit gprims (Cylinder/Sphere/Cube/...) are measured from their
    radius/height/size via the extent plugins, because some assets (Isaac Quadcopter) carry stale 'extent' values."""
    from pxr import Usd, UsdGeom, Gf
    t = Usd.TimeCode.Default()
    c = UsdGeom.BBoxCache(t, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    lo = np.array([1e18] * 3); hi = -lo
    for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if not p.IsA(UsdGeom.Gprim): continue
        if UsdGeom.Imageable(p).ComputePurpose() == UsdGeom.Tokens.guide: continue
        r = None
        if not p.IsA(UsdGeom.PointBased):
            ext = UsdGeom.Boundable.ComputeExtentFromPlugins(UsdGeom.Boundable(p), t)
            if ext is not None and len(ext) == 2:
                r = Gf.BBox3d(Gf.Range3d(Gf.Vec3d(*ext[0]), Gf.Vec3d(*ext[1])), UsdGeom.Xformable(p).ComputeLocalToWorldTransform(t)).ComputeAlignedRange()
        if r is None: r = c.ComputeWorldBound(p).ComputeAlignedRange()
        if r.IsEmpty(): continue
        a, b = np.array(r.GetMin()), np.array(r.GetMax())
        if max_ext and (b - a).max() > max_ext: continue
        lo = np.minimum(lo, a); hi = np.maximum(hi, b)
    return lo, hi

def disable_physics(stage, root):
    from pxr import Usd, UsdPhysics, Sdf
    n = 0
    for p in Usd.PrimRange(stage.GetPrimAtPath(root)):
        if p.HasAPI(UsdPhysics.RigidBodyAPI): p.CreateAttribute("physics:rigidBodyEnabled", Sdf.ValueTypeNames.Bool).Set(False); n += 1
        if p.HasAPI(UsdPhysics.ArticulationRootAPI): p.CreateAttribute("physxArticulation:articulationEnabled", Sdf.ValueTypeNames.Bool).Set(False); n += 1
        if p.HasAPI(UsdPhysics.CollisionAPI): p.CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(False); n += 1
        if p.IsA(UsdPhysics.Scene): p.SetActive(False); n += 1
    return n

LITE_TYPES = ("arl",)   # models whose heavy meshes are decimated into a local cache layer (datasets\stages\<key>_lite_*.usdc)
LITE_MAX_FACES = int(os.environ.get("LITE_MAX_FACES", "60000"))
_url_cache = {}

def asset_url(k):
    """URL the pool references for model k: the original, or for LITE_TYPES a small local layer that references the original
    and overrides its heavy mesh with a vertex-clustered copy (built once, then cached; set LITE_ASSETS=0 to disable)."""
    if k not in _url_cache:
        u = ASSETS[k][0]
        if k in LITE_TYPES and os.environ.get("LITE_ASSETS", "1") != "0":
            try: u = lite_asset(u, k)
            except Exception as e: log("lite_asset failed for", k, repr(e), "-> using the original")
        _url_cache[k] = u
    return _url_cache[k]

def _decimate(P, F, max_faces):
    """Vertex clustering. P (n,3) points, F (m,3) triangles -> new points, new triangles, kept-face index (for GeomSubsets)."""
    lo = P.min(0); ext = float((P.max(0) - lo).max())
    for g in (384, 320, 256, 200, 160, 128, 100, 80):
        key = np.floor((P - lo) / (ext / g)).astype(np.int64).clip(0, g)
        k = (key[:, 0] * (g + 1) + key[:, 1]) * (g + 1) + key[:, 2]
        uq, inv = np.unique(k, return_inverse=True); inv = inv.ravel()
        T = inv[F]; keep = (T[:, 0] != T[:, 1]) & (T[:, 1] != T[:, 2]) & (T[:, 0] != T[:, 2])
        if keep.sum() <= max_faces: break
    cnt = np.bincount(inv, minlength=len(uq)).astype(np.float64)
    NP = np.stack([np.bincount(inv, P[:, d], len(uq)) / cnt for d in range(3)], 1)
    kf = np.nonzero(keep)[0]; T = T[kf]
    used = np.unique(T); remap = -np.ones(len(uq), np.int64); remap[used] = np.arange(len(used))
    return NP[used].astype(np.float32), remap[T].astype(np.int32), kf, g

def lite_asset(url, key):
    from pxr import Usd, UsdGeom, Sdf, Vt
    import hashlib
    out = os.path.join(DS, "stages", f"{key}_lite_{hashlib.md5(url.encode()).hexdigest()[:8]}_{LITE_MAX_FACES}.usdc")
    if os.path.exists(out): return out.replace("\\", "/")
    t0 = time.time(); src = Usd.Stage.Open(url); dp = src.GetDefaultPrim()
    heavy = [p for p in Usd.PrimRange(dp, Usd.TraverseInstanceProxies()) if p.IsA(UsdGeom.Mesh) and len(UsdGeom.Mesh(p).GetFaceVertexCountsAttr().Get() or []) > LITE_MAX_FACES]
    if not heavy: return url
    os.makedirs(os.path.dirname(out), exist_ok=True); tmp = out + ".tmp.usdc"
    lay = Sdf.Layer.CreateNew(tmp); ws = Usd.Stage.Open(lay)
    for kk in ("metersPerUnit", "upAxis"): ws.SetMetadata(kk, src.GetMetadata(kk))
    root = ws.DefinePrim(dp.GetPath(), dp.GetTypeName()); root.GetReferences().AddReference(url); ws.SetDefaultPrim(root)
    for m in heavy:
        anc = []; q = m
        while q and q != dp:
            if q.IsInstance(): anc.append(q.GetPath())
            q = q.GetParent()
        for a in sorted(anc, key=lambda x: len(str(x))): ws.OverridePrim(a).SetInstanceable(False)
        mm = UsdGeom.Mesh(m); fc = np.array(mm.GetFaceVertexCountsAttr().Get())
        if not (fc == 3).all(): log("lite_asset: non-triangle mesh skipped", m.GetPath()); continue
        P = np.array(mm.GetPointsAttr().Get(), np.float64); F = np.array(mm.GetFaceVertexIndicesAttr().Get(), np.int64).reshape(-1, 3)
        NP, NF, kf, g = _decimate(P, F, LITE_MAX_FACES)
        om = UsdGeom.Mesh(ws.OverridePrim(m.GetPath()))
        om.CreatePointsAttr().Set(Vt.Vec3fArray.FromNumpy(NP)); om.CreateFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.full(len(NF), 3, np.int32)))
        om.CreateFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(NF.ravel())); om.CreateExtentAttr().Set(Vt.Vec3fArray.FromNumpy(np.stack([NP.min(0), NP.max(0)])))
        for nm in ("primvars:st", "normals", "primvars:normals"):
            if m.GetAttribute(nm): ws.OverridePrim(m.GetPath()).CreateAttribute(nm, m.GetAttribute(nm).GetTypeName()).Block()
        fmap = -np.ones(len(F), np.int64); fmap[kf] = np.arange(len(kf))
        for sub in UsdGeom.Subset.GetAllGeomSubsets(mm):
            ix = fmap[np.array(sub.GetIndicesAttr().Get() or [], np.int64)]; ix = ix[ix >= 0].astype(np.int32)
            UsdGeom.Subset(ws.OverridePrim(sub.GetPrim().GetPath())).CreateIndicesAttr().Set(Vt.IntArray.FromNumpy(ix))
        log(f"lite_asset {key}: {m.GetPath()} {len(F)} -> {len(NF)} faces, {len(P)} -> {len(NP)} points (grid {g})")
    lay.Save(); del ws, lay
    os.replace(tmp, out); log("lite_asset written", out, f"{time.time() - t0:.1f}s")
    return out.replace("\\", "/")

def make_pool(stage, app, M, mix, nmax, paint, legacy=(400, 100, 100)):
    """mix None -> legacy pool (ghost/cf2x/quad counts from env POOL_GHOST/POOL_CF/POOL_QUAD or `legacy`); else enough prims of
    every type in the mix for nmax drones (POOL_EACH overrides). Returns Pool."""
    import math
    if mix is None:
        g = int(os.environ.get("POOL_GHOST", legacy[0])); cf = int(os.environ.get("POOL_CF", legacy[1])); q = int(os.environ.get("POOL_QUAD", legacy[2]))
        return Pool(stage, app, g, cf, q, M, paint=paint)
    each = os.environ.get("POOL_EACH")
    n = {k: (int(each) if each else int(min(nmax, math.ceil(nmax * p * 1.6) + 6))) for k, p in mix.items()}
    return Pool(stage, app, n.get("ghost", 0), n.get("cf2x", 0), n.get("quad", 0), M, n_extra={k: v for k, v in n.items() if k not in LEGACY_TYPES}, paint=paint)

def appearance_env():
    """COLOR_MODE (none|random|palette), COLOR_STRENGTH (0-1, default 0.85), COLOR_WEAR (1/0: random roughness/metallic)."""
    m = os.environ.get("COLOR_MODE", "none").strip().lower()
    if m not in ("none", "random", "palette"): raise ValueError("COLOR_MODE must be none|random|palette")
    return m, float(os.environ.get("COLOR_STRENGTH", "0.85")), os.environ.get("COLOR_WEAR", "1") != "0"

def short_ap(ap):
    return None if not ap or ap.get("mode") == "none" else {k: ap[k] for k in ("color_name", "rgb", "hsv", "strength", "roughness", "metallic") if k in ap}

class Pool:
    """Pre-built pool of drone prims /Root/DS/d_XXXX (Xform: translate, orient, scale) -> asset (reference, centring translate).
    n_extra: {"arl": n}; paint=True makes every drone individually re-colourable (no instancing)."""
    def __init__(self, stage, app, n_ghost, n_cf, n_quad, M, n_extra=None, paint=False):
        from pxr import Usd, UsdGeom, Gf, Sdf, UsdShade
        from isaacsim.core.utils.semantics import add_labels
        self.stage, self.M, self.paint = stage, M, paint
        counts = {"ghost": n_ghost, "cf2x": n_cf, "quad": n_quad}; counts.update(n_extra or {})
        used = [k for k in ASSETS if counts.get(k, 0) > 0]
        UsdGeom.Xform.Define(stage, "/Root/DS")
        # probe each used asset type once for its raw span and centre (stage units, scale 1)
        self.info = {}
        for k in used:
            url = asset_url(k)
            pp = f"/Root/DS/_probe_{k}"; UsdGeom.Xform.Define(stage, pp); a = UsdGeom.Xform.Define(stage, pp + "/asset"); a.GetPrim().GetReferences().AddReference(url)
        settle(app, 30)
        for k in used:
            pp = f"/Root/DS/_probe_{k}"
            disable_physics(stage, pp)
            try: amp = UsdGeom.GetStageMetersPerUnit(Usd.Stage.Open(ASSETS[k][0], Usd.Stage.LoadNone))
            except Exception: amp = 1.0
            lo, hi = mesh_bounds(stage.GetPrimAtPath(pp), max_ext=3.0 / amp)
            self.info[k] = {"center": ((lo + hi) / 2).tolist(), "size": (hi - lo).tolist(), "span": float(max(hi[0] - lo[0], hi[1] - lo[1]))}
            stage.RemovePrim(pp)
        log("asset raw sizes (stage units)", {k: [round(x, 2) for x in v["size"]] for k, v in self.info.items()})
        types = []
        for k in used: types += [k] * int(counts[k])
        rng = np.random.default_rng(0); rng.shuffle(types)
        self.types = types; self.paths = []; self.t_attr = []; self.o_attr = []; self.s_attr = []; self.vis = []
        for i, k in enumerate(types):
            p = f"/Root/DS/d_{i:04d}"; x = UsdGeom.Xform.Define(stage, p)
            xf = UsdGeom.Xformable(x)
            t = xf.AddTranslateOp(); o = xf.AddOrientOp(); s = xf.AddScaleOp()
            t.Set(Gf.Vec3d(float(i) * 100.0, 0, -1e6)); o.Set(Gf.Quatf(1, 0, 0, 0)); s.Set(Gf.Vec3f(1, 1, 1))
            cx = UsdGeom.Xform.Define(stage, p + "/c"); c = self.info[k]["center"]
            UsdGeom.Xformable(cx).AddTranslateOp().Set(Gf.Vec3d(-c[0], -c[1], -c[2]))
            a = UsdGeom.Xform.Define(stage, p + "/c/asset"); a.GetPrim().GetReferences().AddReference(asset_url(k))
            if ASSETS[k][2] and not paint: a.GetPrim().SetInstanceable(True)
            add_labels(x.GetPrim(), labels=["drone"], instance_name="class")
            self.paths.append(p); self.t_attr.append(t.GetAttr()); self.o_attr.append(o.GetAttr()); self.s_attr.append(s.GetAttr())
            self.vis.append(x.GetPrim().GetAttribute("visibility"))
        settle(app, 30)
        self.slots = [[] for _ in types]
        if paint:   # break instancing inside the assets so every drone owns its materials, then find the paintable inputs.
            # Safety net: instances with heavy geometry stay instanced (copying them costs VRAM per drone). Heavy meshes in
            # ASSETS (the ARL body, 370k faces) are already replaced by a decimated, non-instanced copy (lite_asset).
            heavy = {}
            def proto_faces(pr):
                key = str(pr.GetPath())
                if key not in heavy:
                    heavy[key] = sum(len(UsdGeom.Mesh(q).GetFaceVertexCountsAttr().Get() or []) for q in Usd.PrimRange(pr) if q.IsA(UsdGeom.Mesh))
                return heavy[key]
            kept = []
            for _ in range(3):
                n_inst = 0
                for p in Usd.PrimRange(stage.GetPrimAtPath("/Root/DS")):
                    if p.IsInstance():
                        if proto_faces(p.GetPrototype()) > HEAVY_FACES:
                            if p not in kept: kept.append(p)
                            continue
                        p.SetInstanceable(False); n_inst += 1
                if not n_inst: break
                settle(app, 10)
            if kept: log("WARNING paint: heavy instances kept instanced (these parts are not recoloured):", len(kept))
            for i, p in enumerate(self.paths): self.slots[i] = collect_paint_slots(stage, p)
            log("paint slots per type", {k: len(self.slots[types.index(k)]) for k in used})
        nph = disable_physics(stage, "/Root/DS"); log("pool built", len(types), {k: types.count(k) for k in used}, "physics attrs disabled", nph, "paint", paint)
        self.active = np.zeros(len(types), bool)

    def scale_for(self, i, span_m):
        return span_m * self.M / self.info[self.types[i]]["span"]

    def set_appearance(self, idx, aps):
        """aps[j] (dict from sample_appearance, or None) for pool index idx[j]."""
        if not self.paint: return
        from pxr import Sdf
        with Sdf.ChangeBlock():
            for i, ap in zip(idx, aps): apply_appearance(self.slots[int(i)], ap)

    def set(self, idx, pos_m, quats_wxyz, spans_m):
        """Activate drones idx (pool indices) at positions (metres, world); others hidden."""
        from pxr import Gf, Sdf
        idx = list(idx); on = set(idx)
        # inactive drones stay visible but are parked 10 km below the terrain, beyond the camera far clip
        with Sdf.ChangeBlock():
            for j, i in enumerate(idx):
                self.t_attr[i].Set(Gf.Vec3d(*(np.asarray(pos_m[j], float) * self.M)))
                q = quats_wxyz[j]; self.o_attr[i].Set(Gf.Quatf(float(q[0]), float(q[1]), float(q[2]), float(q[3])))
                s = self.scale_for(i, spans_m[j]); self.s_attr[i].Set(Gf.Vec3f(s, s, s))
            for i in np.nonzero(self.active)[0]:
                if i not in on: self.t_attr[i].Set(Gf.Vec3d(float(i) * 100.0, 0, -1e6))
        self.active[:] = False; self.active[idx] = True

def quat_from_euler_deg(roll, pitch, yaw):
    r, p, y = np.radians([roll, pitch, yaw]) / 2
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([cr*cp*cy + sr*sp*sy, sr*cp*cy - cr*sp*sy, cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy])

def look_at_matrix(eye_m, tgt_m, M):
    from pxr import Gf
    up = Gf.Vec3d(0, 0, 1)
    e = Gf.Vec3d(*(np.asarray(eye_m) * M)); t = Gf.Vec3d(*(np.asarray(tgt_m) * M))
    if abs((t - e).GetNormalized()[2]) > 0.995: up = Gf.Vec3d(0, 1, 0)
    return Gf.Matrix4d().SetLookAt(e, t, up).GetInverse()

class HeightMap:
    def __init__(self, path):
        z = np.load(path); self.h = z["h"]; self.x0, self.y0, self.res = float(z["x0"]), float(z["y0"]), float(z["res"])
        self.ground = z["ground"]; self.valid = z["valid"]; self.inner = z["valid_inner"]
        iy, ix = np.nonzero(self.inner); self._inner_xy = np.stack([self.x0 + (ix + 0.5) * self.res, self.y0 + (iy + 0.5) * self.res], 1)
        self.center = self._inner_xy.mean(0)
    def sample_xy(self, rng):
        return self._inner_xy[rng.integers(len(self._inner_xy))] + rng.uniform(-0.5, 0.5, 2) * self.res
    def __call__(self, x, y, which="h"):
        a = self.h if which == "h" else self.ground
        i = np.clip(((np.asarray(y) - self.y0) / self.res).astype(int), 0, a.shape[0] - 1)
        j = np.clip(((np.asarray(x) - self.x0) / self.res).astype(int), 0, a.shape[1] - 1)
        return a[i, j]
