"""Scale feasibility test: N drones as ONE UsdGeom.PointInstancer (one prototype model), rendered headless with Replicator
annotators (rgb, semantic_segmentation, instance_id_segmentation_fast, bounding_box_2d_tight_fast, camera_params).

    python ds_tools/scale_test.py --n 1000,10000 --frames 6 --model cf2x --views fill,far
    python ds_tools/scale_test.py --n 100000 --frames 4 --light      # 100k: per-instance id / box annotators stall, use --light

For every N and view: swarm = uniform random blob (spacing --spacing m), moving at 8 m/s; camera = ground observer 2 m above the
terrain looking up 20 deg at the swarm (sky background) and following it.
  fill: the swarm fills ~60 % of the image width;  far: camera 4x farther (drones shrink towards / below one pixel)
Writes <out>/<model>_N<n>_<view>/ : rgb_XXX.jpg, swarm_mask_XXX.png (Replicator semantic mask, 255 = drone),
density_XXX.npy/.jpg (drone count per pixel from projected 3D positions, Gaussian sigma 1.5 px), stats.json; <out>/summary.json.
Default <out> = datasets/_scale_test. Measures s/frame (render step + annotator read) and GPU memory (nvidia-smi, 2 s poll)."""
import os, sys, json, time, math, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument("--n", default="1000,10000,100000"); ap.add_argument("--frames", type=int, default=6)
ap.add_argument("--model", default="cf2x"); ap.add_argument("--views", default="fill,far"); ap.add_argument("--spacing", type=float, default=3.0)
ap.add_argument("--span", type=float, default=0.5, help="drone size (m)"); ap.add_argument("--rt", type=int, default=2); ap.add_argument("--out", default="")
ap.add_argument("--light", action="store_true", help="only rgb + semantic_segmentation + camera_params (no per-instance ids / boxes)")
a = ap.parse_args()
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
from isaacsim import SimulationApp
W, H = 1280, 720
app = SimulationApp({"headless": True, "width": W, "height": H})
from common import *
import omni.replicator.core as rep
from pxr import Usd, UsdGeom, UsdLux, Gf, Sdf, Vt
from PIL import Image
from isaacsim.core.utils.semantics import add_labels
OUT = a.out or os.path.join(DS, "_scale_test"); os.makedirs(OUT, exist_ok=True)
log("start vram", peak["cur"])
stage = open_layer(app, os.path.join(DS, "stages", f"scale_{os.getpid()}.usda"), [DESERT])
mpu = UsdGeom.GetStageMetersPerUnit(stage); M = 1.0 / mpu
hm = HeightMap(os.path.join(DS, "stages", "heightmap.npz"))
dome = UsdLux.DomeLight.Define(stage, "/Root/DS_Sky"); dome.CreateTextureFileAttr(SKIES[0]); dome.CreateTextureFormatAttr("latlong"); dome.CreateIntensityAttr(800.0)
# prototype: drone asset, centred, labelled class=drone
inst = UsdGeom.PointInstancer.Define(stage, "/Root/Swarm")
pp = "/Root/Swarm/Protos/drone"; UsdGeom.Scope.Define(stage, "/Root/Swarm/Protos"); UsdGeom.Xform.Define(stage, pp)
cx = UsdGeom.Xform.Define(stage, pp + "/c"); asset = UsdGeom.Xform.Define(stage, pp + "/c/asset"); asset.GetPrim().GetReferences().AddReference(asset_url(a.model))
settle(app, 30); disable_physics(stage, "/Root/Swarm")
try: amp = UsdGeom.GetStageMetersPerUnit(Usd.Stage.Open(ASSETS[a.model][0], Usd.Stage.LoadNone))
except Exception: amp = 1.0
lo, hi = mesh_bounds(stage.GetPrimAtPath(pp), max_ext=3.0 / amp); ctr = (lo + hi) / 2; span0 = float(max(hi[0] - lo[0], hi[1] - lo[1]))
UsdGeom.Xformable(cx).AddTranslateOp().Set(Gf.Vec3d(*(-ctr)))
add_labels(stage.GetPrimAtPath(pp), labels=["drone"], instance_name="class")
inst.CreatePrototypesRel().SetTargets([Sdf.Path(pp)])
SC = a.span * M / span0
log("prototype", a.model, "raw span", round(span0, 3), "scale", round(SC, 4))
cam = UsdGeom.Camera.Define(stage, "/Root/DSCam"); FOC = 18.0; cam.CreateFocalLengthAttr(FOC); cam.CreateHorizontalApertureAttr(20.955); cam.CreateVerticalApertureAttr(20.955 * H / W)
cam.CreateClippingRangeAttr(Gf.Vec2f(0.1 * M, 6000.0 * M)); cam_op = UsdGeom.Xformable(cam).AddTransformOp()
rp = rep.create.render_product(str(cam.GetPath()), (W, H)); A = {}
for nm, ip in [("rgb", None), ("semantic_segmentation", {"colorize": False}), ("instance_id_segmentation_fast", {"colorize": False}), ("bounding_box_2d_tight_fast", None), ("camera_params", None)]:
    if a.light and nm in ("instance_id_segmentation_fast", "bounding_box_2d_tight_fast"): continue
    an = rep.AnnotatorRegistry.get_annotator(nm, init_params=ip) if ip else rep.AnnotatorRegistry.get_annotator(nm); an.attach(rp); A[nm] = an
inst.CreatePositionsAttr().Set(Vt.Vec3fArray.FromNumpy(np.zeros((1, 3), np.float32))); inst.CreateProtoIndicesAttr().Set(Vt.IntArray([0]))
for _ in range(30): app.update()
rep.orchestrator.step(rt_subframes=4)
base_mb = peak["cur"]; log("scene ready vram", base_mb)
TAN = 20.955 / 2 / FOC; FX = FOC / 20.955 * W

def project(Pw, V, Pj):
    X = np.concatenate([Pw * M, np.ones((len(Pw), 1))], 1) @ V
    depth = -X[:, 2] * mpu; Y = X @ Pj; ndc = Y[:, :2] / Y[:, 3:4]
    return np.stack([(ndc[:, 0] + 1) / 2 * W, (1 - ndc[:, 1]) / 2 * H], 1), depth

def blur(img, s=1.5):
    r = int(3 * s); k = np.exp(-0.5 * (np.arange(-r, r + 1) / s) ** 2); k /= k.sum()
    img = np.apply_along_axis(lambda v: np.convolve(v, k, "same"), 0, img); return np.apply_along_axis(lambda v: np.convolve(v, k, "same"), 1, img)

summary = {} if os.environ.get("SCALE_FRESH") == "1" else None
if summary is None: summary = (json.load(open(os.path.join(OUT, "summary.json"))) if os.path.exists(os.path.join(OUT, "summary.json")) else {})
rng = np.random.default_rng(0)
C0 = np.array([hm.center[0], hm.center[1], float(hm(hm.center[0], hm.center[1], "ground")) + 60.0])
for N in [int(x) for x in a.n.split(",")]:
    R = a.spacing * N ** (1 / 3) * 0.62 + a.spacing
    u = rng.normal(size=(N, 3)); u /= np.linalg.norm(u, axis=1, keepdims=True); rr = rng.random(N) ** (1 / 3)
    O = u * rr[:, None] * np.array([R, R, 0.5 * R]); C0n = C0.copy(); C0n[2] = max(C0[2], float(hm(C0[0], C0[1], "ground")) + 0.5 * R + 15)
    for view in a.views.split(","):
        tag = f"{a.model}_N{N}_{view}" + ("_light" if a.light else ""); od = os.path.join(OUT, tag); os.makedirs(od, exist_ok=True)
        st = dict(light=a.light, model=a.model, n=N, view=view, swarm_radius_m=round(R, 1), spacing_m=a.spacing, drone_span_m=a.span, rt_subframes=a.rt, frames=a.frames)
        try:
            t0 = time.time()
            q = np.zeros((N, 4), np.float16); q[:, 0] = 1
            inst.GetProtoIndicesAttr().Set(Vt.IntArray.FromNumpy(np.zeros(N, np.int32)))
            inst.CreateOrientationsAttr().Set(Vt.QuathArray.FromNumpy(q)); inst.CreateScalesAttr().Set(Vt.Vec3fArray.FromNumpy(np.full((N, 3), SC, np.float32)))
            # ground observer looking UP at the swarm (sky background, like a counter-drone camera): eye 2 m above the terrain
            dist = R / (0.6 * TAN) * (4.0 if view == "far" else 1.0); el = math.radians(20)
            dirv = np.array([math.cos(el) * math.cos(0.7), math.cos(el) * math.sin(0.7), -math.sin(el)])
            e0 = C0n[:2] + dirv[:2] * dist; C0n = C0n.copy(); C0n[2] = float(hm(e0[0], e0[1], "h")) + 2 + dist * math.sin(el)
            jit = np.zeros((N, 3)); times = []; peak["mb"] = peak["cur"]; per = []
            for f in range(a.frames + 1):   # frame 0 = warm-up (not timed)
                cen = C0n + np.array([0.8 * f, 0, 0]); jit = 0.9 * jit + rng.normal(0, 0.05, (N, 3)); Pw = cen + O + jit
                ts = time.time()
                inst.GetPositionsAttr().Set(Vt.Vec3fArray.FromNumpy((Pw * M).astype(np.float32)))
                eye = cen + dirv * dist
                cam_op.Set(look_at_matrix(eye, cen, M)); app.update(); rep.orchestrator.step(rt_subframes=a.rt if f else 8)
                rgb = np.asarray(A["rgb"].get_data())[..., :3].astype(np.uint8)
                sd = A["semantic_segmentation"].get_data(); seg = np.asarray(sd["data"]).astype(np.int64)
                did = [int(k) for k, v in sd["info"]["idToLabels"].items() if "drone" in str(v)]
                smask = np.isin(seg, did)
                iseg = np.asarray(A["instance_id_segmentation_fast"].get_data()["data"]).astype(np.int64) if not a.light else None
                bb = A["bounding_box_2d_tight_fast"].get_data() if not a.light else None; cp = A["camera_params"].get_data()
                te = time.time() - ts
                if f: times.append(te)
                V = np.asarray(cp["cameraViewTransform"], float).reshape(4, 4); Pj = np.asarray(cp["cameraProjection"], float).reshape(4, 4)
                uv, depth = project(Pw, V, Pj); inf = (depth > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
                span_px = FX * a.span / np.maximum(depth[inf], 1e-3)
                ys, xs = np.nonzero(smask)
                rec = dict(frame=f, s=round(te, 3), drone_px=int(smask.sum()), n_in_frame=int(inf.sum()), span_px_median=round(float(np.median(span_px)), 2) if inf.any() else None,
                           frac_drones_below_1px=round(float((span_px < 1).mean()), 3) if inf.any() else None,
                           n_instance_ids_on_drone_px=(int(len(np.unique(iseg[smask]))) if smask.any() else 0) if iseg is not None else None, n_bbox_entries=int(len(bb["data"])) if bb is not None else None,
                           box_from_replicator_mask=[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else None,
                           box_from_projected_points=[int(uv[inf, 0].min()), int(uv[inf, 1].min()), int(uv[inf, 0].max()), int(uv[inf, 1].max())] if inf.any() else None)
                if f in (1, a.frames):
                    Image.fromarray(rgb).save(os.path.join(od, f"rgb_{f:03d}.jpg"), quality=92)
                    Image.fromarray((smask * 255).astype(np.uint8)).save(os.path.join(od, f"swarm_mask_{f:03d}.png"), optimize=True)
                    dens = np.zeros((H, W)); ii = uv[inf].astype(int); np.add.at(dens, (ii[:, 1], ii[:, 0]), 1.0); dens = blur(dens)
                    np.save(os.path.join(od, f"density_{f:03d}.npy"), dens.astype(np.float32))
                    v = (255 * np.clip(dens / max(dens.max(), 1e-6), 0, 1) ** 0.5).astype(np.uint8); Image.fromarray(v).save(os.path.join(od, f"density_{f:03d}.jpg"), quality=90)
                    rec["density_sum"] = round(float(dens.sum()), 1)
                per.append(rec); log(tag, rec)
            st.update(s_per_frame_mean=round(float(np.mean(times)), 3), s_per_frame_max=round(float(np.max(times)), 3), setup_s=round(time.time() - t0 - sum(times), 1),
                      gpu_mem_peak_mb=peak["mb"], gpu_mem_scene_ready_mb=base_mb, per_frame=per, ok=True)
        except Exception as e:
            import traceback; st.update(ok=False, error=repr(e), trace=traceback.format_exc()[-2000:], gpu_mem_peak_mb=peak["mb"]); log(tag, "FAILED", repr(e))
        json.dump(st, open(os.path.join(od, "stats.json"), "w"), indent=1); summary[tag] = {k: v for k, v in st.items() if k != "per_frame"}
        if st.get("per_frame"): summary[tag]["last_frame"] = st["per_frame"][-1]
        json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
log("DONE"); app.close()
