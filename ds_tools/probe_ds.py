"""Probe: scene bounds, height maps (full + terrain-only), sky dome look, and a drone-count stress test (VRAM / time / label sanity)."""
import os, sys, json, time, math
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "width": 1280, "height": 720})
from common import *
import omni.usd, omni.replicator.core as rep
from pxr import Usd, UsdGeom, UsdLux, Gf, Sdf
from PIL import Image, ImageDraw
OUT = os.path.join(DS, "_probe"); os.makedirs(OUT, exist_ok=True); os.makedirs(os.path.join(DS, "stages"), exist_ok=True)
res = {}
stage = open_layer(app, os.path.join(DS, "stages", "probe.usda"), [DESERT])
mpu = UsdGeom.GetStageMetersPerUnit(stage); M = 1.0 / mpu
lo, hi = mesh_bounds(stage.GetPrimAtPath("/Root/Landscape"), max_ext=20000 * M); lo, hi = lo * mpu, hi * mpu
slo, shi = mesh_bounds(stage.GetPrimAtPath("/Root/Statics"), max_ext=2000 * M); slo, shi = slo * mpu, shi * mpu
res["landscape_bounds_m"] = [lo.tolist(), hi.tolist()]; res["statics_bounds_m"] = [slo.tolist(), shi.tolist()]
log("landscape", lo.round(1), hi.round(1), "statics", slo.round(1), shi.round(1))
# ---------------- height maps ----------------
HW = 1024; FOOT = 400.0
cam = UsdGeom.Camera.Define(stage, "/Root/HMCam"); cam.CreateFocalLengthAttr(18.0); cam.CreateHorizontalApertureAttr(20.955); cam.CreateVerticalApertureAttr(20.955)
cam.CreateClippingRangeAttr(Gf.Vec2f(1.0 * M, 6000.0 * M)); cxf = UsdGeom.Xformable(cam); cop = cxf.AddTransformOp()
rp = rep.create.render_product(str(cam.GetPath()), (HW, HW))
a_d = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane"); a_d.attach(rp)
a_cp = rep.AnnotatorRegistry.get_annotator("camera_params"); a_cp.attach(rp)
zmax = max(hi[2], shi[2]); camz = zmax + 300.0
f = 20.955 * 300.0 / FOOT; cam.GetFocalLengthAttr().Set(f)
x0, y0 = min(lo[0], slo[0]), min(lo[1], slo[1]); x1, y1 = max(hi[0], shi[0]), max(hi[1], shi[1])
# restrict to the area that actually has statics +100 m margin (landscape may be much larger)
ax0, ay0, ax1, ay1 = lo[0], lo[1], hi[0], hi[1]
RES = 1.0
nx, ny = int(math.ceil((ax1 - ax0) / RES)), int(math.ceil((ay1 - ay0) / RES))
log("heightmap area", [ax0, ay0, ax1, ay1], "grid", nx, ny)
def render_hm():
    H = np.full((ny, nx), -1e9, np.float32)
    xs = np.arange(ax0 + FOOT / 2 - 20, ax1 + FOOT / 2, FOOT - 40); ys = np.arange(ay0 + FOOT / 2 - 20, ay1 + FOOT / 2, FOOT - 40)
    for cx in xs:
        for cy in ys:
            cop.Set(look_at_matrix([cx, cy, camz], [cx, cy, camz - 10], M))
            for _ in range(3): app.update()
            rep.orchestrator.step(rt_subframes=2)
            d = np.asarray(a_d.get_data()).astype(np.float64); cp = a_cp.get_data()
            V = np.asarray(cp["cameraViewTransform"]).reshape(4, 4); fx = HW * f / 20.955
            v, u = np.mgrid[0:HW, 0:HW]
            ok = np.isfinite(d) & (d > 0) & (d < 5000)
            xc = (u + 0.5 - HW / 2) / fx * d; yc = -(v + 0.5 - HW / 2) / fx * d; zc = -d
            P = np.stack([xc[ok] * M, yc[ok] * M, zc[ok] * M, np.ones(ok.sum())], 1) @ np.linalg.inv(V)
            wx, wy, wz = P[:, 0] * mpu, P[:, 1] * mpu, P[:, 2] * mpu
            j = ((wx - ax0) / RES).astype(int); i = ((wy - ay0) / RES).astype(int); k = (j >= 0) & (j < nx) & (i >= 0) & (i < ny)
            np.maximum.at(H, (i[k], j[k]), wz[k].astype(np.float32))
    return H
t = time.time(); Hfull = render_hm(); log("full heightmap", round(time.time() - t, 1), "s")
st = stage.GetPrimAtPath("/Root/Statics"); UsdGeom.Imageable(st).MakeInvisible()
fol = [p for p in Usd.PrimRange(stage.GetPrimAtPath("/Root/Landscape")) if "Foliage" in p.GetName() or "HierarchicalInstanced" in p.GetName()]
for p in fol: UsdGeom.Imageable(p).MakeInvisible()
t = time.time(); Hground = render_hm(); log("terrain heightmap", round(time.time() - t, 1), "s", "foliage prims hidden", len(fol))
UsdGeom.Imageable(st).MakeVisible()
for p in fol: UsdGeom.Imageable(p).MakeVisible()
np.savez_compressed(os.path.join(OUT, "hm_raw.npz"), h=Hfull, g=Hground)
for H in (Hfull, Hground):   # fill holes with nearest-ish (row/col max filter few passes)
    bad = H < -1e8
    log("holes", int(bad.sum()), "of", H.size)
    for _ in range(20):
        if not bad.any(): break
        Hp = np.pad(H, 1, constant_values=-1e9); nb = np.max(np.stack([Hp[1:-1, :-2], Hp[1:-1, 2:], Hp[:-2, 1:-1], Hp[2:, 1:-1]]), 0)
        H[bad] = nb[bad]; bad = H < -1e8
    H[bad] = np.median(H[~bad])
np.savez_compressed(os.path.join(DS, "stages", "heightmap.npz"), h=Hfull, ground=Hground, x0=ax0, y0=ay0, res=RES)
def hmimg(H, fn):
    a = (H - np.percentile(H, 1)) / (np.percentile(H, 99.5) - np.percentile(H, 1) + 1e-6)
    Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)[::-1]).save(os.path.join(OUT, fn))
hmimg(Hfull, "heightmap_full.png"); hmimg(Hground, "heightmap_ground.png")
obj = Hfull - Hground
res["heightmap"] = {"area_m": [ax0, ay0, ax1, ay1], "grid": [nx, ny], "ground_z_range": [float(Hground.min()), float(Hground.max())],
                    "object_height_p99": float(np.percentile(obj, 99)), "frac_open_ground(<0.5m)": float((obj < 0.5).mean()), "frac_roof(>3m)": float((obj > 3).mean())}
log("heightmap", res["heightmap"])
a_d.detach(); a_cp.detach(); rp.destroy()
# ---------------- sky dome ----------------
dome = UsdLux.DomeLight.Define(stage, "/Root/DS_Sky"); dome.CreateTextureFileAttr(SKIES[0]); dome.CreateTextureFormatAttr("latlong"); dome.CreateIntensityAttr(1000.0)
# ---------------- pool + stress ----------------
t = time.time(); pool = Pool(stage, app, 900, 150, 150, M); res["pool_build_s"] = round(time.time() - t, 1); res["asset_info"] = pool.info
log("pool build", res["pool_build_s"], "s vram", peak["cur"])
gc = np.array([(slo[0] + shi[0]) / 2, (slo[1] + shi[1]) / 2]); hm = HeightMap(os.path.join(DS, "stages", "heightmap.npz"))
gz = float(hm(gc[0], gc[1], "ground"))
rc = UsdGeom.Camera.Define(stage, "/Root/DSCam"); rc.CreateFocalLengthAttr(18.0); rc.CreateHorizontalApertureAttr(20.955); rc.CreateVerticalApertureAttr(20.955 * 720 / 1280)
rc.CreateClippingRangeAttr(Gf.Vec2f(0.1 * M, 8000.0 * M)); rcop = UsdGeom.Xformable(rc).AddTransformOp()
rp = rep.create.render_product(str(rc.GetPath()), (1280, 720))
anns = {}
for nm, ip in [("rgb", None), ("instance_segmentation_fast", {"colorize": False}), ("bounding_box_2d_tight_fast", None), ("bounding_box_2d_loose_fast", None), ("instance_id_segmentation_fast", {"colorize": False}), ("occlusion", None), ("camera_params", None)]:
    try:
        a = rep.AnnotatorRegistry.get_annotator(nm, init_params=ip) if ip else rep.AnnotatorRegistry.get_annotator(nm); a.attach(rp); anns[nm] = a
    except Exception as e: log("annotator fail", nm, e)
rng = np.random.default_rng(1)
res["stress"] = []
for N, sky_up in [(50, False), (200, False), (500, False), (1000, False), (300, True)]:
    idx = rng.choice(len(pool.types), N, replace=False)
    R = 2.2 * N ** (1 / 3)
    sc = np.array([gc[0], gc[1], gz + 40.0])
    pts = []
    while len(pts) < N:
        p = sc + rng.uniform(-1, 1, 3) * np.array([R, R, R * 0.5])
        pts.append(p)
    pos = np.array(pts); quats = [quat_from_euler_deg(rng.uniform(-8, 8), rng.uniform(-8, 8), rng.uniform(0, 360)) for _ in range(N)]
    spans = rng.uniform(0.3, 1.0, N)
    t = time.time(); pool.set(idx, pos, quats, spans); t_set = time.time() - t
    if sky_up: eye = sc + np.array([-25, -10, -38.3]); tgt = sc   # camera ~1.7 m above ground looking up
    else: eye = sc + np.array([-(R + 40), -(R + 30), 25]); tgt = sc
    rcop.Set(look_at_matrix(eye, tgt, M))
    times = []
    for k in range(3):
        t = time.time(); app.update(); rep.orchestrator.step(rt_subframes=4); times.append(time.time() - t)
    t = time.time()
    img = np.asarray(anns["rgb"].get_data())[..., :3].astype(np.uint8)
    seg = anns["instance_segmentation_fast"].get_data(); s = np.asarray(seg["data"]); i2l = seg["info"]["idToLabels"]
    bb = anns["bounding_box_2d_tight_fast"].get_data(); bl = anns["bounding_box_2d_loose_fast"].get_data()
    iid = anns["instance_id_segmentation_fast"].get_data(); oc = anns["occlusion"].get_data()
    t_read = time.time() - t
    seg_drone_ids = [k for k, v in i2l.items() if "/Root/DS/d_" in str(v)]
    bpaths = bb["info"].get("primPaths", []); nbd = sum("/Root/DS/d_" in p for p in bpaths)
    iid_map = iid["info"]["idToLabels"]; iid_drone = [k for k, v in iid_map.items() if "/Root/DS/d_" in str(v)]
    ocd = oc["data"] if isinstance(oc, dict) else oc; ocd = np.asarray(ocd)
    occ_drone = [float(o["occlusionRatio"]) for o in ocd if str(int(o["instanceId"])) in set(iid_drone)]
    ent = {"N": N, "sky_up": sky_up, "set_s": round(t_set, 2), "step_s": [round(x, 2) for x in times], "read_s": round(t_read, 2), "vram_cur": peak["cur"], "vram_peak": peak["mb"],
           "seg_instances_drone": len(seg_drone_ids), "seg_label_examples": [str(i2l[k]) for k in seg_drone_ids[:2]], "bbox_drone": nbd, "bbox_examples": bpaths[:2],
           "iid_drone_entries": len(iid_drone), "iid_examples": [str(iid_map[k]) for k in iid_drone[:3]], "occ_drone_n": len(occ_drone),
           "occ_examples": occ_drone[:8], "mean_rgb": float(img.mean()), "n_iid_entries": len(iid_map)}
    res["stress"].append(ent); log("STRESS", ent)
    ov = Image.fromarray(img.copy()); dr = ImageDraw.Draw(ov)
    for b, pth in zip(bb["data"], bpaths):
        if "/Root/DS/d_" in pth: dr.rectangle([int(b["x_min"]), int(b["y_min"]), int(b["x_max"]), int(b["y_max"])], outline=(255, 40, 40), width=1)
    ov.save(os.path.join(OUT, f"stress_{N}{'_skyup' if sky_up else ''}.jpg"), quality=90)
    json.dump(res, open(os.path.join(OUT, "probe.json"), "w"), indent=1, default=str)
log("DONE peak VRAM", peak["mb"])
json.dump(res, open(os.path.join(OUT, "probe.json"), "w"), indent=1, default=str)
app.close()
