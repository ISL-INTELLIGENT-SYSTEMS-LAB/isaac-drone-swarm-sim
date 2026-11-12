import json, numpy as np, os, colorsys
from PIL import Image, ImageDraw, ImageFont
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; from matplotlib import cm
R = "/workspace/isl/raw/ao"; O = "/workspace/renders/all_outputs"
F = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"; FB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
f18 = ImageFont.truetype(F, 18); f16 = ImageFont.truetype(F, 16); fb = ImageFont.truetype(FB, 20)
J = lambda n: json.load(open(f"{R}/{n}.json")); Z = lambda n: np.load(f"{R}/{n}.npz")
rgb = np.array(Image.open(f"{R}/rgb.png").convert("RGB")); H, W = rgb.shape[:2]
cp = J("camera_params"); V = np.array(cp["cameraViewTransform"]).reshape(4, 4); P = np.array(cp["cameraProjection"]).reshape(4, 4)
def proj(X):   # X (N,3) stage units (cm) -> pixels
    h = np.c_[X, np.ones(len(X))] @ V @ P; ndc = h[:, :3] / h[:, 3:4]
    return np.c_[(ndc[:, 0] + 1) / 2 * W, (1 - ndc[:, 1]) / 2 * H], h[:, 3]
def save(name, im): Image.fromarray(im) if isinstance(im, np.ndarray) else im; (Image.fromarray(im) if isinstance(im, np.ndarray) else im).save(f"{O}/{name}.png"); print("saved", name)
def colorize(a, cmap="turbo", lo=None, hi=None, log=False):
    a = a.astype(np.float64); f = np.isfinite(a); b = np.log(np.clip(a, 1e-3, None)) if log else a
    lo = np.nanpercentile(b[f], 1) if lo is None else lo; hi = np.nanpercentile(b[f], 99) if hi is None else hi
    n = np.clip((b - lo) / (hi - lo + 1e-9), 0, 1); out = (matplotlib.colormaps[cmap](n)[..., :3] * 255).astype(np.uint8); out[~f] = 0; return out
def legend(im, items, xy=(10, 10)):
    im = Image.fromarray(im) if isinstance(im, np.ndarray) else im; d = ImageDraw.Draw(im); x, y = xy
    for t, c in items:
        d.rectangle([x, y, x + 18, y + 18], fill=tuple(int(v) for v in c)); d.text((x + 24, y - 1), t, font=f18, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0)); y += 24
    return im
save("01_rgb", rgb)
# depth
for k in ["distance_to_camera", "distance_to_image_plane"]:
    d = Z(k)["data"]; im = colorize(d, "turbo_r", log=True)
    fin = d[np.isfinite(d)]
    im = legend(im, [(f"{k}: {fin.min():.1f} m (red/near) .. {fin.max():.1f} m (blue/far), log scale", (0, 0, 0))])
    save("02_" + k, im)
# normals
n = Z("normals")["data"][..., :3]; save("03_normals", ((n + 1) / 2 * 255).clip(0, 255).astype(np.uint8))
# semantic
ss = Z("semantic_segmentation")["data"]; sl = J("semantic_segmentation")["info"]["idToLabels"]
pal = {"BACKGROUND": (0, 0, 0), "UNLABELLED": (60, 60, 60), "drone": (255, 30, 30), "vegetation": (40, 170, 60), "ground": (190, 150, 90),
       "building": (90, 140, 230), "vehicle": (250, 200, 30), "road": (150, 150, 160), "wall_fence": (200, 90, 200)}
im = np.zeros((H, W, 3), np.uint8); used = []
for k, v in sl.items():
    c = pal.get(v["class"], (255, 255, 255)); m = ss == int(k)
    if m.any(): im[m] = c; used.append((f"{v['class']} ({100*m.mean():.1f}%)", c))
save("04_semantic_segmentation", legend(im, used))
# instance segmentation (per-labelled-prim) and instance id (per mesh)
def idcolor(ids):
    u = np.unique(ids); out = np.zeros(ids.shape + (3,), np.uint8); rng = np.random.default_rng(1)
    for x in u:
        if x == 0: continue
        out[ids == x] = (np.array(colorsys.hsv_to_rgb(rng.random(), 0.75, 1.0)) * 255).astype(np.uint8)
    return out, len(u)
ins = Z("instance_segmentation")["data"]; il = J("instance_segmentation")["info"]["idToLabels"]
im, nu = idcolor(ins); ndr = sum(1 for v in il.values() if "DroneSwarm" in str(v))
save("05_instance_segmentation", legend(im, [(f"{nu} instances (labelled prims), {ndr} drones", (0, 0, 0))]))
iid = Z("instance_id_segmentation")["data"]; im, nu = idcolor(iid)
save("06_instance_id_segmentation", legend(im, [(f"{nu} unique mesh instance IDs in view (every mesh, labelled or not)", (0, 0, 0))]))
# boxes 2D
def boxes(name):
    j = J(name); return j["data"], j["info"]["primPaths"], j["info"]["idToLabels"]
im = Image.fromarray((rgb * 0.8).astype(np.uint8)); d = ImageDraw.Draw(im)
for name, col, wd in [("bounding_box_2d_loose", (60, 255, 60), 2), ("bounding_box_2d_tight", (255, 40, 40), 2)]:
    bd, pp, lab = boxes(name)
    for b, p in zip(bd, pp):
        if "DroneSwarm" not in p: continue
        d.rectangle([b["x_min"], b["y_min"], b["x_max"], b["y_max"]], outline=col, width=wd)
        if name.endswith("tight"): d.text((b["x_min"], b["y_min"] - 18), p.split("/")[-1], font=f16, fill=(255, 255, 0), stroke_width=2, stroke_fill=(0, 0, 0))
save("07_bounding_box_2d_tight_loose", legend(im, [("tight 2D box (visible pixels)", (255, 40, 40)), ("loose 2D box (full extent incl. occluded)", (60, 255, 60))]))
# all-class tight boxes
im = Image.fromarray((rgb * 0.8).astype(np.uint8)); d = ImageDraw.Draw(im); bd, pp, lab = boxes("bounding_box_2d_tight"); cnt = {}
for b in bd:
    c = lab[str(b["semanticId"])]["class"]; cnt[c] = cnt.get(c, 0) + 1
    d.rectangle([b["x_min"], b["y_min"], b["x_max"], b["y_max"]], outline=pal.get(c, (255, 255, 255)), width=2)
save("08_bounding_box_2d_all_classes", legend(im, [(f"{k}: {v} boxes", pal.get(k, (255, 255, 255))) for k, v in cnt.items()]))
# 3D boxes
j = J("bounding_box_3d"); im = Image.fromarray(rgb.copy()); d = ImageDraw.Draw(im); nb = 0
E = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
for b, p in zip(j["data"], j["info"]["primPaths"]):
    c = j["info"]["idToLabels"][str(b["semanticId"])]["class"]
    if c not in ("drone", "vehicle"): continue
    T = np.array(b["transform"]); cs = np.array([[x, y, z] for x in (b["x_min"], b["x_max"]) for y in (b["y_min"], b["y_max"]) for z in (b["z_min"], b["z_max"])])
    Xw = (np.c_[cs, np.ones(8)] @ T)[:, :3]; uv, w = proj(Xw)
    if (w <= 0).any() or not ((uv[:, 0] > -W) & (uv[:, 0] < 2 * W)).all(): continue
    col = (255, 60, 60) if c == "drone" else (255, 220, 40); nb += 1
    for a, e in E: d.line([tuple(uv[a]), tuple(uv[e])], fill=col, width=2)
save("09_bounding_box_3d_projected", legend(im, [(f"3D oriented boxes projected with camera_params ({nb} drawn)", (255, 60, 60))]))
# occlusion
oc = J("occlusion")["data"]; idl = J("instance_id_segmentation")["info"]["idToLabels"]
occ = {}
for o in oc:
    p = idl.get(str(o["instanceId"]), "")
    if "DroneSwarm" in p: occ[p.split("/")[3]] = o["occlusionRatio"]
bd, pp, _ = boxes("bounding_box_2d_tight")
im = Image.fromarray((rgb * 0.75).astype(np.uint8)); d = ImageDraw.Draw(im)
for b, p in zip(bd, pp):
    if "DroneSwarm" in p:
        k = p.split("/")[-1]; v = 1 - occ.get(k, np.nan)
        d.rectangle([b["x_min"], b["y_min"], b["x_max"], b["y_max"]], outline=(0, 255, 255), width=2)
        d.text((b["x_min"], b["y_max"] + 2), f"{k} visible {100*v:.0f}%", font=f16, fill=(0, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
vals = [v for v in occ.values() if v >= 0]
save("10_occlusion_visible_fraction", legend(im, [(f"occlusion annotator: {len(occ)} drones, occlusionRatio range {min(vals):.2f}-{max(vals):.2f}", (0, 255, 255))]))
json.dump({k: {"occlusion_ratio": v, "visible_fraction": 1 - v} for k, v in sorted(occ.items())}, open(f"{O}/drone_visibility.json", "w"), indent=1)
# derived flow (motion_vectors annotator returned zeros)
mv = Z("motion_vectors")["data"]; print("motion annotator max", np.abs(mv[..., :2]).max())
poses = J("drone_world_poses"); A = {p["id"]: np.array(p["position_m"]) for p in poses["frame_A_poses"]}; B = {p["id"]: np.array(p["position_m"]) for p in poses["frame_B_poses"]}
dep = Z("distance_to_image_plane")["data"] * 100.0  # m -> cm
ys, xs = np.mgrid[0:H, 0:W]; ndx = (xs + 0.5) / W * 2 - 1; ndy = 1 - (ys + 0.5) / H * 2
Xc = np.stack([ndx * dep / P[0, 0], ndy * dep / P[1, 1], -dep, np.ones_like(dep)], -1)
Xw = (Xc.reshape(-1, 4) @ np.linalg.inv(V))[:, :3].reshape(H, W, 3)
flow = np.zeros((H, W, 2), np.float32)
for k, p in idl.items():
    if "DroneSwarm" not in p: continue
    did = p.split("/")[3]; m = iid == int(k)
    if not m.any(): continue
    delta = (B[did] - A[did]) * 100.0; XB = Xw[m]; uvB, _ = proj(XB); uvA, _ = proj(XB - delta)
    flow[m] = (uvB - uvA)
np.savez_compressed(f"{O}/derived_flow_A_to_B.npz", flow=flow)
mag = np.linalg.norm(flow, axis=-1); ang = np.arctan2(flow[..., 1], flow[..., 0])
hsv = np.stack([(ang + np.pi) / (2 * np.pi), np.ones_like(mag), np.clip(mag / (mag.max() + 1e-6), 0, 1)], -1)
fim = (matplotlib.colors.hsv_to_rgb(hsv) * 255).astype(np.uint8)
bg = (rgb * 0.25).astype(np.uint8); fm = mag > 0.1; bg[fm] = fim[fm]
im = Image.fromarray(bg); d = ImageDraw.Draw(im)
for k, p in idl.items():
    if "DroneSwarm" not in p: continue
    m = iid == int(k)
    if m.any():
        cy, cx = np.argwhere(m).mean(0); fl = flow[m].mean(0)
        d.line([(cx, cy), (cx + fl[0] * 3, cy + fl[1] * 3)], fill=(255, 255, 255), width=2)
        d.text((cx + 6, cy + 6), f"{np.linalg.norm(fl):.0f}px", font=f16, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
save("11_motion_flow_derived", legend(im, [(f"flow-colour (hue=direction), drones moved 0.1 s; mean {mag[fm].mean():.1f}px, max {mag.max():.1f}px", (0, 0, 0)),
                                          ("motion_vectors annotator itself returned ~0 (max %.4f)" % np.abs(mv[..., :2]).max(), (0, 0, 0))]))
# point cloud rendered from a novel viewpoint
pc = Z("pointcloud"); X = pc["data"].astype(np.float64); C = pc["info.pointRgb"][:, :3]; S = pc["info.pointSemantic"]
c0 = np.median(X, 0); eye0 = np.linalg.inv(V)[3, :3]
ang = np.radians(50); rot = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
eye = c0 + (eye0 - c0) @ rot.T * 0.85 + np.array([0, 0, 600])
fwd = c0 - eye; fwd /= np.linalg.norm(fwd); rt = np.cross(fwd, [0, 0, 1]); rt /= np.linalg.norm(rt); up = np.cross(rt, fwd)
rel = X - eye; z = rel @ fwd; u = rel @ rt / z; v = rel @ up / z; f = 0.75
px = (W / 2 + u / f * W / 2).astype(int); py = (H / 2 - v / f * W / 2).astype(int)
ok = (z > 0) & (px >= 0) & (px < W - 1) & (py >= 0) & (py < H - 1)
order = np.argsort(-z[ok]); img = np.full((H, W, 3), 20, np.uint8)
for dx in (0, 1):
    for dy in (0, 1):
        img[py[ok][order] + dy, px[ok][order] + dx] = C[ok][order]
dr = S == int([k for k, v in J("semantic_segmentation")["info"]["idToLabels"].items() if v["class"] == "drone"][0]) if False else None
save("12_pointcloud_novel_view", legend(img, [(f"pointcloud annotator: {len(X):,} coloured 3D points, re-rendered from a viewpoint rotated 50 deg", (0, 0, 0))]))
# lidar
L = np.load(f"{R}/lidar_IsaacExtractRTXSensorPointCloudNoAccumulator.npz")["data"]
rng_ = np.linalg.norm(L, axis=1); az = np.degrees(np.arctan2(L[:, 1], L[:, 0])); el = np.degrees(np.arctan2(L[:, 2], np.hypot(L[:, 0], L[:, 1])))
fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
ax0 = fig.add_subplot(2, 1, 1); sc = ax0.scatter(az, el, c=rng_, s=0.5, cmap="turbo"); fig.colorbar(sc, ax=ax0, label="range (m)")
ax0.set_title(f"RTX LiDAR (Example_Rotary) single read: {len(L):,} returns - range image (azimuth vs elevation)"); ax0.set_xlabel("azimuth (deg)"); ax0.set_ylabel("elevation (deg)")
ax1 = fig.add_subplot(2, 1, 2); ax1.scatter(L[:, 1], L[:, 2], c=rng_, s=0.3, cmap="turbo"); ax1.set_title("side view of returns, sensor frame"); ax1.set_xlabel("y (m)"); ax1.set_ylabel("z (m)")
plt.tight_layout(); plt.savefig(f"{O}/13_rtx_lidar.png"); plt.close()
# camera params text tile
fx = cp["cameraFocalLength"] / cp["cameraAperture"][0] * W; fy = cp["cameraFocalLength"] / cp["cameraAperture"][1] * H
Vinv = np.linalg.inv(V); pos = Vinv[3, :3] * 0.01
lines = ["camera_params (close camera, pinhole)", "", f"resolution {W}x{H}   focal {cp['cameraFocalLength']} mm   aperture {cp['cameraAperture'][0]:.3f} x {cp['cameraAperture'][1]:.3f} mm",
         f"intrinsics K: fx={fx:.1f} fy={fy:.1f} cx={W/2:.1f} cy={H/2:.1f} px", f"near/far {cp['cameraNearFar']} (scene units, cm)", f"metersPerSceneUnit {cp['metersPerSceneUnit']:.4f}",
         "", f"extrinsics: camera position (m) {np.round(pos, 2).tolist()}", "world->camera view matrix (row-major):"] + ["   " + "  ".join(f"{x:10.4f}" for x in r) for r in V] + \
        ["", "drone world poses (USD, frame B):"] + [f"  {p['id']}: pos(m) {np.round(p['position_m'], 2).tolist()}  quat(wxyz) {np.round(p['orientation_quat_wxyz'], 3).tolist()}" for p in poses["frame_B_poses"]]
im = Image.new("RGB", (W, H), (18, 20, 26)); d = ImageDraw.Draw(im); y = 14
for l in lines: d.text((16, y), l, font=f16 if y > 20 else fb, fill=(230, 230, 230)); y += 22 if y > 20 else 28
save("14_camera_params_and_drone_poses", im)
K = {"fx": fx, "fy": fy, "cx": W / 2, "cy": H / 2, "width": W, "height": H, "world_to_camera_row_major": V.tolist(), "camera_position_m": pos.tolist()}
json.dump(K, open(f"{O}/camera_intrinsics_extrinsics.json", "w"), indent=1)
# DR frames: redraw drone boxes only
for f in range(3):
    im = Image.open(f"{R}/dr_{f}_rgb.png").convert("RGB"); d = ImageDraw.Draw(im); j = J(f"dr_{f}_bounding_box_2d_tight"); n = 0
    for b, p in zip(j["data"], j["info"]["primPaths"]):
        if "DroneSwarm" in p: d.rectangle([b["x_min"], b["y_min"], b["x_max"], b["y_max"]], outline=(255, 40, 40), width=2); n += 1
    im.save(f"{O}/dr_{f}_drone_boxes.png"); print("dr", f, n)
