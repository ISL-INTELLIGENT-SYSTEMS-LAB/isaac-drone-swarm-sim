"""Counting dataset generator (resumable). env: START, END (frame range), OUT (default datasets/counting_v1), RT (rt_subframes), NMAX.
Per frame: rgb jpg, uint16 instance mask png (value = drone index in this frame + 1), labels json."""
import os, sys, json, time, math
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from isaacsim import SimulationApp
W, H = 1280, 720
app = SimulationApp({"headless": True, "width": W, "height": H})
from common import *
import omni.usd, omni.replicator.core as rep
from pxr import Usd, UsdGeom, UsdLux, Gf, Sdf
from PIL import Image
START, END = int(os.environ.get("START", "0")), int(os.environ.get("END", "10"))
OUT = os.environ.get("OUT", os.path.join(DS, "counting_v1")); RT = int(os.environ.get("RT", "6")); NMAX = int(os.environ.get("NMAX", "1000"))
SEED = 20261101
for d in ("images", "masks", "ann"): os.makedirs(os.path.join(OUT, d), exist_ok=True)
todo = [i for i in range(START, END) if not os.path.exists(os.path.join(OUT, "ann", f"{i:06d}.json"))]
log("chunk", START, END, "todo", len(todo))
if not todo: app.close(); sys.exit(0)
stage = open_layer(app, os.path.join(DS, "stages", f"counting_{os.getpid()}.usda"), [DESERT])
mpu = UsdGeom.GetStageMetersPerUnit(stage); M = 1.0 / mpu
hm = HeightMap(os.path.join(DS, "stages", "heightmap.npz"))
dome = UsdLux.DomeLight.Define(stage, "/Root/DS_Sky"); dome.CreateTextureFileAttr(SKIES[0]); dome.CreateTextureFormatAttr("latlong"); dome.CreateIntensityAttr(800.0)
dome_rot = UsdGeom.Xformable(dome).AddRotateZOp()
sun = [p for p in stage.Traverse() if p.IsA(UsdLux.DistantLight)][0]
sun_xf = UsdGeom.Xformable(sun); sun_int = sun.GetAttribute("intensity"); INT0 = sun_int.Get()
sun_ct = sun.GetAttribute("colorTemperature"); sun.GetAttribute("enableColorTemperature").Set(True)
sun_orient = sun.GetAttribute("xformOp:orient"); parentW = sun_xf.ComputeParentToWorldTransform(Usd.TimeCode.Default())
PG = int(os.environ.get("POOL_GHOST", "900")); pool = Pool(stage, app, PG, 150, 150, M)
TYPES = np.array(pool.types); by_type = {k: np.nonzero(TYPES == k)[0] for k in ASSETS}
cam = UsdGeom.Camera.Define(stage, "/Root/DSCam"); cam.CreateFocalLengthAttr(18.0); cam.CreateHorizontalApertureAttr(20.955); cam.CreateVerticalApertureAttr(20.955 * H / W)
cam.CreateClippingRangeAttr(Gf.Vec2f(0.1 * M, 4000.0 * M)); cam_op = UsdGeom.Xformable(cam).AddTransformOp()
rp = rep.create.render_product(str(cam.GetPath()), (W, H))
A = {}
for nm, ip in [("rgb", None), ("instance_id_segmentation_fast", {"colorize": False}), ("bounding_box_2d_tight_fast", None), ("bounding_box_2d_loose_fast", None), ("occlusion", None), ("camera_params", None)]:
    a = rep.AnnotatorRegistry.get_annotator(nm, init_params=ip) if ip else rep.AnnotatorRegistry.get_annotator(nm); a.attach(rp); A[nm] = a
for _ in range(30): app.update()
rep.orchestrator.step(rt_subframes=4)
log("ready vram", peak["cur"])

def set_sun(el, az, inten, ct):
    s = np.array([math.cos(math.radians(el)) * math.cos(math.radians(az)), math.cos(math.radians(el)) * math.sin(math.radians(az)), math.sin(math.radians(el))])
    Wm = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), Gf.Vec3d(*s)))   # local +Z -> towards sun, light shines along -Z
    L = Wm * parentW.GetInverse(); q = L.ExtractRotationQuat().GetNormalized()
    tn = sun_orient.GetTypeName()
    sun_orient.Set(Gf.Quatd(q) if "quatd" in str(tn) else Gf.Quatf(q))
    sun_int.Set(float(INT0 * inten)); sun_ct.Set(float(ct))
    d = sun_xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).TransformDir(Gf.Vec3d(0, 0, -1))
    return [round(x, 3) for x in Gf.Vec3d(d).GetNormalized()]

def formation(rng, N, spacing, span_max):
    kind = rng.choice(["blob", "lattice", "clusters", "ring_line"], p=[0.4, 0.2, 0.2, 0.2])
    flat = rng.uniform(0.15, 1.0); mind = max(0.9 * span_max, 0.55 * spacing)
    if kind == "lattice":
        n = int(math.ceil(N ** (1 / 3))) + 1; nz = max(1, int(round(n * flat)))
        g = np.stack(np.meshgrid(np.arange(n), np.arange(n), np.arange(nz)), -1).reshape(-1, 3).astype(float)
        g = g[rng.permutation(len(g))[:N]] if len(g) >= N else g
        while len(g) < N: g = np.concatenate([g, g[: N - len(g)] + [0, 0, nz]])
        P = (g - g.mean(0)) * spacing + rng.normal(0, 0.12 * spacing, (N, 3))
        return kind, P
    if kind == "ring_line":
        sub = rng.choice(["ring", "line", "vee"])
        L = spacing * N
        if sub == "ring":
            r = L / (2 * math.pi) / max(1, int(math.ceil(N / 60))); t = rng.uniform(0, 2 * math.pi, N)
            P = np.stack([r * np.cos(t), r * np.sin(t), rng.normal(0, 0.3 * spacing, N)], 1) * rng.uniform(0.9, 1.1, (N, 1))
        else:
            rows = max(1, int(math.ceil(N / 80))); t = (np.arange(N) // rows) * spacing; t -= t.mean()
            off = (np.arange(N) % rows) * spacing
            P = np.stack([t, off + (np.abs(t) * 0.6 if sub == "vee" else 0), rng.normal(0, 0.2 * spacing, N)], 1)
        a = rng.uniform(0, 2 * math.pi); Rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
        return kind + ":" + sub, P @ Rz.T
    k = 1 if kind == "blob" else int(rng.integers(2, 6))
    cents = [np.zeros(3)] if k == 1 else [rng.normal(0, 1, 3) * np.array([1, 1, 0.3]) * spacing * N ** (1 / 3) * 1.5 for _ in range(k)]
    R = spacing * (N / k) ** (1 / 3) * 0.62 + spacing
    pts = np.zeros((N, 3)); n = 0; tries = 0; ax = np.array([R, R, R * flat])
    while n < N and tries < N * 60:
        tries += 1
        c = cents[n % k]
        p = c + rng.uniform(-1, 1, 3) * ax
        if np.sum(((p - c) / ax) ** 2) > 1: continue
        if n and np.min(np.sum((pts[:n] - p) ** 2, axis=1)) < mind * mind: continue
        pts[n] = p; n += 1
    while n < N: pts[n] = cents[0] + rng.uniform(-1, 1, 3) * R * 2; n += 1
    return kind, pts

def sample_frame(i, attempt=0):
    rng = np.random.default_rng([SEED, i] + ([attempt] if attempt else []))
    u = rng.random()   # count mixture, weighted towards dense swarms: 40% log-uniform 1..NMAX, 10% U(51,100), 25% U(101,300), 25% log-uniform 301..NMAX
    if u < 0.40: N = int(np.clip(np.floor(np.exp(rng.uniform(0, math.log(NMAX + 1)))), 1, NMAX))
    elif u < 0.50: N = int(rng.integers(51, 101))
    elif u < 0.75: N = int(rng.integers(101, 301))
    else: N = int(np.clip(np.floor(np.exp(rng.uniform(math.log(301), math.log(NMAX + 1)))), 301, NMAX))
    if os.environ.get("NFORCE"): N = int(os.environ["NFORCE"])
    probs = {"ghost": 0.75, "cf2x": 0.125, "quad": 0.125}
    tsel = rng.choice(list(probs), N, p=list(probs.values()))
    idx = []
    for k in ASSETS:
        n = int((tsel == k).sum()); avail = by_type[k]
        if n > len(avail): extra = n - len(avail); n = len(avail); tsel[np.nonzero(tsel == k)[0][:extra]] = "ghost"
    for k in ASSETS:
        n = int((tsel == k).sum()); idx += list(rng.choice(by_type[k], n, replace=False))
    idx = np.array(idx)
    base_span = rng.uniform(0.3, 1.0); spans = np.clip(base_span * rng.uniform(0.85, 1.15, N), 0.3, 1.0) if rng.random() < 0.7 else rng.uniform(0.3, 1.0, N)
    spacing = rng.uniform(1.3, 7.0) * max(0.6, base_span)
    kind, P = formation(rng, N, spacing, spans.max())
    c_xy = hm.sample_xy(rng)
    near_ground = rng.random() < 0.08
    cam_mode = rng.choice(["aerial", "ground_up"], p=[0.8, 0.2])
    alt = rng.uniform(0.4, 3.0) if near_ground else (rng.uniform(15, 90) if cam_mode == "ground_up" else float(np.exp(rng.uniform(math.log(5), math.log(110)))))
    gz = float(hm(c_xy[0], c_xy[1], "ground"))
    P = P - P.mean(0); P[:, :2] += c_xy
    zfloor = hm(P[:, 0], P[:, 1], "h") + 0.4
    P[:, 2] += gz + alt + (P[:, 2].max() - P[:, 2].min()) * 0.5 * (not near_ground)
    P[:, 2] = np.maximum(P[:, 2], zfloor)
    if near_ground: P[:, 2] = zfloor + rng.uniform(0, 1.5, N) * (rng.random() < 0.5)
    sw_yaw = rng.uniform(0, 360); coh = rng.random() < 0.6
    quats = [quat_from_euler_deg(rng.uniform(-15, 15), rng.uniform(-15, 15), (sw_yaw + rng.normal(0, 15)) if coh else rng.uniform(0, 360)) for _ in range(N)]
    C = P.mean(0); ext = np.ptp(P, 0); Rsw = 0.5 * float(np.linalg.norm(ext[:2])) + 1.0
    focal = float(np.exp(rng.uniform(math.log(12), math.log(40)))); fx = focal / 20.955 * W
    if cam_mode == "aerial":
        tgt_px = float(np.exp(rng.uniform(math.log(2.5), math.log(70))))
        dist = fx * float(np.mean(spans)) / tgt_px
        fit = Rsw * fx / (0.42 * W)
        if rng.random() < 0.75: dist = max(dist, fit)
        dist = max(dist, 6.0)
        top_agl = float(P[:, 2].max() - hm(C[0], C[1], "ground"))
        above = rng.random() < 0.6
        for _ in range(40):
            lo_ = max(20.0, top_agl + 8); hc = rng.uniform(lo_, max(lo_ + 10, min(130.0, top_agl + 70))) if above else rng.uniform(20, 120)
            # camera sits on the outward side of the swarm and looks back towards the terrain interior (avoids the HDR horizon/ground)
            dirv = C[:2] - hm.center; nd = np.linalg.norm(dirv)
            ang = (math.atan2(dirv[1], dirv[0]) + rng.normal(0, 0.8)) if (nd > 30 and rng.random() < 0.85) else rng.uniform(0, 2 * math.pi)
            dz = float(hm(C[0], C[1], "ground")) + hc - C[2]
            hd = math.sqrt(max(dist ** 2 - dz ** 2, 0.0))
            if dz < ext[2] / 2 + 5: hd = max(hd, Rsw * 1.1 + 3)          # not above the swarm -> must stand off horizontally
            exy = C[:2] + hd * np.array([math.cos(ang), math.sin(ang)])
            ez = float(hm(exy[0], exy[1], "ground")) + hc
            if ez < float(hm(exy[0], exy[1], "h")) + 3: continue
            if np.min(np.linalg.norm(P - np.array([exy[0], exy[1], ez]), axis=1)) < 4: continue
            break
        eye = np.array([exy[0], exy[1], ez])
    else:
        for _ in range(40):
            ang = rng.uniform(0, 2 * math.pi); hd = rng.uniform(5, 70)
            exy = C[:2] + hd * np.array([math.cos(ang), math.sin(ang)])
            if float(hm(exy[0], exy[1], "h") - hm(exy[0], exy[1], "ground")) < 0.5: break
        eye = np.array([exy[0], exy[1], float(hm(exy[0], exy[1], "ground")) + rng.uniform(1.4, 2.0)])
    tgt = C + rng.uniform(-0.3, 0.3, 3) * np.array([Rsw, Rsw, max(2.0, ext[2])])
    sun = dict(el=float(rng.uniform(8, 80)), az=float(rng.uniform(0, 360)), inten=float(rng.uniform(0.5, 1.6)), ct=float(rng.uniform(4200, 7500)))
    sky = dict(tex=int(rng.integers(len(SKIES))), inten=float(rng.uniform(400, 1300)), rot=float(rng.uniform(0, 360)))
    return dict(N=N, idx=idx, types=[pool.types[j] for j in idx], P=P, quats=quats, spans=spans, kind=kind, near_ground=near_ground, cam_mode=cam_mode,
                alt_agl=alt, focal=focal, eye=eye, tgt=tgt, sun=sun, sky=sky, spacing=spacing)

ZG = float(np.median(hm.ground[hm.valid]))
def hdr_ground_fraction(eye, tgt, focal):
    """fraction of a 13x7 ray grid that looks below the horizon but misses the terrain (would show the HDR dome's lower half)"""
    fwd = np.asarray(tgt, float) - eye; fwd /= np.linalg.norm(fwd)
    rt = np.cross(fwd, [0, 0, 1.0]); rt = rt / np.linalg.norm(rt) if np.linalg.norm(rt) > 1e-6 else np.array([1.0, 0, 0]); up = np.cross(rt, fwd)
    tx = 20.955 / 2 / focal; ty = tx * H / W
    u, v = np.meshgrid(np.linspace(-1, 1, 13), np.linspace(-1, 1, 7)); d = fwd + u.ravel()[:, None] * tx * rt + v.ravel()[:, None] * ty * up
    down = d[:, 2] < -1e-4; bad = 0
    if down.any():
        t = (ZG - eye[2]) / d[down, 2]; p = eye + t[:, None] * d[down]
        ii = ((p[:, 1] - hm.y0) / hm.res).astype(int); jj = ((p[:, 0] - hm.x0) / hm.res).astype(int)
        ok = (t > 0) & (ii >= 0) & (ii < hm.valid.shape[0]) & (jj >= 0) & (jj < hm.valid.shape[1])
        ok[ok] = hm.valid[ii[ok], jj[ok]]
        bad = int((~ok).sum())
    return bad / len(d)

# local corner set for projected (amodal) boxes: centred asset box scaled to span
def corners_world(i_pool, pos, q, span):
    inf = pool.info[pool.types[i_pool]]; s = pool.scale_for(i_pool, span) * mpu
    hx, hy, hz = np.array(inf["size"]) * s / 2
    c = np.array([[sx * hx, sy * hy, sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    w, x, y, z = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return c @ R.T + pos

def project(Pw, V, Pj):
    X = np.concatenate([Pw * M, np.ones((len(Pw), 1))], 1) @ V
    depth = -X[:, 2] * mpu
    Y = X @ Pj; ndc = Y[:, :2] / Y[:, 3:4]
    return np.stack([(ndc[:, 0] + 1) / 2 * W, (1 - ndc[:, 1]) / 2 * H], 1), depth

cur_tex = -1; t_frames = []
for fi, i in enumerate(todo):
    t0 = time.time()
    try:
        for att in range(10):
            S = sample_frame(i, att)
            S["hdr_frac"] = hdr_ground_fraction(S["eye"], S["tgt"], S["focal"])
            if np.min(np.linalg.norm(S["P"] - S["eye"], axis=1)) >= 3.0 and S["hdr_frac"] <= 0.03: break
        S["attempt"] = att
    except Exception as e:
        log("SAMPLE FAILED frame", i, repr(e)); continue
    pool.set(S["idx"], S["P"], S["quats"], S["spans"])
    sdir = set_sun(**S["sun"])
    if S["sky"]["tex"] != cur_tex: dome.GetTextureFileAttr().Set(SKIES[S["sky"]["tex"]]); cur_tex = S["sky"]["tex"]
    dome.GetIntensityAttr().Set(S["sky"]["inten"]); dome_rot.Set(S["sky"]["rot"])
    cam.GetFocalLengthAttr().Set(S["focal"]); cam_op.Set(look_at_matrix(S["eye"], S["tgt"], M))
    app.update(); rep.orchestrator.step(rt_subframes=RT)
    t_r = time.time()
    img = np.asarray(A["rgb"].get_data())[..., :3].astype(np.uint8)
    sd = A["instance_id_segmentation_fast"].get_data(); seg = np.asarray(sd["data"]).astype(np.int64); i2l = sd["info"]["idToLabels"]
    pool2k = {int(p): k for k, p in enumerate(S["idx"])}
    lut = np.zeros(int(seg.max()) + 1 if seg.size else 1, np.int32); iid2k = {}
    for key, pth in i2l.items():
        pth = str(pth)
        if pth.startswith("/Root/DS/d_"):
            pidx = int(pth[11:15]); k = pool2k.get(pidx)
            if k is not None and int(key) < len(lut): lut[int(key)] = k + 1; iid2k[int(key)] = k
    mask = lut[seg].astype(np.uint16)
    N = S["N"]; mpx = np.bincount(mask.ravel(), minlength=N + 1)[1:N + 1]
    # mask-derived tight boxes
    tb = [None] * N
    ys, xs = np.nonzero(mask)
    if len(xs):
        mv = mask[ys, xs].astype(np.int64) - 1
        xmn = np.full(N, 1 << 30); ymn = np.full(N, 1 << 30); xmx = np.full(N, -1); ymx = np.full(N, -1)
        np.minimum.at(xmn, mv, xs); np.minimum.at(ymn, mv, ys); np.maximum.at(xmx, mv, xs); np.maximum.at(ymx, mv, ys)
        for k in np.nonzero(xmx >= 0)[0]: tb[k] = [int(xmn[k]), int(ymn[k]), int(xmx[k]), int(ymx[k])]
    # annotator boxes (semantic, keyed by drone prim)
    occ_bb = {}
    def boxes_by_k(bd, keep_occ=False):
        out = {}; names = bd["data"].dtype.names or ()
        for b, pth in zip(bd["data"], bd["info"].get("primPaths", [])):
            pth = str(pth)
            if pth.startswith("/Root/DS/d_"):
                k = pool2k.get(int(pth[11:15]))
                if k is not None:
                    out[k] = [int(b["x_min"]), int(b["y_min"]), int(b["x_max"]), int(b["y_max"])]
                    if keep_occ and "occlusionRatio" in names: occ_bb[k] = float(b["occlusionRatio"])
        return out
    _bt = A["bounding_box_2d_tight_fast"].get_data()
    if fi == 0: log("DEBUG bbox dtype", _bt["data"].dtype)
    bt = boxes_by_k(_bt, True); bl = boxes_by_k(A["bounding_box_2d_loose_fast"].get_data())
    od = A["occlusion"].get_data(); od = od["data"] if isinstance(od, dict) else od; od = np.asarray(od)
    occ = np.full(N, -1.0)
    for k, v in occ_bb.items():
        if v >= 0: occ[k] = v
    if (occ < 0).all() and od.size and od.dtype.names and "instanceId" in od.dtype.names:
        for o in od:
            k = iid2k.get(int(o["instanceId"]))
            if k is not None: occ[k] = max(occ[k], float(o["occlusionRatio"]))
    cp = A["camera_params"].get_data()
    V = np.asarray(cp["cameraViewTransform"], float).reshape(4, 4); Pj = np.asarray(cp["cameraProjection"], float).reshape(4, 4)
    ctr, depth = project(S["P"], V, Pj)
    fx = S["focal"] / 20.955 * W
    if fi == 0:
        log("DEBUG occlusion dtype", od.dtype if od.size else None, "n", od.size, "first", od[:3] if od.size else None, "iid drone entries", len(iid2k))
    drones = []
    for k in range(N):
        pi = int(S["idx"][k]); inf_ = (depth[k] > 0) and (0 <= ctr[k, 0] < W) and (0 <= ctr[k, 1] < H)
        cw = corners_world(pi, S["P"][k], S["quats"][k], S["spans"][k]); cpx, cd = project(cw, V, Pj)
        amodal = [round(float(cpx[:, 0].min()), 1), round(float(cpx[:, 1].min()), 1), round(float(cpx[:, 0].max()), 1), round(float(cpx[:, 1].max()), 1)] if (cd > 0).all() else None
        vf = (1.0 - occ[k]) if occ[k] >= 0 else None
        drones.append({"k": k, "pool_id": pi, "type": pool.types[pi], "span_m": round(float(S["spans"][k]), 3), "pos_m": [round(float(x), 3) for x in S["P"][k]],
                       "quat_wxyz": [round(float(x), 4) for x in S["quats"][k]], "depth_m": round(float(depth[k]), 2), "center_px": [round(float(ctr[k, 0]), 1), round(float(ctr[k, 1]), 1)],
                       "in_frustum": bool(inf_), "span_px": round(float(fx * S["spans"][k] / max(depth[k], 1e-3)), 2), "mask_px": int(mpx[k]), "bbox_tight": tb[k],
                       "bbox_tight_ann": bt.get(k), "bbox_loose_ann": bl.get(k), "bbox_amodal_proj": amodal, "occlusion_ratio": round(float(occ[k]), 4) if occ[k] >= 0 else None,
                       "visible_fraction": round(float(vf), 4) if vf is not None else None})
    vis = lambda t: int(sum(d["mask_px"] >= t for d in drones))
    rec = {"frame": i, "image": f"images/{i:06d}.jpg", "mask": f"masks/{i:06d}.png", "width": W, "height": H,
           "n_total": N, "n_in_frustum": int(sum(d["in_frustum"] for d in drones)), "n_visible_ge1px": vis(1), "n_visible_ge10px": vis(10), "n_visible_ge50px": vis(50),
           "formation": S["kind"], "sample_attempt": S["attempt"], "offterrain_view_frac": round(S["hdr_frac"], 3), "spacing_m": round(S["spacing"], 2), "near_ground": bool(S["near_ground"]), "alt_agl_m": round(S["alt_agl"], 1), "cam_mode": S["cam_mode"],
           "camera": {"focal_mm": round(S["focal"], 2), "h_aperture_mm": 20.955, "fx": round(fx, 2), "fy": round(fx, 2), "cx": W / 2, "cy": H / 2, "eye_m": [round(float(x), 2) for x in S["eye"]],
                      "target_m": [round(float(x), 2) for x in S["tgt"]], "eye_agl_m": round(float(S["eye"][2] - hm(S["eye"][0], S["eye"][1], "ground")), 1),
                      "cameraViewTransform": V.ravel().tolist(), "cameraProjection": Pj.ravel().tolist(), "metersPerSceneUnit": mpu},
           "sun": dict(S["sun"], dir_world=sdir), "sky": dict(S["sky"], file=SKIES[S["sky"]["tex"]].split("/")[-1]),
           "ann_bbox_tight_count": len(bt), "drones": drones}
    Image.fromarray(img).save(os.path.join(OUT, "images", f"{i:06d}.jpg"), quality=92)
    Image.fromarray(mask).save(os.path.join(OUT, "masks", f"{i:06d}.png"), compress_level=6)
    with open(os.path.join(OUT, "ann", f"{i:06d}.json.tmp"), "w") as f: json.dump(rec, f, separators=(",", ":"))
    os.replace(os.path.join(OUT, "ann", f"{i:06d}.json.tmp"), os.path.join(OUT, "ann", f"{i:06d}.json"))
    dt = time.time() - t0; t_frames.append(dt)
    if fi < 3 or fi % 25 == 0:
        log(f"frame {i} N={N} vis1={rec['n_visible_ge1px']} vis10={rec['n_visible_ge10px']} annbox={len(bt)} maskdrones={int((mpx>0).sum())} mode={S['cam_mode']} {S['kind']} {dt:.2f}s (render {t_r-t0:.2f}) vram {peak['cur']} peak {peak['mb']}")
log("chunk done", len(todo), "frames, mean s/frame", round(float(np.mean(t_frames)), 2), "peak VRAM", peak["mb"])
app.close()
