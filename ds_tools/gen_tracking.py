"""Tracking dataset generator: swarm takeoff -> cruise sequences, ~half with 1-3 breakaway drones. Resumable per sequence.
env: SEQ_START, SEQ_END, OUT (default datasets/tracking_v1), RT, NMAX_T, POOL_GHOST, SEED (default 77001)
Visible-count BIN mode (used by run_tracking.ps1 -Bins ... for re-ID datasets), extra env:
  BIN_LO, BIN_HI   keep a sequence only if the MEDIAN over its rendered frames of n_visible_ge1px is in [BIN_LO, BIN_HI]
  CLIP_FRAMES      if >0, render only a window of this many consecutive frames (taken from the cruise part when possible)
  MAX_ACCEPT       designs tried per sequence before giving up (default 30); PROBE_FRAMES cheap pre-check frames (default 5)
  CAM_DIST         camera distance multiplier (default 1.0; <1 = closer camera = bigger drones / crops)
  MODEL_MIX        legacy (default: 50% all-Ghost, else 80/10/10 Ghost/Crazyflie/Quadcopter) | balanced | isaac | ghost | 'ghost:1,arl:2,...'
  COLOR_MODE       none (default) | random | palette  -- per-drone colour, fixed for the whole sequence (COLOR_STRENGTH, COLOR_WEAR)
  CAM_FRAMING      wide (default, original tracking cameras) | chase (camera follows the swarm, slowly orbiting, drones ~30-60 px wide; used for re-ID)
Swarm scenario options (run_tracking.ps1 -Swarms/-SwarmSep/-Spread/-BrkProb/-BrkCount/-BrkGroup; defaults reproduce the old designs exactly):
  SWARMS (1)       number of swarms: the formation is split into SWARMS spatial groups that fly the same path side by side
  SWARM_SEP (1.0)  gap between neighbouring swarms, in swarm radii
  SPREAD (1.0)     multiplier on the drone spacing inside a formation (<1 tighter, >1 looser)
  BRK_PROB (0.5)   probability that a sequence gets breakaways (FORCE_BRK=1 forces it)
  BRK_COUNT (0)    number of breakaway events (0 = original random 1-3)
  N_DRONES (0)     >0 = fixed drone count (else random / binned)
  BRK_GROUP (1)    drones per breakaway event (>1 = a subgroup of nearest neighbours leaves together)
  SWARM_ANN (1)    write swarm annotations after each sequence (ds_tools/swarm_annotate.py; can also be run later)
  REP_WRITER (0)   also attach a Replicator BasicWriter (tight 2D boxes + instance-id + 'swarm' semantic segmentation) -> seq/replicator/
Team-manifest mode (ds_tools/run_manifest.py; unset = old behaviour, identical designs for the same seed):
  MANIFEST_ROWS    JSON list of manifest rows; sequence s renders row s into OUT/<seq_id>. Each row sets SEED (its own seed, the
                   design rng no longer depends on s), N_DRONES, SWARMS, SWARM_SEP, SPREAD, breakaways (on/off, count, group size,
                   start frame in the clip), CLIP_FRAMES, CAM_FRAMING, CAM_DIST, and these overrides (drawn values are replaced,
                   the rng stream is unchanged): sun elevation / azimuth / intensity / colour temperature, sky HDR index and
                   intensity, chase-camera elevation, chase_fit (frame the whole formation, no 30-60 px drone-size cap) and a
                   turbulence multiplier on the per-drone position jitter (a wind proxy; there is no wind physics).
IDs: drone "id" == "track_id" is persistent for the whole sequence and equals the value of that drone in masks/*.png."""
import os, sys, json, time, math, shutil
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from isaacsim import SimulationApp
W, H = 1280, 720
app = SimulationApp({"headless": True, "width": W, "height": H})
from common import *
import omni.usd, omni.replicator.core as rep
from pxr import Usd, UsdGeom, UsdLux, Gf, Sdf
from PIL import Image
S0, S1 = int(os.environ.get("SEQ_START", "0")), int(os.environ.get("SEQ_END", "1"))
OUT = os.environ.get("OUT", os.path.join(DS, "tracking_v1")); RT = int(os.environ.get("RT", "4")); NMAX = int(os.environ.get("NMAX_T", "600"))
SEED = int(os.environ.get("SEED", "77001"))
BIN_LO = int(os.environ.get("BIN_LO", "0")); BIN_HI = int(os.environ.get("BIN_HI", "0")); BINNED = BIN_HI > 0
VISIBLE_BIN = f"{BIN_LO}-{BIN_HI}" if BINNED else None
CLIP = int(os.environ.get("CLIP_FRAMES", "0")); MAX_ACCEPT = int(os.environ.get("MAX_ACCEPT", "30")); NPROBE = int(os.environ.get("PROBE_FRAMES", "5"))
CAM_DIST = float(os.environ.get("CAM_DIST", "1.0"))
CAM_FRAMING = os.environ.get("CAM_FRAMING", "wide")   # wide = original (whole trajectory in view, small drones); chase = camera follows the swarm at close range
FORCE_BRK = os.environ.get("FORCE_BRK", "0") == "1"   # seq_036+: always add a breakaway; extend the sequence if cruise is too short
SWARMS = max(1, int(os.environ.get("SWARMS", "1"))); SWARM_SEP = float(os.environ.get("SWARM_SEP", "1.0")); SPREAD = float(os.environ.get("SPREAD", "1.0"))
BRK_PROB = float(os.environ.get("BRK_PROB", "0.5")); BRK_COUNT = int(os.environ.get("BRK_COUNT", "0")); BRK_GROUP = max(1, int(os.environ.get("BRK_GROUP", "1")))
N_DRONES = int(os.environ.get("N_DRONES", "0"))   # >0: fixed number of drones (overrides the random / binned swarm size)
SWARM_ANN = os.environ.get("SWARM_ANN", "1") == "1"; REP_WRITER = os.environ.get("REP_WRITER", "0") == "1"
MROWS = json.load(open(os.environ["MANIFEST_ROWS"])) if os.environ.get("MANIFEST_ROWS") else None
OVR = {}   # per-row overrides (manifest mode only; empty = generator draws everything itself)
def seq_name(s): return str(MROWS[s]["seq_id"]) if MROWS else f"seq_{s:03d}"
def apply_row(r):
    """Manifest mode: set the scenario globals for one manifest row (see module docstring)."""
    global SEED, N_DRONES, SWARMS, SWARM_SEP, SPREAD, BRK_PROB, BRK_COUNT, BRK_GROUP, FORCE_BRK, CAM_DIST, CAM_FRAMING, CLIP
    g = lambda k, d: d if r.get(k) in (None, "") else r[k]   # empty CSV cell = default
    SEED = int(r["seed"]); N_DRONES = int(r["n_drones_total"]); SWARMS = int(r["n_swarms"]); SWARM_SEP = float(g("swarm_sep", 1.0))
    SPREAD = float(g("spread", 1.0)); CLIP = int(g("frames", 50)); CAM_FRAMING = str(r["cam_framing"]); CAM_DIST = float(g("cam_dist", 1.0))
    on = str(g("breakaway", "off")) == "on"
    BRK_PROB = 1.0 if on else 0.0; FORCE_BRK = on; BRK_COUNT = int(g("brk_count", 1)) if on else 0; BRK_GROUP = max(1, int(g("brk_size", 1))) if on else 1
    OVR.clear()
    for k in ("sun_el", "sun_az", "sun_int", "sun_ct", "sky_int", "cam_el_deg", "jitter"):
        if r.get(k) not in (None, ""): OVR[k] = float(r[k])
    if r.get("sky_tex") not in (None, ""): OVR["sky_tex"] = int(r["sky_tex"])
    if on and r.get("brk_start_frame") not in (None, ""): OVR["brk_at"] = int(r["brk_start_frame"])
    OVR["chase_fit"] = str(g("chase_fit", "1")) in ("1", "True", "true")
if MROWS: S0, S1 = max(0, S0), min(S1, len(MROWS)) if S1 > 0 else len(MROWS)
todo = [s for s in range(S0, S1) if not os.path.exists(os.path.join(OUT, seq_name(s), "meta.json"))]
log("seqs", S0, S1, "todo", todo)
if not todo: app.close(); sys.exit(0)
stage = open_layer(app, os.path.join(DS, "stages", f"tracking_{os.getpid()}.usda"), [DESERT])
mpu = UsdGeom.GetStageMetersPerUnit(stage); M = 1.0 / mpu
hm = HeightMap(os.path.join(DS, "stages", "heightmap.npz"))
dome = UsdLux.DomeLight.Define(stage, "/Root/DS_Sky"); dome.CreateTextureFileAttr(SKIES[0]); dome.CreateTextureFormatAttr("latlong"); dome.CreateIntensityAttr(800.0)
dome_rot = UsdGeom.Xformable(dome).AddRotateZOp()
sun = [p for p in stage.Traverse() if p.IsA(UsdLux.DistantLight)][0]
sun_xf = UsdGeom.Xformable(sun); sun_int = sun.GetAttribute("intensity"); INT0 = sun_int.Get()
sun_ct = sun.GetAttribute("colorTemperature"); sun.GetAttribute("enableColorTemperature").Set(True)
sun_orient = sun.GetAttribute("xformOp:orient"); parentW = sun_xf.ComputeParentToWorldTransform(Usd.TimeCode.Default())
MIX = parse_mix(os.environ.get("MODEL_MIX", "legacy")); CMODE, CSTR, CWEAR = appearance_env()
NMAX_POOL = (int(math.ceil(BIN_HI * 1.3)) + 2) if BINNED else NMAX
pool = make_pool(stage, app, M, MIX, NMAX_POOL, CMODE != "none", legacy=(600, 100, 100))
TYPES = np.array(pool.types); by_type = {k: np.nonzero(TYPES == k)[0] for k in ASSETS}
cam = UsdGeom.Camera.Define(stage, "/Root/DSCam"); cam.CreateFocalLengthAttr(18.0); cam.CreateHorizontalApertureAttr(20.955); cam.CreateVerticalApertureAttr(20.955 * H / W)
cam.CreateClippingRangeAttr(Gf.Vec2f(0.1 * M, 4000.0 * M)); cam_op = UsdGeom.Xformable(cam).AddTransformOp()
rp = rep.create.render_product(str(cam.GetPath()), (W, H))
A = {}
for nm, ip in [("rgb", None), ("instance_id_segmentation_fast", {"colorize": False}), ("bounding_box_2d_tight_fast", None), ("camera_params", None)]:
    a = rep.AnnotatorRegistry.get_annotator(nm, init_params=ip) if ip else rep.AnnotatorRegistry.get_annotator(nm); a.attach(rp); A[nm] = a
for _ in range(30): app.update()
rep.orchestrator.step(rt_subframes=4)
log("ready vram", peak["cur"])

def set_sun(el, az, inten, ct):
    s = np.array([math.cos(math.radians(el)) * math.cos(math.radians(az)), math.cos(math.radians(el)) * math.sin(math.radians(az)), math.sin(math.radians(el))])
    Wm = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), Gf.Vec3d(*s)))
    L = Wm * parentW.GetInverse(); q = L.ExtractRotationQuat().GetNormalized()
    sun_orient.Set(Gf.Quatd(q) if "quatd" in str(sun_orient.GetTypeName()) else Gf.Quatf(q))
    sun_int.Set(float(INT0 * inten)); sun_ct.Set(float(ct))

def project(Pw, V, Pj):
    X = np.concatenate([Pw * M, np.ones((len(Pw), 1))], 1) @ V
    depth = -X[:, 2] * mpu; Y = X @ Pj; ndc = Y[:, :2] / Y[:, 3:4]
    return np.stack([(ndc[:, 0] + 1) / 2 * W, (1 - ndc[:, 1]) / 2 * H], 1), depth

ZG = float(np.median(hm.ground[hm.valid]))
def hdr_ground_fraction(eye, tgt, focal):
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

def smooth(x): x = np.clip(x, 0, 1); return x * x * (3 - 2 * x)

def catmull(P, t):
    """Catmull-Rom through waypoints P (K,3); t in [0, K-1]"""
    K = len(P); i = np.clip(np.floor(t).astype(int), 0, K - 2); u = (t - i)[:, None]
    p0 = P[np.clip(i - 1, 0, K - 1)]; p1 = P[i]; p2 = P[i + 1]; p3 = P[np.clip(i + 2, 0, K - 1)]
    return 0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u ** 2 + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3)

def split_groups(X, K, rng=None):
    """Split formation slots X (N,2) into K compact, equal-size groups (sizes differ by at most 1) by recursive bisection
    along the principal axis. Returns the group index per drone (0..K-1)."""
    g = np.zeros(len(X), int)
    def rec(idx, k, base):
        if k <= 1: g[idx] = base; return
        P = X[idx] - X[idx].mean(0); ax = np.linalg.svd(P, full_matrices=False)[2][0] if len(idx) > 1 else np.array([1.0, 0.0])
        o = idx[np.argsort(P @ ax, kind="stable")]; k1 = k // 2; n1 = int(round(len(idx) * k1 / k))
        rec(o[:n1], k1, base); rec(o[n1:], k - k1, base + k1)
    rec(np.arange(len(X)), K, 0)
    return g

def design(s, attempt=0):
    sk = 0 if MROWS else s   # manifest mode: the row's own seed fully defines the design
    rng = np.random.default_rng([SEED, sk] if not BINNED else [SEED, BIN_LO, BIN_HI, sk, attempt])
    fps = int(rng.choice([10, 10, 12, 15])); F = int(rng.integers(140, 201)); dt = 1.0 / fps; T = np.arange(F) * dt
    u = rng.random()
    N = int(rng.integers(5, 31)) if u < 0.3 else (int(rng.integers(31, 151)) if u < 0.7 else int(np.exp(rng.uniform(math.log(151), math.log(max(NMAX, 151) + 1)))))
    if BINNED:   # spawn a bit above the bin: some drones are off-screen / occluded / sub-pixel (more so with the chase camera)
        top = int(math.ceil(BIN_HI * 1.3)) + 2 if CAM_FRAMING == "chase" else BIN_HI + int(math.ceil(0.1 * BIN_HI)) + 1
        N = int(rng.integers(BIN_LO, max(BIN_LO, top) + 1))
    if N_DRONES > 0: N = N_DRONES
    N = min(N, len(pool.types))
    if MIX is None:
        tsel = rng.choice(["ghost", "cf2x", "quad"], N, p=[0.8, 0.1, 0.1]) if rng.random() < 0.5 else np.array(["ghost"] * N)
        for k in ("cf2x", "quad"):
            ex = int((tsel == k).sum()) - len(by_type[k])
            if ex > 0: tsel[np.nonzero(tsel == k)[0][:ex]] = "ghost"
    else:
        tsel = fit_types(rng.choice(list(MIX), N, p=list(MIX.values())), by_type)
    idx = np.concatenate([rng.choice(by_type[k], int((tsel == k).sum()), replace=False) for k in ASSETS]).astype(int)
    types = [pool.types[i] for i in idx]
    span = rng.uniform(0.3, 1.0); spans = np.clip(span * rng.uniform(0.9, 1.1, N), 0.3, 1.0)
    pad = rng.uniform(1.2, 3.0) * max(1.0, span * 1.5)
    # launch site: open ground or rooftops
    launch = "rooftop" if rng.random() < 0.3 else "ground"
    for _ in range(200):
        c = hm.sample_xy(rng)
        if launch == "ground":
            side = int(math.ceil(math.sqrt(N))); g = np.stack(np.meshgrid(np.arange(side), np.arange(side)), -1).reshape(-1, 2)[:N] * pad
            g = g - g.mean(0) + c + rng.normal(0, 0.1 * pad, (N, 2)); rough = hm(g[:, 0], g[:, 1], "h") - hm(g[:, 0], g[:, 1], "ground")
            if (rough < 0.3).mean() > 0.97 and hm.inner[np.clip(((g[:, 1] - hm.y0) / hm.res).astype(int), 0, hm.inner.shape[0] - 1), np.clip(((g[:, 0] - hm.x0) / hm.res).astype(int), 0, hm.inner.shape[1] - 1)].all(): break
        else:
            yy, xx = np.mgrid[-40:41:2, -40:41:2]; cand = np.stack([xx.ravel(), yy.ravel()], 1) + c
            r = hm(cand[:, 0], cand[:, 1], "h") - hm(cand[:, 0], cand[:, 1], "ground"); cand = cand[r > 3]
            if len(cand) >= max(4, N // 3):
                g = cand[rng.integers(len(cand), size=N)] + rng.uniform(-0.7, 0.7, (N, 2)); break
    else:
        launch = "ground"
    gz = hm(g[:, 0], g[:, 1], "h")
    P0 = np.concatenate([g, (gz + 0.02 + spans * 0.15)[:, None]], 1)
    c0 = P0.mean(0)
    # formation slots (local frame, x forward)
    form = str(rng.choice(["blob", "lattice", "ring", "vee"], p=[0.4, 0.3, 0.15, 0.15]))
    if (form == "vee" and N > 60) or (form == "ring" and N > 120): form = str(rng.choice(["blob", "lattice"]))
    sp = rng.uniform(1.5, 5.0) * max(0.7, span)
    sp *= SPREAD   # formation spread (default 1.0 = original)
    if form == "lattice":
        n = int(math.ceil(N ** (1 / 3))); L = np.stack(np.meshgrid(np.arange(n), np.arange(n), np.arange(n)), -1).reshape(-1, 3)[:N].astype(float); O = (L - L.mean(0)) * sp
    elif form == "ring":
        t = np.linspace(0, 2 * math.pi, N, endpoint=False); r = sp * N / (2 * math.pi) + 2; O = np.stack([r * np.cos(t), r * np.sin(t), rng.normal(0, 0.3, N)], 1)
    elif form == "vee":
        k = np.arange(N); side = np.where(k % 2 == 0, 1, -1); d = (k + 1) // 2 * sp; O = np.stack([-d * 0.8, side * d * 0.6, rng.normal(0, 0.3, N)], 1)
    else:
        R = sp * N ** (1 / 3) * 0.62 + sp; O = []
        while len(O) < N:
            p = rng.uniform(-1, 1, 3) * [R, R, R * 0.5]
            if np.sum((p / [R, R, R * 0.5]) ** 2) <= 1 and (not O or np.min(np.linalg.norm(np.array(O) - p, axis=1)) > max(sp * 0.6, 1.3 * span)): O.append(p)
        O = np.array(O)
    O = O - O.mean(0)
    # initial travel heading; formation is held at this heading during assembly
    hd_start = rng.uniform(0, 2 * math.pi)
    ch, sh = math.cos(hd_start), math.sin(hd_start); Oxy = np.stack([ch * O[:, 0] - sh * O[:, 1], sh * O[:, 0] + ch * O[:, 1]], 1)
    # locality-preserving launch->slot assignment: sort both into columns along x, then by y inside each column
    def colsort(X):
        o = np.argsort(X[:, 0], kind="stable"); nc = max(1, int(round(math.sqrt(len(X))))); out = []
        for ch_ in np.array_split(o, nc): out += list(ch_[np.argsort(X[ch_, 1], kind="stable")])
        return np.array(out)
    og = colsort(P0[:, :2] - c0[:2]); oo = colsort(Oxy); perm = np.empty(N, int); perm[og] = oo; O = O[perm]; Oxy = Oxy[perm]
    sid = np.zeros(N, int)   # scenario swarm index per drone (0-based); all 0 when SWARMS == 1
    if SWARMS > 1 and N >= 2 * SWARMS:   # split the formation into SWARMS compact groups and push them apart (separate rng stream)
        srng = np.random.default_rng([SEED, 11, BIN_LO, BIN_HI, sk, attempt])
        sid = split_groups(O[:, :2], SWARMS, srng)
        for g in range(SWARMS): O[sid == g] -= O[sid == g].mean(0)
        rg = max(1.5, max(float(np.linalg.norm(O[sid == g, :2], axis=1).max()) for g in range(SWARMS)))
        Rc = (2 + SWARM_SEP) * rg / (2 * math.sin(math.pi / SWARMS)); a0 = srng.uniform(0, 2 * math.pi)
        for g in range(SWARMS):
            a = a0 + 2 * math.pi * g / SWARMS; O[sid == g] += [Rc * math.cos(a), Rc * math.sin(a), srng.uniform(-3, 3)]
        O = O - O.mean(0); Oxy = np.stack([ch * O[:, 0] - sh * O[:, 1], sh * O[:, 0] + ch * O[:, 1]], 1)
    alt = rng.uniform(10, 35); speed = rng.uniform(3, 12); vcl = rng.uniform(3.0, 8.0); vlat = rng.uniform(4.0, 8.0)
    T0 = rng.uniform(0.3, 1.0); stagger = rng.choice(["wave", "random", "simultaneous"])
    if stagger == "wave": tl = T0 + (P0[:, 0] - P0[:, 0].min()) / max(1e-3, np.ptp(P0[:, 0])) * rng.uniform(1.0, 2.5)
    elif stagger == "random": tl = T0 + rng.uniform(0, 2, N)
    else: tl = T0 + rng.uniform(0, 0.3, N)
    # climb at ~vcl m/s (smoothstep peak = 1.5x mean), then join the formation slot laterally at ~vlat m/s
    hz = hm(c0[0], c0[1], "ground") + alt
    Tz = np.maximum(1.0, 1.5 * np.maximum(hz + O[:, 2] - P0[:, 2], 1.0) / vcl)
    Txy = np.maximum(1.5, 1.5 * np.linalg.norm(c0[:2] + Oxy - P0[:, :2], axis=1) / vlat)
    t_end = tl + np.maximum(Tz, 0.2 * Tz + Txy)
    t_cruise = float(np.quantile(t_end, 0.6)) + rng.uniform(0.0, 1.0)   # path motion starts once most drones have joined; late joiners chase their moving slot
    if (FORCE_BRK or BRK_COUNT > 0) and F - int(math.ceil(t_cruise * fps)) < 60:   # extend cruise so a breakaway fits (>= 60 frames after cruise start)
        F = int(math.ceil(t_cruise * fps)) + int(rng.integers(60, 91)); T = np.arange(F) * dt
    # centroid path: hover above launch site, then move along a smooth random path inside the terrain
    hov = np.array([c0[0], c0[1], hm(c0[0], c0[1], "ground") + alt])
    Tm = max(0.0, T[-1] - t_cruise); L = speed * Tm * 1.3 + 5
    for _ in range(100):
        K = 4; hd0 = hd_start; wps = [hov]
        for k in range(1, K):
            hd0 += rng.normal(0, 0.8) if k > 1 else 0.0; st = L / (K - 1)
            q = wps[-1][:2] + st * np.array([math.cos(hd0), math.sin(hd0)])
            wps.append(np.array([q[0], q[1], hm(q[0], q[1], "ground") + alt + rng.normal(0, 5)]))
        wps = np.array(wps); ii = ((wps[:, 1] - hm.y0) / hm.res).astype(int); jj = ((wps[:, 0] - hm.x0) / hm.res).astype(int)
        if ((ii >= 0) & (ii < hm.valid.shape[0]) & (jj >= 0) & (jj < hm.valid.shape[1])).all() and hm.valid[ii, jj].all(): break
        if _ % 5 == 4: hd_start = rng.uniform(0, 2 * math.pi)
    # dense path sampled by arc-length so speed is ~constant (with smooth start)
    tt = np.linspace(0, K - 1, 2000); pp = catmull(wps, tt); seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pp, axis=0), axis=1))]
    def centroid(t):
        tm = np.clip(t - t_cruise, 0, None); ramp = 2.0
        d = speed * np.where(tm < ramp, tm ** 2 / (2 * ramp), tm - ramp / 2)
        d = np.clip(d, 0, seg[-1]); c = np.stack([np.interp(d, seg, pp[:, j]) for j in range(3)], 1)
        dd = np.clip(d + 0.5, 0, seg[-1]); c2 = np.stack([np.interp(dd, seg, pp[:, j]) for j in range(3)], 1)
        hdg = np.arctan2(c2[:, 1] - c[:, 1], c2[:, 0] - c[:, 0]); hdg = np.where(d + 0.5 <= seg[-1], hdg, np.nan)
        return c, hdg
    Cc, hdg = centroid(T)
    fin = np.isfinite(hdg)
    if fin.any():
        last = hdg[np.argmax(fin)]
        for f in range(F):
            if fin[f]: last = hdg[f]
            else: hdg[f] = last
        hdg = np.unwrap(hdg)
    else: hdg = np.zeros(F)
    # rate-limit formation yaw so the formation tips stay below ~4 m/s extra speed
    Rf = max(2.0, float(np.linalg.norm(O[:, :2], axis=1).max())); wmax = float(np.clip(4.0 / Rf, 0.05, 0.5)) / fps
    for f in range(1, F): hdg[f] = hdg[f - 1] + np.clip(hdg[f] - hdg[f - 1], -wmax, wmax)
    # formation rotation: before cruise, keep the initial heading
    h0 = hdg[np.searchsorted(T, t_cruise)] if t_cruise < T[-1] else hdg[-1]
    hdg = np.where(T < t_cruise, h0, hdg)
    cs, sn = np.cos(hdg), np.sin(hdg)
    slot = np.zeros((F, N, 3))
    slot[:, :, 0] = Cc[:, None, 0] + cs[:, None] * O[None, :, 0] - sn[:, None] * O[None, :, 1]
    slot[:, :, 1] = Cc[:, None, 1] + sn[:, None] * O[None, :, 0] + cs[:, None] * O[None, :, 1]
    slot[:, :, 2] = Cc[:, None, 2] + O[None, :, 2]
    # takeoff blend: vertical first, then lateral
    tr = T[:, None] - tl[None, :]
    bz = smooth(tr / Tz[None]); bxy = smooth((tr - 0.2 * Tz[None]) / Txy[None])
    Pos = np.empty((F, N, 3))
    Pos[:, :, :2] = P0[None, :, :2] + (slot[:, :, :2] - P0[None, :, :2]) * bxy[:, :, None]
    Pos[:, :, 2] = P0[None, :, 2] + (slot[:, :, 2] - P0[None, :, 2]) * bz
    # small per-drone jitter (OU) while airborne
    jit = np.zeros((F, N, 3)); th = 0.6; sg = 0.25 * min(1.0, sp / 3) * OVR.get("jitter", 1.0)   # jitter: manifest turbulence multiplier (wind proxy)
    for f in range(1, F): jit[f] = jit[f - 1] - th * jit[f - 1] * dt + sg * math.sqrt(dt) * rng.normal(0, 1, (N, 3))
    Pos += jit * smooth(tr / 1.5)[:, :, None]
    phase = np.where(tr <= 0, 0, np.where(T[:, None] < t_end[None], 1, 2))   # 0 ground, 1 takeoff, 2 cruise
    # breakaways
    brk = []; is_brk = np.zeros((F, N), bool); bstart = np.full(N, -1)
    f_cr = int(math.ceil(t_cruise * fps))
    if (rng.random() < BRK_PROB or FORCE_BRK) and F - f_cr > 35:
        nb = int(rng.integers(1, 4)) if N >= 6 else 1
        if BRK_COUNT > 0: nb = max(1, min(BRK_COUNT, N // (BRK_GROUP + 1)))
        who = rng.choice(N, nb, replace=False)
        for b in who:
            if BRK_GROUP > 1 and is_brk[:, b].any(): continue   # already taken by an earlier subgroup
            fb = int(rng.integers(f_cr + 5, F - 25)); kind = str(rng.choice(["dive", "climb", "sharp_turn"]))
            v0 = (Pos[fb, b] - Pos[fb - 1, b]) * fps; vs = max(np.linalg.norm(v0[:2]), 3.0)
            if kind == "dive":
                tgt = Pos[fb, b] + np.r_[v0[:2] / max(1e-3, np.linalg.norm(v0[:2])) * rng.uniform(20, 60) + rng.normal(0, 10, 2), 0]; tgt[2] = hm(tgt[0], tgt[1], "h") + 1.0
                vd = (tgt - Pos[fb, b]); vd = vd / np.linalg.norm(vd) * rng.uniform(1.3, 2.0) * vs
            elif kind == "climb":
                vd = np.r_[v0[:2] * rng.uniform(0.3, 1.0), rng.uniform(4, 9)]; tgt = None
            else:
                a = math.atan2(v0[1], v0[0]) + rng.choice([-1, 1]) * math.radians(rng.uniform(60, 150)); sp2 = vs * rng.uniform(1.2, 1.8)
                vd = np.r_[sp2 * math.cos(a), sp2 * math.sin(a), rng.normal(0, 1)]; tgt = Pos[fb, b] + vd * (F - fb) * dt
            tb = rng.uniform(0.7, 1.5); p = Pos[fb, b].copy(); v = v0.copy()
            mem = [int(b)]
            if BRK_GROUP > 1:   # subgroup: nearest same-swarm drones that are not breakaways yet leave rigidly with the leader
                cand = np.nonzero((sid == sid[b]) & ~is_brk[fb] & (np.arange(N) != b))[0]
                cand = cand[np.argsort(np.linalg.norm(Pos[fb, cand] - Pos[fb, b], axis=1))][:BRK_GROUP - 1]; mem += [int(c) for c in cand]
            base = Pos[fb, mem].copy(); jb = jit[fb, mem].copy()
            for f in range(fb + 1, F):
                w = min(1.0, (f - fb) * dt / tb); v = v0 * (1 - w) + vd * w
                p = p + v * dt; zmin = float(hm(p[0], p[1], "h")) + 0.8
                if p[2] < zmin: p[2] = min(zmin, Pos[f - 1, b, 2] + 8 * dt)
                Pos[f, b] = p
                for j, m in enumerate(mem[1:], 1):
                    q = base[j] + (p - base[0]) + (jit[f, m] - jb[j]); q[2] = max(q[2], float(hm(q[0], q[1], "h")) + 0.8); Pos[f, m] = q
            is_brk[fb:, mem] = True; bstart[mem] = fb
            brk.append({"id": int(b) + 1, "start_frame": fb + 1, "start_time_s": round(fb * dt, 3), "type": kind, "target_m": None if tgt is None else [round(float(x), 2) for x in tgt],
                        "members": [m + 1 for m in mem], "swarm_id": int(sid[b]) + 1})
    Vel = np.gradient(Pos, dt, axis=0)
    # orientation: yaw along motion (or launch yaw), tilt with acceleration
    Acc = np.gradient(Vel, dt, axis=0); yaw0 = rng.uniform(0, 360)
    hv = np.linalg.norm(Vel[:, :, :2], axis=2); yaw = np.where(hv > 0.8, np.degrees(np.arctan2(Vel[:, :, 1], Vel[:, :, 0])), np.nan)
    for n in range(N):
        y = yaw[:, n]; last = yaw0
        for f in range(F):
            if np.isnan(y[f]): y[f] = last
            last = y[f]
    pitch = np.clip(np.degrees(np.arctan2(np.linalg.norm(Acc[:, :, :2], axis=2), 9.81)), 0, 25)
    # camera
    allC = Pos[:, ~np.any(is_brk, 0)].reshape(-1, 3) if (~np.any(is_brk, 0)).any() else Pos.reshape(-1, 3)
    lo, hi = np.percentile(allC, 2, 0), np.percentile(allC, 98, 0); mid = (lo + hi) / 2; ext = float(np.linalg.norm(hi[:2] - lo[:2])) + 10
    for cam_try in range(40 if CAM_FRAMING != "chase" else 0):
        cmode = str(rng.choice(["static", "static_pan", "moving"], p=[0.45, 0.3, 0.25])); ground_cam = rng.random() < 0.15
        for _ in range(100):
            a = rng.uniform(0, 2 * math.pi); dist = (ext * rng.uniform(0.9, 1.8) + rng.uniform(30, 120)) * CAM_DIST
            exy = mid[:2] + dist * np.array([math.cos(a), math.sin(a)])
            ez = hm(exy[0], exy[1], "ground") + (rng.uniform(1.5, 2.0) if ground_cam else rng.uniform(20, 100))
            if ez > hm(exy[0], exy[1], "h") + (0.3 if ground_cam else 3): break
        eye0 = np.array([exy[0], exy[1], ez])
        cam_vel = np.zeros(3)
        if cmode == "moving":
            b = rng.uniform(0, 2 * math.pi); cam_vel = np.r_[rng.uniform(0.5, 3) * np.array([math.cos(b), math.sin(b)]), rng.uniform(-0.3, 0.3)]
        eyes = eye0[None] + T[:, None] * cam_vel[None]
        eyes[:, 2] = np.maximum(eyes[:, 2], hm(eyes[:, 0], eyes[:, 1], "h") + (1.5 if ground_cam else 3))
        if cmode == "static":
            tg = np.repeat(((lo + hi) / 2)[None], F, 0)
            # focal so the whole trajectory fits
            d0 = np.linalg.norm(mid - eye0); fov_need = 2 * math.atan2(ext * 0.6, d0)
            focal = float(np.clip(20.955 / 2 / math.tan(fov_need / 2), 10, 45))
        else:
            swarmc = np.array([Pos[f][~is_brk[f]].mean(0) if (~is_brk[f]).any() else Pos[f].mean(0) for f in range(F)])
            tg = swarmc.copy(); a_s = 1 - math.exp(-dt / 1.5)
            for f in range(1, F): tg[f] = tg[f - 1] + a_s * (swarmc[f] - tg[f - 1])
            d0 = np.linalg.norm(swarmc - eyes, axis=1).mean(); R = max(np.ptp(O[:, :2], 0).max() / 2, 3) * 3 + 15
            focal = float(np.clip(20.955 / 2 / math.tan(math.atan2(R, d0)), 10, 60))
        hf = max(hdr_ground_fraction(eyes[f], tg[f], focal) for f in (0, F // 2, F - 1))
        if hf <= 0.03: break
    if CAM_FRAMING == "chase":
        swarmc = np.array([Pos[f][~is_brk[f]].mean(0) if (~is_brk[f]).any() else Pos[f].mean(0) for f in range(F)])
        tg = swarmc.copy(); a_s = 1 - math.exp(-dt / 0.15)   # aim at the swarm centre (light smoothing only, a lagging aim point pushes a close-up swarm off-screen)
        for f in range(1, F): tg[f] = tg[f - 1] + a_s * (swarmc[f] - tg[f - 1])
        Dform = float(np.ptp(O[:, :2], 0).max()) + 2 * span
        for cam_try in range(40):
            cmode = "chase"; ground_cam = False
            focal = float(rng.uniform(18, 50)); fxc = focal / 20.955 * W
            d = fxc * Dform / (rng.uniform(0.35, 0.85) * W)            # formation fills 35-85 % of the image width
            cap = fxc * span / rng.uniform(30, 60)   # (several swarms: frame them all, drop the drone-size cap)
            d = float(np.clip(min(d, cap) if (SWARMS == 1 and not OVR.get("chase_fit")) else d, 6.0, 400.0)) * CAM_DIST   # ...but keep drones ~30-60 px wide (short side ~16+ px)
            el = rng.uniform(8, 40); el = math.radians(OVR.get("cam_el_deg", el)); az = rng.uniform(0, 2 * math.pi) + math.radians(rng.uniform(-4, 4)) * T   # slow orbit
            off_ = d * np.stack([math.cos(el) * np.cos(az), math.cos(el) * np.sin(az), np.full(F, math.sin(el))], 1)
            eyes = tg + off_ + np.cumsum(rng.normal(0, 0.02, (F, 3)), 0)   # small hand-held drift
            eyes[:, 2] = np.maximum(eyes[:, 2], hm(eyes[:, 0], eyes[:, 1], "h") + 3)
            hf = max(hdr_ground_fraction(eyes[f], tg[f], focal) for f in (0, F // 2, F - 1))
            if hf <= 0.03: break
    sunp = dict(el=float(rng.uniform(10, 80)), az=float(rng.uniform(0, 360)), inten=float(rng.uniform(0.5, 1.6)), ct=float(rng.uniform(4200, 7500)))
    sky = dict(tex=int(rng.integers(len(SKIES))), inten=float(rng.uniform(400, 1300)), rot=float(rng.uniform(0, 360)))
    for k, o in (("el", "sun_el"), ("az", "sun_az"), ("inten", "sun_int"), ("ct", "sun_ct")):
        if o in OVR: sunp[k] = float(OVR[o])
    if "sky_tex" in OVR: sky["tex"] = int(OVR["sky_tex"]) % len(SKIES)
    if "sky_int" in OVR: sky["inten"] = float(OVR["sky_int"])
    arng = np.random.default_rng([SEED, 7, BIN_LO, BIN_HI, sk, attempt])   # separate stream: colours never change the scene design
    aps = [sample_appearance(arng, CMODE, CSTR, CWEAR) for _ in range(N)]
    return dict(sid=sid, rng=rng, aps=aps, fps=fps, F=F, T=T, N=N, idx=idx, types=types, spans=spans, Pos=Pos, Vel=Vel, yaw=yaw, pitch=pitch, phase=phase, is_brk=is_brk, bstart=bstart, brk=brk,
                launch=launch, form=form, spacing=sp, alt=alt, vcl=vcl, speed=speed, stagger=stagger, t_cruise=t_cruise, cmode=cmode, ground_cam=ground_cam, hdr_frac=hf, eyes=eyes, tg=tg, focal=focal, sun=sunp, sky=sky)

PH = ["ground", "takeoff", "cruise"]

def setup_scene(D):
    set_sun(**D["sun"]); dome.GetTextureFileAttr().Set(SKIES[D["sky"]["tex"]]); dome.GetIntensityAttr().Set(D["sky"]["inten"]); dome_rot.Set(D["sky"]["rot"])
    cam.GetFocalLengthAttr().Set(D["focal"])
    pool.set_appearance(D["idx"], D["aps"])
    if REP_WRITER:   # Replicator semantic taxonomy 'swarm' = scenario swarm id (pool prims are reused, so relabel every sequence)
        from isaacsim.core.utils.semantics import add_labels
        for k, i in enumerate(D["idx"]): add_labels(stage.GetPrimAtPath(pool.paths[int(i)]), labels=[f"swarm_{int(D['sid'][k]) + 1}"], instance_name="swarm")

def render_frame(D, f, rt):
    N = D["N"]; pool2k = D["pool2k"]
    quats = [quat_from_euler_deg(0.0, float(D["pitch"][f, n]), float(D["yaw"][f, n])) for n in range(N)]
    pool.set(D["idx"], D["Pos"][f], quats, D["spans"])
    cam_op.Set(look_at_matrix(D["eyes"][f], D["tg"][f], M))
    app.update(); rep.orchestrator.step(rt_subframes=rt)
    sdat = A["instance_id_segmentation_fast"].get_data(); seg = np.asarray(sdat["data"]).astype(np.int64)
    lut = np.zeros(int(seg.max()) + 1, np.int32)
    for key, pth in sdat["info"]["idToLabels"].items():
        pth = str(pth)
        if pth.startswith("/Root/DS/d_"):
            k = pool2k.get(int(pth[11:15]))
            if k is not None and int(key) < len(lut): lut[int(key)] = k + 1
    mask = lut[seg].astype(np.uint16)   # mask value = track_id (k + 1), stable for the whole sequence
    mpx = np.bincount(mask.ravel(), minlength=N + 1)[1:N + 1]
    return mask, mpx

def pick_window(D):
    F = D["F"]
    if CLIP <= 0 or CLIP >= F: return 0, F
    f_cr = int(math.ceil(D["t_cruise"] * D["fps"])); lo = min(f_cr, F - CLIP)
    if BRK_COUNT > 0 and D["brk"]:   # scripted breakaways: put the first breakaway start ~1/4 into the clip
        fb = min(b["start_frame"] for b in D["brk"]) - 1; return int(np.clip(fb - ((OVR["brk_at"] - 1) if "brk_at" in OVR else CLIP // 4), 0, F - CLIP)), CLIP
    return int(D["rng"].integers(lo, F - CLIP + 1)), CLIP

def render_seq(s, D, off, Fw):
    """Render frames [off, off+Fw) of design D into OUT/seq_sss (output frames numbered 1..Fw). Returns per-frame n_visible_ge1px."""
    F, N = D["F"], D["N"]
    sd = os.path.join(OUT, seq_name(s))
    if os.path.exists(sd): shutil.rmtree(sd)
    for d in ("img", "masks", "gt", "ann"): os.makedirs(os.path.join(sd, d))
    fx = D["focal"] / 20.955 * W
    gt = []; nvis = []; fj = open(os.path.join(sd, "ann", "frames.jsonl"), "w")
    writer = None
    if REP_WRITER:   # Replicator BasicWriter: one written frame per orchestrator.step below (file index = output frame - 1)
        writer = rep.WriterRegistry.get("BasicWriter")
        writer.initialize(output_dir=os.path.join(sd, "replicator"), semantic_types=["swarm"], rgb=False, bounding_box_2d_tight=True,
                          instance_id_segmentation=True, semantic_segmentation=True, colorize_semantic_segmentation=False, colorize_instance_id_segmentation=False)
        writer.attach([rp])
    for f in range(off, off + Fw):
        fo = f - off + 1
        mask, mpx = render_frame(D, f, RT if f > off else 12)
        img = np.asarray(A["rgb"].get_data())[..., :3].astype(np.uint8)
        tb = [None] * N; ys, xs = np.nonzero(mask)
        if len(xs):
            mv = mask[ys, xs].astype(np.int64) - 1
            xmn = np.full(N, 1 << 30); ymn = np.full(N, 1 << 30); xmx = np.full(N, -1); ymx = np.full(N, -1)
            np.minimum.at(xmn, mv, xs); np.minimum.at(ymn, mv, ys); np.maximum.at(xmx, mv, xs); np.maximum.at(ymx, mv, ys)
            for k in np.nonzero(xmx >= 0)[0]: tb[k] = [int(xmn[k]), int(ymn[k]), int(xmx[k]), int(ymx[k])]
        occ = {}
        bd = A["bounding_box_2d_tight_fast"].get_data()
        for b, pth in zip(bd["data"], bd["info"].get("primPaths", [])):
            pth = str(pth)
            if pth.startswith("/Root/DS/d_"):
                k = D["pool2k"].get(int(pth[11:15]))
                if k is not None and b["occlusionRatio"] >= 0: occ[k] = float(b["occlusionRatio"])
        cp = A["camera_params"].get_data(); V = np.asarray(cp["cameraViewTransform"], float).reshape(4, 4); Pj = np.asarray(cp["cameraProjection"], float).reshape(4, 4)
        ctr, depth = project(D["Pos"][f], V, Pj)
        sw = ~D["is_brk"][f]; cen = D["Pos"][f][sw].mean(0) if sw.any() else D["Pos"][f].mean(0)
        dc = np.linalg.norm(D["Pos"][f] - cen, axis=1)
        drones = []
        for k in range(N):
            vf = (1 - occ[k]) if k in occ else None
            q = quat_from_euler_deg(0.0, float(D["pitch"][f, k]), float(D["yaw"][f, k]))
            drones.append({"id": k + 1, "track_id": k + 1, "type": D["types"][k], "model": ASSETS[D["types"][k]][0].split("/")[-1], "pool_id": int(D["idx"][k]), "appearance": short_ap(D["aps"][k]),
                           "span_m": round(float(D["spans"][k]), 3), "pos_m": [round(float(x), 3) for x in D["Pos"][f, k]], "vel_mps": [round(float(x), 3) for x in D["Vel"][f, k]],
                           "quat_wxyz": [round(float(x), 4) for x in q], "yaw_deg": round(float(D["yaw"][f, k]), 2), "pitch_deg": round(float(D["pitch"][f, k]), 2),
                           "phase": PH[int(D["phase"][f, k])], "swarm_id": int(D["sid"][k]) + 1, "is_breakaway": bool(D["is_brk"][f, k]), "breakaway_start_frame": int(D["bstart"][k]) - off + 1 if D["bstart"][k] >= 0 else -1,
                           "dist_to_swarm_centroid_m": round(float(dc[k]), 3), "center_px": [round(float(ctr[k, 0]), 1), round(float(ctr[k, 1]), 1)], "depth_m": round(float(depth[k]), 2),
                           "span_px": round(float(fx * D["spans"][k] / max(depth[k], 1e-3)), 2), "mask_px": int(mpx[k]), "bbox_tight": tb[k],
                           "occlusion_ratio": round(occ[k], 4) if k in occ else None, "visible_fraction": round(vf, 4) if vf is not None else None})
            if tb[k] is not None:
                x1, y1, x2, y2 = tb[k]; gt.append(f"{fo},{k + 1},{x1},{y1},{x2 - x1 + 1},{y2 - y1 + 1},1,1,{(vf if vf is not None else 1.0):.3f}")
        nv = int((mpx > 0).sum()); nvis.append(nv)
        rec = {"frame": fo, "source_frame": f + 1, "time_s": round(float(D["T"][f]), 4), "image": f"img/{fo:06d}.jpg", "mask": f"masks/{fo:06d}.png", "swarm_centroid_m": [round(float(x), 3) for x in cen],
               "n_total": N, "n_visible_ge1px": nv, "n_breakaway_active": int(D["is_brk"][f].sum()),
               "camera": {"eye_m": [round(float(x), 3) for x in D["eyes"][f]], "target_m": [round(float(x), 3) for x in D["tg"][f]], "focal_mm": round(D["focal"], 3), "fx": round(fx, 2), "cx": W / 2, "cy": H / 2,
                          "cameraViewTransform": V.ravel().tolist(), "cameraProjection": Pj.ravel().tolist()}, "drones": drones}
        fj.write(json.dumps(rec, separators=(",", ":")) + "\n")
        Image.fromarray(img).save(os.path.join(sd, "img", f"{fo:06d}.jpg"), quality=92)
        Image.fromarray(mask).save(os.path.join(sd, "masks", f"{fo:06d}.png"), compress_level=6)
    fj.close()
    if writer is not None: rep.orchestrator.wait_until_complete(); writer.detach()
    open(os.path.join(sd, "gt", "gt.txt"), "w").write("\n".join(gt) + "\n")
    open(os.path.join(sd, "seqinfo.ini"), "w").write(f"[Sequence]\nname={seq_name(s)}\nimDir=img\nframeRate={D['fps']}\nseqLength={Fw}\nimWidth={W}\nimHeight={H}\nimExt=.jpg\n")
    return sd, nvis

for s in todo:
    t_seq = time.time(); done = False
    for att in range(MAX_ACCEPT if BINNED else 1):
        if MROWS: apply_row(MROWS[s])
        try: D = design(s, att)
        except Exception as e:
            import traceback; log("DESIGN FAILED seq", s, "try", att, repr(e), traceback.format_exc()[-1500:]); continue
        D["pool2k"] = {int(p): k for k, p in enumerate(D["idx"])}
        off, Fw = pick_window(D); setup_scene(D)
        if BINNED and NPROBE > 0:   # cheap pre-check on a few frames before rendering the whole sequence
            pf = sorted(set(np.linspace(off, off + Fw - 1, min(NPROBE, Fw)).round().astype(int).tolist()))
            pv = [int((render_frame(D, f, 2)[1] > 0).sum()) for f in pf]
            if not (BIN_LO <= float(np.median(pv)) <= BIN_HI):
                log(f"seq {s} try {att} probe reject N={D['N']} probe_visible={pv} want {VISIBLE_BIN}"); continue
        sd, nvis = render_seq(s, D, off, Fw)
        med = float(np.median(nvis))
        if BINNED and not (BIN_LO <= med <= BIN_HI):
            log(f"seq {s} try {att} full-render reject N={D['N']} median_visible={med} want {VISIBLE_BIN}"); shutil.rmtree(sd, ignore_errors=True); continue
        F, N = D["F"], D["N"]
        brk = [dict(b, start_frame=b["start_frame"] - off) for b in D["brk"]]
        meta = {"seq": s, "fps": D["fps"], "frames": Fw, "n_drones": N, "types": {t: D["types"].count(t) for t in set(D["types"])}, "launch": D["launch"], "formation": D["form"], "spacing_m": round(D["spacing"], 2),
                "cruise_alt_agl_m": round(D["alt"], 1), "climb_rate_mps": round(D["vcl"], 2), "cruise_speed_mps": round(D["speed"], 2), "takeoff_stagger": D["stagger"], "cruise_start_time_s": round(D["t_cruise"], 2),
                "camera_mode": D["cmode"], "offterrain_view_frac": round(D["hdr_frac"], 3), "ground_camera": bool(D["ground_cam"]), "focal_mm": round(D["focal"], 2), "sun": D["sun"], "sky": dict(D["sky"], file=SKIES[D["sky"]["tex"]].split("/")[-1]),
                "has_breakaway": bool(brk), "breakaways": brk, "gt_format": "MOT: frame,id,x,y,w,h,conf,class,visibility (visible drones only, tight mask box)",
                "seed": SEED, "manifest_row": MROWS[s] if MROWS else None, "design_try": att, "camera_framing": CAM_FRAMING, "cam_dist": CAM_DIST, "source_frames": F, "source_frame_offset": off,
                "visible_bin": VISIBLE_BIN, "bin_lo": BIN_LO if BINNED else None, "bin_hi": BIN_HI if BINNED else None,
                "bin_rule": "median over rendered frames of n_visible_ge1px", "visible_median": med, "visible_first_frame": nvis[0], "visible_min": min(nvis), "visible_max": max(nvis),
                "id_note": "drones[].id == drones[].track_id is persistent for the whole sequence and equals the mask pixel value; unique only within this sequence (global id = sequence + track_id)",
                "track_ids": [{"track_id": k + 1, "type": D["types"][k], "model": ASSETS[D["types"][k]][0].split("/")[-1], "pool_id": int(D["idx"][k]), "span_m": round(float(D["spans"][k]), 3), "license": ASSETS[D["types"][k]][1], "appearance": short_ap(D["aps"][k])} for k in range(N)],
                "model_mix": os.environ.get("MODEL_MIX", "legacy"), "color_mode": CMODE, "color_strength": CSTR, "color_wear": CWEAR,
                "render_seconds": round(time.time() - t_seq, 1),
                "scenario": {"swarms": SWARMS, "swarm_sep_radii": SWARM_SEP, "spread": SPREAD, "brk_prob": BRK_PROB, "brk_count": BRK_COUNT, "brk_group": BRK_GROUP, "replicator_writer": REP_WRITER},
                "swarms": [{"swarm_id": g + 1, "n_drones": int((D["sid"] == g).sum()), "track_ids": [int(k) + 1 for k in np.nonzero(D["sid"] == g)[0]]} for g in range(int(D["sid"].max()) + 1)]}
        json.dump(meta, open(os.path.join(sd, "meta.json"), "w"), indent=1)
        if SWARM_ANN:
            try:
                import swarm_annotate; r = swarm_annotate.annotate(sd); log(f"seq {s} swarm annotations: {r}")
            except Exception as e: log(f"seq {s} SWARM ANNOTATION FAILED (re-run ds_tools/swarm_annotate.py {sd}):", repr(e))
        log(f"seq {s} done try={att} N={N} frames={Fw}/{F} off={off} vis_median={med} first={nvis[0]} bin={VISIBLE_BIN} fps={D['fps']} brk={[(b['id'], b['start_frame'], b['type']) for b in brk]} cam={D['cmode']} {time.time()-t_seq:.0f}s vram peak {peak['mb']}")
        done = True; break
    if not done: log("FAILED seq", s, "after", MAX_ACCEPT if BINNED else 1, "designs; leaving it for resume")
app.close()
