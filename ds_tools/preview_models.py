"""Render every drone model with a few appearance variants (row = model, columns = original + COLOR_MODE variants).
  .\\env\\Scripts\\python.exe ds_tools\\preview_models.py [out.jpg]      env: COLOR_MODE (random|palette), COLOR_STRENGTH, SEED
Used to check that each model loads, is upright, scaled to ~0.8 m, and takes the colour variation."""
import os, sys, json, math
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from isaacsim import SimulationApp
W, H = 1600, 1000
app = SimulationApp({"headless": True, "width": W, "height": H})
from common import *
import omni.replicator.core as rep
from pxr import UsdGeom, UsdLux, Gf
from PIL import Image
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DS, "model_preview.jpg")
MODE = os.environ.get("COLOR_MODE", "random"); STR = float(os.environ.get("COLOR_STRENGTH", "0.85")); NV = 4
stage = open_layer(app, os.path.join(DS, "stages", f"preview_{os.getpid()}.usda"), [DESERT])
mpu = UsdGeom.GetStageMetersPerUnit(stage); M = 1.0 / mpu
hm = HeightMap(os.path.join(DS, "stages", "heightmap.npz"))
dome = UsdLux.DomeLight.Define(stage, "/Root/DS_Sky"); dome.CreateTextureFileAttr(SKIES[0]); dome.CreateTextureFormatAttr("latlong"); dome.CreateIntensityAttr(800.0)
types = list(ASSETS)
pool = Pool(stage, app, NV + 1, NV + 1, NV + 1, M, n_extra={k: NV + 1 for k in types if k not in LEGACY_TYPES}, paint=True)
rng = np.random.default_rng(int(os.environ.get("SEED", "1")))
c = hm.center; gz = float(hm(c[0], c[1], "ground")); z0 = gz + 3.0
idx, P, Q, S, aps, rows = [], [], [], [], [], []
for r, k in enumerate(types):
    cand = [i for i, t in enumerate(pool.types) if t == k][:NV + 1]
    for j, i in enumerate(cand):
        idx.append(i); P.append([c[0] + (j - NV / 2) * 1.4, c[1], z0 + (len(types) - 1 - r) * 1.1]); Q.append(quat_from_euler_deg(0, 0, 30)); S.append(0.8)
        aps.append(None if j == 0 else sample_appearance(rng, MODE, STR)); rows.append((k, j))
pool.set(idx, P, Q, S); pool.set_appearance(idx, aps)
cam = UsdGeom.Camera.Define(stage, "/Root/PCam"); cam.CreateFocalLengthAttr(24.0); cam.CreateHorizontalApertureAttr(20.955); cam.CreateVerticalApertureAttr(20.955 * H / W)
cam.CreateClippingRangeAttr(Gf.Vec2f(0.1 * M, 4000.0 * M))
tgt = np.array([c[0], c[1], z0 + (len(types) - 1) * 0.55]); eye = tgt + np.array([0.0, -9.5, 2.5])
UsdGeom.Xformable(cam).AddTransformOp().Set(look_at_matrix(eye, tgt, M))
rp = rep.create.render_product("/Root/PCam", (W, H)); rgb = rep.AnnotatorRegistry.get_annotator("rgb"); rgb.attach(rp)
for _ in range(60): app.update()
rep.orchestrator.step(rt_subframes=16)
img = np.asarray(rgb.get_data())[..., :3].astype(np.uint8); Image.fromarray(img).save(OUT, quality=92)
json.dump({"rows": types, "sizes_stage_units": {k: v["size"] for k, v in pool.info.items()}, "appearances": [{"type": k, "col": j, "ap": a} for (k, j), a in zip(rows, aps)]},
          open(os.path.splitext(OUT)[0] + ".json", "w"), indent=1)
log("preview written", OUT, "slots", {k: len(pool.slots[pool.types.index(k)]) for k in types})
app.close()
