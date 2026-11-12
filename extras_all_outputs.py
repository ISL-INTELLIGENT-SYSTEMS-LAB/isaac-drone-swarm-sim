"""Extras for the all-outputs sample: motion-vector capture variants, RTX lidar/radar single scan, skeleton_data test (last: it crashed Kit before)."""
import os, time, json, threading, subprocess, inspect, traceback
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "samples", "all_outputs")
LAYER = os.path.join(ROOT, "stages", "all_outputs.usda")
W, H = 1280, 720
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
app = SimulationApp({"headless": True, "width": W, "height": H})
import omni.usd, omni.kit.commands, omni.timeline
import numpy as np
from pxr import Usd, UsdGeom, Gf, Sdf
import omni.replicator.core as rep
from isaacsim.core.utils.stage import is_stage_loading
status = {}
def ok(k, v): status[k] = v; log("STATUS", k, "->", v)
def save_status():
    json.dump(status, open(os.path.join(OUT, "extras_status.json"), "w"), indent=1, default=str)
ctx = omni.usd.get_context(); ctx.open_stage(LAYER)
n = 0
while True:
    app.update(); n += 1
    if not is_stage_loading() and n > 60: break
for _ in range(60): app.update()
stage = ctx.get_stage(); mpu = UsdGeom.GetStageMetersPerUnit(stage); M = 1.0 / mpu
log("step signature", str(inspect.signature(rep.orchestrator.step)))
drones = sorted(stage.GetPrimAtPath("/Root/DroneSwarm").GetChildren(), key=lambda p: p.GetName())
tops = [UsdGeom.Xformable(p).GetOrderedXformOps()[0].GetAttr() for p in drones]
base = [a.Get() for a in tops]
rng = np.random.default_rng(3)
vel = [np.array([3.0, 1.0, 0.0]) + rng.normal(0, 1.5, 3) * np.array([1, 1, 0.5]) for _ in drones]   # same as sample script
DT = 0.1
cam_path = "/Root/RenderCams/close"
rp = rep.create.render_product(cam_path, (W, H))
mv = rep.AnnotatorRegistry.get_annotator("motion_vectors"); mv.attach(rp)
seg = rep.AnnotatorRegistry.get_annotator("instance_id_segmentation_fast", init_params={"colorize": False}); seg.attach(rp)
cp = rep.AnnotatorRegistry.get_annotator("camera_params"); cp.attach(rp)
for _ in range(30): app.update()
def stats(tag):
    d = np.asarray(mv.get_data()); a = np.abs(d[..., :2])
    s = (float(a.max()), float(a.mean()), int((a.max(-1) > 1e-3).sum()))
    log("motion", tag, "max/mean/npx>1e-3", s); return d, s
best = None
def setpos(k):
    for a, p0, v in zip(tops, base, vel): a.Set(p0 + Gf.Vec3d(*(v * DT * k * M)))
# variant A: orchestrator step then move + plain app.update()
try:
    setpos(0); rep.orchestrator.step(rt_subframes=8)
    setpos(1)
    for j in range(3):
        app.update(); d, s = stats(f"A_update{j+1}")
        if s[2] > 50 and (best is None or s[2] > best[1][2]): best = (d.copy(), s, f"A_update{j+1}")
except Exception as e: log("variant A fail", e)
# variant B: time-sampled translate + timeline time change
try:
    stage.SetTimeCodesPerSecond(10); stage.SetStartTimeCode(0); stage.SetEndTimeCode(10)
    for a, p0, v in zip(tops, base, vel):
        a.Set(p0, Usd.TimeCode(0)); a.Set(p0 + Gf.Vec3d(*(v * DT * M)), Usd.TimeCode(1)); a.Set(p0 + Gf.Vec3d(*(v * DT * 2 * M)), Usd.TimeCode(2))
    tl = omni.timeline.get_timeline_interface()
    tl.set_current_time(0.0); rep.orchestrator.step(rt_subframes=8, pause_timeline=True) if "pause_timeline" in str(inspect.signature(rep.orchestrator.step)) else rep.orchestrator.step(rt_subframes=8)
    tl.set_current_time(0.1); app.update(); d, s = stats("B_time0.1_update")
    if s[2] > 50 and (best is None or s[2] > best[1][2]): best = (d.copy(), s, "B_time0.1_update")
    tl.set_current_time(0.2); rep.orchestrator.step(rt_subframes=1); d, s = stats("B_time0.2_step_rt1")
    if s[2] > 50 and (best is None or s[2] > best[1][2]): best = (d.copy(), s, "B_time0.2_step_rt1")
    tl.play()
    for j in range(3):
        app.update(); d, s = stats(f"B_play_update{j+1}")
        if s[2] > 50 and (best is None or s[2] > best[1][2]): best = (d.copy(), s, f"B_play_update{j+1}")
    tl.stop()
except Exception as e: log("variant B fail", traceback.format_exc()[-500:])
if best is not None:
    np.savez_compressed(os.path.join(OUT, "motion_vectors_moving.npz"), data=best[0])
    sd = seg.get_data(); np.savez_compressed(os.path.join(OUT, "motion_vectors_moving_instance_id.npz"), data=np.asarray(sd["data"]))
    json.dump({"variant": best[2], "stats_max_mean_npx": best[1], "idToLabels": sd["info"]["idToLabels"], "velocity_mps": [v.tolist() for v in vel], "dt_s": DT,
               "camera_params": {k: np.asarray(v).tolist() if hasattr(v, "__len__") and not isinstance(v, str) else v for k, v in cp.get_data().items()}},
              open(os.path.join(OUT, "motion_vectors_moving.json"), "w"), indent=1, default=str)
    ok("motion_vectors", f"non-zero via {best[2]} stats {best[1]}")
else:
    ok("motion_vectors", "still ~zero in all variants")
save_status()
for a in (mv, seg, cp):
    try: a.detach()
    except Exception: pass
try: rp.destroy()
except Exception: pass
# clear time samples so the stage is static again
for a, p0 in zip(tops, base):
    a.Clear(); a.Set(p0)

# ---------- RTX lidar / radar ----------
try: rep.orchestrator.set_capture_on_play(False)
except Exception: pass
cam = stage.GetPrimAtPath(cam_path); cx = UsdGeom.Xformable(cam).ComputeLocalToWorldTransform(Usd.TimeCode.Default()); eye = cx.ExtractTranslation()
def dump_any(name, d):
    big = {}; info = {}
    for k, v in (d.items() if isinstance(d, dict) else [("data", d)]):
        if isinstance(v, dict):
            info[k] = {kk: (np.asarray(vv).tolist() if isinstance(vv, np.ndarray) and vv.size <= 64 else (f"<array {np.asarray(vv).shape} in npz>" if isinstance(vv, np.ndarray) else vv)) for kk, vv in v.items()}
            for kk, vv in v.items():
                if isinstance(vv, np.ndarray) and vv.size > 64: big[f"{k}.{kk}"] = vv
        elif isinstance(v, np.ndarray) and v.size > 64: big[k] = v; info[k] = f"<array {v.shape} {v.dtype} in npz>"
        else: info[k] = np.asarray(v).tolist() if isinstance(v, np.ndarray) else v
    if big: np.savez_compressed(os.path.join(OUT, f"{name}.npz"), **big)
    json.dump(info, open(os.path.join(OUT, f"{name}.json"), "w"), indent=1, default=str)
def sensor_try(kind, cmd, configs, ann_names, frames=40):
    for cfg in configs:
        try:
            kw = dict(path=f"/Root/Sensors/{kind}_{cfg or 'default'}", parent=None, translation=Gf.Vec3d(*eye), orientation=Gf.Quatd(1, 0, 0, 0))
            if cfg: kw["config"] = cfg
            res = omni.kit.commands.execute(cmd, **kw)
            sensor = res[1] if isinstance(res, tuple) else res
            if sensor is None: log(kind, cfg, "command returned None"); continue
            spath = str(sensor.GetPath())
            log(kind, "created", spath, "config", cfg, "type", stage.GetPrimAtPath(spath).GetTypeName())
            srp = rep.create.render_product(spath, [1, 1], name=f"{kind}_rp")
            for an in ann_names:
                try:
                    a = rep.AnnotatorRegistry.get_annotator(an); a.attach([srp])
                    tl = omni.timeline.get_timeline_interface(); tl.play()
                    best = None; bsz = 0
                    for _ in range(frames):
                        app.update(); d = a.get_data()
                        arr = d.get("data") if isinstance(d, dict) else d
                        sz = np.asarray(arr).size if arr is not None else 0
                        if sz > bsz: best, bsz = d, sz
                    tl.stop()
                    log(kind, an, "max data size", bsz, "keys", list(best.keys()) if isinstance(best, dict) else type(best))
                    if bsz:
                        dump_any(f"{kind}_{an}", best); ok(kind, f"config={cfg} annotator={an} data shape {np.asarray(best['data']).shape}"); save_status(); return
                    a.detach()
                except Exception as e: log(kind, an, "annot fail", str(e)[:300])
            ok(kind, f"sensor created (config={cfg}) but annotators returned no points in {frames} frames"); save_status(); return
        except Exception as e:
            log(kind, "config", cfg, "fail", traceback.format_exc()[-400:])
    ok(kind, "FAILED to create sensor"); save_status()
sensor_try("lidar", "IsaacSensorCreateRtxLidar", ["Example_Rotary", "OS1_REV6", None],
           ["IsaacExtractRTXSensorPointCloudNoAccumulator", "IsaacCreateRTXLidarScanBuffer"])
sensor_try("radar", "IsaacSensorCreateRtxRadar", ["Example", None], ["IsaacExtractRTXSensorPointCloudNoAccumulator", "IsaacCreateRTXRadarPointCloud"])
log("vram so far", peak["mb"])
# ---------- skeleton_data (crashed Kit in the main run) ----------
if os.environ.get("TRY_SKELETON", "1") == "1":
    log("skeleton attempt start (if the log ends here, skeleton_data crashed Kit)"); ok("skeleton_data", "attempt started - crashed if not updated"); save_status()
    try:
        rp2 = rep.create.render_product(cam_path, (W, H))
        sk = rep.AnnotatorRegistry.get_annotator("skeleton_data"); sk.attach(rp2)
        for _ in range(10): app.update()
        rep.orchestrator.step(rt_subframes=4)
        d = sk.get_data(); dump_any("skeleton_data", d); ok("skeleton_data", f"ok keys {list(d.keys()) if isinstance(d, dict) else type(d)}")
    except Exception as e: ok("skeleton_data", "FAILED " + str(e)[:300])
save_status()
log("DONE peak VRAM", peak["mb"])
app.close()
