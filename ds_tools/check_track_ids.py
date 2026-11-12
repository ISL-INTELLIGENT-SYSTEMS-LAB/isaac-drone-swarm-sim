"""Check that track IDs are persistent in tracking / re-ID output. Plain python (numpy + Pillow).
  python ds_tools/check_track_ids.py <tracking_root>
For every sequence: each drone's mask value == its track_id in every frame (pixel count == mask_px),
type/model/appearance (colour) never change for a track_id, the drone list is the same in every frame, and the max per-frame jump of
each track's centre (px) is small (an ID swap would show up as a big jump)."""
import os, sys, json
import numpy as np
from PIL import Image
root = sys.argv[1]; bad = 0; res = []
for dp, dn, fn in sorted(os.walk(root)):
    if "meta.json" not in fn or not os.path.exists(os.path.join(dp, "ann", "frames.jsonl")): continue
    recs = [json.loads(l) for l in open(os.path.join(dp, "ann", "frames.jsonl"))]
    ids0 = [d["track_id"] for d in recs[0]["drones"]]; key = lambda d: (d["type"], d.get("model"), json.dumps(d.get("appearance"), sort_keys=True))
    types = {d["track_id"]: key(d) for d in recs[0]["drones"]}
    mism = 0; tchg = 0; setchg = 0; last = {}; jump = 0.0; seen = {}
    for r in recs:
        if [d["track_id"] for d in r["drones"]] != ids0: setchg += 1
        m = np.asarray(Image.open(os.path.join(dp, r["mask"])))
        cnt = np.bincount(m.ravel(), minlength=max(ids0) + 1)
        for d in r["drones"]:
            t = d["track_id"]
            if d["id"] != t or int(cnt[t]) != d["mask_px"]: mism += 1
            if types[t] != key(d): tchg += 1
            if d["mask_px"] > 0:
                seen[t] = seen.get(t, 0) + 1
                c = np.array(d["center_px"])
                if t in last and last[t][0] == r["frame"] - 1: jump = max(jump, float(np.linalg.norm(c - last[t][1])))
                last[t] = (r["frame"], c)
    meta = json.load(open(os.path.join(dp, "meta.json")))
    mt = {x["track_id"]: (x["type"], x.get("model"), json.dumps(x.get("appearance"), sort_keys=True)) for x in meta.get("track_ids", [])}
    if mt and mt != types: tchg += 1
    cols = {v[2] for v in types.values()}
    ok = mism == 0 and tchg == 0 and setchg == 0; bad += not ok
    res.append({"seq": os.path.relpath(dp, root), "frames": len(recs), "tracks": len(ids0), "tracks_visible_2plus_frames": sum(v >= 2 for v in seen.values()),
                "visible_bin": meta.get("visible_bin"), "visible_median": meta.get("visible_median"), "models": sorted({v[0] for v in types.values()}), "distinct_appearances": len(cols), "mask_ann_mismatch": mism, "type_or_appearance_changes": tchg, "id_set_changes": setchg,
                "max_center_jump_px": round(jump, 1), "persistent_ids_ok": ok})
    print(json.dumps(res[-1]))
print("ALL OK" if bad == 0 and res else f"PROBLEMS in {bad} of {len(res)} sequences")
sys.exit(1 if bad or not res else 0)
