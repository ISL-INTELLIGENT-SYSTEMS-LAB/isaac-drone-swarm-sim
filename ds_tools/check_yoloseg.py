import json, sys, numpy as np, cv2
from PIL import Image
D = sys.argv[1]; fr = sys.argv[2]
r = json.load(open(f"{D}/ann/{fr}.json")); mask = np.asarray(Image.open(f"{D}/masks/{fr}.png"))
ks = [d["k"] for d in r["drones"] if d["mask_px"] > 0 and d["bbox_tight"]]
L = open(f"{D}/labels/{fr}.txt").read().split("\n"); L = [l for l in L if l]
assert len(L) == len(ks), (len(L), len(ks))
H, W = mask.shape; ious = []; pxs = []
for k, l in zip(ks, L):
    v = np.array(l.split()[1:], float).reshape(-1, 2) * [W, H]
    x1, y1, x2, y2 = r["drones"][k]["bbox_tight"]; pad = 3
    sub = mask[max(0,y1-pad):y2+pad+1, max(0,x1-pad):x2+pad+1] == k + 1
    can = np.zeros(sub.shape, np.uint8); off = [max(0,x1-pad), max(0,y1-pad)]
    # rasterise at 8x then downsample: pixel counted if its centre is inside
    S = 8; big = np.zeros((sub.shape[0]*S, sub.shape[1]*S), np.uint8)
    cv2.fillPoly(big, [np.round((v - off) * S).astype(np.int32)], 1)
    can = big[S//2::S, S//2::S][:sub.shape[0], :sub.shape[1]] > 0
    can |= big.reshape(sub.shape[0], S, sub.shape[1], S).mean((1, 3)) > 0.25
    i = (can & sub).sum(); u = (can | sub).sum(); ious.append(i / u); pxs.append(sub.sum())
ious = np.array(ious); pxs = np.array(pxs)
for lo, hi in [(1, 2), (3, 9), (10, 49), (50, 10**9)]:
    m = (pxs >= lo) & (pxs <= hi)
    if m.any(): print(f"mask px {lo}-{hi}: n={m.sum()} median IoU {np.median(ious[m]):.2f} p10 {np.percentile(ious[m],10):.2f}")
