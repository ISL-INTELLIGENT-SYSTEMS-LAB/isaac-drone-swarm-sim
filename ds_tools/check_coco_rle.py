import json, numpy as np, sys
from PIL import Image
def dec(s):
    cnts = []; p = 0
    while p < len(s):
        x = 0; k = 0; more = 1
        while more:
            c = ord(s[p]) - 48; x |= (c & 0x1f) << (5 * k); more = c & 0x20; p += 1; k += 1
            if not more and (c & 0x10): x |= -1 << (5 * k)
        if len(cnts) > 2: x += cnts[-2]
        cnts.append(x)
    return cnts
def decode(seg):
    H, W = seg["size"]; f = np.zeros(H * W, bool); pos = 0; v = False
    for c in dec(seg["counts"]):
        if v: f[pos:pos + c] = True
        pos += c; v = not v
    return f.reshape(W, H).T
c = json.load(open("coco/instances_test.json"))
bad = 0; tot = 0
for im in c["images"][:5]:
    m = np.asarray(Image.open(im["file_name"].replace("images/", "masks/").replace(".jpg", ".png")))
    for a in [a for a in c["annotations"] if a["image_id"] == im["id"]]:
        tot += 1; bad += int(not (decode(a["segmentation"]) == (m == a["drone_k"] + 1)).all())
print("checked", tot, "instances in 5 test images, mismatches", bad)
