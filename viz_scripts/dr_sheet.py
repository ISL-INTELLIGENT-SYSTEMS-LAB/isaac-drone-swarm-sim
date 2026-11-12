import json, numpy as np
from PIL import Image, ImageDraw, ImageFont
R="/workspace/isl/raw/dr2"; O="/workspace/renders/all_outputs"
m=json.load(open(f"{R}/domain_randomization_v2.json"))
FB=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",18)
W2,H2,HB=800,450,30; S=Image.new("RGB",(4*W2,2*(H2+HB)),(10,10,10)); d=ImageDraw.Draw(S)
for c,e in enumerate(m):
    f=e["frame"]
    for r,suf in enumerate(["rgb","boxes"]):
        S.paste(Image.open(f"{R}/dr_{f}_{suf}.png").convert("RGB").resize((W2,H2),Image.LANCZOS),(c*W2,r*(H2+HB)+HB))
    t=("reference (no randomisation)" if f=="ref" else f"DR {f}: sun az {e['sun_azimuth_rot_deg']:+.0f}deg el {e['sun_elevation_deg']}deg x{e['sun_intensity_scale']}, cam jitter")
    d.text((c*W2+8,6),t,fill=(255,255,255),font=FB); d.text((c*W2+8,H2+HB+6),f"drone 2D tight boxes: {len(e['drone_boxes_xyxy'])}/10 (rest off-screen/hidden)",fill=(255,255,255),font=FB)
    # box vs mask agreement
    diffs=[max(abs(a-b) for a,b in zip(e['drone_boxes_xyxy'][k],e['drone_mask_extent_xyxy'][k])) for k in e['drone_boxes_xyxy'] if k in e['drone_mask_extent_xyxy']]
    print(f, "boxes",len(e['drone_boxes_xyxy']),"mask drones",len(e['drone_mask_extent_xyxy']),"max box-vs-mask px diff",max(diffs) if diffs else None, "mean_rgb",round(e['mean_rgb'],1))
S.save("/workspace/renders/all_outputs_dr_sheet.png")
import shutil
for e in m:
    shutil.copy(f"{R}/dr_{e['frame']}_boxes.png",f"{O}/dr_{e['frame']}_drone_boxes.png")
shutil.copy(f"{R}/domain_randomization_v2.json",f"{O}/domain_randomization_v2.json")
