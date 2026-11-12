from PIL import Image, ImageDraw, ImageFont
O="/workspace/renders/all_outputs"; TW,TH=640,360; HB=34
FB=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",20); FR=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",17)
def text_tile(lines,col):
    im=Image.new("RGB",(TW,TH),(25,25,25)); d=ImageDraw.Draw(im); y=40
    for l in lines: d.text((20,y),l,fill=col,font=FR); y+=26
    return im
T=[("rgb","01_rgb.png"),("distance_to_camera","02_distance_to_camera.png"),("distance_to_image_plane","02_distance_to_image_plane.png"),("normals","03_normals.png"),
("semantic_segmentation","04_semantic_segmentation.png"),("instance_segmentation","05_instance_segmentation.png"),("instance_id_segmentation","06_instance_id_segmentation.png"),("bounding_box_2d tight+loose (drones)","07_bounding_box_2d_tight_loose.png"),
("bounding_box_2d_tight (all classes)","08_bounding_box_2d_all_classes.png"),("bounding_box_3d (projected)","09_bounding_box_3d_projected.png"),("occlusion / visible fraction","10_occlusion_visible_fraction.png"),("motion flow (DERIVED, annotator=0)","11_motion_flow_derived.png"),
("pointcloud (novel view)","12_pointcloud_novel_view.png"),("RTX lidar (Example_Rotary)","13_rtx_lidar.png"),("camera_params + drone poses","14_camera_params_and_drone_poses.png"),
("skeleton_data: FAILED",["Kit crashed natively (exit -1073740791)","in omni.graph.image.core, in the full run","and when run alone. Excluded.","(drones have no skeleton/joints anyway)"]),
("RTX radar: FAILED",["IsaacSensorCreateRtxRadar returned None","for config 'Example' and default.","No radar data produced."]),
("motion_vectors annotator: ZERO",["Raw annotator max |v| = 0.00037 px","across 8 capture variants (rt steps,","time-sampled xforms, timeline play).","Tile 11 is analytic flow computed from","depth + camera + USD poses instead.","Raw zeros kept in motion_vectors.npz"])]
cols=3; rows=(len(T)+cols-1)//cols
S=Image.new("RGB",(cols*TW,rows*(TH+HB)),(10,10,10)); d=ImageDraw.Draw(S)
for i,(name,src) in enumerate(T):
    x,y=(i%cols)*TW,(i//cols)*(TH+HB)
    bad=isinstance(src,list); im=text_tile(src,(255,160,160) if "ZERO" not in name else (255,220,120)) if bad else Image.open(f"{O}/{src}").convert("RGB").resize((TW,TH),Image.LANCZOS)
    S.paste(im,(x,y+HB)); d.text((x+8,y+6),name,fill=(255,90,90) if bad else (255,255,255),font=FB)
S.save("/workspace/renders/all_outputs_sheet.png"); print(S.size)
# DR sheet
W2,H2=800,450; D=Image.new("RGB",(3*W2,2*(H2+HB)),(10,10,10)); d=ImageDraw.Draw(D)
import json; dr=json.load(open("/workspace/isl/raw/ao/domain_randomization.json"))
for k in range(3):
    for r,(suf,lab) in enumerate([("_rgb.png","rgb"),("_drone_boxes.png","drone 2D boxes")]):
        p=f"/workspace/isl/raw/ao/dr_{k}{suf}" if r==0 else f"{O}/dr_{k}_drone_boxes.png"
        D.paste(Image.open(p).convert("RGB").resize((W2,H2),Image.LANCZOS),(k*W2,r*(H2+HB)+HB))
        e=dr[k] if isinstance(dr,list) else dr.get("frames",[{}]*3)[k]
        sun=e.get("sun",e) if isinstance(e,dict) else {}
        d.text((k*W2+8,r*(H2+HB)+6),f"DR frame {k}: {lab}  | sun el {e['sun_elevation_deg']} az {e['sun_azimuth_deg']} I {e['sun_intensity']:.0f}",fill=(255,255,255),font=FB)
D.save("/workspace/renders/all_outputs_dr_sheet.png"); print(D.size)
