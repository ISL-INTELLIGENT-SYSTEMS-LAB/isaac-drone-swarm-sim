import json,glob,sys
for p in sorted(glob.glob(sys.argv[1]+r'\ann\*.json'))+sorted(glob.glob(sys.argv[1]+r'\labels\*.json')):
    r=json.load(open(p)); d=r['drones']; occ=[x['occlusion_ratio'] for x in d if x['mask_px']>0]
    print(r['frame'],r['n_total'],r['n_in_frustum'],r['n_visible_ge1px'],r['n_visible_ge10px'],r['n_visible_ge50px'],'ann',r['ann_bbox_tight_count'],r['cam_mode'],r['formation'],'occNone',sum(o is None for o in occ),'occ',round(sum(o for o in occ if o is not None)/max(1,len(occ)),2),'eyeagl',r['camera']['eye_agl_m'],'f',r['camera']['focal_mm'],'sun',r['sun']['dir_world'],round(r['sun']['el']))
