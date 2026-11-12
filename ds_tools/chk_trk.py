import json,sys,numpy as np,collections
L=[json.loads(l) for l in open(sys.argv[1]+'/ann/frames.jsonl')]
for f in [0,30,60,100,140,160,180,195]:
    d=L[f]['drones']; ph=collections.Counter(x['phase'] for x in d)
    sp=[np.linalg.norm(x['vel_mps']) for x in d]; b=[x for x in d if x['is_breakaway']]
    spx=np.array([x['span_px'] for x in d if x['mask_px']>0])
    print(f,dict(ph),'vis',L[f]['n_visible_ge1px'],'span_px med %.1f'%(np.median(spx) if len(spx) else 0),'vmax %.1f'%max(sp),'brk',[(x['id'],x['breakaway_start_frame'],round(x['dist_to_swarm_centroid_m'],1)) for x in b])
print('global vmax',max(np.linalg.norm(x['vel_mps']) for l in L for x in l['drones']))
