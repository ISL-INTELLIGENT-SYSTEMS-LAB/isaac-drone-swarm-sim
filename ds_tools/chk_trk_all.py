import os
import json,glob,os
D=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets", "tracking_v1"); rows=[]
for m in sorted(glob.glob(D+r"\seq_*\meta.json")):
    M=json.load(open(m)); sd=os.path.dirname(m)
    L=open(sd+r"\ann\frames.jsonl").readlines(); f0=json.loads(L[0]); fl=json.loads(L[-1])
    keys=set(f0["drones"][0].keys()); need={"pos_m","vel_mps","phase","is_breakaway","breakaway_start_frame","dist_to_swarm_centroid_m"}
    gt=os.path.getsize(sd+r"\gt\gt.txt")>0
    ph0=set(d["phase"] for d in f0["drones"]); 
    brkflag=any(d["is_breakaway"] for l in L[-5:] for d in json.loads(l)["drones"])
    rows.append((M["seq"],M["n_drones"],M["frames"],M["launch"],len(M["breakaways"]),need<=keys,gt,len(L)==M["frames"],ph0=={"ground"},brkflag==bool(M["breakaways"])))
bad=[r for r in rows if not all(r[5:])]
print("seqs",len(rows),"frames",sum(r[2] for r in rows),"with_brk",sum(1 for r in rows if r[4]),"brk_drones",sum(r[4] for r in rows))
print("N sorted",sorted(r[1] for r in rows)); print("launch",{k:sum(1 for r in rows if r[3]==k) for k in ("ground","rooftop")}); print("bad",bad)
