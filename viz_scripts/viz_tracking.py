import json, numpy as np, subprocess, os, shutil
from PIL import Image, ImageDraw, ImageFont
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R="/workspace/isl/raw/trk"; O="/workspace/renders/tracking"; os.makedirs(O+"/overlay",exist_ok=True)
meta=json.load(open(f"{R}/sequence_meta.json")); BF=meta["breakaway_start_frame"]; BIDS=set(meta["breakaway_ids"])
F=[json.loads(l) for l in open(f"{R}/frames.jsonl")]
FB=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",14); FS=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",22)
pal=[(0,200,255),(80,255,120),(255,220,0),(255,140,0),(180,120,255),(255,255,255),(0,255,200),(255,0,180),(160,255,0),(120,180,255)]
trail={}
for f in F:
    im=Image.open(f"{R}/frames/{f['frame']:04d}.jpg").convert("RGB"); d=ImageDraw.Draw(im)
    for dr in f["drones"]:
        tid=dr["track_id"]; br=dr["id"] in BIDS; active=br and f["frame"]>=BF
        col=(255,30,30) if active else ((255,120,120) if br else (40,255,90))
        if dr["bbox_xywh"] and dr["visible"]:
            x,y,w,h=dr["bbox_xywh"]; cx,cy=x+w/2,y+h/2; trail.setdefault(tid,[]).append((cx,cy))
            d.rectangle([x-2,y-2,x+w+2,y+h+2],outline=col,width=3 if active else 2)
            lab=f"{dr['id'].replace('drone_','id')}"+(" BREAKAWAY" if active else "")
            d.text((x,y-17),lab,fill=col,font=FB,stroke_width=2,stroke_fill=(0,0,0))
        t=trail.get(tid,[])[-25:]
        if len(t)>1: d.line(t,fill=col,width=2)
    vis=sum(dr["visible"] for dr in f["drones"])
    txt=f"frame {f['frame']:03d}  t={f['t']:.1f}s  visible drones: {vis}/10   " + (f"BREAKAWAY ACTIVE (start frame {BF})" if f['frame']>=BF else f"swarm (breakaway at frame {BF})")
    d.rectangle([0,0,1280,34],fill=(0,0,0)); d.text((10,5),txt,fill=(255,80,80) if f['frame']>=BF else (255,255,255),font=FS)
    im.save(f"{O}/overlay/{f['frame']:04d}.png")
fps=meta["fps"]
subprocess.run(["ffmpeg","-y","-loglevel","error","-framerate",str(fps),"-i",f"{O}/overlay/%04d.png","-c:v","libx264","-pix_fmt","yuv420p","-crf","23","-preset","slow",f"{O}/tracking_overlay.mp4"],check=True)
subprocess.run(["ffmpeg","-y","-loglevel","error","-framerate",str(fps),"-i",f"{O}/overlay/%04d.png","-vf","scale=640:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse",f"{O}/tracking_overlay.gif"],check=True)
for k in (0,BF-1,BF+15,len(F)-1): shutil.copy(f"{O}/overlay/{k:04d}.png",f"{O}/overlay_frame_{k:03d}.png")
shutil.rmtree(f"{O}/overlay")
# trajectories
P=np.array([[dr["position_m"] for dr in f["drones"]] for f in F]); ids=[dr["id"] for dr in F[0]["drones"]]
D=np.array([[dr["dist_to_swarm_centroid_m"] for dr in f["drones"]] for f in F]); T=np.array([f["t"] for f in F])
C=np.array(meta["swarm_centroid_m"])
fig,ax=plt.subplots(1,2,figsize=(14,6.5))
for i,n in enumerate(ids):
    br=n in BIDS; c="red" if br else "tab:green"
    ax[0].plot(P[:,i,0],P[:,i,1],color=c,lw=2.2 if br else 1,alpha=1 if br else .7,label=("breakaway "+n) if br else (None if i else "swarm drones"))
    ax[0].scatter(P[0,i,0],P[0,i,1],color=c,marker="o",s=15); ax[0].scatter(P[BF,i,0],P[BF,i,1],color="k",marker="x",s=25)
    ax[1].plot(T,P[:,i,2],color=c,lw=2.2 if br else 1,alpha=1 if br else .7)
ax[0].plot(C[:,0],C[:,1],"k--",lw=1.5,label="swarm centroid (non-breakaway)")
ax[0].set_aspect("equal"); ax[0].set_xlabel("world X (m)"); ax[0].set_ylabel("world Y (m)"); ax[0].set_title(f"Top-down 3D trajectories (o = start, x = frame {BF})"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
ax[1].axvline(BF/fps,color="r",ls="--",label=f"breakaway frame {BF}"); ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("world Z (m)"); ax[1].set_title("Altitude vs time"); ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{O}/trajectories_topdown.png",dpi=110); plt.close()
fig,ax=plt.subplots(figsize=(11,5))
for i,n in enumerate(ids):
    br=n in BIDS; ax.plot(np.arange(len(F)),D[:,i],color="red" if br else "tab:green",lw=2.2 if br else 1,label=n if br else (None if i else "swarm drones"))
ax.axvline(BF,color="k",ls="--",label=f"breakaway start (frame {BF})"); ax.set_xlabel("frame (10 fps)"); ax.set_ylabel("distance to swarm centroid (m)")
ax.set_title("Distance from swarm centroid over time"); ax.legend(); ax.grid(alpha=.3); plt.tight_layout(); plt.savefig(f"{O}/centroid_distance.png",dpi=110); plt.close()
for fn in ["gt_mot.txt","trajectories_3d.csv","frames.jsonl","sequence_meta.json"]: shutil.copy(f"{R}/{fn}",f"{O}/{fn}")
print("max swarm-drone dist",D[:,[i for i,n in enumerate(ids) if n not in BIDS]].max(),"breakaway final",D[-1,[i for i,n in enumerate(ids) if n in BIDS]])
print("visible counts min",min(sum(dr["visible"] for dr in f["drones"]) for f in F))
