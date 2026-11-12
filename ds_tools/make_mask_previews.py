import os
r"""Write color previews of counting_v1 uint16 instance masks.
masks_preview\NNNNNN.png      : each drone a distinct color on black
masks_preview_overlay\NNNNNN.jpg : colors blended onto the rendered frame
Usage: python make_mask_previews.py"""
import glob, os, numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
D=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets", "counting_v1")
O1=os.path.join(D,"masks_preview"); O2=os.path.join(D,"masks_preview_overlay")
rng=np.random.default_rng(1); LUT=rng.integers(60,256,(65536,3),dtype=np.uint8); LUT[0]=0
def one(f):
    n=os.path.splitext(os.path.basename(f))[0]
    a=np.array(Image.open(f)); col=LUT[a]
    Image.fromarray(col).save(os.path.join(O1,n+".png"))
    imgs=glob.glob(os.path.join(D,"images",n+".*"))
    if imgs:
        rgb=np.array(Image.open(imgs[0]).convert("RGB"))
        m=a>0; out=rgb.copy(); out[m]=col[m]
        Image.fromarray(out).save(os.path.join(O2,n+".jpg"),quality=92)
if __name__=="__main__":
    os.makedirs(O1,exist_ok=True); os.makedirs(O2,exist_ok=True)
    fs=sorted(glob.glob(os.path.join(D,"masks","*.png")))
    with ProcessPoolExecutor(6) as ex: list(ex.map(one,fs,chunksize=50))
    print("done",len(fs))

