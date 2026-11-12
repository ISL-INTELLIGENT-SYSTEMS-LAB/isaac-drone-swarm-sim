import os, collections, re
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
import omni.usd, omni.replicator.core as rep
from isaacsim.core.utils.stage import is_stage_loading
from pxr import Usd, UsdGeom
ctx = omni.usd.get_context(); ctx.open_stage(os.path.join(os.path.dirname(os.path.abspath(__file__)), "stages", "desert_drones.usda"))
n = 0
while True:
    app.update(); n += 1
    if not is_stage_loading() and n > 60: break
st = ctx.get_stage()
print("ANNOTATORS", sorted(rep.AnnotatorRegistry.get_registered_annotators()), flush=True)
for p in st.GetPrimAtPath("/Root").GetChildren():
    print("TOP", p.GetPath(), p.GetTypeName(), len(p.GetChildren()))
def walk(pr, depth):
    ch = pr.GetChildren()
    toks = collections.Counter(re.sub(r"[_\d]+$", "", c.GetName()) for c in ch)
    print("NAMES", pr.GetPath(), len(ch), toks.most_common(60))
    if depth > 0:
        for c in ch[:6]:
            if len(c.GetChildren()) > 3: walk(c, depth - 1)
for top in ["/Root/Statics", "/Root/Landscape"]:
    pr = st.GetPrimAtPath(top)
    if pr: walk(pr, 1)
cnt = collections.Counter(p.GetTypeName() for p in st.Traverse())
print("COUNT", cnt.most_common(15))
for mod in ["isaacsim.sensors.rtx", "omni.replicator.core.scripts.annotators"]:
    try:
        __import__(mod); print("IMPORT ok", mod)
    except Exception as e: print("IMPORT fail", mod, e)
app.close()
