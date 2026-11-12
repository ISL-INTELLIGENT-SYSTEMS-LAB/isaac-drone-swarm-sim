"""Print what a USD stage is made of (layers, units, lights, cameras, top prims). Needs the Isaac Sim env.
  .\\env\\python.exe ds_tools\\inspect_stage.py                        # the desert stage the generators use
  .\\env\\python.exe ds_tools\\inspect_stage.py stages\\desert_drones.usda
  .\\env\\python.exe ds_tools\\inspect_stage.py <file-or-url> --prims 200   # also list the first 200 prim paths"""
import os, sys
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
from common import DESERT, ASSETS, settle
import omni.usd
from pxr import Usd, UsdGeom, UsdLux
args = [x for x in sys.argv[1:] if not x.startswith("--")]; nprim = int(sys.argv[sys.argv.index("--prims") + 1]) if "--prims" in sys.argv else 0
args = [x for x in args if not x.isdigit()]
path = os.path.abspath(args[0]) if args and not args[0].startswith("http") else (args[0] if args else DESERT)
ctx = omni.usd.get_context(); ctx.open_stage(path); settle(app); st = ctx.get_stage()
print("STAGE", path)
print("metersPerUnit", UsdGeom.GetStageMetersPerUnit(st), "upAxis", UsdGeom.GetStageUpAxis(st), "defaultPrim", st.GetDefaultPrim().GetPath() if st.GetDefaultPrim() else None)
print("LAYER STACK (strongest first):")
for l in st.GetLayerStack(includeSessionLayers=False): print("  ", l.identifier)
print("TOP-LEVEL PRIMS:", [str(p.GetPath()) for p in st.GetPseudoRoot().GetChildren()])
n = 0; lights = []; cams = []
for p in st.Traverse():
    n += 1
    if p.HasAPI(UsdLux.LightAPI) or "Light" in p.GetTypeName(): lights.append(f"{p.GetPath()} ({p.GetTypeName()})")
    if p.IsA(UsdGeom.Camera): cams.append(str(p.GetPath()))
    if n <= nprim: print("  prim", p.GetPath(), p.GetTypeName())
print("PRIMS", n); print("LIGHTS", lights); print("CAMERAS", cams)
print("DRONE ASSETS used by the generators:"); [print("  ", k, v[0], "|", v[1]) for k, v in ASSETS.items()]
app.close()
