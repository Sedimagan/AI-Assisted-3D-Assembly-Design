import sys, numpy as np, trimesh, pyvista as pv
sys.path.insert(0, "/Users/mbp/Documents/MTECH/Sem4/Individual_project/AI_Assisted_3D_Assembly_Design/AI-Assisted-3D-Assembly-Design/back_end")
import rebuild as RB, slot_detector as sd
pv.OFF_SCREEN = True
PROJ="/Users/mbp/Documents/MTECH/Sem4/Individual_project/AI_Assisted_3D_Assembly_Design/AI-Assisted-3D-Assembly-Design"
SRC=f"{PROJ}/Source_3d_models/Best_models_for_training/Tool_Post/Tool_Post_10/Tool_Post_10.stp"
PART=f"{PROJ}/Test_3D_models/Partial_tool_post.step"
OUT=f"{PROJ}/Review_files/toolpost_reconstruction"

GT_EXTRA={74985.6:"shaft_central",9024.5:"shaft_side",32670.1:"knob"}
def fam(v):
    f=sd.family_of_volume(v)
    if f!="unknown": return f
    for k,n in GT_EXTRA.items():
        if abs(v-k)/k<0.01: return n
    return "unknown"

# host bodies are removed by load_part_meshes, so load them separately at coarse size
import gmsh
def host_meshes(path):
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal",0)
    gmsh.option.setNumber("Mesh.MeshSizeMax",6.0)
    gmsh.model.add("h"); gmsh.model.occ.importShapes(path); gmsh.model.occ.synchronize()
    ents=gmsh.model.getEntities(3); keep=[]; drop=[]
    for _,t in ents:
        v=abs(gmsh.model.occ.getMass(3,t)); (keep if v>=sd.HOST_MIN_VOLUME else drop).append((3,t))
    gmsh.model.occ.remove(drop,recursive=True); gmsh.model.occ.synchronize(); gmsh.model.mesh.generate(2)
    out=[]
    for _,t in gmsh.model.getEntities(3):
        V,F,vm=[],[],{}
        for _,s in gmsh.model.getBoundary([(3,t)],oriented=False):
            s=abs(s); nt,nc,_=gmsh.model.mesh.getNodes(2,s,True); _,en=gmsh.model.mesh.getElementsByType(2,s)
            for a,x in zip(nt,np.array(nc).reshape(-1,3)):
                if a not in vm: vm[a]=len(V); V.append(x)
            for tri in np.array(en,int).reshape(-1,3):
                if all(int(q) in vm for q in tri): F.append([vm[int(q)] for q in tri])
        out.append(trimesh.Trimesh(np.array(V),np.array(F),process=False))
    gmsh.finalize(); return out

hosts_o=host_meshes(SRC); hosts_p=host_meshes(PART)
orig=[(fam(s["volume"]),s["mesh"]) for s in RB.load_part_meshes(SRC,1.5)]
have=[(s["family"],s["mesh"]) for s in RB.load_part_meshes(PART,1.5)]
res=RB.rebuild_missing(PART)
rebuilt=[(p["family"],p["mesh"]) for p in res["placed"]]
COL=RB.COLOR_OF
def add(pl,m,color,op):
    f=np.hstack([np.full((len(m.faces),1),3),m.faces]).ravel()
    pl.add_mesh(pv.PolyData(np.asarray(m.vertices),f),color=np.array(color)/255.0,opacity=op,smooth_shading=True,specular=0.25)
allv=np.vstack([np.asarray(m.vertices) for _,m in orig]+[np.asarray(h.vertices) for h in hosts_o])
lo,hi=allv.min(0),allv.max(0); ctr=(lo+hi)/2; rad=float(np.linalg.norm(hi-lo))/2
views={"iso":(1.0,-1.0,0.62),"front":(0.03,-1.0,0.10),"top":(0.03,-0.2,1.0)}
for vn,d in views.items():
    d=np.array(d,float); d/=np.linalg.norm(d); eye=ctr+d*rad*2.7
    pl=pv.Plotter(off_screen=True,window_size=(1500,780),shape=(1,2))
    for k,(title,hosts,parts) in enumerate([("ORIGINAL model (Tool_Post_10)",hosts_o,orig),
                                             ("YOUR TEST FILE + rebuilt parts",hosts_p,have+rebuilt)]):
        pl.subplot(0,k); pl.set_background("white")
        for h in hosts: add(pl,h,(150,158,172),0.16)
        for f,m in parts: add(pl,m,COL.get(f,(140,140,140)),1.0)
        pl.add_text(title,font_size=13,color="black",position="upper_edge")
        pl.camera_position=[tuple(eye),tuple(ctr),(0,0,1)]; pl.camera.zoom(1.12)
    pl.screenshot(f"{OUT}/match_{vn}.png"); pl.close(); print("wrote match_%s.png"%vn)
