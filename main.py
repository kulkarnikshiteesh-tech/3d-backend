from __future__ import annotations
import os, tempfile, traceback, math, gc
from pathlib import Path
from uuid import uuid4
import cascadio, trimesh, numpy as np
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

STATIC_DIR = Path("static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
async def root():
    return {"status": "ok", "message": "DFM Analyzer Online"}

def analyze_undercuts_advanced(mesh, pull_axis):
    direction = np.array(pull_axis, dtype=float)
    direction /= np.linalg.norm(direction)
    epsilon = mesh.scale * 0.005 # Increased epsilon to jump over mesh noise
    
    # 1. UP CHECK: Are we blocked by something far away?
    origins_up = mesh.triangles_center + (direction * epsilon)
    locations_up, index_ray_up, _ = mesh.ray.intersects_location(origins_up, np.tile(direction, (len(mesh.faces), 1)))
    
    # 2. DOWN CHECK: Are we blocked by something far away?
    origins_down = mesh.triangles_center - (direction * epsilon)
    locations_down, index_ray_down, _ = mesh.ray.intersects_location(origins_down, np.tile(-direction, (len(mesh.faces), 1)))

    # Logic: Only count as an undercut if the obstruction is at least 2mm away.
    # This ignores 'self-collisions' caused by jagged low-poly edges.
    blocked_up = np.zeros(len(mesh.faces), dtype=bool)
    for loc, idx in zip(locations_up, index_ray_up):
        dist = np.linalg.norm(loc - origins_up[idx])
        if dist > 2.0: blocked_up[idx] = True

    blocked_down = np.zeros(len(mesh.faces), dtype=bool)
    for loc, idx in zip(locations_down, index_ray_down):
        dist = np.linalg.norm(loc - origins_down[idx])
        if dist > 2.0: blocked_down[idx] = True

    # Real undercut = Trapped from both sides by distant geometry
    undercut_mask = blocked_up & blocked_down
    
    # Cluster check: Ignore tiny islands of error
    hit_indices = np.where(undercut_mask)[0]
    if len(hit_indices) < 20: return 0.0
    
    try:
        blocked_mesh = mesh.submesh([hit_indices], append=True)
        islands = blocked_mesh.split(only_watertight=False)
        return float(sum(island.area for island in islands if len(island.faces) > 20))
    except:
        return 0.0

@app.post("/upload")
async def upload_step(request: Request, file: UploadFile = File(...)):
    gc.collect()
    tmp_step = Path(tempfile.gettempdir()) / f"{uuid4()}.step"
    glb_path = STATIC_DIR / f"{uuid4()}.glb"
    
    try:
        content = await file.read()
        with tmp_step.open("wb") as f: f.write(content)
        
        # 0.5 precision: Slightly faster and less memory-intensive
        cascadio.step_to_glb(str(tmp_step), str(glb_path), 0.5)
        
        scene = trimesh.load(str(glb_path))
        mesh = trimesh.util.concatenate([g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)])
        
        # Test Z, X, Y
        dirs = {"Z": [0,0,1], "X": [1,0,0], "Y": [0,1,0]}
        results = {n: analyze_undercuts_advanced(mesh, v) for n, v in dirs.items()}
        
        # Best axis with Z-bias
        weighted = {n: v for n, v in results.items()}
        weighted["Z"] *= 0.1
        best_axis = min(weighted, key=weighted.get)
        
        # Area Threshold: 100mm2 of clustered, distant obstructions
        has_u = bool((results[best_axis] * 1e6) > 100.0)

        vol = abs(float(mesh.volume)) * 1e9
        cost = 28000 + (vol/1000 * 0.08) + (45000 if has_u else 0)

        return {
            "glb_url": f"{str(request.base_url).rstrip('/')}/static/{glb_path.name}",
            "volume_cubic_mm": round(vol, 2),
            "has_undercuts": has_u,
            "optimal_axis": best_axis,
            "undercut_message": f"Optimal Pull: {best_axis}. " + ("Side-actions required." if has_u else "Straight-pull compatible."),
            "mold_cost_inr": round(cost, 2)
        }
    except Exception as e:
        return {"error": str(e)}
    finally:
        if tmp_step.exists(): tmp_step.unlink()
        gc.collect()
