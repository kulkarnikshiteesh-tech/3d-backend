from __future__ import annotations
import os, shutil, tempfile, traceback, math, gc
from pathlib import Path
from uuid import uuid4
import cascadio, trimesh, numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

STATIC_DIR = Path("static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

def analyze_undercuts_bidirectional(mesh, pull_axis):
    """
    True Undercut Logic: A face is only an undercut if it is blocked
    from BOTH the pull direction and the opposite direction.
    """
    direction = np.array(pull_axis, dtype=float)
    direction /= np.linalg.norm(direction)
    epsilon = mesh.scale * 0.001
    
    # 1. Check if blocked going 'UP'
    origins_up = mesh.triangles_center + (direction * epsilon)
    hits_up = mesh.ray.intersects_any(origins_up, np.tile(direction, (len(mesh.faces), 1)))
    
    # 2. Check if blocked going 'DOWN'
    origins_down = mesh.triangles_center - (direction * epsilon)
    hits_down = mesh.ray.intersects_any(origins_down, np.tile(-direction, (len(mesh.faces), 1)))
    
    # A true undercut is trapped in both directions (cannot be in core or cavity)
    undercut_mask = hits_up & hits_down
    
    if not np.any(undercut_mask):
        return 0.0
        
    # Apply the island filter to ignore mesh noise (must be a cluster of >10 faces)
    hit_indices = np.where(undercut_mask)[0]
    if len(hit_indices) < 10: return 0.0
    
    blocked_mesh = mesh.submesh([hit_indices], append=True)
    islands = blocked_mesh.split(only_watertight=False)
    
    valid_area = sum(island.area for island in islands if len(island.faces) > 10)
    return float(valid_area)

@app.post("/upload")
async def upload_step(request: Request, file: UploadFile = File(...)):
    gc.collect()
    tmp_step = Path(tempfile.gettempdir()) / f"{uuid4()}.step"
    glb_path = STATIC_DIR / f"{uuid4()}.glb"
    
    try:
        content = await file.read()
        with tmp_step.open("wb") as f: f.write(content)
        cascadio.step_to_glb(str(tmp_step), str(glb_path), 0.8)
        
        scene = trimesh.load(str(glb_path))
        mesh = trimesh.util.concatenate([g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)])
        
        test_dirs = {"X": [1,0,0], "Y": [0,1,0], "Z": [0,0,1]} # Bi-directional covers +/- automatically
        raw_results = {n: analyze_undercuts_bidirectional(mesh, v) for n, v in test_dirs.items()}
        
        # Weighted Z-preference
        weighted = {n: v for n, v in raw_results.items()}
        weighted["Z"] *= 0.1
        best_axis = min(weighted, key=weighted.get)
        
        # Real undercut threshold
        has_u = bool((raw_results[best_axis] * 1e6) > 50.0) 

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
    except Exception as e: return {"error": str(e)}
    finally:
        if tmp_step.exists(): tmp_step.unlink()
        gc.collect()
