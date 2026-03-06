from __future__ import annotations
import os, shutil, tempfile, traceback, math, gc
from pathlib import Path
from uuid import uuid4
import cascadio, trimesh, numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path("static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

def analyze_undercuts_raytrace(mesh, pull_axis):
    direction = np.array(pull_axis, dtype=float)
    direction /= np.linalg.norm(direction)
    
    # Core/Cavity logic: Does plastic block the exit?
    up_mask = np.dot(mesh.face_normals, direction) > 0.1
    down_mask = np.dot(mesh.face_normals, direction) < -0.1
    
    epsilon = mesh.scale * 0.0001
    undercut_area = 0.0
    
    if np.any(up_mask):
        hits_up = mesh.ray.intersects_any(mesh.triangles_center[up_mask] + (direction * epsilon), np.tile(direction, (np.sum(up_mask), 1)))
        undercut_area += np.sum(mesh.area_faces[up_mask][hits_up])
        
    if np.any(down_mask):
        hits_down = mesh.ray.intersects_any(mesh.triangles_center[down_mask] - (direction * epsilon), np.tile(-direction, (np.sum(down_mask), 1)))
        undercut_area += np.sum(mesh.area_faces[down_mask][hits_down])
        
    return float(undercut_area)

@app.post("/upload")
async def upload_step(request: Request, file: UploadFile = File(...)):
    # FORCE CLEANUP AT START
    gc.collect() 
    
    tmp_step = Path(tempfile.gettempdir()) / f"{uuid4()}.step"
    glb_filename = f"{uuid4()}.glb"
    glb_path = STATIC_DIR / glb_filename
    
    try:
        with tmp_step.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # LOWER TOLERANCE = LESS MEMORY. 0.1 or 0.5 is usually fine for DFM.
        cascadio.step_to_glb(str(tmp_step), str(glb_path), 0.5) 
        
        scene = trimesh.load(str(glb_path))
        mesh = trimesh.util.concatenate([g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)])

        # Calculate bounding box
        dims = mesh.extents * 1000
        
        # 6-Axis Brute Force
        test_dirs = {"X": [1,0,0], "-X": [-1,0,0], "Y": [0,1,0], "-Y": [0,-1,0], "Z": [0,0,1], "-Z": [0,0,-1]}
        results = {n: analyze_undercuts_raytrace(mesh, v) for n, v in test_dirs.items()}
        
        best_axis = min(results, key=results.get)
        # Convert m^2 to mm^2 and check against a 50mm^2 threshold (size of a small port)
        has_u = bool((results[best_axis] * 1e6) > 50.0)

        # Basic volume and cost
        v_mm3 = abs(float(mesh.volume)) * 1e9
        cost = 28000 + (v_mm3/1000 * 0.08) + (45000 if has_u else 0)

        return {
            "glb_url": f"{str(request.base_url).rstrip('/')}/static/{glb_filename}",
            "volume_cubic_mm": round(v_mm3, 2),
            "has_undercuts": has_u,
            "optimal_axis": best_axis.replace("-", ""),
            "undercut_message": f"Optimal Pull: {best_axis.replace('-', '')}. " + ("Sliders required." if has_u else "Straight-pull.")
        }

    finally:
        if tmp_step.exists(): tmp_step.unlink()
        # RELEASE MESH FROM RAM IMMEDIATELY
        del mesh
        gc.collect()
