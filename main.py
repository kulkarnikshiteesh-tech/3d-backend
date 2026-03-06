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

def sanitize(val):
    try:
        if isinstance(val, (float, int, np.float64, np.int64)):
            if math.isnan(val) or math.isinf(val): return 0.0
            return float(val)
    except:
        pass
    return 0.0

def analyze_undercuts_raytrace(mesh, pull_axis):
    direction = np.array(pull_axis, dtype=float)
    direction /= np.linalg.norm(direction)
    
    up_mask = np.dot(mesh.face_normals, direction) > 0.1
    down_mask = np.dot(mesh.face_normals, direction) < -0.1
    
    epsilon = mesh.scale * 0.001
    undercut_area = 0.0
    
    if np.any(up_mask):
        origins = mesh.triangles_center[up_mask] + (direction * epsilon)
        hits_up = mesh.ray.intersects_any(origins, np.tile(direction, (len(origins), 1)))
        undercut_area += np.sum(mesh.area_faces[up_mask][hits_up])
        
    if np.any(down_mask):
        origins = mesh.triangles_center[down_mask] - (direction * epsilon)
        hits_down = mesh.ray.intersects_any(origins, np.tile(-direction, (len(origins), 1)))
        undercut_area += np.sum(mesh.area_faces[down_mask][hits_down])
        
    return float(undercut_area)

@app.get("/")
async def health():
    return {"status": "alive"}

@app.post("/upload")
async def upload_step(request: Request, file: UploadFile = File(...)):
    gc.collect()
    tmp_step = Path(tempfile.gettempdir()) / f"{uuid4()}.step"
    glb_filename = f"{uuid4()}.glb"
    glb_path = STATIC_DIR / glb_filename
    
    try:
        content = await file.read()
        with tmp_step.open("wb") as f:
            f.write(content)
        del content
        
        # 0.8 precision to keep Hair Dryer from crashing RAM
        cascadio.step_to_glb(str(tmp_step), str(glb_path), 0.8) 
        
        scene = trimesh.load(str(glb_path))
        geometry = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geometry: raise ValueError("No geometry")
        mesh = trimesh.util.concatenate(geometry)
        
        # Physical stats
        v_raw = abs(float(mesh.volume)) * 1e9
        volume_mm3 = v_raw if v_raw > 100 else (float(mesh.area) * 1e6 / 2.0) * 1.5
        dims = mesh.extents * 1000
        
        # 6-Axis logic with Hair Dryer Fix
        test_dirs = {"X": [1,0,0], "-X": [-1,0,0], "Y": [0,1,0], "-Y": [0,-1,0], "Z": [0,0,1], "-Z": [0,0,-1]}
        results = {n: analyze_undercuts_raytrace(mesh, v) for n, v in test_dirs.items()}
        
        # Preference to Z axis
        results["Z"] *= 0.8
        results["-Z"] *= 0.8

        best_axis = min(results, key=results.get)
        # 100mm2 threshold for organic parts
        has_u = bool((results[best_axis] * 1e6) > 100.0)

        cost = 28000 + (volume_mm3/1000 * 0.08) + (45000 if has_u else 0)

        return {
            "glb_url": f"{str(request.base_url).rstrip('/')}/static/{glb_filename}",
            "volume_cubic_mm": round(volume_mm3, 2),
            "bounding_box_mm": {"x": round(dims[0],1), "y": round(dims[1],1), "z": round(dims[2],1)},
            "has_undercuts": has_u,
            "optimal_axis": best_axis.replace("-", ""),
            "undercut_message": f"Optimal Pull: {best_axis.replace('-', '')}. " + ("Side-actions required." if has_u else "Straight-pull compatible."),
            "mold_cost_inr": round(cost, 2)
        }
    except Exception as e:
        return {"error": str(e)}
    finally:
        if tmp_step.exists(): tmp_step.unlink()
        gc.collect()
