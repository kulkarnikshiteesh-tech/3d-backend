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
    if isinstance(val, (float, int, np.float64, np.int64)):
        val = float(val)
        return 0.0 if math.isnan(val) or math.isinf(val) else val
    return val

def analyze_undercuts_raytrace(mesh, pull_axis):
    """
    Memory-efficient Core/Cavity simulation.
    """
    direction = np.array(pull_axis, dtype=float)
    direction /= np.linalg.norm(direction)
    
    # Identify faces pointing towards the Cavity and Core
    up_mask = np.dot(mesh.face_normals, direction) > 0.1
    down_mask = np.dot(mesh.face_normals, direction) < -0.1
    
    epsilon = mesh.scale * 0.0001
    undercut_area = 0.0
    
    # Process Cavity side
    if np.any(up_mask):
        hits_up = mesh.ray.intersects_any(
            mesh.triangles_center[up_mask] + (direction * epsilon), 
            np.tile(direction, (np.sum(up_mask), 1))
        )
        undercut_area += np.sum(mesh.area_faces[up_mask][hits_up])
        
    # Process Core side
    if np.any(down_mask):
        hits_down = mesh.ray.intersects_any(
            mesh.triangles_center[down_mask] - (direction * epsilon), 
            np.tile(-direction, (np.sum(down_mask), 1))
        )
        undercut_area += np.sum(mesh.area_faces[down_mask][hits_down])
        
    return float(undercut_area)

@app.post("/upload")
async def upload_step(request: Request, file: UploadFile = File(...)):
    # 1. Clean RAM before starting
    gc.collect() 
    
    tmp_step = Path(tempfile.gettempdir()) / f"{uuid4()}.step"
    glb_filename = f"{uuid4()}.glb"
    glb_path = STATIC_DIR / glb_filename
    
    try:
        with tmp_step.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # 2. REDUCED PRECISION: 0.5 creates a much lighter mesh for the 512MB limit
        cascadio.step_to_glb(str(tmp_step), str(glb_path), 0.5) 
        
        scene = trimesh.load(str(glb_path))
        mesh = trimesh.util.concatenate([g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)])

        # Bounding Box
        extents = mesh.extents
        dims = extents * 1000
        bbox = {"x": sanitize(round(float(dims[0]), 1)), "y": sanitize(round(float(dims[1]), 1)), "z": sanitize(round(float(dims[2]), 1))}
        
        # 6-Axis Search
        test_dirs = {"X": [1,0,0], "-X": [-1,0,0], "Y": [0,1,0], "-Y": [0,-1,0], "Z": [0,0,1], "-Z": [0,0,-1]}
        results = {n: analyze_undercuts_raytrace(mesh, v) for n, v in test_dirs.items()}
        
        best_axis_raw = min(results, key=results.get)
        min_undercut_area_mm2 = results[best_axis_raw] * 1e6
        
        # 3. FIXED THRESHOLD: Catch real features, ignore mesh artifacts
        has_u = bool(min_undercut_area_mm2 > 50.0)
        display_axis = best_axis_raw.replace("-", "")

        # Volume and Cost
        v_raw = abs(float(mesh.volume)) * 1e9
        volume_mm3 = sanitize(v_raw if v_raw > 500 else (float(mesh.area) * 1e6 / 2.0) * 1.5)
        
        # Warpage risk for flat parts
        sorted_dims = sorted(extents)
        warpage_risk = bool((sorted_dims[2] / sorted_dims[0]) > 15.0 if sorted_dims[0] > 0 else False)

        slider_penalty = 45000.0 if has_u else 0.0
        mold_cost = float(28000.0 + (volume_mm3/1000 * 0.08) + slider_penalty)

        return {
            "glb_url": f"{str(request.base_url).rstrip('/')}/static/{glb_filename}",
            "volume_cubic_mm": round(volume_mm3, 2),
            "bounding_box_mm": bbox,
            "has_undercuts": has_u,
            "optimal_axis": display_axis,
            "undercut_message": f"Optimal Pull: {display_axis}. " + ("Side-actions required." if has_u else "Straight-pull compatible."),
            "warpage_risk": warpage_risk
        }

    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_step.exists(): tmp_step.unlink()
        # 4. FINAL MEMORY PURGE
        if 'mesh' in locals(): del mesh
        if 'scene' in locals(): del scene
        gc.collect()
