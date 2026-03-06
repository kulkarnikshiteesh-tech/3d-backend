from __future__ import annotations
import os, shutil, tempfile, traceback, math
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

def cleanup_static():
    import time
    now = time.time()
    for f in STATIC_DIR.glob("*.glb"):
        if f.stat().st_mtime < now - 600:
            try: f.unlink()
            except: pass

def analyze_undercuts_raytrace(mesh, pull_axis):
    direction = np.array(pull_axis) / np.linalg.norm(pull_axis)
    dots = np.dot(mesh.face_normals, direction)
    
    potential_u_idx = np.where(np.abs(dots) < 0.05)[0]
    if len(potential_u_idx) == 0:
        return 0.0

    origins = mesh.triangles_center[potential_u_idx]
    epsilon = mesh.scale * 0.001
    
    hits_up = mesh.ray.intersects_any(origins + direction * epsilon, np.tile(direction, (len(origins), 1)))
    hits_down = mesh.ray.intersects_any(origins - direction * epsilon, np.tile(-direction, (len(origins), 1)))
    
    undercut_mask = np.logical_and(hits_up, hits_down)
    return float(np.sum(mesh.area_faces[potential_u_idx][undercut_mask]))

@app.post("/upload")
async def upload_step(request: Request, file: UploadFile = File(...)):
    cleanup_static()
    tmp_step = Path(tempfile.gettempdir()) / f"{uuid4()}.step"
    glb_filename = f"{uuid4()}.glb"
    glb_path = STATIC_DIR / glb_filename
    
    try:
        with tmp_step.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        
        cascadio.step_to_glb(str(tmp_step), str(glb_path))
        scene = trimesh.load(str(glb_path))
        mesh = trimesh.util.concatenate([g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)])

        # 1. Dimensions
        extents = mesh.extents
        dims = extents * 1000
        bbox = {
            "x": sanitize(round(float(dims[0]), 1)), 
            "y": sanitize(round(float(dims[1]), 1)), 
            "z": sanitize(round(float(dims[2]), 1))
        }
        
        v_raw = abs(float(mesh.volume)) * 1e9
        volume_mm3 = sanitize(v_raw if v_raw > 500 else (float(mesh.area) * 1e6 / 2.0) * 1.5)

        # 2. Orientation Logic (Shortest-Axis Priority)
        # Identify which index (0=X, 1=Y, 2=Z) is the thinnest part of the part
        shortest_dim_idx = int(np.argmin(extents)) 
        axes_list = [[1,0,0], [0,1,0], [0,0,1]]
        axes_names = ["X-Axis", "Y-Axis", "Z-Axis"]
        
        # Calculate undercut area for all 3 axes
        areas = {axes_names[i]: float(analyze_undercuts_raytrace(mesh, axes_list[i])) for i in range(3)}
        threshold = float(mesh.area) * 0.0005 
        
        # Check the shortest axis first (the likely pull direction)
        preferred_axis = axes_names[shortest_dim_idx]
        
        if areas[preferred_axis] <= threshold:
            best_axis = preferred_axis
            has_u = False
        else:
            # Fallback only if the shortest axis is actually blocked
            best_axis = str(min(areas, key=areas.get))
            has_u = bool(areas[best_axis] > threshold)

        # 3. Cost Estimation
        slider_penalty = 45000.0 if has_u else 0.0
        mold_cost = float(28000.0 + (volume_mm3/1000 * 0.08) + slider_penalty)

        return {
            "glb_url": str(request.base_url).rstrip('/') + f"/static/{glb_filename}",
            "volume_cubic_mm": float(volume_mm3),
            "bounding_box_mm": bbox,
            "has_undercuts": bool(has_u),
            "optimal_axis": str(best_axis),
            "undercut_message": f"Optimal Pull: {best_axis}. " + 
                               ("Side-actions required." if has_u else "Straight-pull compatible.")
        }

    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_step.exists(): tmp_step.unlink()
