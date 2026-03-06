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
    """Ensure numerical values are JSON-serializable and finite."""
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
    """
    Performs ray-tracing to find geometry trapped in both directions 
    along a specific pull axis.
    """
    direction = np.array(pull_axis) / np.linalg.norm(pull_axis)
    dots = np.dot(mesh.face_normals, direction)
    
    # Target faces roughly parallel to the pull direction
    potential_u_idx = np.where(np.abs(dots) < 0.05)[0]
    if len(potential_u_idx) == 0:
        return 0.0

    origins = mesh.triangles_center[potential_u_idx]
    epsilon = mesh.scale * 0.001
    
    # Check for obstructions in BOTH positive and negative directions
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

        # 1. Bounding Box & Basic Dimensions
        extents = mesh.extents
        dims = extents * 1000
        bbox = {
            "x": sanitize(round(float(dims[0]), 1)), 
            "y": sanitize(round(float(dims[1]), 1)), 
            "z": sanitize(round(float(dims[2]), 1))
        }
        
        # 2. THE FIX: 6-Axis Brute Force Search
        # We test every possible mold opening direction to find the 'cleanest' one.
        test_dirs = {
            "X-Axis": [1,0,0], "-X-Axis": [-1,0,0],
            "Y-Axis": [0,1,0], "-Y-Axis": [0,-1,0],
            "Z-Axis": [0,0,1], "-Z-Axis": [0,0,-1]
        }
        
        # Dictionary to store undercut area for each direction
        results = {name: analyze_undercuts_raytrace(mesh, vec) for name, vec in test_dirs.items()}
        
        # Find the direction with the absolute minimum shadow area
        best_axis_raw = min(results, key=results.get)
        min_undercut_area = results[best_axis_raw]
        
        # Use a relative threshold (0.1% of total area) to ignore mesh noise
        threshold = float(mesh.area) * 0.001
        has_u = bool(min_undercut_area > threshold)
        
        # Clean the axis name for the UI (e.g., "-Z-Axis" -> "Z-Axis")
        display_axis = best_axis_raw.replace("-", "")

        # 3. Costing & DFM Logic
        v_raw = abs(float(mesh.volume)) * 1e9
        volume_mm3 = sanitize(v_raw if v_raw > 500 else (float(mesh.area) * 1e6 / 2.0) * 1.5)
        
        # High Aspect Ratio Check (Warpage Risk)
        # Only trigger if the part is extremely long compared to its height
        sorted_dims = sorted(extents)
        aspect_ratio = sorted_dims[2] / sorted_dims[0] if sorted_dims[0] > 0 else 0
        warpage_risk = bool(aspect_ratio > 15.0) # Loosened from 10.0 to 15.0 to reduce noise

        slider_penalty = 45000.0 if has_u else 0.0
        mold_cost = float(28000.0 + (volume_mm3/1000 * 0.08) + slider_penalty)

        return {
            "glb_url": str(request.base_url).rstrip('/') + f"/static/{glb_filename}",
            "volume_cubic_mm": float(volume_mm3),
            "bounding_box_mm": bbox,
            "has_undercuts": has_u,
            "optimal_axis": display_axis,
            "undercut_message": f"Optimal Pull: {display_axis}. " + 
                               ("Side-actions (sliders) required." if has_u else "Straight-pull compatible."),
            "warpage_risk": warpage_risk
        }

    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_step.exists():
            tmp_step.unlink()
