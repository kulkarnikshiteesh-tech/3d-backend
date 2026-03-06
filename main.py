from __future__ import annotations
import os, shutil, tempfile, traceback, math
from pathlib import Path
from uuid import uuid4
import cascadio, trimesh, numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles

app = FastAPI()

# Enable CORS for Vercel frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup static directory for GLB files
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
    """Remove files older than 10 minutes to save disk space on Render."""
    import time
    now = time.time()
    for f in STATIC_DIR.glob("*.glb"):
        if f.stat().st_mtime < now - 600:
            try: f.unlink()
            except: pass

def analyze_undercuts_raytrace(mesh, pull_axis):
    """
    Professional Ray-Tracing: Identifies if a feature is trapped by geometry 
    above and below it relative to the mold opening direction.
    """
    direction = np.array(pull_axis) / np.linalg.norm(pull_axis)
    dots = np.dot(mesh.face_normals, direction)
    
    # Filter for vertical walls (potential undercuts)
    potential_u_idx = np.where(np.abs(dots) < 0.05)[0]
    if len(potential_u_idx) == 0:
        return 0.0

    origins = mesh.triangles_center[potential_u_idx]
    epsilon = mesh.scale * 0.001
    
    # Fire rays in both directions of the pull axis
    hits_up = mesh.ray.intersects_any(origins + direction * epsilon, np.tile(direction, (len(origins), 1)))
    hits_down = mesh.ray.intersects_any(origins - direction * epsilon, np.tile(-direction, (len(origins), 1)))
    
    # An undercut is confirmed only if blocked in both directions
    undercut_mask = np.logical_and(hits_up, hits_down)
    return float(np.sum(mesh.area_faces[potential_u_idx][undercut_mask]))

@app.post("/upload")
async def upload_step(request: Request, file: UploadFile = File(...)):
    cleanup_static()
    tmp_step = Path(tempfile.gettempdir()) / f"{uuid4()}.step"
    glb_filename = f"{uuid4()}.glb"
    glb_path = STATIC_DIR / glb_filename
    
    try:
        # Save uploaded file
        with tmp_step.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # Convert STEP to GLB
        cascadio.step_to_glb(str(tmp_step), str(glb_path))
        
        # Load Mesh with Trimesh
        scene = trimesh.load(str(glb_path))
        mesh = trimesh.util.concatenate([g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)])

        # 1. Bounding Box & Volume
        dims = mesh.extents * 1000
        bbox = {
            "x": sanitize(round(float(dims[0]), 1)), 
            "y": sanitize(round(float(dims[1]), 1)), 
            "z": sanitize(round(float(dims[2]), 1))
        }
        
        v_raw = abs(float(mesh.volume)) * 1e9
        # Fallback for shells/enclosures where volume calculation might struggle
        volume_mm3 = sanitize(v_raw if v_raw > 500 else (float(mesh.area) * 1e6 / 2.0) * 1.5)

        # 2. Orientation Logic (Z-Axis Preference)
        axes = {"X-Axis": [1,0,0], "Y-Axis": [0,1,0], "Z-Axis": [0,0,1]}
        areas = {n: float(analyze_undercuts_raytrace(mesh, v)) for n, v in axes.items()}
        
        # Threshold: Undercuts smaller than 0.05% of total area are ignored as noise
        threshold = float(mesh.area) * 0.0005 
        
        # FORCE Z-Axis if it is clean (Standard Enclosure logic)
        if areas["Z-Axis"] <= threshold:
            best_axis = "Z-Axis"
            has_u = False
        else:
            # Otherwise, find the axis with the absolute minimum undercut area
            best_axis = str(min(areas, key=areas.get))
            has_u = bool(areas[best_axis] > threshold)

        # 3. Cost Estimation
        slider_penalty = 45000.0 if has_u else 0.0
        mold_cost = float(28000.0 + (volume_mm3/1000 * 0.08) + slider_penalty)

        # 4. Final Payload (All values explicitly cast to Python primitives)
        return {
            "glb_url": str(request.base_url).rstrip('/') + f"/static/{glb_filename}",
            "volume_cubic_mm": float(volume_mm3),
            "bounding_box_mm": bbox,
            "has_undercuts": bool(has_u),
            "optimal_axis": str(best_axis),
            "undercut_message": f"Optimal Pull: {best_axis}. " + 
                               ("Side-actions (sliders) required." if has_u else "Straight-pull compatible.")
        }

    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_step.exists():
            tmp_step.unlink()
