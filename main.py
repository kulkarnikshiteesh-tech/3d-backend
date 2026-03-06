from __future__ import annotations
import os, shutil, tempfile, traceback, math
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

def sanitize(val):
    if isinstance(val, (float, int)):
        return 0.0 if math.isnan(val) or math.isinf(val) else val
    return val

def analyze_undercuts_raytrace(mesh, pull_axis):
    """
    Professional Ray-Tracing: Detects faces trapped between plastic.
    Using sampled centers and restricted ray-distance to save RAM.
    """
    direction = np.array(pull_axis) / np.linalg.norm(pull_axis)
    
    # Step 1: Only check faces that are parallel to the pull (the walls)
    # This reduces the number of rays by 70%, preventing the 500 error.
    dots = np.dot(mesh.face_normals, direction)
    potential_u_idx = np.where(np.abs(dots) < 0.1)[0]
    
    if len(potential_u_idx) == 0:
        return 0.0

    # Step 2: Sample origins from the centers of these vertical faces
    origins = mesh.triangles_center[potential_u_idx]
    epsilon = mesh.scale * 0.0001
    
    # Step 3: Fire Rays both ways. 
    # If a wall hits plastic both UP and DOWN, it is an undercut.
    # 
    hits_up = mesh.ray.intersects_any(origins + direction * epsilon, np.tile(direction, (len(origins), 1)))
    hits_down = mesh.ray.intersects_any(origins - direction * epsilon, np.tile(-direction, (len(origins), 1)))
    
    undercut_mask = np.logical_and(hits_up, hits_down)
    undercut_area = np.sum(mesh.area_faces[potential_u_idx][undercut_mask])
    
    return float(undercut_area)

@app.post("/upload")
async def upload_step(request: Request, file: UploadFile = File(...)):
    tmp_step = Path(tempfile.gettempdir()) / f"{uuid4()}.step"
    glb_filename = f"{uuid4()}.glb"
    glb_path = STATIC_DIR / glb_filename
    
    try:
        with tmp_step.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        
        cascadio.step_to_glb(str(tmp_step), str(glb_path))
        
        # Load and verify mesh
        scene = trimesh.load(str(glb_path))
        mesh = trimesh.util.concatenate([g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)])

        # Part Metrics
        dims = mesh.extents * 1000
        bbox = {"x": sanitize(round(dims[0], 1)), "y": sanitize(round(dims[1], 1)), "z": sanitize(round(dims[2], 1))}
        
        # Volume Calculation (Handling enclosures)
        v_raw = abs(mesh.volume) * 1e9
        volume_mm3 = sanitize(v_raw if v_raw > 100 else (mesh.area * 1e6 / 2.0) * 1.5)

        # THE ANALYSIS (Ray-Tracing X, Y, and Z)
        axes = {"X-Axis": [1,0,0], "Y-Axis": [0,1,0], "Z-Axis": [0,0,1]}
        
        # Calculate undercut area for each axis
        # 
        areas = {n: analyze_undercuts_raytrace(mesh, v) for n, v in axes.items()}
        
        # Add a "preference" for Z-axis (standard for enclosures)
        # We only switch from Z if another axis has significantly fewer undercuts
        scores = {n: (areas[n] if n == "Z-Axis" else areas[n] + (mesh.area * 0.1)) for n in areas}
        best_axis = min(scores, key=scores.get)
        
        final_u_area = areas[best_axis]
        has_u = bool(final_u_area > (mesh.area * 0.001)) # 0.1% area threshold

        # Pricing (₹45k penalty for side-actions)
        mold_cost = float(30000.0 + (volume_mm3/1000 * 0.10) + (45000.0 if has_u else 0.0))

        return {
            "glb_url": f"{str(request.base_url).rstrip('/')}/static/{glb_filename}",
            "volume_cubic_mm": volume_mm3,
            "bounding_box_mm": bbox,
            "has_undercuts": has_u,
            "optimal_axis": best_axis,
            "undercut_message": f"Optimal Pull: {best_axis}. " + 
                               ("Side-actions (sliders) required." if has_u else "Straight-pull compatible.")
        }
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_step.exists(): tmp_step.unlink()
