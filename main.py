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
    """
    Simulates a 2-part mold (Core & Cavity).
    Checks if upward-facing geometry is blocked from above, 
    and downward-facing geometry is blocked from below.
    """
    direction = np.array(pull_axis, dtype=float)
    direction /= np.linalg.norm(direction)
    
    # 1. Identify faces pointing towards the Cavity (+ direction) and Core (- direction)
    up_mask = np.dot(mesh.face_normals, direction) > 0.01
    down_mask = np.dot(mesh.face_normals, direction) < -0.01
    
    # Use a tiny offset to prevent rays from hitting the face they start on
    epsilon = mesh.scale * 1e-4
    undercut_area = 0.0
    
    # 2. Check Cavity side (+ direction)
    if np.any(up_mask):
        up_centers = mesh.triangles_center[up_mask]
        up_origins = up_centers + (direction * epsilon)
        hits_up = mesh.ray.intersects_any(up_origins, np.tile(direction, (len(up_centers), 1)))
        undercut_area += np.sum(mesh.area_faces[up_mask][hits_up])
        
    # 3. Check Core side (- direction)
    if np.any(down_mask):
        down_centers = mesh.triangles_center[down_mask]
        down_origins = down_centers - (direction * epsilon)
        hits_down = mesh.ray.intersects_any(down_origins, np.tile(-direction, (len(down_centers), 1)))
        undercut_area += np.sum(mesh.area_faces[down_mask][hits_down])
        
    return float(undercut_area)

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

        extents = mesh.extents
        dims = extents * 1000
        bbox = {"x": sanitize(round(float(dims[0]), 1)), "y": sanitize(round(float(dims[1]), 1)), "z": sanitize(round(float(dims[2]), 1))}
        
        # 6-Axis Search: Find the direction with the minimum true undercut area
        test_dirs = {
            "X-Axis": [1,0,0], "-X-Axis": [-1,0,0],
            "Y-Axis": [0,1,0], "-Y-Axis": [0,-1,0],
            "Z-Axis": [0,0,1], "-Z-Axis": [0,0,-1]
        }
        
        results = {name: analyze_undercuts_raytrace(mesh, vec) for name, vec in test_dirs.items()}
        
        best_axis_raw = min(results, key=results.get)
        min_undercut_area = results[best_axis_raw]
        
        # Threshold: Only flag as undercut if trapped area is > 0.5% of total area
        threshold = float(mesh.area) * 0.005
        has_u = bool(min_undercut_area > threshold)
        
        display_axis = best_axis_raw.replace("-", "")

        v_raw = abs(float(mesh.volume)) * 1e9
        volume_mm3 = sanitize(v_raw if v_raw > 500 else (float(mesh.area) * 1e6 / 2.0) * 1.5)
        
        # Fixed Aspect Ratio check (now only flags extremely long, flat parts > 15:1)
        sorted_dims = sorted(extents)
        aspect_ratio = sorted_dims[2] / sorted_dims[0] if sorted_dims[0] > 0 else 0
        warpage_risk = bool(aspect_ratio > 15.0) 

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
        if tmp_step.exists(): tmp_step.unlink()
