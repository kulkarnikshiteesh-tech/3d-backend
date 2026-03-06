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

def cleanup_static():
    import time
    now = time.time()
    for f in STATIC_DIR.glob("*.glb"):
        if f.stat().st_mtime < now - 600:
            try: f.unlink()
            except: pass

def analyze_undercuts_raytrace(mesh, pull_axis):
    """
    Detects if vertical walls are 'trapped' by geometry above and below.
    This is the only way to accurately distinguish between a hole in the top
    (straight-pull) and a hole in the side (undercut).
    """
    direction = np.array(pull_axis) / np.linalg.norm(pull_axis)
    dots = np.dot(mesh.face_normals, direction)
    
    # Target only vertical walls relative to this axis
    potential_u_idx = np.where(np.abs(dots) < 0.05)[0]
    if len(potential_u_idx) == 0:
        return 0.0

    origins = mesh.triangles_center[potential_u_idx]
    epsilon = mesh.scale * 0.001
    
    # Fire rays: An undercut exists only if the wall is blocked BOTH ways
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

        # 1. Dimensions & Volume
        dims = mesh.extents * 1000
        bbox = {"x": sanitize(round(dims[0], 1)), "y": sanitize(round(dims[1], 1)), "z": sanitize(round(dims[2], 1))}
        v_raw = abs(mesh.volume) * 1e9
        volume_mm3 = sanitize(v_raw if v_raw > 500 else (mesh.area * 1e6 / 2.0) * 1.5)

        # 2. Advanced Orientation Logic
        axes = {"X-Axis": [1,0,0], "Y-Axis": [0,1,0], "Z-Axis": [0,0,1]}
        areas = {n: analyze_undercuts_raytrace(mesh, v) for n, v in axes.items()}
        
        # LOGIC FIX: Enforce Z-Axis priority for enclosures
        # If Z-Axis has negligible undercut area, force it as the optimal axis
        z_threshold = mesh.area * 0.0005 # 0.05% threshold
        if areas["Z-Axis"] <= z_threshold:
            best_axis = "Z-Axis"
            has_u = False
        else:
            # Only if Z is blocked do we check if X or Y is better
            best_axis = min(areas, key=areas.get)
            has_u = areas[best_axis] > z_threshold

        # 3. Costing (Indian Market Rates)
        # Base Tooling + Volume Factor + Slider Penalty
        slider_penalty = 45000.0 if has_u else 0.0
        mold_cost = float(28000.0 + (volume_mm3/1000 * 0.08) + slider_penalty)

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
