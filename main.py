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
    Robust Core/Cavity simulation using face-normal offsets to prevent 
    self-intersection noise on flat walls.
    """
    direction = np.array(pull_axis, dtype=float)
    direction /= np.linalg.norm(direction)
    
    # Dot product to find faces pointing roughly in the pull direction
    up_mask = np.dot(mesh.face_normals, direction) > 0.1
    down_mask = np.dot(mesh.face_normals, direction) < -0.1
    
    # Epsilon: Offset the ray start point so it doesn't 'hit' its own face
    epsilon = mesh.scale * 0.001
    undercut_area = 0.0
    
    if np.any(up_mask):
        origins = mesh.triangles_center[up_mask] + (direction * epsilon)
        hits_up = mesh.ray.intersects_any(origins, np.tile(direction, (np.sum(up_mask), 1)))
        undercut_area += np.sum(mesh.area_faces[up_mask][hits_up])
        
    if np.any(down_mask):
        origins = mesh.triangles_center[down_mask] - (direction * epsilon)
        hits_down = mesh.ray.intersects_any(origins, np.tile(-direction, (np.sum(down_mask), 1)))
        undercut_area += np.sum(mesh.area_faces[down_mask][hits_down])
        
    return float(undercut_area)

@app.post("/upload")
async def upload_step(request: Request, file: UploadFile = File(...)):
    # 1. Force RAM purge from any previous failed sessions
    gc.collect() 
    
    tmp_step = Path(tempfile.gettempdir()) / f"{uuid4()}.step"
    glb_filename = f"{uuid4()}.glb"
    glb_path = STATIC_DIR / glb_filename
    
    try:
        # Save uploaded file
        content = await file.read()
        with tmp_step.open("wb") as f:
            f.write(content)
        del content
        
        # 2. LOW-POLY CONVERSION (0.8)
        # This keeps the Hair Dryer under 512MB RAM while preserving dimensions.
        cascadio.step_to_glb(str(tmp_step), str(glb_path), 0.8) 
        
        # 3. MEMORY EFFICIENT LOADING
        scene = trimesh.load(str(glb_path))
        geometry = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geometry:
            raise ValueError("No valid geometry in STEP file.")
            
        mesh = trimesh.util.concatenate(geometry)
        
        # Optional: 'Fix' normals for smoother visual rendering despite low-poly count
        mesh.fix_normals()
        
        # Clear scene from RAM immediately
        del scene
        gc.collect()

        # Bounding Box Logic (Absolute Dimensions)
        dims = mesh.extents * 1000
        bbox = {
            "x": sanitize(round(float(dims[0]), 1)),
            "y": sanitize(round(float(dims[1]), 1)),
            "z": sanitize(round(float(dims[2]), 1))
        }
        
        # 4. UNIVERSAL 6-AXIS SEARCH
        test_dirs = {"X": [1,0,0], "-X": [-1,0,0], "Y": [0,1,0], "-Y": [0,-1,0], "Z": [0,0,1], "-Z": [0,0,-1]}
        results = {n: analyze_undercuts_raytrace(mesh, v) for n, v in test_dirs.items()}
        
        best_axis_raw = min(results, key=results.get)
        min_u_area_mm2 = results[best_axis_raw] * 1e6
        
        # 5. THE 'PROJECT 08' FIX: 40mm2 threshold catches windows but ignores noise.
        has_u = bool(min_u_area_mm2 > 40.0) 
        display_axis = best_axis_raw.replace("-", "")

        # Volume Logic
        v_raw = abs(float(mesh.volume)) * 1e9
        volume_mm3 = sanitize(v_raw if v_raw > 100 else (float(mesh.area) * 1e6 / 2.0) * 1.5)
        
        # Cost Logic
        slider_penalty = 45000.0 if has_u else 0.0
        mold_cost = float(28000.0 + (volume_mm3/1000 * 0.08) + slider_penalty)

        return {
            "glb_url": f"{str(request.base_url).rstrip('/')}/static/{glb_filename}",
            "volume_cubic_mm": round(volume_mm3, 2),
            "bounding_box_mm": bbox,
            "has_undercuts": has_u,
            "optimal_axis": display_axis,
            "undercut_message": f"Optimal Pull: {display_axis}. " + ("Side-actions required." if has_u else "Straight-pull compatible."),
            "mold_cost_inr": round(mold_cost, 2)
        }

    except Exception as e:
        print(traceback.format_exc())
        return {
            "error": str(e),
            "has_undercuts": False,
            "undercut_message": "Analysis failed. Model might be too large for free-tier memory."
        }
    finally:
        # 6. ABSOLUTE MEMORY WIPE
        if tmp_step.exists(): tmp_step.unlink()
        if 'mesh' in locals(): del mesh
        gc.collect()
