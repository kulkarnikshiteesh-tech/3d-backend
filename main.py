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
    """Prevents NaN/Infinity from breaking the Frontend JSON."""
    if isinstance(val, (float, int)):
        return 0.0 if math.isnan(val) or math.isinf(val) else val
    return val

def cleanup_static():
    """Deletes files older than 10 minutes to save Render disk space."""
    import time
    now = time.time()
    for f in STATIC_DIR.glob("*.glb"):
        if f.stat().st_mtime < now - 600:
            try: f.unlink()
            except: pass

def calculate_undercut_stats(mesh, pull_axis):
    direction = np.array(pull_axis) / np.linalg.norm(pull_axis)
    dots = np.dot(mesh.face_normals, direction)
    epsilon = mesh.scale * 0.001 
    
    # Shadow analysis
    up_idx = np.where(dots > 0.1)[0]
    blocked_up = mesh.ray.intersects_any(mesh.triangles_center[up_idx] + direction * epsilon, np.tile(direction, (len(up_idx), 1)))
    
    down_idx = np.where(dots < -0.1)[0]
    blocked_down = mesh.ray.intersects_any(mesh.triangles_center[down_idx] - direction * epsilon, np.tile(-direction, (len(down_idx), 1)))
    
    side_idx = np.where(np.abs(dots) <= 0.1)[0]
    hits_f = mesh.ray.intersects_any(mesh.triangles_center[side_idx] + direction * epsilon, np.tile(direction, (len(side_idx), 1)))
    hits_b = mesh.ray.intersects_any(mesh.triangles_center[side_idx] - direction * epsilon, np.tile(-direction, (len(side_idx), 1)))
    blocked_side = np.logical_and(hits_f, hits_b)

    total_area = (np.sum(mesh.area_faces[up_idx][blocked_up]) + 
                  np.sum(mesh.area_faces[down_idx][blocked_down]) + 
                  np.sum(mesh.area_faces[side_idx][blocked_side]))
    return float(total_area)

@app.post("/upload")
async def upload_step(request: Request, file: UploadFile = File(...)):
    cleanup_static() # Free up space
    tmp_step = Path(tempfile.gettempdir()) / f"{uuid4()}.step"
    glb_filename = f"{uuid4()}.glb"
    glb_path = STATIC_DIR / glb_filename
    
    try:
        with tmp_step.open("wb") as f: shutil.copyfileobj(file.file, f)
        cascadio.step_to_glb(str(tmp_step), str(glb_path))
        
        scene = trimesh.load(str(glb_path))
        mesh = trimesh.util.concatenate([g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)])

        # 1. Calculation
        dims = mesh.extents * 1000
        bbox = {"x": sanitize(round(dims[0], 1)), "y": sanitize(round(dims[1], 1)), "z": sanitize(round(dims[2], 1))}
        measured_vol = abs(mesh.volume) * 1e9
        volume_mm3 = sanitize((mesh.area * 1e6 / 2.0) * 1.5 if measured_vol > (np.prod(dims) * 0.4) else measured_vol)

        # 2. Undercut Analysis
        axes = {"X-Axis": [1,0,0], "Y-Axis": [0,1,0], "Z-Axis": [0,0,1]}
        scores = {n: calculate_undercut_stats(mesh, v) + (0 if n == "Z-Axis" else mesh.area*0.05) for n, v in axes.items()}
        best_axis = min(scores, key=scores.get)
        actual_u_area = calculate_undercut_stats(mesh, axes[best_axis])
        
        has_u = actual_u_area > (mesh.area * 0.005)
        
        # 3. Create ABSOLUTE URL
        # This fixes the Vercel loading issue by telling the frontend exactly where to find the file
        base_url = str(request.base_url).rstrip('/')
        full_glb_url = f"{base_url}/static/{glb_filename}"

        return {
            "glb_url": full_glb_url,
            "volume_cubic_mm": volume_mm3,
            "bounding_box_mm": bbox,
            "has_undercuts": bool(has_u),
            "optimal_axis": best_axis,
            "undercut_message": f"Optimal Pull: {best_axis}. " + ("Requires sliders." if has_u else "Straight-pull.")
        }
    except Exception as e:
        print(f"ERROR: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal processing error")
    finally:
        if tmp_step.exists(): tmp_step.unlink()
