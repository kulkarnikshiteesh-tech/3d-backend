from __future__ import annotations
import os, shutil, tempfile, traceback
from pathlib import Path
from uuid import uuid4
import cascadio, trimesh, numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

STATIC_DIR = Path("static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

def get_advanced_analysis(mesh):
    """Detects best pull axis and trapped side-features (undercuts)."""
    axes = {"X-Axis": [1,0,0], "Y-Axis": [0,1,0], "Z-Axis": [0,0,1]}
    results = {}
    
    for name, vec in axes.items():
        pull = np.array(vec)
        # Find faces perpendicular to pull (walls)
        dots = np.abs(np.dot(mesh.face_normals, pull))
        side_idx = np.where(dots < 0.1)[0]
        if len(side_idx) == 0:
            results[name] = 0
            continue
        
        origins = mesh.triangles_center[side_idx]
        try:
            # Sandwich Test: Is it blocked both ways along the pull axis?
            f = mesh.ray.intersects_any(origins + (pull * 0.1), np.tile(pull, (len(origins), 1)))
            b = mesh.ray.intersects_any(origins - (pull * 0.1), np.tile(-pull, (len(origins), 1)))
            results[name] = int(np.sum(np.logical_and(f, b)))
        except:
            results[name] = 999
            
    best_axis = min(results, key=results.get)
    u_count = results[best_axis]
    has_undercuts = u_count > 50 # Threshold to ignore noise
    # Severity calculation for your UI
    severity = "High" if u_count > 500 else "Medium" if u_count > 50 else "None"
    
    return has_undercuts, u_count, best_axis, severity

@app.post("/upload")
async def upload_step(file: UploadFile = File(...)):
    tmp_step = Path(tempfile.gettempdir()) / f"{uuid4()}.step"
    glb_path = STATIC_DIR / f"{uuid4()}.glb"
    try:
        with tmp_step.open("wb") as f: shutil.copyfileobj(file.file, f)
        cascadio.step_to_glb(str(tmp_step), str(glb_path))
        
        scene = trimesh.load(str(glb_path))
        mesh = trimesh.util.concatenate([g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)])

        # Volume & Bounding Box for Index.tsx
        dims = mesh.extents * 1000 # Convert to mm
        bbox = {"x": round(dims[0], 1), "y": round(dims[1], 1), "z": round(dims[2], 1)}
        
        measured_vol = abs(mesh.volume) * 1e9
        # Enclosure shell logic
        volume_mm3 = (mesh.area * 1e6 / 2.0) * 1.5 if measured_vol > (np.prod(dims) * 0.5) else measured_vol

        has_u, u_count, axis, severity = get_advanced_analysis(mesh)
        
        # Match keys in Index.tsx exactly
        return {
            "glb_url": f"/static/{glb_path.name}",
            "volume_cubic_mm": round(volume_mm3, 2),
            "bounding_box_mm": bbox,
            "has_undercuts": has_u,
            "undercut_severity": severity,
            "optimal_axis": axis,
            "undercut_message": f"Optimal Pull: {axis}. " + ("Side-actions required." if has_u else "Straight-pull compatible.")
        }
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_step.exists(): tmp_step.unlink()
