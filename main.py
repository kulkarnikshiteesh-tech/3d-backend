from __future__ import annotations
import os, shutil, tempfile, traceback
from pathlib import Path
from uuid import uuid4
import cascadio, trimesh, numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
STATIC_DIR = Path("static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

def calculate_mold_cost_inr(volume_mm3, has_undercuts):
    base_cost = 28000.0 
    volume_cm3 = volume_mm3 / 1000.0
    size_factor = volume_cm3 * 0.05 
    undercut_penalty = 45000.0 if has_undercuts else 0.0
    return round(base_cost + size_factor + undercut_penalty, 2)

def analyze_axis_for_trapped_faces(mesh, axis_vector):
    pull_dir = np.array(axis_vector) / np.linalg.norm(axis_vector)
    dots = np.abs(np.dot(mesh.face_normals, pull_dir))
    side_indices = np.where(dots < 0.1)[0]
    
    if len(side_indices) == 0: return 0

    origins = mesh.triangles_center[side_indices]
    try:
        # Sandwich check: Look 'Forward' and 'Backward' along the axis
        hits_f = mesh.ray.intersects_any(origins + (pull_dir * 0.1), np.tile(pull_dir, (len(origins), 1)))
        hits_b = mesh.ray.intersects_any(origins - (pull_dir * 0.1), np.tile(-pull_dir, (len(origins), 1)))
        return int(np.sum(np.logical_and(hits_f, hits_b)))
    except:
        return 9999

def get_best_orientation_analysis(mesh):
    axes = {"X-Axis": [1, 0, 0], "Y-Axis": [0, 1, 0], "Z-Axis": [0, 0, 1]}
    results = {name: analyze_axis_for_trapped_faces(mesh, vec) for name, vec in axes.items()}
    optimal_axis = min(results, key=results.get)
    undercut_count = results[optimal_axis]
    return undercut_count > 40, undercut_count, optimal_axis

@app.post("/upload")
async def upload_step(file: UploadFile = File(...)):
    tmp_step = Path(tempfile.gettempdir()) / f"{uuid4()}.step"
    glb_path = STATIC_DIR / f"{uuid4()}.glb"
    try:
        with tmp_step.open("wb") as f: shutil.copyfileobj(file.file, f)
        cascadio.step_to_glb(str(tmp_step), str(glb_path))
        
        scene = trimesh.load(str(glb_path))
        mesh = trimesh.util.concatenate([g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)])

        # Volume Guardrail
        bbox_vol = np.prod(mesh.extents) * 1e9
        measured_vol = abs(mesh.volume) * 1e9
        volume_mm3 = (mesh.area * 1e6 / 2.0) * 1.5 if measured_vol > (bbox_vol * 0.5) else measured_vol

        # Orientation Analysis
        has_undercut, u_count, best_axis = get_best_orientation_analysis(mesh)
        
        # Costs
        mold_cost = calculate_mold_cost_inr(volume_mm3, has_undercut)
        per_piece = round((volume_mm3 / 69311.0) * 18.29, 2)

        # Return JSON with exact keys expected by Frontend
        return {
            "glb_url": f"/static/{glb_path.name}",
            "volume_cubic_mm": volume_mm3,
            "estimated_mold_cost_inr": mold_cost,
            "estimated_per_piece_inr": per_piece,
            "has_undercuts": has_undercut,
            "undercut_face_count": u_count,
            "optimal_axis": best_axis,
            "undercut_message": f"Optimal Pull: {best_axis}. " + ("Side-actions required." if has_undercut else "Straight-pull compatible.")
        }
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_step.exists(): tmp_step.unlink()
