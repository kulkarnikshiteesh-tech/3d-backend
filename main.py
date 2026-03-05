from __future__ import annotations
import os, shutil, tempfile, traceback
from pathlib import Path
from uuid import uuid4
import cascadio, trimesh, numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles

STATIC_DIR = Path("static")
os.makedirs("static", exist_ok=True)
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

def calculate_mold_cost_inr(volume_mm3, has_undercuts, undercut_count):
    # Standard Indian Market Base Rates
    base_cost_inr = 150000.0 
    volume_cm3 = volume_mm3 / 1000.0
    size_factor_inr = volume_cm3 * 5.0 # Complexity scaling
    
    undercut_penalty_inr = 0.0
    if has_undercuts:
        # Only penalize for significant geometry issues
        num_sliders = max(1, undercut_count // 200)
        undercut_penalty_inr = num_sliders * 55000.0 
    return round(base_cost_inr + size_factor_inr + undercut_penalty_inr, 2)

def analyze_undercuts_geometric(mesh, pull_axis):
    pull_vec = np.array(pull_axis) / np.linalg.norm(pull_axis)
    dots = np.dot(mesh.face_normals, pull_vec)
    
    # Only flag faces pointing "down" significantly (-0.2 to -0.98)
    # This ignores vertical walls and the flat bottom floors of your holes.
    undercut_mask = (dots < -0.2) & (dots > -0.98)
    indices = np.where(undercut_mask)[0]
    return {"count": len(indices), "area": float(mesh.area_faces[indices].sum()) if len(indices) > 0 else 0}

def get_best_mold_analysis(mesh):
    axes = {"X-Axis": [1,0,0], "Y-Axis": [0,1,0], "Z-Axis": [0,0,1]}
    results = []
    for name, vec in axes.items():
        res = analyze_undercuts_geometric(mesh, vec)
        results.append({"axis": name, "data": res})
    
    best = min(results, key=lambda x: x["data"]["count"])
    # Threshold (100 faces) ignores tiny mesh artifacts around hole edges
    has_undercuts = best["data"]["count"] > 100 
    return {
        "has_undercuts": has_undercuts,
        "undercut_face_count": int(best["data"]["count"]),
        "optimal_axis": best["axis"],
        "undercut_message": f"Straight-pull compatible (Optimal: {best['axis']})" if not has_undercuts else f"Undercuts detected on {best['axis']}."
    }

@app.post("/upload")
async def upload_step(file: UploadFile = File(...)):
    tmp_step = Path(tempfile.gettempdir()) / f"{uuid4()}.step"
    glb_path = STATIC_DIR / f"{uuid4()}.glb"
    try:
        with tmp_step.open("wb") as f: shutil.copyfileobj(file.file, f)
        cascadio.step_to_glb(str(tmp_step), str(glb_path))
        
        loaded = trimesh.load(str(glb_path))
        # FIX: Merge all parts of the STEP file to get a single, accurate volume
        if isinstance(loaded, trimesh.Scene):
            mesh = trimesh.util.concatenate([g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)])
        else:
            mesh = loaded

        # FIX: Robust volume calculation for enclosures
        volume_mm3 = abs(float(mesh.volume)) * 1e9 if mesh.is_watertight else abs(mesh.convex_hull.volume) * 0.85 * 1e9
        
        undercut_data = get_best_mold_analysis(mesh)
        cost_inr = calculate_mold_cost_inr(volume_mm3, undercut_data["has_undercuts"], undercut_data["undercut_face_count"])

        return {
            "glb_url": f"/static/{glb_path.name}",
            "volume_cubic_mm": volume_mm3,
            "estimated_mold_cost_inr": cost_inr,
            "bounding_box_mm": {"x": mesh.extents[0]*1000, "y": mesh.extents[1]*1000, "z": mesh.extents[2]*1000},
            **undercut_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_step.exists(): tmp_step.unlink()
