from __future__ import annotations

import io
import os
import shutil
import tempfile
import traceback
from pathlib import Path
from uuid import uuid4

import cascadio
import trimesh
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.staticfiles import StaticFiles

STATIC_DIR = Path("static")
os.makedirs("static", exist_ok=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    error_msg = f"{type(exc).__name__}: {str(exc)}"
    print(f"CRITICAL ERROR: {error_msg}")
    return JSONResponse(status_code=500, content={"detail": error_msg})

def calculate_mold_cost_inr(volume_mm3, has_undercuts, undercut_count):
    """
    Calculates cost in INR based on volume and complexity.
    """
    base_cost_inr = 150000.0 # Realistic starting point for Indian tooling
    volume_cm3 = volume_mm3 / 1000.0
    size_factor_inr = volume_cm3 * 5.0 
    
    undercut_penalty_inr = 0.0
    if has_undercuts:
        num_sliders = max(1, undercut_count // 150)
        undercut_penalty_inr = num_sliders * 55000.0 

    return round(base_cost_inr + size_factor_inr + undercut_penalty_inr, 2)

def analyze_undercuts_geometric(mesh, pull_axis):
    pull_vec = np.array(pull_axis) / np.linalg.norm(pull_axis)
    dots = np.dot(mesh.face_normals, pull_vec)
    # Filter to ignore flat floors and vertical walls
    undercut_mask = (dots < -0.18) & (dots > -0.98)
    undercut_indices = np.where(undercut_mask)[0]
    return {
        "count": len(undercut_indices),
        "area": float(mesh.area_faces[undercut_indices].sum()) if len(undercut_indices) > 0 else 0.0
    }

def get_best_mold_analysis(mesh):
    axes = {"X-Axis": [1, 0, 0], "Y-Axis": [0, 1, 0], "Z-Axis": [0, 0, 1]}
    results = []
    for name, vector in axes.items():
        res = analyze_undercuts_geometric(mesh, vector)
        results.append({"axis": name, "data": res})

    best = min(results, key=lambda x: x["data"]["count"])
    has_undercuts = best["data"]["count"] > 100 
    
    return {
        "has_undercuts": has_undercuts,
        "undercut_face_count": int(best["data"]["count"]),
        "optimal_axis": best["axis"],
        "undercut_message": f"Straight-pull compatible (Optimal: {best['axis']})" if not has_undercuts else f"Undercuts detected on {best['axis']}."
    }

@app.post("/upload")
async def upload_step(file: UploadFile = File(...)):
    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()
    
    tmp_step_path = Path(tempfile.gettempdir()) / f"{uuid4()}{suffix}"
    glb_name = f"{uuid4()}.glb"
    glb_path = STATIC_DIR / glb_name

    try:
        with tmp_step_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        # 1. Convert STEP to GLB
        cascadio.step_to_glb(str(tmp_step_path), str(glb_path))

        # 2. Load and Force Concatenation (Fixes missing model issue)
        loaded = trimesh.load(str(glb_path))
        if isinstance(loaded, trimesh.Scene):
            mesh = trimesh.util.concatenate([g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)])
        else:
            mesh = loaded

        # 3. Robust Volume (Fixes 0.00 mm3 issue)
        volume_mm3 = abs(float(mesh.volume)) * 1e9 if mesh.is_watertight else abs(mesh.convex_hull.volume) * 0.9 * 1e9

        # 4. Analysis & Costing
        undercut_data = get_best_mold_analysis(mesh)
        cost_inr = calculate_mold_cost_inr(volume_mm3, undercut_data["has_undercuts"], undercut_data["undercut_face_count"])

        return {
            "glb_url": f"/static/{glb_name}",
            "volume_cubic_mm": volume_mm3,
            "estimated_mold_cost_inr": cost_inr,
            "bounding_box_mm": {"x": mesh.extents[0]*1000, "y": mesh.extents[1]*1000, "z": mesh.extents[2]*1000},
            **undercut_data
        }

    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_step_path.exists(): tmp_step_path.unlink()
