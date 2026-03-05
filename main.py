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
    print(traceback.format_exc())
    return JSONResponse(status_code=500, content={"detail": error_msg})

@app.get("/health")
async def health():
    return {"status": "ok"}

def calculate_mold_cost(volume_mm3, has_undercuts, undercut_count):
    """
    Refined Cost Logic:
    - Base price covers setup and CNC time.
    - Volume adds material/size complexity.
    - Undercuts add 'Side Action' costs (approx. $1,800 per slider).
    """
    base_cost = 2500.0 
    size_factor = (volume_mm3 / 1000) * 0.05 
    
    undercut_penalty = 0.0
    if has_undercuts:
        # Every ~150 faces suggests a distinct side-action requirement
        num_sliders = max(1, undercut_count // 150)
        undercut_penalty = num_sliders * 1800.0 

    total_usd = base_cost + size_factor + undercut_penalty
    return round(total_usd, 2)

def analyze_undercuts_geometric(mesh, pull_axis):
    """
    Strict Geometric Filter: Ignores vertical walls and flat floors.
    Only flags faces angled against the pull that would be 'trapped'.
    """
    pull_vec = np.array(pull_axis) / np.linalg.norm(pull_axis)
    dots = np.dot(mesh.face_normals, pull_vec)
    
    # Filter: Ignore vertical walls (dots near 0.0) and flat floors (dots near -1.0)
    # This specifically fixes the false positives seen in hole bottoms.
    undercut_mask = (dots < -0.15) & (dots > -0.98)
    undercut_indices = np.where(undercut_mask)[0]
    
    return {
        "count": len(undercut_indices),
        "area": float(mesh.area_faces[undercut_indices].sum()) if len(undercut_indices) > 0 else 0.0
    }

def get_best_mold_analysis(mesh):
    """
    ORIENTATION INDEPENDENT LOGIC:
    Tests X, Y, and Z axes to find the pull direction with the fewest undercuts.
    """
    axes = {"X-Axis": [1, 0, 0], "Y-Axis": [0, 1, 0], "Z-Axis": [0, 0, 1]}
    results = []
    
    for name, vector in axes.items():
        res = analyze_undercuts_geometric(mesh, vector)
        results.append({"axis": name, "data": res})

    # Find the axis that minimizes 'trapped' geometry
    best = min(results, key=lambda x: x["data"]["count"])
    
    # Threshold to ignore minor mesh noise (80 faces)
    has_undercuts = best["data"]["count"] > 80 
    
    return {
        "has_undercuts": has_undercuts,
        "undercut_face_count": int(best["data"]["count"]),
        "undercut_severity": "high" if best["data"]["count"] > 300 else "low",
        "optimal_axis": best["axis"],
        "undercut_message": (
            f"Undercut detected. Optimal pull is {best['axis']}. Side-actions required."
            if has_undercuts else f"Straight-pull compatible (Optimal Axis: {best['axis']})."
        )
    }

@app.post("/upload")
async def upload_step(file: UploadFile = File(...)):
    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()
    if suffix not in (".step", ".stp"):
        raise HTTPException(status_code=400, detail="Only .step/.stp files are supported")

    tmp_step_path = Path(tempfile.gettempdir()) / f"{uuid4()}{suffix}"
    glb_name = f"{uuid4()}.glb"
    glb_path = STATIC_DIR / glb_name

    try:
        with tmp_step_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        # 1. Convert STEP to GLB
        conv_res = cascadio.step_to_glb(str(tmp_step_path), str(glb_path))
        if conv_res != 0:
            raise RuntimeError(f"cascadio failed: {conv_res}")

        # 2. Load Mesh & Merge Scene Bodies
        loaded = trimesh.load(str(glb_path), force="mesh")
        if isinstance(loaded, trimesh.Scene):
            mesh = trimesh.util.concatenate([g for g in loaded.geometry.values()])
        else:
            mesh = loaded

        # 3. Robust Volume Calculation (mm3)
        volume_mm3 = abs(float(mesh.volume)) * 1e9

        # 4. Analyze Undercuts & Best Orientation
        undercut_data = get_best_mold_analysis(mesh)

        # 5. Refined Cost Estimation
        estimated_mold_cost_usd = calculate_mold_cost(
            volume_mm3, 
            undercut_data["has_undercuts"], 
            undercut_data["undercut_face_count"]
        )

        # 6. Dimensions (mm)
        extents = (mesh.extents * 1000.0)
        
        return {
            "glb_url": f"/static/{glb_name}",
            "volume_cubic_mm": volume_mm3,
            "bounding_box_mm": {"x": extents[0], "y": extents[1], "z": extents[2]},
            "estimated_mold_cost_usd": estimated_mold_cost_usd,
            **undercut_data,
        }

    except Exception as e:
        print(f"Error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_step_path.exists():
            tmp_step_path.unlink()
