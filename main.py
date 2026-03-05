from __future__ import annotations

import io
import os
import re
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

def analyze_undercuts_geometric(mesh, pull_axis):
    """
    Detects undercuts by analyzing face accessibility along a specific axis.
    """
    pull_vec = np.array(pull_axis) / np.linalg.norm(pull_axis)
    
    # Calculate dot product of normals vs pull direction
    dots = np.dot(mesh.face_normals, pull_vec)
    
    # 1. Side Walls (Perpendicular to pull) -> Needs Side Action
    perpendicular_faces = np.abs(dots) < 0.05
    
    # 2. Back-facing surfaces (Pointed against the pull) -> Trapped
    back_faces = dots < -0.05
    
    undercut_mask = perpendicular_faces | back_faces
    undercut_indices = np.where(undercut_mask)[0]
    
    return {
        "count": len(undercut_indices),
        "area": float(mesh.area_faces[undercut_indices].sum()) if len(undercut_indices) > 0 else 0.0
    }

def get_best_mold_analysis(mesh):
    """
    Tests X, Y, and Z axes to find the orientation with the lowest manufacturing cost.
    """
    axes = {
        "X-Axis": [1, 0, 0],
        "Y-Axis": [0, 1, 0],
        "Z-Axis": [0, 0, 1]
    }
    
    results = []
    for name, vector in axes.items():
        res = analyze_undercuts_geometric(mesh, vector)
        results.append({"axis": name, "data": res})

    # Find the axis that minimizes the number of 'trapped' faces
    best = min(results, key=lambda x: x["data"]["count"])
    
    # Threshold for detection (ignoring minor mesh noise)
    has_undercuts = best["data"]["count"] > 25 
    
    return {
        "has_undercuts": has_undercuts,
        "undercut_face_count": int(best["data"]["count"]),
        "undercut_severity": "high" if has_undercuts else "low",
        "optimal_axis": best["axis"],
        "undercut_message": (
            f"Undercut detected. Optimal mold pull is the {best['axis']}. "
            "Side-action sliders or lifters will be required."
            if has_undercuts else "No significant undercut risk. Part is straight-pull compatible."
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
            raise RuntimeError(f"cascadio conversion failed: {conv_res}")

        # 2. Load Mesh & Handle Scene (Multi-body files)
        loaded = trimesh.load(str(glb_path), force="mesh")
        if isinstance(loaded, trimesh.Scene):
            if not loaded.geometry:
                raise ValueError("The STEP file contains no valid geometry.")
            mesh = trimesh.util.concatenate([g for g in loaded.geometry.values()])
        else:
            mesh = loaded

        # 3. Geometric Undercut Analysis
        undercut_data = get_best_mold_analysis(mesh)

        # 4. Physical Calculations
        # Convert volume to cubic mm (trimesh defaults to meters for STEP-to-GLB)
        volume_mm3 = (float(mesh.volume) if mesh.is_watertight else 0.0) * 1e9
        extents = (mesh.extents * 1000.0)
        
        return {
            "glb_url": f"/static/{glb_name}",
            "volume_cubic_mm": volume_mm3,
            "bounding_box_mm": {"x": extents[0], "y": extents[1], "z": extents[2]},
            **undercut_data,
        }

    except Exception as e:
        print(f"Error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_step_path.exists():
            tmp_step_path.unlink()
