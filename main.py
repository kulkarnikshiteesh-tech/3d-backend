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

def calculate_mold_cost_inr(volume_mm3, has_undercuts, undercut_count):
    """
    Refined Cost Logic based on Indian Industrial Standards:
    - Base Cost: Initial setup, design, and base plates.
    - Volume Factor: Complexity of the cavity/core based on cm3.
    - Slider Penalty: Cost of side-action mechanisms in INR.
    """
    # Base setup for a small-to-medium precision mold in India
    base_cost_inr = 150000.0 
    
    # Conversion to cm3 for more intuitive scaling (₹5.0 per cm3)
    volume_cm3 = volume_mm3 / 1000.0
    size_factor_inr = volume_cm3 * 5.0 
    
    undercut_penalty_inr = 0.0
    if has_undercuts:
        # Every ~150 faces suggests a distinct side-action mechanism
        num_sliders = max(1, undercut_count // 150)
        # Average cost of an Indian-made slider/lifter unit
        undercut_penalty_inr = num_sliders * 55000.0 

    total_inr = base_cost_inr + size_factor_inr + undercut_penalty_inr
    return round(total_inr, 2)

def analyze_undercuts_geometric(mesh, pull_axis):
    """
    Improved Filter: Ignores hole bottoms and vertical walls.
    Focuses only on geometry that 'traps' the mold plate.
    """
    pull_vec = np.array(pull_axis) / np.linalg.norm(pull_axis)
    dots = np.dot(mesh.face_normals, pull_vec)
    
    # THRESHOLD FIX:
    # 1. Ignore vertical walls (dots near 0.0)
    # 2. Ignore perfectly flat bottoms (dots near -1.0)
    # Only flag faces between -0.18 and -0.98.
    undercut_mask = (dots < -0.18) & (dots > -0.98)
    undercut_indices = np.where(undercut_mask)[0]
    
    return {
        "count": len(undercut_indices),
        "area": float(mesh.area_faces[undercut_indices].sum()) if len(undercut_indices) > 0 else 0.0
    }

def get_best_mold_analysis(mesh):
    """
    ORIENTATION INDEPENDENT:
    Tests X, Y, and Z axes to automatically find the cheapest pull direction.
    """
    axes = {"X-Axis": [1, 0, 0], "Y-Axis": [0, 1, 0], "Z-Axis": [0, 0, 1]}
    results = []
    
    for name, vector in axes.items():
        res = analyze_undercuts_geometric(mesh, vector)
        results.append({"axis": name, "data": res})

    # Pick the axis with the least amount of problematic geometry
    best = min(results, key=lambda x: x["data"]["count"])
    
    # Noise Buffer: STEP-to-GLB conversion often leaves tiny artifacts at hole edges.
    # 100 faces is a safe buffer for complex parts.
    has_undercuts = best["data"]["count"] > 100 
    
    return {
        "has_undercuts": has_undercuts,
        "undercut_face_count": int(best["data"]["count"]),
        "undercut_severity": "high" if best["data"]["count"] > 400 else "low",
        "optimal_axis": best["axis"],
        "undercut_message": (
            f"Undercut detected. Optimal pull is {best['axis']}. Side-actions required."
            if has_undercuts else f"Straight-pull compatible (Optimal Axis: {best['axis']})."
        )
    }

@app.post("/upload")
async def upload_step(file: UploadFile = File(...)):
    # [Same upload and conversion logic as your current file]
    # ...
    try:
        # Load mesh, calculate volume, etc.
        # volume_mm3 = abs(float(mesh.volume)) * 1e9
        
        undercut_data = get_best_mold_analysis(mesh)
        
        estimated_mold_cost_inr = calculate_mold_cost_inr(
            volume_mm3, 
            undercut_data["has_undercuts"], 
            undercut_data["undercut_face_count"]
        )

        return {
            "glb_url": f"/static/{glb_name}",
            "volume_cubic_mm": volume_mm3,
            "estimated_mold_cost_inr": estimated_mold_cost_inr,
            **undercut_data,
        }
    except Exception as e:
        # Error handling...
        pass
