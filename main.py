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

def calculate_undercut_stats(mesh, pull_axis):
    """
    Professional Shadow Analysis: A face is an undercut if it is 'shadowed' 
    by other geometry along the pull axis.
    """
    direction = np.array(pull_axis) / np.linalg.norm(pull_axis)
    
    # 1. Determine which faces are 'up-facing' and 'down-facing' relative to pull
    dots = np.dot(mesh.face_normals, direction)
    
    # Small epsilon based on part size to avoid 'self-hitting' rays
    epsilon = mesh.scale * 0.001 
    
    # 2. Ray Trace: Check if faces are blocked
    # Check faces pointing 'up' for obstructions above them
    up_idx = np.where(dots > 0.1)[0]
    blocked_up = mesh.ray.intersects_any(
        mesh.triangles_center[up_idx] + direction * epsilon, 
        np.tile(direction, (len(up_idx), 1))
    )
    
    # Check faces pointing 'down' for obstructions below them
    down_idx = np.where(dots < -0.1)[0]
    blocked_down = mesh.ray.intersects_any(
        mesh.triangles_center[down_idx] - direction * epsilon, 
        np.tile(-direction, (len(down_idx), 1))
    )
    
    # Check vertical walls for the 'sandwich' (blocked both ways)
    side_idx = np.where(np.abs(dots) <= 0.1)[0]
    hits_f = mesh.ray.intersects_any(mesh.triangles_center[side_idx] + direction * epsilon, np.tile(direction, (len(side_idx), 1)))
    hits_b = mesh.ray.intersects_any(mesh.triangles_center[side_idx] - direction * epsilon, np.tile(-direction, (len(side_idx), 1)))
    blocked_side = np.logical_and(hits_f, hits_b)

    # 3. Calculate Total Undercut Area
    total_area = (
        np.sum(mesh.area_faces[up_idx][blocked_up]) + 
        np.sum(mesh.area_faces[down_idx][blocked_down]) + 
        np.sum(mesh.area_faces[side_idx][blocked_side])
    )
    
    return float(total_area)

def get_best_orientation(mesh):
    axes = {"X-Axis": [1,0,0], "Y-Axis": [0,1,0], "Z-Axis": [0,0,1]}
    results = {}
    
    for name, vec in axes.items():
        area = calculate_undercut_stats(mesh, vec)
        # We apply a 'Common Sense Bias' for enclosures: 
        # Z-axis is preferred unless another axis is significantly better.
        bias = 0.0 if name == "Z-Axis" else (mesh.area * 0.05)
        results[name] = area + bias
        
    best_axis_name = min(results, key=results.get)
    # Return the REAL area (without bias) for the chosen axis
    actual_area = calculate_undercut_stats(mesh, axes[best_axis_name])
    
    # Threshold: Undercuts are real only if they cover more than 0.5% of the total part area
    has_undercuts = actual_area > (mesh.area * 0.005)
    
    severity = "None"
    if has_undercuts:
        severity = "High" if actual_area > (mesh.area * 0.05) else "Medium"
        
    return bool(has_undercuts), best_axis_name, severity

@app.post("/upload")
async def upload_step(file: UploadFile = File(...)):
    tmp_step = Path(tempfile.gettempdir()) / f"{uuid4()}.step"
    glb_path = STATIC_DIR / f"{uuid4()}.glb"
    try:
        with tmp_step.open("wb") as f: shutil.copyfileobj(file.file, f)
        cascadio.step_to_glb(str(tmp_step), str(glb_path))
        
        scene = trimesh.load(str(glb_path))
        mesh = trimesh.util.concatenate([g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)])

        # Part Dimensions & Volume (Standardized to mm)
        dims = mesh.extents * 1000
        bbox = {"x": float(round(dims[0], 1)), "y": float(round(dims[1], 1)), "z": float(round(dims[2], 1))}
        
        # Enclosure volume logic: if it's too 'hollow', estimate shell volume
        measured_vol = abs(mesh.volume) * 1e9
        if measured_vol > (np.prod(dims) * 0.4):
            volume_mm3 = float((mesh.area * 1e6 / 2.0) * 1.5) # 1.5mm wall estimate
        else:
            volume_mm3 = float(measured_vol)

        # Advanced Orientation & Undercut Analysis
        has_u, axis, severity = get_best_orientation(mesh)
        
        # Costing (Indian Market Logic)
        mold_cost = float(28000.0 + (volume_mm3/1000 * 0.05) + (45000.0 if has_u else 0.0))
        per_piece = float(round((volume_mm3 / 69311.0) * 18.29, 2))

        return {
            "glb_url": f"/static/{glb_path.name}",
            "volume_cubic_mm": volume_mm3,
            "bounding_box_mm": bbox,
            "has_undercuts": has_u,
            "undercut_severity": severity,
            "optimal_axis": axis,
            "undercut_message": f"Optimal Pull: {axis}. " + 
                               ("Requires side-action sliders (undercuts detected)." if has_u 
                                else "Straight-pull compatible.")
        }
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_step.exists(): tmp_step.unlink()
