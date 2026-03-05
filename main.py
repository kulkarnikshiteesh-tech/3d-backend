from __future__ import annotations
import os, shutil, tempfile, traceback
from pathlib import Path
from uuid import uuid4
import cascadio, trimesh, numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles

app = FastAPI()
# Allowing all origins for Vercel/Render compatibility
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

STATIC_DIR = Path("static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

def calculate_mold_cost_inr(volume_mm3, has_undercuts):
    """Cost logic aligned with Indian soft-tooling market."""
    base_cost = 28000.0 
    volume_cm3 = volume_mm3 / 1000.0
    size_factor = volume_cm3 * 0.05 
    
    # Side-action penalty (Sliders/Lifters)
    undercut_penalty = 45000.0 if has_undercuts else 0.0
    return round(base_cost + size_factor + undercut_penalty, 2)

def analyze_axis_for_trapped_faces(mesh, axis_vector):
    """Counts faces that are physically blocked along a specific axis."""
    pull_dir = np.array(axis_vector) / np.linalg.norm(axis_vector)
    dots = np.abs(np.dot(mesh.face_normals, pull_dir))
    side_idx = np.where(dots < 0.1)[0]
    
    if len(side_idx) == 0: return 0

    origins = mesh.triangles_center[side_idx]
    try:
        # Sandwich check: Look 'Forward' and 'Backward' along the axis
        hits_f = mesh.ray.intersects_any(origins + (pull_dir * 0.1), np.tile(pull_dir, (len(origins), 1)))
        hits_b = mesh.ray.intersects_any(origins - (pull_dir * 0.1), np.tile(-pull_dir, (len(origins), 1)))
        return int(np.sum(np.logical_and(hits_f, hits_b)))
    except:
        return 9999

def get_advanced_analysis(mesh):
    """Evaluates X, Y, Z axes with a bias toward the Z-Axis (Common Sense logic)."""
    axes = {"X-Axis": [1,0,0], "Y-Axis": [0,1,0], "Z-Axis": [0,0,1]}
    results_with_bias = {}
    
    for name, vec in axes.items():
        count = analyze_axis_for_trapped_faces(mesh, vec)
        
        # BIAS: Enclosures are almost always molded on the Z-axis.
        # We penalize X and Y axes to prevent the 'sideways pull' false positive.
        bias_count = count
        if name != "Z-Axis":
            bias_count += 5000 
            
        results_with_bias[name] = bias_count
            
    # Select the best axis based on biased results
    best_axis = min(results_with_bias, key=results_with_bias.get)
    
    # Get the ACTUAL undercut count for the chosen axis for reporting
    final_u_count = analyze_axis_for_trapped_faces(mesh, axes[best_axis])
    
    has_u = final_u_count > 50
    severity = "High" if final_u_count > 500 else "Medium" if final_u_count > 50 else "None"
    
    return has_u, final_u_count, best_axis, severity

@app.post("/upload")
async def upload_step(file: UploadFile = File(...)):
    tmp_step = Path(tempfile.gettempdir()) / f"{uuid4()}.step"
    glb_path = STATIC_DIR / f"{uuid4()}.glb"
    try:
        with tmp_step.open("wb") as f: shutil.copyfileobj(file.file, f)
        cascadio.step_to_glb(str(tmp_step), str(glb_path))
        
        scene = trimesh.load(str(glb_path))
        mesh = trimesh.util.concatenate([g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)])

        # Bounding Box for Frontend
        dims = mesh.extents * 1000
        bbox = {"x": round(dims[0], 1), "y": round(dims[1], 1), "z": round(dims[2], 1)}
        
        # Volume logic (Shell estimation for enclosures)
        measured_vol = abs(mesh.volume) * 1e9
        if measured_vol > (np.prod(dims) * 0.5):
            volume_mm3 = (mesh.area * 1e6 / 2.0) * 1.5 
        else:
            volume_mm3 = measured_vol

        # DFM Analysis
        has_u, u_count, axis, severity = get_advanced_analysis(mesh)
        
        mold_cost = calculate_mold_cost_inr(volume_mm3, has_u)
        per_piece = round((volume_mm3 / 69311.0) * 18.29, 2)

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
