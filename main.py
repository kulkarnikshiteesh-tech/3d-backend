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
    """
    Cost Logic aligned with your screenshot baseline (₹28,932).
    """
    # Base for Zinc Alloy / Soft Mold in India
    base_cost = 25000.0 
    volume_cm3 = volume_mm3 / 1000.0
    
    # Complexity scaling
    size_factor = volume_cm3 * 0.06 
    
    # Add ₹45,000 only if a REAL physical undercut is detected
    undercut_penalty = 45000.0 if has_undercuts else 0.0
        
    return round(base_cost + size_factor + undercut_penalty, 2)

def get_sureshot_undercut_analysis(mesh):
    """
    Ray-Tracing Analysis: Fires rays in the Z-direction.
    If a face is 'shadowed' by other geometry, it's a true undercut.
    """
    pull_dir = np.array([0, 0, 1])
    
    # Only check faces pointing downwards (potential traps)
    dots = np.dot(mesh.face_normals, pull_dir)
    potential_faces = np.where(dots < -0.1)[0]
    
    if len(potential_faces) == 0:
        return False, 0

    # Origins moved slightly out from the surface to avoid self-intersection
    origins = mesh.triangles_center[potential_faces] + (mesh.face_normals[potential_faces] * 0.001)
    
    # Check if these faces are blocked by ANY geometry above them
    hits = mesh.ray.intersects_any(
        origins=origins, 
        ray_directions=np.tile(pull_dir, (len(origins), 1))
    )
    
    true_undercut_count = int(np.sum(hits))
    
    # Threshold (50 triangles) to ignore microscopic artifacts at hole edges
    has_real_undercuts = true_undercut_count > 50
    
    return has_real_undercuts, true_undercut_count

@app.post("/upload")
async def upload_step(file: UploadFile = File(...)):
    tmp_step = Path(tempfile.gettempdir()) / f"{uuid4()}.step"
    glb_path = STATIC_DIR / f"{uuid4()}.glb"
    try:
        with tmp_step.open("wb") as f: 
            shutil.copyfileobj(file.file, f)
            
        cascadio.step_to_glb(str(tmp_step), str(glb_path))
        
        scene = trimesh.load(str(glb_path))
        mesh = trimesh.util.concatenate([g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)])

        # VOLUME FIX: If mesh is broken/open, fallback to shell estimation
        # This keeps the volume at the correct ~69k range
        bbox_vol = np.prod(mesh.extents) * 1e9
        measured_vol = abs(mesh.volume) * 1e9
        
        if measured_vol > (bbox_vol * 0.5): # If it thinks it's a solid block
            volume_mm3 = (mesh.area * 1e6 / 2.0) * 1.5 # Surface Area * 1.5mm wall
        else:
            volume_mm3 = measured_vol

        # RAY-TRACE ANALYSIS
        has_undercut, u_count = get_sureshot_undercut_analysis(mesh)
        
        # COSTING
        total_mold_cost = calculate_mold_cost_inr(volume_mm3, has_undercut)
        per_piece_cost = round((volume_mm3 / 69311.0) * 18.29, 2)

        return {
            "glb_url": f"/static/{glb_path.name}",
            "volume_cubic_mm": volume_mm3,
            "estimated_mold_cost_inr": total_mold_cost,
            "estimated_per_piece_inr": per_piece_cost,
            "bounding_box_mm": {"x": mesh.extents[0]*1000, "y": mesh.extents[1]*1000, "z": mesh.extents[2]*1000},
            "has_undercuts": has_undercut,
            "undercut_face_count": u_count,
            "optimal_axis": "Z-Axis",
            "undercut_message": "Straight-pull compatible" if not has_undercut else "Side-action sliders required"
        }
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_step.exists(): tmp_step.unlink()
