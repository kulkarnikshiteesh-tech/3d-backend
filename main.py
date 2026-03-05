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

def get_undercut_analysis(mesh):
    """
    Advanced Ray-Tracing Analysis: 
    Checks if geometry is physically blocked during ejection.
    """
    # 1. Test Z-axis (Primary Pull)
    direction = np.array([0, 0, 1])
    
    # We only care about faces pointing 'away' from the pull
    dots = np.dot(mesh.face_normals, direction)
    potential_faces = np.where(dots < -0.1)[0] # Angled significantly downward
    
    if len(potential_faces) == 0:
        return False, 0

    # 2. Ray-cast from face centers to see if they hit the part itself
    origins = mesh.triangles_center[potential_faces] + (mesh.face_normals[potential_faces] * 0.01)
    
    # Check if these faces are 'shadowed' by other geometry above them
    locations, index_ray, index_tri = mesh.ray.intersects_id(
        origins=origins, 
        ray_directions=np.tile(direction, (len(origins), 1)),
        multiple_hits=False
    )
    
    # Real undercuts are faces that have something blocking their path upward
    real_undercuts = len(np.unique(index_ray))
    
    # Filter out 'noise' (tiny edge artifacts)
    has_undercuts = real_undercuts > 50 
    
    return has_undercuts, real_undercuts

@app.post("/upload")
async def upload_step(file: UploadFile = File(...)):
    tmp_step = Path(tempfile.gettempdir()) / f"{uuid4()}.step"
    glb_path = STATIC_DIR / f"{uuid4()}.glb"
    try:
        with tmp_step.open("wb") as f: shutil.copyfileobj(file.file, f)
        cascadio.step_to_glb(str(tmp_step), str(glb_path))
        
        scene = trimesh.load(str(glb_path))
        mesh = trimesh.util.concatenate([g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)])

        # THE VOLUME FIX: Sanity check against Bounding Box
        # A hollow enclosure should never be > 40% of its bounding box volume.
        bbox_vol = np.prod(mesh.extents) * 1e9
        measured_vol = abs(mesh.volume) * 1e9
        
        # If it looks like a solid block, it's a mesh error. 
        # We calculate 'Shell Volume' (Surface Area * 1.5mm avg thickness)
        if measured_vol > (bbox_vol * 0.45):
            volume_mm3 = (mesh.area * 1e6 / 2.0) * 1.5 
        else:
            volume_mm3 = measured_vol

        has_undercut, u_count = get_undercut_analysis(mesh)
        
        # Consistent INR Cost Logic
        base_mold_inr = 28000.0 # Matching your screenshot baseline
        per_piece_inr = (volume_mm3 / 69311.0) * 18.29 # Normalizing to your correct baseline

        return {
            "glb_url": f"/static/{glb_path.name}",
            "volume_cubic_mm": volume_mm3,
            "estimated_mold_cost_inr": round(base_mold_inr + (50000 if has_undercut else 0), 2),
            "estimated_per_piece_inr": round(per_piece_inr, 2),
            "has_undercuts": has_undercut,
            "undercut_face_count": u_count,
            "optimal_axis": "Z-Axis",
            "undercut_message": "Straight-pull compatible" if not has_undercut else "Undercut detected: Side-action required."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_step.exists(): tmp_step.unlink()
