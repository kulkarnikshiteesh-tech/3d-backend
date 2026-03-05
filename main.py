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

def calculate_mold_cost_inr(volume_mm3, has_undercuts, undercut_count):
    # Reverted to more conservative base logic
    base_cost = 120000.0 
    volume_cm3 = volume_mm3 / 1000.0
    # Scaled complexity: Large hollow parts shouldn't scale as fast as solids
    size_factor = volume_cm3 * 3.5 
    
    undercut_penalty = 0.0
    if has_undercuts and undercut_count > 150:
        undercut_penalty = 45000.0 # Standard side-action cost
        
    return round(base_cost + size_factor + undercut_penalty, 2)

def get_refined_analysis(mesh):
    # Test only Z-axis as primary pull (as seen in your screenshots)
    pull_vec = np.array([0, 0, 1])
    dots = np.dot(mesh.face_normals, pull_vec)
    
    # STRICT FILTER: 
    # Ignore vertical walls (dot ~ 0)
    # Ignore flat floors (dot ~ -1)
    # Only flag significant 'Trapped' overhangs
    undercut_mask = (dots < -0.3) & (dots > -0.95)
    undercut_count = np.sum(undercut_mask)
    
    # Thresholding: 150 faces allows for mesh noise on hole rims
    has_undercuts = int(undercut_count) > 150
    
    return {
        "has_undercuts": has_undercuts,
        "undercut_face_count": int(undercut_count),
        "optimal_axis": "Z-Axis",
        "undercut_message": "Straight-pull compatible" if not has_undercuts else "Undercut detected: Side-action required."
    }

@app.post("/upload")
async def upload_step(file: UploadFile = File(...)):
    tmp_step = Path(tempfile.gettempdir()) / f"{uuid4()}.step"
    glb_path = STATIC_DIR / f"{uuid4()}.glb"
    try:
        with tmp_step.open("wb") as f: shutil.copyfileobj(file.file, f)
        cascadio.step_to_glb(str(tmp_step), str(glb_path))
        
        scene = trimesh.load(str(glb_path))
        mesh = trimesh.util.concatenate([g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)])

        # THE VOLUME FIX: 
        # If volume is > 50% of bounding box, it's likely counting 'air'. 
        # We use surface area * estimated wall thickness (2mm) as a sanity check.
        raw_vol = abs(mesh.volume) * 1e9
        bbox_vol = np.prod(mesh.extents) * 1e9
        
        if raw_vol > (bbox_vol * 0.4): # Sanity check for hollow parts
            # Estimate: (Surface Area / 2) * 2mm wall thickness
            volume_mm3 = (mesh.area * 1e6 / 2.0) * 2.0 
        else:
            volume_mm3 = raw_vol

        analysis = get_refined_analysis(mesh)
        cost_inr = calculate_mold_cost_inr(volume_mm3, analysis["has_undercuts"], analysis["undercut_face_count"])

        return {
            "glb_url": f"/static/{glb_path.name}",
            "volume_cubic_mm": volume_mm3,
            "estimated_mold_cost_inr": cost_inr,
            "bounding_box_mm": {"x": mesh.extents[0]*1000, "y": mesh.extents[1]*1000, "z": mesh.extents[2]*1000},
            **analysis
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_step.exists(): tmp_step.unlink()
