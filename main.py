from __future__ import annotations
import os, shutil, tempfile, traceback, math, gc
from pathlib import Path
from uuid import uuid4
import cascadio, trimesh, numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path("static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

def analyze_undercuts_strict(mesh, pull_axis):
    direction = np.array(pull_axis, dtype=float)
    direction /= np.linalg.norm(direction)
    
    # Identify faces pointing 'down' into the core
    back_mask = np.dot(mesh.face_normals, direction) < -0.1
    if not np.any(back_mask):
        return 0.0

    epsilon = mesh.scale * 0.001
    origins = mesh.triangles_center[back_mask] - (direction * epsilon)
    
    # Ray-cast to find internal obstructions
    hits = mesh.ray.intersects_any(origins, np.tile(direction, (len(origins), 1)))
    
    blocked_area = np.sum(mesh.area_faces[back_mask][hits])
    return float(blocked_area)

@app.get("/")
async def health(): return {"status": "online"}

@app.post("/upload")
async def upload_step(request: Request, file: UploadFile = File(...)):
    gc.collect()
    tmp_step = Path(tempfile.gettempdir()) / f"{uuid4()}.step"
    glb_filename = f"{uuid4()}.glb"
    glb_path = STATIC_DIR / glb_filename
    
    try:
        content = await file.read()
        with tmp_step.open("wb") as f:
            f.write(content)
        del content
        
        # 0.8 Precision for Render stability
        cascadio.step_to_glb(str(tmp_step), str(glb_path), 0.8)
        
        scene = trimesh.load(str(glb_path))
        geometry = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geometry: raise ValueError("No mesh found")
        mesh = trimesh.util.concatenate(geometry)
        
        # 1. TOTAL SURFACE AREA for relative thresholding
        total_area = mesh.area
        
        # 2. 6-AXIS SEARCH
        test_dirs = {"X": [1,0,0], "-X": [-1,0,0], "Y": [0,1,0], "-Y": [0,-1,0], "Z": [0,0,1], "-Z": [0,0,-1]}
        raw_results = {n: analyze_undercuts_strict(mesh, v) for n, v in test_dirs.items()}
        
        # 3. Z-PREFERENCE (80% bias to favor standard CAD orientation)
        weighted_results = raw_results.copy()
        weighted_results["Z"] *= 0.2  # Massive bias to favor Z-pull for shells
        weighted_results["-Z"] *= 0.2

        best_axis = min(weighted_results, key=weighted_results.get)
        
        # 4. PERCENTAGE THRESHOLD (The Noise Filter)
        # Only flag if undercut area > 1.5% of total part area
        undercut_ratio = raw_results[best_axis] / total_area
        has_u = bool(undercut_ratio > 0.015) 

        # Physicals
        vol = abs(float(mesh.volume)) * 1e9
        dims = mesh.extents * 1000
        
        # Pricing
        slider_cost = 45000 if has_u else 0
        total_cost = 28000 + (vol/1000 * 0.08) + slider_cost

        return {
            "glb_url": f"{str(request.base_url).rstrip('/')}/static/{glb_filename}",
            "volume_cubic_mm": round(vol, 2),
            "bounding_box_mm": {"x": round(dims[0],1), "y": round(dims[1],1), "z": round(dims[2],1)},
            "has_undercuts": has_u,
            "optimal_axis": best_axis.replace("-", ""),
            "undercut_message": f"Optimal Pull: {best_axis.replace('-', '')}. " + ("Side-actions required." if has_u else "Straight-pull compatible."),
            "mold_cost_inr": round(total_cost, 2)
        }
    except Exception as e:
        return {"error": str(e)}
    finally:
        if tmp_step.exists(): tmp_step.unlink()
        if 'mesh' in locals(): del mesh
        gc.collect()
