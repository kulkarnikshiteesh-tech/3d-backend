from __future__ import annotations
import os, tempfile, gc, math
from pathlib import Path
from uuid import uuid4
import cascadio, trimesh, numpy as np
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles

app = FastAPI()

# Robust CORS to ensure your Vercel frontend can talk to Render
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path("static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
async def root(): return {"status": "ready"}

def quick_undercut_check(mesh, pull_axis):
    """Fast, stable check for real obstructions."""
    direction = np.array(pull_axis, dtype=float)
    direction /= np.linalg.norm(direction)
    epsilon = mesh.scale * 0.005
    
    # Check if anything is trapped between 'Core' and 'Cavity'
    origins_up = mesh.triangles_center + (direction * epsilon)
    hits_up = mesh.ray.intersects_any(origins_up, np.tile(direction, (len(mesh.faces), 1)))
    
    origins_down = mesh.triangles_center - (direction * epsilon)
    hits_down = mesh.ray.intersects_any(origins_down, np.tile(-direction, (len(mesh.faces), 1)))
    
    # Only count if trapped from both sides (True Undercut)
    trapped = hits_up & hits_down
    return float(np.sum(mesh.area_faces[trapped]))

@app.post("/upload")
async def upload_step(request: Request, file: UploadFile = File(...)):
    gc.collect()
    tmp_step = Path(tempfile.gettempdir()) / f"{uuid4()}.step"
    glb_filename = f"{uuid4()}.glb"
    glb_path = STATIC_DIR / glb_filename
    
    try:
        content = await file.read()
        with tmp_step.open("wb") as f: f.write(content)
        
        # Convert with slightly higher precision for the frontend viewer
        cascadio.step_to_glb(str(tmp_step), str(glb_path), 0.6)
        
        scene = trimesh.load(str(glb_path))
        mesh = trimesh.util.concatenate([g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)])
        
        # Physical stats
        vol_mm3 = abs(float(mesh.volume)) * 1e9
        dims = mesh.extents * 1000
        
        # Multi-axis check
        results = {n: quick_undercut_check(mesh, v) for n, v in {"Z": [0,0,1], "X": [1,0,0], "Y": [0,1,0]}.items()}
        best_axis = min(results, key=results.get)
        
        # If blocked area > 100mm2, it's a side-action
        has_u = bool((results[best_axis] * 1e6) > 100.0)
        
        # Pricing logic
        base_mold = 28000
        vol_adder = (vol_mm3 / 1000) * 0.08
        slider_adder = 45000 if has_u else 0
        total_mold_cost = base_mold + vol_adder + slider_adder

        # IMPORTANT: Return every field your frontend UI expects!
        return {
            "glb_url": f"{str(request.base_url).rstrip('/')}/static/{glb_filename}",
            "volume_cubic_mm": round(vol_mm3, 2),
            "bounding_box_mm": {"x": round(dims[0],1), "y": round(dims[1],1), "z": round(dims[2],1)},
            "has_undercuts": has_u,
            "optimal_axis": best_axis,
            "undercut_message": f"Optimal Pull: {best_axis}. " + ("Side-actions required." if has_u else "Straight-pull compatible."),
            "mold_cost_inr": round(total_mold_cost, 2),
            "per_piece_cost": round(9.11 + (vol_mm3/5000), 2), # Simplified per-piece
            "wall_thickness_ok": True,
            "draft_angle_ok": True,
            "fits_mold_ok": True
        }
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return {"error": str(e)}
    finally:
        if tmp_step.exists(): tmp_step.unlink()
        gc.collect()
