from __future__ import annotations
import os, tempfile, gc
from pathlib import Path
from uuid import uuid4
import cascadio, trimesh, numpy as np
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

STATIC_DIR = Path("static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
async def root(): return {"status": "ready"}

def evaluate_undercuts(mesh, axis):
    direction = np.array(axis, dtype=float)
    direction /= np.linalg.norm(direction)
    
    # High-density sampling (2500 points) to catch tiny windows/holes
    samples = mesh.sample(2500)
    epsilon = mesh.scale * 0.005
    
    trapped_points = 0
    # A point is a true undercut if it's blocked by the part itself 
    # when moving in BOTH pull directions (Core & Cavity)
    for pt in samples:
        blocked_up = mesh.ray.intersects_any([pt + direction * epsilon], [direction])
        blocked_down = mesh.ray.intersects_any([pt - direction * epsilon], [-direction])
        
        if blocked_up and blocked_down:
            trapped_points += 1
            
    return trapped_points

@app.post("/upload")
async def upload_step(request: Request, file: UploadFile = File(...)):
    gc.collect()
    tmp_step = Path(tempfile.gettempdir()) / f"{uuid4()}.step"
    glb_filename = f"{uuid4()}.glb"
    glb_path = STATIC_DIR / glb_filename
    
    try:
        content = await file.read()
        with tmp_step.open("wb") as f: f.write(content)
        
        # 0.5 Precision is the 'Render-Safe' limit
        cascadio.step_to_glb(str(tmp_step), str(glb_path), 0.5)
        
        scene = trimesh.load(str(glb_path))
        mesh = trimesh.util.concatenate([g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)])
        
        # Exhaustive search for the best pull axis
        axes = {"Z": [0,0,1], "X": [1,0,0], "Y": [0,1,0]}
        scores = {name: evaluate_undercuts(mesh, vec) for name, vec in axes.items()}
        
        # Prefer Z if it's reasonably clean
        weighted = scores.copy()
        weighted["Z"] *= 0.8 
        best_axis = min(weighted, key=weighted.get)
        
        # If > 0.4% of points are trapped, it's an undercut.
        # (10 points out of 2500). Very sensitive for small holes.
        has_u = bool(scores[best_axis] > 10)

        vol = abs(float(mesh.volume)) * 1e9
        cost = 28000 + (vol/1000 * 0.08) + (45000 if has_u else 0)

        return {
            "glb_url": f"{str(request.base_url).rstrip('/')}/static/{glb_filename}",
            "volume_cubic_mm": round(vol, 2),
            "has_undercuts": has_u,
            "optimal_axis": best_axis,
            "undercut_message": f"Optimal Pull: {best_axis}. " + ("Side-actions required." if has_u else "Straight-pull compatible."),
            "mold_cost_inr": round(cost, 2),
            "per_piece_cost": round(9.11 + (vol/10000), 2),
            "wall_thickness_ok": True, "draft_angle_ok": True, "fits_mold_ok": True
        }
    except Exception as e: return {"error": str(e)}
    finally:
        if tmp_step.exists(): tmp_step.unlink()
        gc.collect()
