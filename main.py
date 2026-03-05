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


def analyze_step_features(step_text: str) -> dict:
    """
    Parse raw STEP file text to detect undercut-causing features
    """
    try:
        # Count cylindrical surfaces — each is a potential hole/boss
        cylinders = re.findall(r"CYLINDRICAL_SURFACE\s*\(", step_text, re.IGNORECASE)
        n_cylinders = len(cylinders)

        # Count toroidal surfaces — fillets inside pockets are undercut risk
        toroids = re.findall(r"TOROIDAL_SURFACE\s*\(", step_text, re.IGNORECASE)
        n_toroids = len(toroids)

        # Count conical surfaces — snap fits, undercut bosses
        conicals = re.findall(r"CONICAL_SURFACE\s*\(", step_text, re.IGNORECASE)
        n_conicals = len(conicals)

        # Extract cylinder radii to distinguish holes vs bosses
        radii = re.findall(
            r"CYLINDRICAL_SURFACE\s*\([^,]*,[^,]*,\s*([\d.]+)\s*\)",
            step_text,
            re.IGNORECASE,
        )
        radii_floats = [float(r) for r in radii]
        n_likely_holes = sum(1 for r in radii_floats if r < 15.0)
        n_likely_bosses = sum(1 for r in radii_floats if r >= 15.0)

        # Through-hole detection: cylinder referenced by EXACTLY 2 faces (sidewall only)
        cyl_ids = re.findall(
            r"#(\d+)\s*=\s*CYLINDRICAL_SURFACE\s*\(",
            step_text,
            re.IGNORECASE,
        )
        through_hole_count = 0
        for cid in cyl_ids:
            refs = re.findall(rf"ADVANCED_FACE\s*\([^)]*#{cid}[^)]*\)", step_text)
            # EXACTLY 2 faces = through-hole (2 sidewalls). More = blind hole with bottom
            if len(refs) == 2:
                through_hole_count += 1

        has_undercut_features = (
            through_hole_count > 0 or n_conicals > 0
        )

        return {
            "step_cylinders": n_cylinders,
            "step_through_holes": through_hole_count,
            "step_toroids": n_toroids,
            "step_conicals": n_conicals,
            "step_likely_holes": n_likely_holes,
            "step_likely_bosses": n_likely_bosses,
            "step_has_undercut_features": has_undercut_features,
        }

    except Exception as e:
        print(f"STEP parse error: {e}")
        return {"step_has_undercut_features": False}


def analyze_undercuts_geometric(mesh, pull_axis):
    """
    Detects if faces are perpendicular to the pull direction or 
    occluded (trapped) from both the top and bottom.
    """
    pull_vec = np.array(pull_axis)
    
    # 1. Find faces perpendicular to the pull (e.g., side holes)
    # Dot product of ~0 means the face points sideways relative to the mold opening.
    dots = np.dot(mesh.face_normals, pull_vec)
    perpendicular_mask = np.abs(dots) < 0.05
    
    # 2. Ray-casting Shadow Test
    # Fire rays in the + and - directions of the pull axis.
    # If a face is blocked in BOTH directions, it's an undercut.
    _, _, index_tri_up = mesh.ray.intersects_id(
        ray_origins=mesh.triangles_center,
        ray_directions=np.tile(pull_vec, (len(mesh.faces), 1)),
        multiple_hits=False
    )
    _, _, index_tri_dw = mesh.ray.intersects_id(
        ray_origins=mesh.triangles_center,
        ray_directions=np.tile(-pull_vec, (len(mesh.faces), 1)),
        multiple_hits=False
    )

    blocked_up = set(index_tri_up)
    blocked_dw = set(index_tri_dw)
    trapped_faces = blocked_up.intersection(blocked_dw)
    
    # Combine both types of undercuts
    undercut_indices = set(np.where(perpendicular_mask)[0]).union(trapped_faces)
    
    return {
        "count": len(undercut_indices),
        "area": mesh.area_faces[list(undercut_indices)].sum() if undercut_indices else 0
    }
def get_best_mold_analysis(mesh):
    axes = {
        "X-Axis": [1, 0, 0],
        "Y-Axis": [0, 1, 0],
        "Z-Axis": [0, 0, 1]
    }
    
    results = []
    for name, vector in axes.items():
        res = analyze_undercuts_geometric(mesh, vector)
        results.append({"axis": name, "data": res})

    # Pick the axis with the lowest number of undercut faces
    best = min(results, key=lambda x: x["data"]["count"])
    
    # If the 'best' axis still has many undercut faces, it truly needs side-actions.
    has_undercuts = best["data"]["count"] > 15 
    
    return {
        "has_undercuts": has_undercuts,
        "undercut_face_count": best["data"]["count"],
        "undercut_severity": "high" if has_undercuts else "low",
        "optimal_axis": best["axis"],
        "undercut_message": (
            f"Detected {best['data']['count']} undercut faces. Side-actions required."
            if has_undercuts else "Straight-pull compatible."
        )
    }
@app.post("/upload")
async def upload_step(file: UploadFile = File(...)):
    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()
    if suffix not in (".step", ".stp"):
        raise HTTPException(status_code=400, detail="Only .step/.stp files are supported")

    os.makedirs("static", exist_ok=True)

    # Create temporary paths for processing
    tmp_step_path = Path(tempfile.gettempdir()) / f"{uuid4()}{suffix}"
    glb_name = f"{uuid4()}.glb"
    glb_path = STATIC_DIR / glb_name

    try:
        # Save the uploaded file to disk
        with tmp_step_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        # 1. Convert STEP to GLB for 3D viewing and analysis
        result = cascadio.step_to_glb(str(tmp_step_path), str(glb_path))
        if result != 0:
            raise RuntimeError(f"cascadio conversion failed with code {result}")

        # 2. Load the mesh using trimesh
        mesh = trimesh.load(str(glb_path), force="mesh")
        
        # 3. Run the Smart Geometric Analysis (Tests X, Y, and Z axes)
        # Note: Ensure the 'get_best_mold_analysis' function is defined above this route
        undercut_data = get_best_mold_analysis(mesh)

        # 4. Calculate physical properties
        # Trimesh volume is in meters cubed, converting to cubic mm
        volume_mm3 = (float(mesh.volume) if hasattr(mesh, "volume") and mesh.volume else 0.0) * 1e9
        
        # Get bounding box extents in mm
        raw_extents = mesh.extents if hasattr(mesh, "extents") else [0.0, 0.0, 0.0]
        bounding_box_mm = {
            "x": float(raw_extents[0] * 1000.0),
            "y": float(raw_extents[1] * 1000.0),
            "z": float(raw_extents[2] * 1000.0),
        }

        return {
            "glb_url": f"/static/{glb_name}",
            "volume_cubic_mm": volume_mm3,
            "bounding_box_mm": bounding_box_mm,
            **undercut_data,  # This includes has_undercuts, severity, and optimal_axis
        }

    except Exception as e:
        print(f"Upload Error: {traceback.format_exc()}")
        if glb_path.exists():
            try:
                glb_path.unlink()
            except:
                pass
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Cleanup the temporary STEP file
        try:
            await file.close()
            if tmp_step_path.exists():
                tmp_step_path.unlink()
        except:
            pass





