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


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    error_msg = f"{type(exc).__name__}: {str(exc)}"
    print(f"CRITICAL ERROR: {error_msg}")
    print(traceback.format_exc())
    return JSONResponse(status_code=500, content={"detail": error_msg})


@app.get("/health")
async def health():
    return {"status": "ok"}


def analyze_undercuts(mesh) -> dict:
    try:
        # Handle Scene or multi-mesh objects
        if isinstance(mesh, trimesh.scene.Scene):
            mesh = trimesh.util.concatenate(list(mesh.dump()))
        elif not isinstance(mesh, trimesh.Trimesh):
            mesh = trimesh.util.concatenate(list(mesh.dump()))


        pull_dir = np.array([0.0, 0.0, 1.0])

        points, face_indices = trimesh.sample.sample_surface(mesh, 1000)
        normals = mesh.face_normals[face_indices]

        undercut_hits = 0
        checked = 0

        for point, normal in zip(points, normals):
            if np.dot(normal, pull_dir) > 0.3:
                continue
            checked += 1
            origin = point + pull_dir * 0.1
            locations, _, _ = mesh.ray.intersects_location(
                ray_origins=np.array([origin]),
                ray_directions=np.array([pull_dir])
            )
            if len(locations) > 0:
                undercut_hits += 1

        ratio = undercut_hits / checked if checked > 0 else 0

        if ratio > 0.25:
            return {
                "has_undercuts": True,
                "undercut_face_count": undercut_hits,
                "undercut_severity": "high",
                "undercut_message": f"High undercut risk — {undercut_hits} points blocked from pull direction. Side-action sliders likely required, increasing tooling cost by ~25–40%.",
            }
        elif ratio > 0.10:
            return {
                "has_undercuts": True,
                "undercut_face_count": undercut_hits,
                "undercut_severity": "moderate",
                "undercut_message": f"Moderate undercut risk — {undercut_hits} points may be occluded. Review part for side holes or overhangs.",
            }
        else:
            return {
                "has_undercuts": False,
                "undercut_face_count": undercut_hits,
                "undercut_severity": "low",
                "undercut_message": "No undercut risk — part is fully compatible with straight-pull mold.",
            }
    except Exception as e:
        print(f"Undercut analysis error: {type(e).__name__}: {str(e)}")
        print(traceback.format_exc())
        return {
            "has_undercuts": None,
            "undercut_face_count": None,
            "undercut_severity": "unknown",
            "undercut_message": "Undercut analysis could not be completed for this part.",
        }


@app.post("/upload")
async def upload_step(file: UploadFile = File(...)):
    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()
    if suffix not in (".step", ".stp"):
        raise HTTPException(status_code=400, detail="Only .step/.stp files are supported")

    os.makedirs("static", exist_ok=True)

    tmp_step_path = Path(tempfile.gettempdir()) / f"{uuid4()}{suffix}"
    glb_name = f"{uuid4()}.glb"
    glb_path = STATIC_DIR / glb_name

    try:
        with tmp_step_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        result = cascadio.step_to_glb(str(tmp_step_path), str(glb_path))
        if result != 0:
            raise RuntimeError(f"cascadio conversion failed with code {result}")

        mesh = trimesh.load(str(glb_path), force="mesh")
        glb_url = f"/static/{glb_name}"

        raw_volume_m3 = float(mesh.volume) if hasattr(mesh, "volume") and mesh.volume else 0.0
        volume_cubic_mm = raw_volume_m3 * 1e9

        raw_extents_m = mesh.extents if hasattr(mesh, "extents") else [0.0, 0.0, 0.0]
        bounding_box_mm = {
            "x": float(raw_extents_m[0] * 1000.0),
            "y": float(raw_extents_m[1] * 1000.0),
            "z": float(raw_extents_m[2] * 1000.0),
        }

        undercut_data = analyze_undercuts(mesh)

        return {
            "glb_url": glb_url,
            "volume_cubic_mm": volume_cubic_mm,
            "bounding_box_mm": bounding_box_mm,
            **undercut_data,
        }
    except HTTPException:
        raise
    except Exception:
        if glb_path.exists():
            try:
                glb_path.unlink()
            except Exception:
                pass
        raise
    finally:
        try:
            await file.close()
        except Exception:
            pass
        if tmp_step_path.exists():
            try:
                tmp_step_path.unlink()
            except Exception:
                pass



