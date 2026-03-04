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


def analyze_undercuts(mesh: trimesh.Trimesh) -> dict:
    try:
        # Sample points on the mesh surface
        points, face_indices = trimesh.sample.sample_surface(mesh, 2000)
        normals = mesh.face_normals[face_indices]

        pull_dir = np.array([0, 0, 1])  # pull from top
        pull_dir_neg = np.array([0, 0, -1])

        undercut_count = 0

        for i, (point, normal) in enumerate(zip(points, normals)):
            # Only check faces that face sideways (not top/bottom)
            dot = abs(np.dot(normal, pull_dir))
            if dot > 0.3:  # skip faces that mostly face up or down
                continue

            # Shoot ray upward from this point
            offset_point = point + pull_dir * 0.01
            locs, _, _ = mesh.ray.intersects_location(
                ray_origins=[offset_point],
                ray_directions=[pull_dir]
            )
            if len(locs) > 0:
                undercut_count += 1  # something blocks the exit = undercut

        total_checked = sum(1 for n in normals if abs(np.dot(n, pull_dir)) <= 0.3)
        ratio = undercut_count / total_checked if total_checked > 0 else 0

        if ratio > 0.15:
            return {
                "has_undercuts": True,
                "undercut_face_count": undercut_count,
                "undercut_severity": "high",
                "undercut_message": f"High undercut risk — {undercut_count} surface points are occluded from pull direction. Side-action sliders likely required, increasing tooling cost by ~25–40%.",
            }
        elif ratio > 0.05:
            return {
                "has_undercuts": True,
                "undercut_face_count": undercut_count,
                "undercut_severity": "moderate",
                "undercut_message": f"Moderate undercut risk — {undercut_count} surface points may be occluded. Review part for side holes or overhangs.",
            }
        else:
            return {
                "has_undercuts": False,
                "undercut_face_count": undercut_count,
                "undercut_severity": "low",
                "undercut_message": f"Low undercut risk — part appears compatible with straight-pull mold.",
            }
    except Exception as e:
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


