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
        if hasattr(mesh, 'dump'):
            meshes = mesh.dump()
            mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]

        normals = np.array(mesh.face_normals)
        areas = np.array(mesh.area_faces)
        total_area = float(np.sum(areas))

        directions = [
            np.array([0, 0, 1]),
            np.array([0, 0, -1]),
            np.array([1, 0, 0]),
            np.array([-1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, -1, 0]),
        ]

        reachable = np.zeros(len(normals), dtype=bool)
        for d in directions:
            dot = np.dot(normals, d)
            reachable |= (dot > 0.15)

        unreachable_area = float(np.sum(areas[~reachable]))
        ratio = unreachable_area / total_area if total_area > 0 else 0
        pct = ratio * 100

        if ratio > 0.06:
            return {
                "has_undercuts": True,
                "undercut_face_count": int(np.sum(~reachable)),
                "undercut_severity": "high",
                "undercut_message": f"Undercut detected — {pct:.1f}% of surface area is not reachable from any pull direction. Side-action sliders required, increasing tooling cost by ~25–40%.",
            }
        elif ratio > 0.02:
            return {
                "has_undercuts": True,
                "undercut_face_count": int(np.sum(~reachable)),
                "undercut_severity": "moderate",
                "undercut_message": f"Possible undercut — {pct:.1f}% of surface area may need sliders. Review side holes or internal features.",
            }
        else:
            return {
                "has_undercuts": False,
                "undercut_face_count": 0,
                "undercut_severity": "low",
                "undercut_message": f"No undercut risk — {pct:.1f}% unreachable area. Part is compatible with straight-pull mold.",
            }
    except Exception as e:
        print(f"Undercut error: {type(e).__name__}: {str(e)}")
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




