from __future__ import annotations

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
from pydantic import BaseModel
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


# ── Pydantic model for reanalyze request ──────────────────────────────────────

class PullDirection(BaseModel):
    x: float
    y: float
    z: float

class ReanalyzeRequest(BaseModel):
    glb_filename: str
    pull_direction: PullDirection


# ── STEP feature parser ───────────────────────────────────────────────────────

def analyze_step_features(step_text: str) -> dict:
    try:
        cylinders = re.findall(r"CYLINDRICAL_SURFACE\s*\(", step_text, re.IGNORECASE)
        n_cylinders = len(cylinders)

        toroids = re.findall(r"TOROIDAL_SURFACE\s*\(", step_text, re.IGNORECASE)
        n_toroids = len(toroids)

        conicals = re.findall(r"CONICAL_SURFACE\s*\(", step_text, re.IGNORECASE)
        n_conicals = len(conicals)

        radii = re.findall(
            r"CYLINDRICAL_SURFACE\s*\([^,]*,[^,]*,\s*([\d.]+)\s*\)",
            step_text, re.IGNORECASE,
        )
        radii_floats = [float(r) for r in radii]
        n_likely_holes = sum(1 for r in radii_floats if r < 15.0)
        n_likely_bosses = sum(1 for r in radii_floats if r >= 15.0)

        cyl_ids = re.findall(r"#(\d+)\s*=\s*CYLINDRICAL_SURFACE\s*\(", step_text, re.IGNORECASE)
        through_hole_count = 0
        for cid in cyl_ids:
            refs = re.findall(rf"ADVANCED_FACE\s*\([^)]*#{cid}[^)]*\)", step_text)
            if len(refs) == 2:
                through_hole_count += 1

        # Only conicals are a true undercut signal — through-holes are NOT
        has_undercut_features = n_conicals > 0

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


# ── Undercut analyser ─────────────────────────────────────────────────────────

def analyze_undercuts(mesh, step_features: dict, pull_direction: list | None = None) -> dict:
    """
    Analyse undercuts relative to a given pull direction.

    If pull_direction is provided (user-confirmed surface), only that axis
    and its opposite are used. Otherwise all 6 axes are tested.
    """
    try:
        if hasattr(mesh, "dump"):
            meshes = mesh.dump()
            mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]

        normals = np.array(mesh.face_normals)
        areas = np.array(mesh.area_faces)
        total_area = float(np.sum(areas))

        # Build direction set
        if pull_direction is not None:
            pd = np.array(pull_direction, dtype=float)
            pd = pd / np.linalg.norm(pd)
            directions = [pd, -pd]
        else:
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
            reachable |= (np.dot(normals, d) > 0.0)

        unreachable_area = float(np.sum(areas[~reachable]))
        ratio = unreachable_area / total_area if total_area > 0 else 0
        pct = ratio * 100
        unreachable_count = int(np.sum(~reachable))

        step_has_undercut = step_features.get("step_has_undercut_features", False)
        conicals = step_features.get("step_conicals", 0)

        print(f"Undercut: ratio={pct:.2f}%, conicals={conicals}, pull={pull_direction}")

        # High confidence — mesh ratio alone is conclusive
        if ratio > 0.10:
            return {
                "has_undercuts": True,
                "undercut_face_count": unreachable_count,
                "undercut_severity": "high",
                "undercut_message": (
                    f"Undercut detected — {pct:.1f}% of surface area unreachable "
                    f"from the pull direction. Side-action sliders or lifters required, "
                    f"increasing tooling cost by ~25–40%."
                ),
            }

        # Confirmed by both mesh and STEP conicals
        if ratio > 0.04 and step_has_undercut:
            return {
                "has_undercuts": True,
                "undercut_face_count": unreachable_count,
                "undercut_severity": "high",
                "undercut_message": (
                    f"Undercut detected — {pct:.1f}% unreachable area confirmed by "
                    f"{conicals} conical feature(s) in geometry. "
                    f"Side-action sliders required, increasing tooling cost by ~25–40%."
                ),
            }

        # Moderate mesh signal alone
        if ratio > 0.04:
            return {
                "has_undercuts": True,
                "undercut_face_count": unreachable_count,
                "undercut_severity": "moderate",
                "undercut_message": (
                    f"Possible undercut — {pct:.1f}% of surface area may be difficult "
                    f"to demould. Review side holes or internal features manually."
                ),
            }

        # STEP-only signal — conicals but low mesh ratio
        if step_has_undercut:
            return {
                "has_undercuts": True,
                "undercut_face_count": unreachable_count,
                "undercut_severity": "moderate",
                "undercut_message": (
                    f"Possible undercut — {conicals} conical surface(s) detected. "
                    f"Verify if angled features require side-action tooling."
                ),
            }

        # Clean
        return {
            "has_undercuts": False,
            "undercut_face_count": 0,
            "undercut_severity": "low",
            "undercut_message": (
                f"No undercut risk — {pct:.1f}% unreachable area is within tolerance. "
                f"Part is compatible with a straight-pull mold."
            ),
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


# ── /upload endpoint ──────────────────────────────────────────────────────────

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

        step_text = tmp_step_path.read_text(errors="ignore")
        step_features = analyze_step_features(step_text)
        print(f"STEP features: {step_features}")

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

        undercut_data = analyze_undercuts(mesh, step_features, pull_direction=None)

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


# ── /reanalyze endpoint ───────────────────────────────────────────────────────

@app.post("/reanalyze")
async def reanalyze(req: ReanalyzeRequest):
    """
    Re-run undercut analysis using a user-confirmed pull direction.
    The GLB already exists in /static — no re-conversion needed.
    """
    glb_path = STATIC_DIR / req.glb_filename

    if not glb_path.exists():
        raise HTTPException(status_code=404, detail="GLB file not found. Please re-upload the model.")

    try:
        mesh = trimesh.load(str(glb_path), force="mesh")

        pull = [req.pull_direction.x, req.pull_direction.y, req.pull_direction.z]

        # No STEP features available at reanalyze time — pass empty dict
        undercut_data = analyze_undercuts(mesh, {}, pull_direction=pull)

        raw_volume_m3 = float(mesh.volume) if hasattr(mesh, "volume") and mesh.volume else 0.0
        volume_cubic_mm = raw_volume_m3 * 1e9

        raw_extents_m = mesh.extents if hasattr(mesh, "extents") else [0.0, 0.0, 0.0]
        bounding_box_mm = {
            "x": float(raw_extents_m[0] * 1000.0),
            "y": float(raw_extents_m[1] * 1000.0),
            "z": float(raw_extents_m[2] * 1000.0),
        }

        return {
            "glb_url": f"/static/{req.glb_filename}",
            "volume_cubic_mm": volume_cubic_mm,
            "bounding_box_mm": bounding_box_mm,
            **undercut_data,
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Reanalyze error: {e}")
        print(traceback.format_exc())
        raise
