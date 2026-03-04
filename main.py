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


def analyze_undercuts(mesh, step_features: dict) -> dict:
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
            reachable |= (dot > 0.25)

        unreachable_area = float(np.sum(areas[~reachable]))
        ratio = unreachable_area / total_area if total_area > 0 else 0
        pct = ratio * 100

        # STEP features override mesh analysis if they clearly detect undercuts
        step_override = step_features.get("step_has_undercut_features", False)
        through_holes = step_features.get("step_through_holes", 0)
        toroids = step_features.get("step_toroids", 0)
        conicals = step_features.get("step_conicals", 0)

        if step_override:
            details = []
            if through_holes > 0:
                details.append(f"{through_holes} through-hole(s)")
            if toroids > 0:
                details.append(f"{toroids} internal fillet(s)/pocket(s)")
            if conicals > 0:
                details.append(f"{conicals} conical feature(s)")
            feature_str = ", ".join(details)

            return {
                "has_undercuts": True,
                "undercut_face_count": int(np.sum(~reachable)),
                "undercut_severity": "high",
                "undercut_message": (
                    f"Undercut detected — {feature_str} found in STEP geometry. "
                    f"Side-action sliders or lifters required, increasing tooling cost by ~25–40%."
                ),
            }

        # Fall back to mesh ratio analysis
        if ratio > 0.03:
            return {
                "has_undercuts": True,
                "undercut_face_count": int(np.sum(~reachable)),
                "undercut_severity": "high",
                "undercut_message": (
                    f"Undercut detected — {pct:.1f}% of surface area unreachable "
                    f"from any pull direction. Side-action sliders required, "
                    f"increasing tooling cost by ~25–40%."
                ),
            }
        elif ratio > 0.015:
            return {
                "has_undercuts": True,
                "undercut_face_count": int(np.sum(~reachable)),
                "undercut_severity": "moderate",
                "undercut_message": (
                    f"Possible undercut — {pct:.1f}% of surface area may need sliders. "
                    f"Review side holes or internal features."
                ),
            }
        else:
            return {
                "has_undercuts": False,
                "undercut_face_count": 0,
                "undercut_severity": "low",
                "undercut_message": (
                    f"No undercut risk — {pct:.1f}% unreachable area. "
                    f"Part is compatible with straight-pull mold."
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

        # Parse STEP text BEFORE conversion
        step_text = tmp_step_path.read_text(errors="ignore")
        step_features = analyze_step_features(step_text)
        print(f"STEP features detected: {step_features}")

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

        undercut_data = analyze_undercuts(mesh, step_features)

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




