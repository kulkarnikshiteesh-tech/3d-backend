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
    """
    Use STEP feature parsing only.
    Mesh-based area ratios are ignored to avoid false positives.
    """
    try:
        has_features = step_features.get("step_has_undercut_features", False)
        through_holes = int(step_features.get("step_through_holes", 0) or 0)
        conicals = int(step_features.get("step_conicals", 0) or 0)

        if has_features:
            details = []
            if through_holes > 0:
                details.append(f"{through_holes} through-hole(s)")
            if conicals > 0:
                details.append(f"{conicals} conical feature(s)")
            feature_str = ", ".join(details) if details else "undercut features"

            return {
                "has_undercuts": True,
                "undercut_face_count": None,
                "undercut_severity": "high",
                "undercut_message": (
                    f"Undercut detected — {feature_str} found in STEP geometry. "
                    "Side-action sliders or lifters required, increasing tooling cost by ~25–40%."
                ),
            }

        # No undercut features in STEP → treat as straight‑pull
        return {
            "has_undercuts": False,
            "undercut_face_count": 0,
            "undercut_severity": "low",
            "undercut_message": (
                "No undercut risk detected from STEP geometry. "
                "Part is compatible with a straight‑pull mold."
            ),
        }

    except Exception as e:
        print(f"Undercut error: {type(e).__name__}: {str(e)}")
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





