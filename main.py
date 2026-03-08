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
    Parse raw STEP file text to detect undercut-causing features.
    Only conical surfaces are flagged as true undercut indicators.
    Through-holes and toroids are counted for info but NOT used to trigger undercut.
    """
    try:
        # Count cylindrical surfaces
        cylinders = re.findall(r"CYLINDRICAL_SURFACE\s*\(", step_text, re.IGNORECASE)
        n_cylinders = len(cylinders)

        # Count toroidal surfaces (fillets/rounds — NOT undercuts, just info)
        toroids = re.findall(r"TOROIDAL_SURFACE\s*\(", step_text, re.IGNORECASE)
        n_toroids = len(toroids)

        # Count conical surfaces — snap fits, angled bosses — TRUE undercut risk
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

        # Through-hole count — informational only, NOT an undercut trigger
        cyl_ids = re.findall(
            r"#(\d+)\s*=\s*CYLINDRICAL_SURFACE\s*\(",
            step_text,
            re.IGNORECASE,
        )
        through_hole_count = 0
        for cid in cyl_ids:
            refs = re.findall(rf"ADVANCED_FACE\s*\([^)]*#{cid}[^)]*\)", step_text)
            if len(refs) == 2:
                through_hole_count += 1

        # Only conical surfaces are a reliable STEP-level undercut signal
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


def analyze_undercuts(mesh, step_features: dict) -> dict:
    """
    Determine if the part has true undercuts that require side-action tooling.

    Strategy:
    - Use 6-axis mesh ray analysis with a strict threshold (dot > 0.0) to
      identify faces genuinely unreachable from any straight-pull direction.
    - Only flag as undercut if BOTH the mesh ratio is significant AND
      conical STEP features are present — or if the mesh ratio alone is
      very high (>10%), indicating a clearly complex geometry.
    - Through-holes and fillets (toroids) are explicitly excluded as triggers.
    """
    try:
        if hasattr(mesh, 'dump'):
            meshes = mesh.dump()
            mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]

        normals = np.array(mesh.face_normals)
        areas = np.array(mesh.area_faces)
        total_area = float(np.sum(areas))

        # 6 primary pull directions for straight-pull mold analysis
        directions = [
            np.array([0, 0, 1]),
            np.array([0, 0, -1]),
            np.array([1, 0, 0]),
            np.array([-1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, -1, 0]),
        ]

        # Strict threshold: dot > 0.0 means any face with ANY component
        # in a pull direction is considered reachable. This avoids false positives
        # on near-vertical walls that are actually fine for straight-pull molds.
        reachable = np.zeros(len(normals), dtype=bool)
        for d in directions:
            dot = np.dot(normals, d)
            reachable |= (dot > 0.0)

        unreachable_mask = ~reachable
        unreachable_area = float(np.sum(areas[unreachable_mask]))
        ratio = unreachable_area / total_area if total_area > 0 else 0
        pct = ratio * 100
        unreachable_count = int(np.sum(unreachable_mask))

        # STEP feature context
        has_step_undercut = step_features.get("step_has_undercut_features", False)
        conicals = step_features.get("step_conicals", 0)

        print(f"Undercut mesh ratio: {pct:.2f}%, step_conicals: {conicals}, step_override: {has_step_undercut}")

        # --- Decision logic ---

        # HIGH confidence undercut: mesh ratio is very high (>10%) regardless of STEP
        # This catches truly complex geometry even without STEP metadata
        if ratio > 0.10:
            return {
                "has_undercuts": True,
                "undercut_face_count": unreachable_count,
                "undercut_severity": "high",
                "undercut_message": (
                    f"Undercut detected — {pct:.1f}% of surface area is unreachable "
                    f"from any straight-pull direction. Side-action sliders or lifters "
                    f"required, increasing tooling cost by ~25–40%."
                ),
            }

        # MEDIUM confidence: mesh ratio is moderate AND conical STEP features confirm it
        if ratio > 0.04 and has_step_undercut:
            details = []
            if conicals > 0:
                details.append(f"{conicals} conical feature(s)")
            feature_str = f" ({', '.join(details)})" if details else ""
            return {
                "has_undercuts": True,
                "undercut_face_count": unreachable_count,
                "undercut_severity": "high",
                "undercut_message": (
                    f"Undercut detected — {pct:.1f}% unreachable surface area "
                    f"confirmed by STEP geometry{feature_str}. "
                    f"Side-action sliders required, increasing tooling cost by ~25–40%."
                ),
            }

        # POSSIBLE undercut: moderate mesh ratio but no strong STEP confirmation
        if ratio > 0.04:
            return {
                "has_undercuts": True,
                "undercut_face_count": unreachable_count,
                "undercut_severity": "moderate",
                "undercut_message": (
                    f"Possible undercut — {pct:.1f}% of surface area may be difficult "
                    f"to reach from straight-pull directions. Review side holes or "
                    f"internal features manually."
                ),
            }

        # STEP-only signal (conicals present but mesh ratio is low)
        # Flag as moderate — conicals are real but geometry may still be mouldable
        if has_step_undercut:
            return {
                "has_undercuts": True,
                "undercut_face_count": unreachable_count,
                "undercut_severity": "moderate",
                "undercut_message": (
                    f"Possible undercut — {conicals} conical surface(s) detected in STEP "
                    f"geometry. Verify if angled features require side-action tooling."
                ),
            }

        # No undercut
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
