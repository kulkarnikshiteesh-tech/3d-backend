from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import traceback
from pathlib import Path
from uuid import uuid4

import cascadio
import trimesh
import trimesh.visual
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


# ── Pydantic models ───────────────────────────────────────────────────────────

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


# ── Assembly & moldability validator ─────────────────────────────────────────

def validate_part(step_text: str, mesh) -> dict | None:
    """
    Returns an error dict if the part fails validation, or None if it's clean.
    Checks: assembly, aspect ratio, negative volume, absolute thin wall.
    """

    # 1. Assembly detection — count distinct solid bodies
    solid_bodies = re.findall(
        r"MANIFOLD_SOLID_BREP\s*\(|CLOSED_SHELL\s*\(",
        step_text, re.IGNORECASE
    )
    # Heuristic: CLOSED_SHELL count > 2 strongly suggests assembly
    closed_shells = re.findall(r"CLOSED_SHELL\s*\(", step_text, re.IGNORECASE)
    if len(closed_shells) > 2:
        return {
            "error": "assembly",
            "message": (
                f"This looks like an assembly with {len(closed_shells)} separate bodies. "
                f"Please upload individual parts one at a time for accurate DFM analysis."
            ),
        }

    # 2. Get bounding box dimensions from mesh
    try:
        extents = mesh.extents if hasattr(mesh, "extents") else None
        if extents is not None:
            dims_mm = sorted([
                float(extents[0]) * 1000.0,
                float(extents[1]) * 1000.0,
                float(extents[2]) * 1000.0,
            ])
            smallest, largest = dims_mm[0], dims_mm[2]

            # 3. Aspect ratio check — pencil/rod/thin sheet guard
            if smallest > 0 and (largest / smallest) > 15:
                return {
                    "error": "not_moldable",
                    "message": (
                        f"This part has an extreme aspect ratio "
                        f"({largest:.1f}mm vs {smallest:.1f}mm). "
                        f"It looks like an extruded, turned, or sheet metal part — "
                        f"injection molding is unlikely to be the right process for this geometry."
                    ),
                }

            # 4. Absolute thin wall — smallest dimension < 0.5mm
            if smallest < 0.5:
                return {
                    "error": "not_moldable",
                    "message": (
                        f"The smallest dimension of this part is {smallest:.2f}mm, "
                        f"which is below the minimum wall thickness for injection molding (0.5mm). "
                        f"Please review your geometry."
                    ),
                }
    except Exception as e:
        print(f"Validation dimension error: {e}")

    # 5. Negative or zero volume — inside-out or empty mesh
    try:
        vol = float(mesh.volume) if hasattr(mesh, "volume") else 0.0
        if vol <= 0:
            return {
                "error": "geometry_error",
                "message": (
                    "The model has a geometry error (invalid or zero volume). "
                    "This usually means the mesh is inside-out or has open surfaces. "
                    "Try re-exporting from your CAD tool with 'Export as solid' enabled."
                ),
            }
    except Exception as e:
        print(f"Validation volume error: {e}")

    return None  # all checks passed


# ── Undercut analyser ─────────────────────────────────────────────────────────

def analyze_undercuts(mesh, step_features: dict, pull_direction: list | None = None) -> dict:
    """
    Analyse undercuts relative to a given pull direction.
    Returns undercut data including face indices of unreachable faces.
    """
    try:
        if hasattr(mesh, "dump"):
            meshes = mesh.dump()
            mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]

        normals = np.array(mesh.face_normals)
        areas = np.array(mesh.area_faces)
        total_area = float(np.sum(areas))

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

        unreachable_mask = ~reachable
        unreachable_area = float(np.sum(areas[unreachable_mask]))
        ratio = unreachable_area / total_area if total_area > 0 else 0
        pct = ratio * 100
        unreachable_count = int(np.sum(unreachable_mask))

        # Face indices for highlighting
        undercut_face_indices = np.where(unreachable_mask)[0].tolist()

        step_has_undercut = step_features.get("step_has_undercut_features", False)
        conicals = step_features.get("step_conicals", 0)

        print(f"Undercut: ratio={pct:.2f}%, conicals={conicals}, pull={pull_direction}")

        if ratio > 0.10:
            return {
                "has_undercuts": True,
                "undercut_face_count": unreachable_count,
                "undercut_face_indices": undercut_face_indices,
                "undercut_severity": "high",
                "undercut_message": (
                    f"Undercut detected — {pct:.1f}% of surface area unreachable "
                    f"from the pull direction. Side-action sliders or lifters required, "
                    f"increasing tooling cost by ~25–40%."
                ),
            }

        if ratio > 0.04 and step_has_undercut:
            return {
                "has_undercuts": True,
                "undercut_face_count": unreachable_count,
                "undercut_face_indices": undercut_face_indices,
                "undercut_severity": "high",
                "undercut_message": (
                    f"Undercut detected — {pct:.1f}% unreachable area confirmed by "
                    f"{conicals} conical feature(s) in geometry. "
                    f"Side-action sliders required, increasing tooling cost by ~25–40%."
                ),
            }

        if ratio > 0.04:
            return {
                "has_undercuts": True,
                "undercut_face_count": unreachable_count,
                "undercut_face_indices": undercut_face_indices,
                "undercut_severity": "moderate",
                "undercut_message": (
                    f"Possible undercut — {pct:.1f}% of surface area may be difficult "
                    f"to demould. Review side holes or internal features manually."
                ),
            }

        if step_has_undercut:
            return {
                "has_undercuts": True,
                "undercut_face_count": unreachable_count,
                "undercut_face_indices": undercut_face_indices,
                "undercut_severity": "moderate",
                "undercut_message": (
                    f"Possible undercut — {conicals} conical surface(s) detected. "
                    f"Verify if angled features require side-action tooling."
                ),
            }

        return {
            "has_undercuts": False,
            "undercut_face_count": 0,
            "undercut_face_indices": [],
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
            "undercut_face_indices": [],
            "undercut_severity": "unknown",
            "undercut_message": "Undercut analysis could not be completed for this part.",
        }


# ── Colored GLB exporter ──────────────────────────────────────────────────────

def export_colored_glb(source_glb: Path, undercut_face_indices: list[int], output_glb: Path):
    """
    Load the source GLB, color undercut faces orange and clean faces blue,
    then export to output_glb.
    Falls back to copying source if coloring fails.
    """
    try:
        mesh = trimesh.load(str(source_glb), force="mesh")

        if hasattr(mesh, "dump"):
            meshes = mesh.dump()
            mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]

        n_faces = len(mesh.faces)

        # Build per-face color array — RGBA uint8
        # Default: blue #3b6bca
        colors = np.full((n_faces, 4), [59, 107, 202, 255], dtype=np.uint8)

        # Undercut faces: orange #f97316
        if undercut_face_indices:
            idx = np.array(undercut_face_indices, dtype=int)
            # Clamp to valid range
            idx = idx[idx < n_faces]
            colors[idx] = [249, 115, 22, 255]

        mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, face_colors=colors)
        mesh.export(str(output_glb))
        print(f"Colored GLB exported: {output_glb.name} "
              f"({len(undercut_face_indices)} undercut faces highlighted)")

    except Exception as e:
        print(f"Colored GLB export failed: {e} — falling back to source GLB")
        shutil.copy2(str(source_glb), str(output_glb))


# ── /upload endpoint ──────────────────────────────────────────────────────────

@app.post("/upload")
async def upload_step(file: UploadFile = File(...)):
    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()
    if suffix not in (".step", ".stp"):
        raise HTTPException(status_code=400, detail="Only .step/.stp files are supported")

    os.makedirs("static", exist_ok=True)

    uid = str(uuid4())
    tmp_step_path = Path(tempfile.gettempdir()) / f"{uid}{suffix}"
    raw_glb_path = STATIC_DIR / f"{uid}_raw.glb"
    colored_glb_name = f"{uid}.glb"
    colored_glb_path = STATIC_DIR / colored_glb_name
    meta_path = STATIC_DIR / f"{uid}.json"

    try:
        with tmp_step_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        step_text = tmp_step_path.read_text(errors="ignore")
        step_features = analyze_step_features(step_text)
        print(f"STEP features: {step_features}")

        # Convert STEP → raw GLB
        result = cascadio.step_to_glb(str(tmp_step_path), str(raw_glb_path))
        if result != 0:
            raise RuntimeError(f"cascadio conversion failed with code {result}")

        mesh = trimesh.load(str(raw_glb_path), force="mesh")

        # ── Validation ────────────────────────────────────────────────────────
        validation_error = validate_part(step_text, mesh)
        if validation_error:
            return JSONResponse(status_code=422, content=validation_error)

        # ── Dimensions ───────────────────────────────────────────────────────
        raw_volume_m3 = float(mesh.volume) if hasattr(mesh, "volume") and mesh.volume else 0.0
        volume_cubic_mm = raw_volume_m3 * 1e9

        raw_extents_m = mesh.extents if hasattr(mesh, "extents") else [0.0, 0.0, 0.0]
        bounding_box_mm = {
            "x": float(raw_extents_m[0] * 1000.0),
            "y": float(raw_extents_m[1] * 1000.0),
            "z": float(raw_extents_m[2] * 1000.0),
        }

        # ── Undercut analysis ─────────────────────────────────────────────────
        undercut_data = analyze_undercuts(mesh, step_features, pull_direction=None)

        # ── Colored GLB export ────────────────────────────────────────────────
        export_colored_glb(raw_glb_path, undercut_data.get("undercut_face_indices", []), colored_glb_path)

        # ── Persist STEP features for reanalyze ──────────────────────────────
        meta_path.write_text(json.dumps(step_features))

        # Clean up raw GLB
        try:
            raw_glb_path.unlink()
        except Exception:
            pass

        return {
            "glb_url": f"/static/{colored_glb_name}",
            "volume_cubic_mm": volume_cubic_mm,
            "bounding_box_mm": bounding_box_mm,
            **undercut_data,
        }

    except HTTPException:
        raise
    except Exception:
        for p in [raw_glb_path, colored_glb_path, meta_path]:
            if p.exists():
                try:
                    p.unlink()
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
    Reloads persisted STEP features, re-colors the GLB with updated undercut faces.
    """
    uid = req.glb_filename.replace(".glb", "")
    colored_glb_path = STATIC_DIR / req.glb_filename
    meta_path = STATIC_DIR / f"{uid}.json"

    # We always re-export a new colored GLB — load from a temp raw copy if needed
    # Since we deleted the raw GLB, we reload the existing colored one as source
    if not colored_glb_path.exists():
        raise HTTPException(
            status_code=404,
            detail="GLB file not found. Please re-upload the model."
        )

    try:
        mesh = trimesh.load(str(colored_glb_path), force="mesh")
        pull = [req.pull_direction.x, req.pull_direction.y, req.pull_direction.z]

        # Reload persisted STEP features
        step_features = {}
        if meta_path.exists():
            try:
                step_features = json.loads(meta_path.read_text())
            except Exception:
                pass

        undercut_data = analyze_undercuts(mesh, step_features, pull_direction=pull)

        # Re-export colored GLB with updated undercut faces
        export_colored_glb(
            colored_glb_path,
            undercut_data.get("undercut_face_indices", []),
            colored_glb_path   # overwrite in place
        )

        raw_volume_m3 = float(mesh.volume) if hasattr(mesh, "volume") and mesh.volume else 0.0
        volume_cubic_mm = raw_volume_m3 * 1e9

        raw_extents_m = mesh.extents if hasattr(mesh, "extents") else [0.0, 0.0, 0.0]
        bounding_box_mm = {
            "x": float(raw_extents_m[0] * 1000.0),
            "y": float(raw_extents_m[1] * 1000.0),
            "z": float(raw_extents_m[2] * 1000.0),
        }

        # Add cache-bust timestamp so frontend reloads the updated GLB
        from time import time
        glb_url = f"/static/{req.glb_filename}?t={int(time())}"

        return {
            "glb_url": glb_url,
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
