from __future__ import annotations

import gc
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
        cylinders  = re.findall(r"CYLINDRICAL_SURFACE\s*\(", step_text, re.IGNORECASE)
        toroids    = re.findall(r"TOROIDAL_SURFACE\s*\(",    step_text, re.IGNORECASE)
        conicals   = re.findall(r"CONICAL_SURFACE\s*\(",     step_text, re.IGNORECASE)

        radii = re.findall(
            r"CYLINDRICAL_SURFACE\s*\([^,]*,[^,]*,\s*([\d.]+)\s*\)",
            step_text, re.IGNORECASE,
        )
        radii_floats    = [float(r) for r in radii]
        n_likely_holes  = sum(1 for r in radii_floats if r < 15.0)
        n_likely_bosses = sum(1 for r in radii_floats if r >= 15.0)

        cyl_ids = re.findall(r"#(\d+)\s*=\s*CYLINDRICAL_SURFACE\s*\(", step_text, re.IGNORECASE)
        through_hole_count = 0
        for cid in cyl_ids:
            refs = re.findall(rf"ADVANCED_FACE\s*\([^)]*#{cid}[^)]*\)", step_text)
            if len(refs) == 2:
                through_hole_count += 1

        return {
            "step_cylinders":            len(cylinders),
            "step_through_holes":        through_hole_count,
            "step_toroids":              len(toroids),
            "step_conicals":             len(conicals),
            "step_likely_holes":         n_likely_holes,
            "step_likely_bosses":        n_likely_bosses,
            "step_has_undercut_features": len(conicals) > 0,
        }
    except Exception as e:
        print(f"STEP parse error: {e}")
        return {"step_has_undercut_features": False}


# ── Assembly / multi-part detector ───────────────────────────────────────────

def detect_step_parts(step_text: str) -> dict:
    """
    Detect if a STEP file contains multiple parts or is an assembly.
    Returns: { is_assembly: bool, part_count: int, error_code: str | None }
    """
    # NEXT_ASSEMBLY_USAGE_OCCURENCE is only present in assemblies
    nauo = re.findall(
        r"NEXT_ASSEMBLY_USAGE_OCCURENCE\s*\(", step_text, re.IGNORECASE
    )
    # Count PRODUCT entries — each part/body has one
    products = re.findall(
        r"=\s*PRODUCT\s*\(", step_text, re.IGNORECASE
    )
    # Count manifold solid bodies
    solids = re.findall(
        r"MANIFOLD_SOLID_BREP\s*\(", step_text, re.IGNORECASE
    )

    part_count = max(len(products), len(solids))
    is_assembly = len(nauo) > 0

    if is_assembly:
        return {
            "is_assembly": True,
            "part_count": len(products),
            "error_code": "assembly",
        }
    if len(solids) > 1:
        return {
            "is_assembly": False,
            "part_count": len(solids),
            "error_code": "multiple_parts",
        }
    return {
        "is_assembly": False,
        "part_count": 1,
        "error_code": None,
    }


# ── Memory-safe mesh loader ───────────────────────────────────────────────────

def load_mesh_safe(path: str) -> trimesh.Trimesh:
    """
    Load a GLB/STEP mesh and immediately flatten Scene → single Trimesh.
    Explicit gc.collect() after load keeps peak RSS low on the free tier.
    """
    raw = trimesh.load(path, force="mesh")
    if isinstance(raw, trimesh.Scene):
        meshes = [g for g in raw.geometry.values() if isinstance(g, trimesh.Trimesh)]
        mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
        del raw, meshes
    elif hasattr(raw, "dump"):
        parts = raw.dump()
        mesh = trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]
        del raw, parts
    else:
        mesh = raw
    gc.collect()
    return mesh


# ── Undercut analyser ─────────────────────────────────────────────────────────

def analyze_undercuts(mesh: trimesh.Trimesh, step_features: dict,
                      pull_direction: list | None = None) -> dict:
    try:
        normals = np.array(mesh.face_normals)
        areas   = np.array(mesh.area_faces)
        total_area = float(np.sum(areas))

        if pull_direction is not None:
            pd = np.array(pull_direction, dtype=float)
            pd = pd / np.linalg.norm(pd)
            directions = [pd, -pd]
        else:
            directions = [
                np.array([0, 0,  1]), np.array([0, 0, -1]),
                np.array([1, 0,  0]), np.array([-1, 0, 0]),
                np.array([0, 1,  0]), np.array([0, -1, 0]),
            ]

        reachable = np.zeros(len(normals), dtype=bool)
        for d in directions:
            reachable |= (np.dot(normals, d) > 0.0)

        unreachable_mask  = ~reachable
        unreachable_area  = float(np.sum(areas[unreachable_mask]))
        ratio             = unreachable_area / total_area if total_area > 0 else 0
        pct               = ratio * 100
        unreachable_count = int(np.sum(unreachable_mask))

        # Return face indices for coloured GLB export
        undercut_face_indices = np.where(unreachable_mask)[0].tolist()

        step_has_undercut = step_features.get("step_has_undercut_features", False)
        conicals          = step_features.get("step_conicals", 0)

        print(f"Undercut: ratio={pct:.2f}%, conicals={conicals}, pull={pull_direction}")

        # Free large arrays now that we have what we need
        del normals, areas, reachable, unreachable_mask
        gc.collect()

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
                    f"{conicals} conical feature(s). "
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


# ── /upload endpoint ──────────────────────────────────────────────────────────

@app.post("/upload")
async def upload_step(file: UploadFile = File(...)):
    filename = file.filename or ""
    suffix   = Path(filename).suffix.lower()
    if suffix not in (".step", ".stp"):
        raise HTTPException(status_code=400, detail="Only .step/.stp files are supported")

    os.makedirs("static", exist_ok=True)

    uid           = str(uuid4())
    tmp_step_path = Path(tempfile.gettempdir()) / f"{uid}{suffix}"
    glb_name      = f"{uid}.glb"
    glb_path      = STATIC_DIR / glb_name

    try:
        # 1. Save uploaded file
        with tmp_step_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        # 2. Parse STEP features (text only — cheap)
        step_text     = tmp_step_path.read_text(errors="ignore")
        step_features = analyze_step_features(step_text)
        print(f"STEP features: {step_features}")

        # 3. Assembly / multi-part check ──────────────────────────────
        part_info = detect_step_parts(step_text)
        if part_info["error_code"] == "assembly":
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "assembly",
                    "part_count": part_info["part_count"],
                    "message": (
                        f"This file contains {part_info['part_count']} parts assembled together. "
                        f"Makeable analyses single parts only. "
                        f"Open your CAD software, isolate one part, and export it as a separate STEP file."
                    ),
                }
            )
        if part_info["error_code"] == "multiple_parts":
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "multiple_parts",
                    "part_count": part_info["part_count"],
                    "message": (
                        f"This file contains {part_info['part_count']} separate bodies. "
                        f"Makeable analyses one body at a time. "
                        f"Export each body as its own STEP file and upload them one by one."
                    ),
                }
            )
        # ─────────────────────────────────────────────────────────────

        del step_text
        gc.collect()

        # 4. Convert STEP → GLB
        result = cascadio.step_to_glb(str(tmp_step_path), str(glb_path))
        if result != 0:
            raise RuntimeError(f"cascadio conversion failed with code {result}")

        # 5. Load mesh safely and immediately free cascadio objects
        mesh = load_mesh_safe(str(glb_path))

        raw_volume_m3  = float(mesh.volume) if hasattr(mesh, "volume") and mesh.volume else 0.0
        volume_cubic_mm = raw_volume_m3 * 1e9
        raw_extents_m  = mesh.extents if hasattr(mesh, "extents") else [0.0, 0.0, 0.0]
        bounding_box_mm = {
            "x": float(raw_extents_m[0] * 1000.0),
            "y": float(raw_extents_m[1] * 1000.0),
            "z": float(raw_extents_m[2] * 1000.0),
        }

        # 6. Undercut analysis (no pull direction on initial upload)
        undercut_data = analyze_undercuts(mesh, step_features, pull_direction=None)

        # 7. Free mesh now — we're done with it
        del mesh
        gc.collect()

        return {
            "glb_url":        f"/static/{glb_name}",
            "volume_cubic_mm": volume_cubic_mm,
            "bounding_box_mm": bounding_box_mm,
            **undercut_data,
        }

    except HTTPException:
        raise
    except Exception:
        if glb_path.exists():
            try: glb_path.unlink()
            except Exception: pass
        raise
    finally:
        try: await file.close()
        except Exception: pass
        if tmp_step_path.exists():
            try: tmp_step_path.unlink()
            except Exception: pass
        gc.collect()


# ── /reanalyze endpoint ───────────────────────────────────────────────────────

@app.post("/reanalyze")
async def reanalyze(req: ReanalyzeRequest):
    """
    Re-run undercut analysis with a user-confirmed pull direction.
    GLB already exists in /static — no re-conversion needed.
    """
    # Strip query-string cache busters if accidentally included
    glb_filename = req.glb_filename.split("?")[0]
    glb_path     = STATIC_DIR / glb_filename

    if not glb_path.exists():
        raise HTTPException(
            status_code=404,
            detail="GLB file not found. The server may have restarted — please re-upload your model."
        )

    try:
        # Load mesh, free immediately after use
        mesh = load_mesh_safe(str(glb_path))

        pull = [req.pull_direction.x, req.pull_direction.y, req.pull_direction.z]

        raw_volume_m3   = float(mesh.volume) if hasattr(mesh, "volume") and mesh.volume else 0.0
        volume_cubic_mm = raw_volume_m3 * 1e9
        raw_extents_m   = mesh.extents if hasattr(mesh, "extents") else [0.0, 0.0, 0.0]
        bounding_box_mm = {
            "x": float(raw_extents_m[0] * 1000.0),
            "y": float(raw_extents_m[1] * 1000.0),
            "z": float(raw_extents_m[2] * 1000.0),
        }

        # Load STEP features from saved JSON if available (better accuracy)
        json_path = STATIC_DIR / glb_filename.replace(".glb", ".json")
        if json_path.exists():
            import json
            step_features = json.loads(json_path.read_text())
        else:
            step_features = {}

        undercut_data = analyze_undercuts(mesh, step_features, pull_direction=pull)

        del mesh
        gc.collect()

        return {
            "glb_url":         f"/static/{glb_filename}",
            "volume_cubic_mm":  volume_cubic_mm,
            "bounding_box_mm":  bounding_box_mm,
            **undercut_data,
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Reanalyze error: {e}")
        print(traceback.format_exc())
        raise
    finally:
        gc.collect()
