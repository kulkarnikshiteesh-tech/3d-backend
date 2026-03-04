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


@app.post("/upload")
async def upload_step(file: UploadFile = File(...)):
    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()
    if suffix not in (".step", ".stp"):
        raise HTTPException(status_code=400, detail="Only .step/.stp files are supported")

    # Ensure static directory exists before we attempt to write into it.
    os.makedirs("static", exist_ok=True)

    tmp_step_path = Path(tempfile.gettempdir()) / f"{uuid4()}{suffix}"
    glb_name = f"{uuid4()}.glb"
    glb_path = STATIC_DIR / glb_name

    try:
        with tmp_step_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        # Convert STEP to GLB bytes using cascadio 0.0.13
        glb_bytes = cascadio.step_to_glb(str(tmp_step_path))

        # Save GLB to disk
        with open(str(glb_path), "wb") as f:
            f.write(glb_bytes)

        # Load GLB into trimesh for geometry calculations
        mesh = trimesh.load(io.BytesIO(glb_bytes), file_type="glb", force="mesh")
        volume = float(mesh.volume) if hasattr(mesh, "volume") and mesh.volume else 0.0
        extents = [float(x) for x in mesh.extents] if hasattr(mesh, "extents") else [0, 0, 0]

        return {
            "glb_url": f"/static/{glb_name}",
            "volume_cubic_mm": volume,
            "bounding_box_mm": {"x": extents[0], "y": extents[1], "z": extents[2]},
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
