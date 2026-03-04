from __future__ import annotations

import os
import shutil
import tempfile
import traceback
from pathlib import Path
from uuid import uuid4

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

    stl_path = tmp_step_path.with_suffix(".stl")

    try:
        with tmp_step_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        # Geometry processing via pythonocc-core (STEP -> tessellated shape -> STL -> GLB)
        from OCC.Core.Bnd import Bnd_Box
        from OCC.Core.BRepGProp import brepgprop
        from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
        from OCC.Core.BRepBndLib import brepbndlib
        from OCC.Core.GProp import GProp_GProps
        from OCC.Core.IFSelect import IFSelect_RetDone
        from OCC.Core.STEPControl import STEPControl_Reader
        from OCC.Core.StlAPI import StlAPI_Writer

        import numpy as np  # noqa: F401
        import trimesh

        # 1. Read the STEP file
        reader = STEPControl_Reader()
        status = reader.ReadFile(str(tmp_step_path))
        if status != IFSelect_RetDone:
            raise RuntimeError("Failed to read STEP file")
        reader.TransferRoots()
        shape = reader.OneShape()

        # 2. Tessellate the shape into a mesh
        mesh = BRepMesh_IncrementalMesh(shape, 0.1)
        mesh.Perform()

        # 3. Calculate volume
        props = GProp_GProps()
        brepgprop.VolumeProperties(shape, props)
        volume = props.Mass()

        # 4. Calculate bounding box
        bbox = Bnd_Box()
        brepbndlib.Add(shape, bbox)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
        extents = [xmax - xmin, ymax - ymin, zmax - zmin]

        # 5. Export to GLB using trimesh (via temporary STL)
        writer = StlAPI_Writer()
        writer.Write(shape, str(stl_path))
        mesh_tri = trimesh.load(str(stl_path), force="mesh")
        mesh_tri.export(str(glb_path))

        return {
            "glb_url": f"/static/{glb_name}",
            "volume_cubic_mm": float(volume),
            "bounding_box_mm": {"x": float(extents[0]), "y": float(extents[1]), "z": float(extents[2])},
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

        try:
            if stl_path.exists():
                stl_path.unlink()
        except Exception:
            pass

        if tmp_step_path.exists():
            try:
                tmp_step_path.unlink()
            except Exception:
                pass
