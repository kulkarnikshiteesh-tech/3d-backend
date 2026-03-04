from __future__ import annotations

import traceback
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable
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


def _call_with_fallbacks(func: Callable[..., Any], fallbacks: list[tuple[tuple[Any, ...], dict[str, Any]]]) -> None:
    last_exc: BaseException | None = None
    for args, kwargs in fallbacks:
        try:
            func(*args, **kwargs)
            return
        except TypeError as e:
            last_exc = e
            continue
    if last_exc is not None:
        raise last_exc


def convert_step_to_glb(step_path: Path, glb_path: Path) -> None:
    """
    Convert a STEP file into GLB using cascadio.

    cascadio's public API can vary between versions, so we try a few common call
    shapes and raise a helpful error if none match.
    """
    try:
        import cascadio  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Failed to import cascadio. Is it installed?") from e

    convert = getattr(cascadio, "convert", None)
    if callable(convert):
        fallbacks: list[tuple[tuple[Any, ...], dict[str, Any]]] = [
            ((str(step_path), str(glb_path)), {}),
            ((step_path, glb_path), {}),
            ((str(step_path),), {"output_path": str(glb_path)}),
            ((str(step_path),), {"output_file": str(glb_path)}),
            ((), {"input_path": str(step_path), "output_path": str(glb_path)}),
            ((), {"input_file": str(step_path), "output_file": str(glb_path)}),
        ]
        _call_with_fallbacks(convert, fallbacks)
        if not glb_path.exists():
            raise RuntimeError("cascadio.convert did not produce an output .glb file")
        return

    # Some versions may expose a submodule/function like cascadio.converter.convert
    for attr_chain in ("converter.convert", "conversion.convert", "io.convert"):
        obj: Any = cascadio
        ok = True
        for part in attr_chain.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                ok = False
                break
        if ok and callable(obj):
            fallbacks = [
                ((str(step_path), str(glb_path)), {}),
                ((step_path, glb_path), {}),
                ((), {"input_path": str(step_path), "output_path": str(glb_path)}),
                ((), {"input_file": str(step_path), "output_file": str(glb_path)}),
            ]
            _call_with_fallbacks(obj, fallbacks)
            if not glb_path.exists():
                raise RuntimeError(f"{attr_chain} did not produce an output .glb file")
            return

    raise RuntimeError(
        "Unsupported cascadio API: expected a callable convert function (e.g. cascadio.convert). "
        "Please check your cascadio version's documentation."
    )


def load_glb_as_mesh(glb_path: Path):
    try:
        import trimesh  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Failed to import trimesh. Is it installed?") from e

    loaded = trimesh.load(str(glb_path), force=None)

    if isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    elif isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise RuntimeError("Loaded GLB scene has no geometry")
        geometries = list(loaded.geometry.values())
        meshes = [g for g in geometries if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise RuntimeError("Loaded GLB scene contains no mesh geometry")
        mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
    else:
        raise RuntimeError(f"Unsupported trimesh load type: {type(loaded)!r}")

    return mesh


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

        convert_step_to_glb(tmp_step_path, glb_path)

        mesh = load_glb_as_mesh(glb_path)

        # Assumes the converted geometry is in millimeters.
        extents = [float(x) for x in mesh.extents]
        volume = float(mesh.volume)

        return {
            "glb_url": f"/static/{glb_name}",
            "volume_cubic_mm": volume,
            "bounding_box_mm": {"x": extents[0], "y": extents[1], "z": extents[2]},
        }
    except HTTPException:
        # Re-raise known HTTP errors unchanged.
        raise
    except Exception:
        # Clean up partially-written outputs, but let the global exception handler
        # print the traceback and return the exact error message.
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
