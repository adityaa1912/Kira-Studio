import logging
import secrets
import zipfile
from pathlib import Path

import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.dataset import dataset_status
from app.generator import (
    MODEL_PATH,
    UPSCALED_SIZE,
    UPSCALE_MODEL_PATH,
    generate_faces,
    get_generator,
    get_upscaler,
)


logger = logging.getLogger(__name__)
app = FastAPI(
    title="Anime Face Generator API",
    version="1.1.0",
    description="Local anime portrait generation with reproducible seeded inference.",
)


class GenerateRequest(BaseModel):
    count: int = Field(default=4, ge=1, le=12)
    seed: int | None = Field(default=None, ge=0, le=2_147_483_647)
    variation: float = Field(default=1.0, ge=0.5, le=1.5)
    ai_upscale: bool = True


@app.get("/api/health")
def health():
    model_available = MODEL_PATH.is_file()
    if model_available:
        try:
            get_generator()
        except (
            FileNotFoundError,
            OSError,
            ValueError,
            TypeError,
            RuntimeError,
            zipfile.BadZipFile,
            tf.errors.OpError,
        ):
            logger.exception("Generator readiness check failed")
            model_available = False
    ai_upscale_available = UPSCALE_MODEL_PATH.is_file()
    if ai_upscale_available:
        try:
            get_upscaler()
        except (FileNotFoundError, ImportError, OSError, RuntimeError, ValueError):
            logger.exception("AI upscaler readiness check failed")
            ai_upscale_available = False
    return {
        "status": "ok",
        "ready": model_available,
        "model_available": model_available,
        "model_name": MODEL_PATH.name,
        "output_size": 256,
        "ai_upscale_available": ai_upscale_available,
        "ai_upscale_model": UPSCALE_MODEL_PATH.name,
        "ai_upscale_output_size": UPSCALED_SIZE,
    }


@app.get("/api/dataset")
def dataset():
    return dataset_status()


@app.post("/api/generate")
def generate(request: GenerateRequest):
    seed = request.seed if request.seed is not None else secrets.randbelow(2_147_483_648)
    try:
        images = generate_faces(
            request.count, seed, request.variation, request.ai_upscale
        )
    except (
        FileNotFoundError,
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        zipfile.BadZipFile,
        tf.errors.OpError,
    ) as error:
        logger.exception("Face generation failed")
        raise HTTPException(
            status_code=503,
            detail="Generation is unavailable. Check that the model is valid and review the backend logs.",
        ) from error
    return {
        "images": images,
        "seed": seed,
        "count": len(images),
        "variation": request.variation,
        "ai_upscaled": request.ai_upscale,
        "output_size": UPSCALED_SIZE if request.ai_upscale else 256,
    }


FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")