from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.core.config import get_settings
from app.services.inference import InferenceService, ModelNotReadyError


def create_app(inference_service: InferenceService | None = None) -> FastAPI:
    settings = get_settings()
    service = inference_service or InferenceService()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.inference_service = service

        if not service.is_loaded:
            try:
                service.load()
            except ModelNotReadyError:
                # Allow startup so the API can still expose a helpful 503 response.
                pass

        yield

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Production-style FastAPI service for PyTorch text classification.",
        lifespan=lifespan,
    )

    app.state.inference_service = service
    app.include_router(router)
    return app


app = create_app()
