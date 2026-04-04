"""
Media routes — ``/api/media/*``.

Endpoints
---------
GET /api/media/latest         → ``LatestMediaResponse``
GET /api/media/stage-images   → ``StageImagesResponse``
GET /api/media/disease-scans  → ``DiseaseImagesResponse``

Responsibility
--------------
All three endpoints delegate to ``MediaService``, which:

1. Queries the ``image_metadata`` table via
   ``agritwin_gh.mpc.image_streamer.ImageStreamer`` to obtain MinIO object
   keys and bucket names.
2. Generates pre-signed URLs via the MinIO client configured in
   ``config/minio_config.py``.
3. Returns ``ImageEntry`` objects with the ``src`` field set to the
   pre-signed URL (valid for a configurable TTL, default 1 hour).

When the DB or MinIO is unavailable, ``MediaService`` returns ``ImageEntry``
objects with empty ``src`` strings — the frontend gracefully shows a
placeholder image.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from agritwin_gh.api.dependencies import get_media_service
from agritwin_gh.schemas.media_schemas import (
    DiseaseImagesResponse,
    LatestMediaResponse,
    StageImagesResponse,
)
from agritwin_gh.services.media_service import MediaService

logger = logging.getLogger("agritwin.api.media")

router = APIRouter(tags=["Media"])


@router.get("/media/latest", response_model=LatestMediaResponse, status_code=200)
async def get_latest_media(
    svc: MediaService = Depends(get_media_service),
) -> LatestMediaResponse:
    """Return the latest stage camera and leaf scan images.

    Pre-signed MinIO URLs are valid for 1 hour.  Returns ``ImageEntry``
    objects with empty ``src`` strings when MinIO is unavailable — the
    frontend renders its local placeholder image in that case.
    """
    try:
        return svc.get_latest()
    except Exception as exc:
        logger.error("GET /media/latest failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Media service temporarily unavailable.")


@router.get("/media/stage-images", response_model=StageImagesResponse, status_code=200)
async def get_stage_images(
    svc: MediaService = Depends(get_media_service),
) -> StageImagesResponse:
    """Return one representative image per growth stage.

    Gallery contains up to 5 entries.  Pre-signed URLs valid for 1 hour.
    """
    try:
        return svc.get_stage_images()
    except Exception as exc:
        logger.error("GET /media/stage-images failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Stage images temporarily unavailable.")


@router.get("/media/disease-scans", response_model=DiseaseImagesResponse, status_code=200)
async def get_disease_images(
    svc: MediaService = Depends(get_media_service),
) -> DiseaseImagesResponse:
    """Return one representative scan image per disease category.

    Gallery contains up to 5 entries.  Pre-signed URLs valid for 1 hour.
    """
    try:
        return svc.get_disease_images()
    except Exception as exc:
        logger.error("GET /media/disease-scans failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Disease scan images temporarily unavailable.")
