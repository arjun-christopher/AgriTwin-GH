"""
Media routes — ``/api/media/*``.

Endpoints
---------
GET /api/media/latest              → ``LatestMediaResponse``
GET /api/media/stage-images        → ``StageImagesResponse``
GET /api/media/disease-scans       → ``DiseaseImagesResponse``
GET /api/media/local/{file_path}   → FileResponse (local CNN image)

Responsibility
--------------
All three collection endpoints delegate to ``MediaService``, which:

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

``/api/media/local/{file_path}`` serves local image files from the
``data/external/`` directory.  Only files within that directory are served
(directory-traversal attempts return 403).
"""

from __future__ import annotations

import logging
import pathlib

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from agritwin_gh.api.dependencies import get_media_service
from agritwin_gh.schemas.media_schemas import (
    DiseaseImagesResponse,
    LatestMediaResponse,
    StageImagesResponse,
)
from agritwin_gh.services.media_service import MediaService

logger = logging.getLogger("agritwin.api.media")

# Absolute path to the local image directory relative to the repo root.
# The server must be launched from the repo root (where data/external/ lives).
_LOCAL_MEDIA_BASE = (pathlib.Path(__file__).resolve().parents[4] / "data" / "external").resolve()

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


@router.get("/media/local/{file_path:path}", status_code=200)
async def get_local_media_file(file_path: str) -> FileResponse:
    """Serve a local image file from the ``data/external/`` directory.

    Called by the frontend when MinIO is unavailable and the DT loop has
    stored a local CNN image path in ``MediaSnapshot``.  Only files under
    ``data/external/`` are served; any attempt to escape that directory
    (e.g. via ``../``) returns 403.
    """
    try:
        requested = (_LOCAL_MEDIA_BASE / file_path).resolve()
        requested.relative_to(_LOCAL_MEDIA_BASE)  # raises ValueError on traversal
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied.")
    except Exception as exc:
        logger.warning("local media path error for '%s': %s", file_path, exc)
        raise HTTPException(status_code=400, detail="Invalid path.")

    if not requested.is_file():
        raise HTTPException(status_code=404, detail="Image not found.")

    return FileResponse(str(requested))
