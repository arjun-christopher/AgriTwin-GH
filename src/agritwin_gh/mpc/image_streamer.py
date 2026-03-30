"""
Image streamer — retrieves random image metadata at each control step.

Returns only metadata (keys / bucket names); actual image bytes are fetched
by the dashboard layer via MinIO presigned URLs.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from sqlalchemy.orm import Session

from .constants import IMAGE_SUBCATEGORY_MAP
from .mpc_input_preparation import MPCInputPreparation
from .state import ImagePayload

logger = logging.getLogger(__name__)

# Default cache TTL — images are refreshed at most every 5 minutes.
_DEFAULT_CACHE_TTL_SEC = 300.0


class ImageStreamer:
    """Periodic random-image retrieval from the ``image_metadata`` table.

    Parameters
    ----------
    session:
        A live SQLAlchemy session.
    """

    def __init__(self, session: Session, cache_ttl_sec: float = _DEFAULT_CACHE_TTL_SEC) -> None:
        self._prep = MPCInputPreparation(session)
        self._cache_ttl = cache_ttl_sec
        self._cache: dict[str, tuple[float, ImagePayload]] = {}  # key → (ts, payload)

    # ── Public interface ───────────────────────────────────────────────

    def get_random_disease_image(self, disease_label: str) -> ImagePayload | None:
        """Fetch one random disease image matching *disease_label* (canonical).

        Returns a cached result if the last fetch for this label was less than
        *cache_ttl_sec* seconds ago.  Returns ``None`` if no matching image is
        found or the table is not yet wired.
        """
        cache_key = f"disease:{disease_label}"
        cached = self._cache.get(cache_key)
        if cached is not None and (time.monotonic() - cached[0]) < self._cache_ttl:
            return cached[1]

        sub = IMAGE_SUBCATEGORY_MAP.get("disease", {}).get(disease_label)
        if sub is None:
            logger.warning("No image subcategory mapping for disease '%s'", disease_label)
            return None

        row = self._prep.get_random_image_for_category("disease", sub)
        if row is None:
            return None

        payload = ImagePayload(
            image_key=row.get("image_key", ""),
            bucket_name=row.get("bucket_name", ""),
            file_name=row.get("file_name", ""),
            label=disease_label,
            category="disease",
        )
        self._cache[cache_key] = (time.monotonic(), payload)
        return payload

    def get_random_growth_stage_image(self, stage_label: str) -> ImagePayload | None:
        """Fetch one random growth-stage image matching *stage_label* (canonical).

        Returns a cached result if the last fetch for this label was less than
        *cache_ttl_sec* seconds ago.  Returns ``None`` if no matching image is
        found or the table is not yet wired.
        """
        cache_key = f"growth_stage:{stage_label}"
        cached = self._cache.get(cache_key)
        if cached is not None and (time.monotonic() - cached[0]) < self._cache_ttl:
            return cached[1]
        sub = IMAGE_SUBCATEGORY_MAP.get("growth_stage", {}).get(stage_label)
        if sub is None:
            logger.warning("No image subcategory mapping for stage '%s'", stage_label)
            return None

        row = self._prep.get_random_image_for_category("growth_stage", sub)
        if row is None:
            return None

        payload = ImagePayload(
            image_key=row.get("image_key", ""),
            bucket_name=row.get("bucket_name", ""),
            file_name=row.get("file_name", ""),
            label=stage_label,
            category="growth_stage",
        )
        self._cache[cache_key] = (time.monotonic(), payload)
        return payload
