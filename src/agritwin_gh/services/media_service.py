"""
MediaService — retrieves crop and disease images from MinIO via ImageStreamer.

Responsibility
--------------
* Query the ``image_metadata`` PostgreSQL table via
  ``agritwin_gh.mpc.image_streamer.ImageStreamer`` to retrieve random
  image keys and bucket names for the current growth stage and dominant
  disease label.
* Generate MinIO pre-signed URLs (default TTL: 3600 s) via the MinIO Python
  SDK client configured in ``config/minio_config.py``.
* Return ``LatestMediaResponse``, ``StageImagesResponse``, and
  ``DiseaseImagesResponse`` for the three ``/api/media/*`` endpoints.

MinIO client
------------
The MinIO client is created lazily from ``MinIOConfig`` on the first request.
If MinIO is not reachable, each ``ImageEntry`` is returned with an empty
``src`` field and the frontend shows a placeholder image.

Modules reused
--------------
* ``agritwin_gh.mpc.image_streamer.ImageStreamer``   — image_metadata DB queries.
* ``agritwin_gh.mpc.constants.GROWTH_STAGES``       — canonical stage labels.
* ``agritwin_gh.mpc.constants.DISEASE_CATEGORIES``  — canonical disease labels.
* ``agritwin_gh.config.minio_config``               — MinIO connection params.

Graceful degradation
--------------------
All MinIO and DB errors are caught and logged.  Empty ``src`` strings signal
the frontend to render its local placeholder image.  All gallery lists may be
shorter than the nominal max (5) when the DB has fewer records.

Thread safety
-------------
``_get_minio_client()`` and ``_get_streamer()`` perform lazy init and are
guarded by ``threading.Lock``.  After the lock the client reference is stable
for the lifetime of the service instance.
"""

from __future__ import annotations

import datetime as _dt
import logging
import threading
from typing import Any

from sqlalchemy.orm import Session

from agritwin_gh.core.runtime_store import MediaSnapshot, RuntimeStore, get_store
from agritwin_gh.mpc.constants import DISEASE_CATEGORIES, GROWTH_STAGES
from agritwin_gh.schemas.enums import GROWTH_STAGE_DISPLAY_NAME
from agritwin_gh.schemas.media_schemas import (
    DiseaseImagesResponse,
    ImageEntry,
    LatestMediaResponse,
    LeafScanEntry,
    StageImageEntry,
    StageImagesResponse,
)

logger = logging.getLogger("agritwin.services.media")

# Presigned URL expiry (seconds)
_PRESIGN_TTL_SEC = 3_600

# Number of frames returned per gallery endpoint
_GALLERY_SIZE = 5


def _human_ago(ts: _dt.datetime | None) -> str:
    """Return a human-readable "X ago" string for *ts* relative to now.

    Returns ``"just now"`` if *ts* is ``None`` or in the future.
    """
    if ts is None:
        return "just now"
    delta = _dt.datetime.now() - ts
    seconds = max(0, int(delta.total_seconds()))
    if seconds < 60:
        return f"{seconds} sec ago"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} min ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours} hr ago"
    return f"{hours // 24} day ago"


class MediaService:
    """Retrieves and serves signed image URLs for the media API endpoints.

    Parameters
    ----------
    store:
        Shared ``RuntimeStore`` singleton; ``None`` → ``get_store()`` at init.
    session:
        SQLAlchemy session passed to ``ImageStreamer``.  When ``None``, all DB
        calls are skipped and galleries are empty (presign-only fallback).
    minio_client:
        Pre-built ``minio.Minio`` client for testing.  ``None`` → lazy init
        from ``MinIOConfig`` env vars on first call to ``_get_minio_client()``.
    """

    def __init__(
        self,
        store: RuntimeStore | None = None,
        session: Session | None = None,
        minio_client: Any | None = None,
    ) -> None:
        self._store: RuntimeStore = store or get_store()
        self._session: Session | None = session
        self._minio: Any = minio_client
        self._streamer: Any = None
        self._init_lock = threading.Lock()

    # ─────────────────────────────────────────────────────────────────────────
    # Internal lazy initialisers
    # ─────────────────────────────────────────────────────────────────────────

    def _get_minio_client(self) -> Any | None:
        """Return (or lazily init) the MinIO client.

        Returns ``None`` (and logs a warning) if the MinIO SDK is not
        installed or credentials are missing.
        """
        with self._init_lock:
            if self._minio is not None:
                return self._minio
            try:
                import minio  # type: ignore[import-untyped]
                from agritwin_gh.config.minio_config import MinIOConfig
                cfg = MinIOConfig()
                if not cfg.endpoint:
                    logger.warning(
                        "MinIO endpoint not configured — presigned URLs disabled."
                    )
                    return None
                self._minio = minio.Minio(
                    cfg.endpoint,
                    access_key=cfg.access_key,
                    secret_key=cfg.secret_key,
                    secure=cfg.endpoint.startswith("https"),
                )
            except Exception as exc:
                logger.warning("MinIO client init failed: %s", exc)
                return None
        return self._minio

    def _get_streamer(self) -> Any | None:
        """Return (or lazily create) the ``ImageStreamer`` if a session is set."""
        with self._init_lock:
            if self._streamer is None and self._session is not None:
                try:
                    from agritwin_gh.mpc.image_streamer import ImageStreamer
                    self._streamer = ImageStreamer(self._session)
                except Exception as exc:
                    logger.warning("ImageStreamer init failed: %s", exc)
        return self._streamer

    # ─────────────────────────────────────────────────────────────────────────
    # Presign helper
    # ─────────────────────────────────────────────────────────────────────────

    def _presign(self, image_key: str, bucket: str) -> str:
        """Generate a MinIO pre-signed GET URL.

        Returns ``""`` on any error so callers can always check truthiness.
        """
        if not image_key or not bucket:
            return ""
        client = self._get_minio_client()
        if client is None:
            return ""
        try:
            import datetime
            url = client.presigned_get_object(
                bucket,
                image_key,
                expires=datetime.timedelta(seconds=_PRESIGN_TTL_SEC),
            )
            return str(url)
        except Exception as exc:
            logger.warning(
                "presign failed for key=%s bucket=%s: %s", image_key, bucket, exc
            )
            return ""

    # ─────────────────────────────────────────────────────────────────────────
    # GET /api/media/latest
    # ─────────────────────────────────────────────────────────────────────────

    def get_latest(self) -> LatestMediaResponse:
        """Return one stage image and one leaf scan image for the home dashboard.

        Reads the current ``growth_stage`` and ``disease.dominant_label`` from
        the store, fetches one random image per category from ``ImageStreamer``,
        presigns the URLs, updates ``MediaSnapshot`` in the store, and returns
        a ``LatestMediaResponse``.

        Falls back to an empty ``ImageEntry`` on any error.
        """
        state = self._store.get_latest_state()
        now = _dt.datetime.now()
        streamer = self._get_streamer()

        # ── Stage image ─────────────────────────────────────────────────────
        stage = state.growth.current_stage or "seedling"
        stage_display = GROWTH_STAGE_DISPLAY_NAME.get(stage, stage.title())
        stage_entry = ImageEntry(
            src="",
            alt=f"Tomato plant — {stage_display} stage",
            badge=f"Day {max(1, int(state.growth.hours_to_next_stage // 24))}",
            location="GH-04 · Camera 2",
            captured=_human_ago(now),
        )
        if streamer is not None:
            try:
                payload = streamer.get_random_growth_stage_image(stage)
                if payload:
                    src = self._presign(payload.image_key, payload.bucket_name)
                    stage_entry = ImageEntry(
                        src=src,
                        alt=f"Tomato plant — {stage_display} stage",
                        badge=f"Day {max(1, int(state.growth.hours_to_next_stage // 24))}",
                        location="GH-04 · Camera 2",
                        captured=_human_ago(now),
                        image_key=payload.image_key,
                        bucket_name=payload.bucket_name,
                    )
            except Exception as exc:
                logger.warning("get_latest stage image failed: %s", exc)

        # ── Leaf scan image ─────────────────────────────────────────────────
        disease_label = state.disease.dominant_label or "healthy leaves"
        leaf_entry = ImageEntry(
            src="",
            alt=f"Leaf scan — {disease_label}",
            badge="",
            location="Sector B · Leaf #7",
            captured=_human_ago(now),
        )
        if streamer is not None:
            try:
                payload = streamer.get_random_disease_image(disease_label)
                if payload:
                    src = self._presign(payload.image_key, payload.bucket_name)
                    confidence = 0.0
                    if state.disease.pathogens:
                        top = max(
                            state.disease.pathogens,
                            key=lambda p: p.get("probability", 0.0),
                            default={},
                        )
                        confidence = float(top.get("probability", 0.0)) * 100.0
                    leaf_entry = ImageEntry(
                        src=src,
                        alt=f"Leaf scan — {disease_label}",
                        badge=f"{confidence:.1f}% Conf." if confidence else "",
                        location="Sector B · Leaf #7",
                        captured=_human_ago(now),
                        image_key=payload.image_key,
                        bucket_name=payload.bucket_name,
                    )
            except Exception as exc:
                logger.warning("get_latest leaf image failed: %s", exc)

        # ── Persist to store so WebSocket layer can read without re-fetching ─
        self._store.update_latest_state(
            media=MediaSnapshot(
                latest_stage_src=stage_entry.src,
                latest_stage_alt=stage_entry.alt,
                latest_leaf_src=leaf_entry.src,
                latest_leaf_alt=leaf_entry.alt,
                stage_images=self._store.get_latest_state().media.stage_images,
                disease_scans=self._store.get_latest_state().media.disease_scans,
            )
        )

        return LatestMediaResponse(stage=stage_entry, leaf=leaf_entry)

    # ─────────────────────────────────────────────────────────────────────────
    # GET /api/media/stage-images
    # ─────────────────────────────────────────────────────────────────────────

    def get_stage_images(self) -> StageImagesResponse:
        """Return up to ``_GALLERY_SIZE`` stage-camera frames, one per growth stage.

        Iterates over ``GROWTH_STAGES`` (most-recent-looking first: ripe →
        seedling), fetches one random image per stage from ``ImageStreamer``,
        and presigns each URL.

        Returns an ``StageImagesResponse`` with an empty list when the DB or
        MinIO is unavailable.
        """
        streamer = self._get_streamer()
        entries: list[StageImageEntry] = []
        now = _dt.datetime.now()

        # Walk stages newest-to-oldest so the most advanced stage appears first.
        for stage in reversed(GROWTH_STAGES):
            if len(entries) >= _GALLERY_SIZE:
                break
            stage_display = GROWTH_STAGE_DISPLAY_NAME.get(stage, stage.title())
            entry = StageImageEntry(
                src="",
                alt=f"Greenhouse stage camera — {stage_display}",
                captured_ago=_human_ago(now),
                stage=stage_display,
                confidence=0.0,
            )
            if streamer is not None:
                try:
                    payload = streamer.get_random_growth_stage_image(stage)
                    if payload:
                        src = self._presign(payload.image_key, payload.bucket_name)
                        entry = StageImageEntry(
                            src=src,
                            alt=f"Greenhouse stage camera — {stage_display}",
                            captured_ago=_human_ago(now),
                            stage=stage_display,
                            confidence=0.0,
                            image_key=payload.image_key,
                            bucket_name=payload.bucket_name,
                        )
                except Exception as exc:
                    logger.warning(
                        "get_stage_images fetch failed for stage '%s': %s", stage, exc
                    )
            entries.append(entry)

        # Update store gallery for 3D layer / WebSocket caching
        self._store.update_latest_state(
            media=MediaSnapshot(
                latest_stage_src=self._store.get_latest_state().media.latest_stage_src,
                latest_stage_alt=self._store.get_latest_state().media.latest_stage_alt,
                latest_leaf_src=self._store.get_latest_state().media.latest_leaf_src,
                latest_leaf_alt=self._store.get_latest_state().media.latest_leaf_alt,
                stage_images=[e.model_dump() for e in entries],
                disease_scans=self._store.get_latest_state().media.disease_scans,
            )
        )
        return StageImagesResponse(images=entries)

    # ─────────────────────────────────────────────────────────────────────────
    # GET /api/media/disease-scans
    # ─────────────────────────────────────────────────────────────────────────

    def get_disease_images(self) -> DiseaseImagesResponse:
        """Return up to ``_GALLERY_SIZE`` leaf-scan frames, one per disease category.

        Iterates over ``DISEASE_CATEGORIES``, fetches one random image per
        category from ``ImageStreamer``, and presigns each URL.

        Returns a ``DiseaseImagesResponse`` with an empty list when the DB or
        MinIO is unavailable.
        """
        streamer = self._get_streamer()
        entries: list[LeafScanEntry] = []
        now = _dt.datetime.now()

        state = self._store.get_latest_state()
        # Build quick lookup: disease label → risk score from current state
        risk_lookup: dict[str, float] = {}
        for p in state.disease.pathogens:
            label = p.get("label", "")
            risk_lookup[label] = float(p.get("risk_score", 0.0)) * 100.0

        for disease in DISEASE_CATEGORIES:
            if len(entries) >= _GALLERY_SIZE:
                break
            risk_pct = risk_lookup.get(disease, 0.0)
            entry = LeafScanEntry(
                src="",
                alt=f"Leaf scan — {disease}",
                captured_ago=_human_ago(now),
                classification=disease,
                risk=round(risk_pct, 1),
                confidence=0.0,
            )
            if streamer is not None:
                try:
                    payload = streamer.get_random_disease_image(disease)
                    if payload:
                        src = self._presign(payload.image_key, payload.bucket_name)
                        entry = LeafScanEntry(
                            src=src,
                            alt=f"Leaf scan — {disease}",
                            captured_ago=_human_ago(now),
                            classification=disease,
                            risk=round(risk_pct, 1),
                            confidence=0.0,
                            image_key=payload.image_key,
                            bucket_name=payload.bucket_name,
                        )
                except Exception as exc:
                    logger.warning(
                        "get_disease_images fetch failed for '%s': %s", disease, exc
                    )
            entries.append(entry)

        # Update store gallery
        self._store.update_latest_state(
            media=MediaSnapshot(
                latest_stage_src=self._store.get_latest_state().media.latest_stage_src,
                latest_stage_alt=self._store.get_latest_state().media.latest_stage_alt,
                latest_leaf_src=self._store.get_latest_state().media.latest_leaf_src,
                latest_leaf_alt=self._store.get_latest_state().media.latest_leaf_alt,
                stage_images=self._store.get_latest_state().media.stage_images,
                disease_scans=[e.model_dump() for e in entries],
            )
        )
        return DiseaseImagesResponse(images=entries)
