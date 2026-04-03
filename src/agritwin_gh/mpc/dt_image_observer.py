"""
Stage-aware image observation — decoupled from DT state physics.

This module provides an ``ImageObserver`` protocol and two implementations:

* ``SyntheticImageObserver`` — generates placeholder image metadata from
  the canonical ``IMAGE_SUBCATEGORY_MAP`` without touching MinIO or a
  database.  Used for quick offline evaluation runs.

* ``MinIOImageObserver`` — wraps ``ImageStreamer`` (DB-backed, TTL-cached)
  to pull real crop images from the MinIO ``image_metadata`` table.
  Pass to ``DTLoop(image_observer=MinIOImageObserver(session))`` — zero
  changes to the loop itself.

The observer is designed to be called at image-refresh cadence (every
30–60 minutes) from the ``DTLoop``.  It takes the current growth stage
and disease risk as inputs and returns an ``ImageObservation`` value
object that the loop can propagate into its step result.

Design principle
----------------
Image logic is fully decoupled from the DT engine (physics).  The
observer does *not* modify greenhouse state — it only *observes* and
produces metadata that the output layer persists.
"""

from __future__ import annotations

import datetime as _dt
import logging
from typing import Protocol, runtime_checkable

from sqlalchemy.orm import Session

from .constants import (
    DISEASE_IMAGE_SUBCATEGORY,
    GROWTH_STAGE_IMAGE_SUBCATEGORY,
)
from .dt_input_provider import ImageObservation
from .image_streamer import ImageStreamer
from .state import GreenhouseState

logger = logging.getLogger(__name__)


# ── Protocol ──────────────────────────────────────────────────────────────────


@runtime_checkable
class ImageObserver(Protocol):
    """Contract for stage-aware image observation at refresh cadence."""

    def observe(
        self,
        growth_stage: str,
        state: GreenhouseState,
        step_index: int,
        timestamp: _dt.datetime,
    ) -> ImageObservation:
        """Produce an image observation for the current step.

        Parameters
        ----------
        growth_stage:
            Canonical growth-stage label used to select the image class.
        state:
            Current greenhouse state — allows disease-risk-aware selection.
        step_index:
            Loop step number (for logging).
        timestamp:
            Logical simulation time.
        """
        ...


# ── Synthetic implementation (no DB / MinIO required) ─────────────────────────


class SyntheticImageObserver:
    """Generates placeholder image observations from subcategory maps.

    For each refresh, builds synthetic image keys following the naming
    convention ``<subcategory>_<step>.jpg``.  The disease label is
    inferred from the current risk score:

    * risk < 0.3 → ``"healthy leaves"``
    * risk < 0.5 → ``"early blight"``
    * risk ≥ 0.5 → ``"late blight"``
    """

    def observe(
        self,
        growth_stage: str,
        state: GreenhouseState,
        step_index: int,
        timestamp: _dt.datetime,
    ) -> ImageObservation:
        # Growth-stage image key.
        gs_sub = GROWTH_STAGE_IMAGE_SUBCATEGORY.get(growth_stage, growth_stage)
        gs_key = f"{gs_sub}/synthetic_{step_index:04d}.jpg"

        # Disease label from risk score.
        risk = state.disease_risk_score
        if risk < 0.3:
            disease_label = "healthy leaves"
        elif risk < 0.5:
            disease_label = "early blight"
        else:
            disease_label = "late blight"

        dis_sub = DISEASE_IMAGE_SUBCATEGORY.get(disease_label, disease_label)
        dis_key = f"{dis_sub}/synthetic_{step_index:04d}.jpg"

        obs = ImageObservation(
            growth_stage_image_key=gs_key,
            growth_stage_label=growth_stage,
            disease_image_key=dis_key,
            disease_label=disease_label,
            timestamp=timestamp,
            source="synthetic",
        )
        logger.debug(
            "Step %d image observation: stage=%s disease=%s",
            step_index,
            growth_stage,
            disease_label,
        )
        return obs


# ── MinIO-backed implementation (real images, TTL-cached) ─────────────────────


class MinIOImageObserver:
    """Real crop images from MinIO via ``ImageStreamer`` (TTL-cached).

    Pulls random growth-stage and disease images from the
    ``image_metadata`` table.  ``ImageStreamer`` already applies a TTL
    cache, so repeated calls within the cache window are free.

    Falls back to a synthetic observation when no matching image is
    found in the database (e.g. missing category, empty table).

    Parameters
    ----------
    session:
        A live SQLAlchemy ``Session``.
    cache_ttl_sec:
        TTL for ``ImageStreamer``'s internal per-category cache (seconds).
        Defaults to 300 s (5 minutes).
    """

    def __init__(
        self,
        session: Session,
        cache_ttl_sec: float = 300.0,
    ) -> None:
        self._streamer = ImageStreamer(session, cache_ttl_sec=cache_ttl_sec)
        self._fallback = SyntheticImageObserver()

    def observe(
        self,
        growth_stage: str,
        state: GreenhouseState,
        step_index: int,
        timestamp: _dt.datetime,
    ) -> ImageObservation:
        # ── Disease label from current risk score ─────────────────
        risk = state.disease_risk_score
        if risk < 0.3:
            disease_label = "healthy leaves"
        elif risk < 0.5:
            disease_label = "early blight"
        else:
            disease_label = "late blight"

        # ── Fetch from MinIO (TTL-cached) ─────────────────────────
        gs_payload = self._streamer.get_random_growth_stage_image(growth_stage)
        dis_payload = self._streamer.get_random_disease_image(disease_label)

        if gs_payload is None or dis_payload is None:
            logger.info(
                "Step %d: MinIO image miss (gs=%s, dis=%s) — falling back to synthetic.",
                step_index,
                gs_payload is not None,
                dis_payload is not None,
            )
            return self._fallback.observe(growth_stage, state, step_index, timestamp)

        obs = ImageObservation(
            growth_stage_image_key=gs_payload.image_key,
            growth_stage_label=growth_stage,
            disease_image_key=dis_payload.image_key,
            disease_label=disease_label,
            timestamp=timestamp,
            source="minio",
        )
        logger.debug(
            "Step %d MinIO observation: stage=%s disease=%s",
            step_index,
            growth_stage,
            disease_label,
        )
        return obs
