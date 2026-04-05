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
import pathlib as _pl
import random
from typing import Protocol, runtime_checkable

from sqlalchemy.orm import Session

from .dt_input_provider import ImageObservation
from .image_streamer import ImageStreamer
from .state import GreenhouseState

logger = logging.getLogger(__name__)


# ── Repo-root navigation & image folder constants ─────────────────────────────

_REPO_ROOT = _pl.Path(__file__).resolve().parents[3]  # …/src/agritwin_gh/mpc → repo root

_GROWTH_STAGE_FOLDER: dict[str, str] = {
    "seedling":               "Stage1_Seedling",
    "early vegetative":       "Stage2_Early_Vegetative",
    "flowering initiation":   "Stage3_Flowering_Initiation",
    "flowering":              "Stage4_Flowering",
    "unripe":                 "Stage5_Unripe",
    "ripe":                   "Stage6_Ripe",
}

_DISEASE_FOLDER: dict[str, str] = {
    "early_blight":   "Tomato_Early_Blight",
    "late_blight":    "Tomato_Late_Blight",
    "leaf_mold":      "Tomato_Leaf_Mold",
    "powdery_mildew": "Tomato_Powdery_Mildew",
    "spider_mites":   "Tomato_Spider_Mites",
}

_DISEASE_MODEL_CACHE: dict = {}


def _pick_random_image(folder: _pl.Path) -> "_pl.Path | None":
    """Return a random image file from *folder*, or None if none found."""
    if not folder.is_dir():
        return None
    imgs = [f for f in folder.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    return random.choice(imgs) if imgs else None


def _load_disease_model() -> tuple:
    """Load the disease CNN once and cache it at the module level."""
    if _DISEASE_MODEL_CACHE:
        return _DISEASE_MODEL_CACHE["model"], _DISEASE_MODEL_CACHE["label_map"]
    models_dir = _pl.Path(__file__).resolve().parents[1] / "models"
    candidates = sorted(models_dir.glob("disease_*_best.keras"))
    if not candidates:
        candidates = sorted(models_dir.glob("disease_*.keras"))
    model_path = candidates[-1]
    run_id = model_path.stem          # e.g. "disease_20260226_141843_best"
    artifact_id = run_id.removesuffix("_best")   # "disease_20260226_141843"
    label_map_path = models_dir / "artifacts" / artifact_id / "label_map.json"
    from agritwin_gh.models.disease_inference import load_inference_assets  # noqa: PLC0415
    model, lmap = load_inference_assets(str(model_path), str(label_map_path))
    _DISEASE_MODEL_CACHE["model"] = model
    _DISEASE_MODEL_CACHE["label_map"] = lmap
    return model, lmap


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
        *,
        model_growth_result: "dict | None" = None,
        model_disease_result: "dict | None" = None,
    ) -> ImageObservation:
        """Produce an image observation for the current step.

        Parameters
        ----------
        growth_stage:
            Canonical growth-stage label — fallback if model_growth_result
            is unavailable.
        state:
            Current greenhouse state.
        step_index:
            Loop step number (for logging).
        timestamp:
            Logical simulation time.
        model_growth_result:
            Output dict from the growth-progression LSTM.  Its
            ``current_stage`` key is used to select the image source folder.
        model_disease_result:
            Output dict from the disease-progression LSTM.  Used to choose
            the disease image source folder and drive CNN selection.
        """
        ...


# ── Synthetic implementation (no DB / MinIO required) ─────────────────────────


class SyntheticImageObserver:
    """Picks a real crop image from disk based on LSTM model outputs and
    runs the appropriate CNN classifier.

    Growth-stage image:
        A random image is selected from the folder matching the current
        growth stage reported by the growth-progression LSTM.  The
        growth-stage CNN then classifies that image.

    Disease image:
        Uses ``model_disease_result`` (from the disease-progression LSTM)
        to choose the source folder:

        * All diseases False  → ``Tomato Healthy Leaves/``
        * Exactly one True    → ``Tomato Diseases/<disease_folder>/``
        * Multiple True       → folder for disease with highest ``severity_24h``

        The disease CNN classifies the selected image.
    """

    def observe(
        self,
        growth_stage: str,
        state: GreenhouseState,
        step_index: int,
        timestamp: _dt.datetime,
        *,
        model_growth_result: dict | None = None,
        model_disease_result: dict | None = None,
    ) -> ImageObservation:
        # ── Growth-stage image ────────────────────────────────────────────────
        # Always use the DT-authoritative growth_stage (passed from DTLoop) to
        # select the image folder.  The LSTM current_stage may disagree; the
        # penalty correction in loop_service.py canonicalises it only after the
        # step result is yielded — too late to affect image selection here.
        gs_folder_name = _GROWTH_STAGE_FOLDER.get(growth_stage, "Stage1_Seedling")
        gs_folder = (
            _REPO_ROOT / "data" / "external" / "Tomato Growth Stages" / gs_folder_name
        )
        gs_image_path = _pick_random_image(gs_folder)

        # ── Disease image: pick source folder from LSTM result ────────────────
        present: dict[str, float] = {}  # disease_key → severity_24h
        if model_disease_result:
            for d, v in model_disease_result.items():
                if v.get("present", False):
                    present[d] = float(v.get("severity_24h", 0.0))

        if not present:
            dis_folder = _REPO_ROOT / "data" / "external" / "Tomato Healthy Leaves"
        else:
            chosen = max(present, key=present.__getitem__)  # highest severity when >1
            folder_name = _DISEASE_FOLDER.get(chosen, "Tomato_Early_Blight")
            dis_folder = (
                _REPO_ROOT / "data" / "external" / "Tomato Diseases" / folder_name
            )
        dis_image_path = _pick_random_image(dis_folder)

        # ── Growth-stage CNN + Disease CNN: run concurrently ──────────────────
        # Pre-load the disease model cache on the main thread before forking,
        # so both futures only do pure inference (no model-load race condition).
        from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415
        from agritwin_gh.models.growth_stage_inference import predict_growth_stage  # noqa: PLC0415
        from agritwin_gh.models.disease_inference import predict_image  # noqa: PLC0415

        dis_model, dis_lmap = _load_disease_model()  # cached after first call

        def _growth_cnn() -> dict:
            return predict_growth_stage(str(gs_image_path))

        def _disease_cnn() -> dict:
            return predict_image(str(dis_image_path), dis_model, dis_lmap)

        gs_label = growth_stage
        gs_confidence = 0.0
        dis_label = "tomato_leaf_healthy"
        dis_confidence = 0.0

        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="cnn") as _pool:
            _fgs = _pool.submit(_growth_cnn) if gs_image_path else None
            _fdi = _pool.submit(_disease_cnn) if dis_image_path else None

        if _fgs is not None:
            try:
                res = _fgs.result()
                gs_label = res["class_name"]
                gs_confidence = float(res["confidence"])
            except Exception as exc:  # noqa: BLE001
                logger.warning("Step %d: growth-stage CNN failed: %s", step_index, exc)

        if _fdi is not None:
            try:
                res = _fdi.result()
                dis_label = res["class_name"]
                dis_confidence = float(res["confidence"])
            except Exception as exc:  # noqa: BLE001
                logger.warning("Step %d: disease CNN failed: %s", step_index, exc)

        # ── Build relative keys for logging ───────────────────────────────────
        try:
            gs_key = (
                str(gs_image_path.relative_to(_REPO_ROOT)).replace("\\", "/")
                if gs_image_path
                else "—"
            )
        except ValueError:
            gs_key = str(gs_image_path) if gs_image_path else "—"
        try:
            dis_key = (
                str(dis_image_path.relative_to(_REPO_ROOT)).replace("\\", "/")
                if dis_image_path
                else "—"
            )
        except ValueError:
            dis_key = str(dis_image_path) if dis_image_path else "—"

        logger.debug(
            "Step %d image: gs=%s [%s]  dis=%s [%s]",
            step_index, gs_key, gs_label, dis_key, dis_label,
        )
        return ImageObservation(
            growth_stage_image_key=gs_key,
            growth_stage_label=gs_label,
            growth_stage_confidence=gs_confidence,
            disease_image_key=dis_key,
            disease_label=dis_label,
            disease_confidence=dis_confidence,
            timestamp=timestamp,
            source="synthetic",
        )


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
        *,
        model_growth_result: dict | None = None,
        model_disease_result: dict | None = None,
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
            return self._fallback.observe(
                growth_stage,
                state,
                step_index,
                timestamp,
                model_growth_result=model_growth_result,
                model_disease_result=model_disease_result,
            )

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
