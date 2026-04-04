"""
ControlService — apply manual overrides to the DT loop.

Responsibility
--------------
``ControlService`` is the single authority for writing to the
``RuntimeStore`` override slot.  All operator-initiated mutations (DT
parameter tweaks, simulation resets, preset application, individual
actuator-level changes) flow through this service.

The service **does not** interact with the MPC solver or any database.
It only:
1. Validates the incoming request (unknown IDs, out-of-range values).
2. Constructs or updates an ``OverrideConfig``.
3. Writes the config to the ``RuntimeStore`` via ``set_override()`` or
   clears it via ``clear_override()``.

The ``LoopService`` background loop reads the active ``OverrideConfig``
after each step and applies actuator overrides to the ``ActuatorSnapshot``.

Preset actuator definitions
---------------------------
Three built-in presets are supported:

``day-cycle``
    Balanced daytime growing profile — LED on, fan and CO₂ moderate.

``night-cycle``
    Low-light night profile — LED off, heater slightly raised, all else reduced.

``emergency-flush``
    Maximum ventilation flush — fan and vent at full, CO₂ scrubbed, LED off.

The levels (0–100 duty-cycle) are chosen to be visually meaningful in the
React ``ManualOverride`` panel; they are approximate and can be tuned by
the operator post-apply.

Thread safety
-------------
All writes use ``RuntimeStore._lock`` under the hood (via the public API).
This service is stateless and can be instantiated once per request or once
globally — both patterns are safe.
"""

from __future__ import annotations

import logging
from typing import Any

from agritwin_gh.core.runtime_store import OverrideConfig, RuntimeStore, get_store
from agritwin_gh.schemas.actuator_schemas import (
    ActuatorSetRequest,
    ActuatorSetResponse,
)
from agritwin_gh.schemas.dt_schemas import (
    DtOverrideResponse,
    DtParamOverrideRequest,
    DtPresetResponse,
    DtSimOverrideRequest,
)
from agritwin_gh.schemas.enums import ACTUATOR_IDS, DT_PRESET_DESCRIPTIONS, STAGE_DURATION_DAYS

logger = logging.getLogger("agritwin.services.control")

# ---------------------------------------------------------------------------
# Preset actuator profiles
# ---------------------------------------------------------------------------

_PRESET_ACTUATOR_OVERRIDES: dict[str, dict[str, float]] = {
    "day-cycle": {
        "fan":        60.0,
        "vent":       40.0,
        "led":        80.0,
        "heater":     20.0,
        "co2":        50.0,
        "fogger":     20.0,
        "irrigation": 30.0,
    },
    "night-cycle": {
        "fan":        30.0,
        "vent":       20.0,
        "led":         0.0,
        "heater":     40.0,
        "co2":        20.0,
        "fogger":     10.0,
        "irrigation":  0.0,
    },
    "emergency-flush": {
        "fan":        100.0,
        "vent":       100.0,
        "led":          0.0,
        "heater":       0.0,
        "co2":          0.0,
        "fogger":       0.0,
        "irrigation":  80.0,
    },
}


def _clamp_level(level: float) -> float:
    """Return ``level`` clamped to [0.0, 100.0]."""
    return max(0.0, min(100.0, level))


# ---------------------------------------------------------------------------
# ControlService
# ---------------------------------------------------------------------------


class ControlService:
    """Apply manual overrides issued by the React ``ManualOverride`` page.

    Parameters
    ----------
    store:
        Shared ``RuntimeStore`` singleton.  ``None`` → ``get_store()`` is
        called at construction time; this is the correct production mode.
    """

    def __init__(self, store: RuntimeStore | None = None) -> None:
        self._store: RuntimeStore = store or get_store()

    # ─────────────────────────────────────────────────────────────────────────
    # DT parameter override  (POST /api/dt/override  — single param variant)
    # ─────────────────────────────────────────────────────────────────────────

    def apply_dt_param_override(self, req: DtParamOverrideRequest) -> DtOverrideResponse:
        """Set a single DT parameter override.

        Merges the new ``{param: value}`` entry into any existing
        ``param_overrides`` so that concurrent overrides are not clobbered.

        Parameters
        ----------
        req:
            Validated ``DtParamOverrideRequest`` from the route handler.

        Returns
        -------
        DtOverrideResponse
            Echoes the applied param and value; ``ok=True`` on success.

        Raises
        ------
        ValueError
            If ``req.param`` is empty.
        """
        if not req.param:
            raise ValueError("Override param name must not be empty.")

        existing: OverrideConfig = self._store.get_override() or OverrideConfig()

        updated_params = dict(existing.param_overrides)
        updated_params[req.param] = req.value

        self._store.set_override(
            OverrideConfig(
                param_overrides=updated_params,
                actuator_overrides=dict(existing.actuator_overrides),
                sim_stage=existing.sim_stage,
                sim_day_in_stage=existing.sim_day_in_stage,
                sim_start_date=existing.sim_start_date,
                sim_start_hour=existing.sim_start_hour,
                preset_id=existing.preset_id,
            )
        )
        logger.info("DT param override applied: %s = %.3f", req.param, req.value)

        return DtOverrideResponse(
            ok=True,
            applied_param=req.param,
            applied_value=req.value,
            applied_stage="",
            message=f"Parameter '{req.param}' updated to {req.value}.",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # DT simulation override  (POST /api/dt/override — sim params variant)
    # ─────────────────────────────────────────────────────────────────────────

    def apply_dt_sim_override(self, req: DtSimOverrideRequest) -> DtOverrideResponse:
        """Override the simulation stage, day-in-stage, start date, and hour.

        Validates that ``req.stage`` is a known growth-stage label (from
        ``STAGE_DURATION_DAYS``).  Clamps ``req.day_in_stage`` to the maximum
        allowed by the stage (so the caller never exceeds the stage duration).

        Parameters
        ----------
        req:
            Validated ``DtSimOverrideRequest`` from the route handler.

        Returns
        -------
        DtOverrideResponse
            Echoes ``applied_stage``; ``ok=True`` on success.

        Raises
        ------
        ValueError
            If ``req.stage`` is not a known growth-stage label.
        """
        known_stages = set(STAGE_DURATION_DAYS.keys())
        if req.stage not in known_stages:
            raise ValueError(
                f"Unknown growth stage '{req.stage}'. "
                f"Valid stages: {sorted(known_stages)}"
            )

        max_days: float = STAGE_DURATION_DAYS.get(req.stage, 30.0)
        clamped_day = max(1, min(req.day_in_stage, int(max_days)))

        existing: OverrideConfig = self._store.get_override() or OverrideConfig()

        self._store.set_override(
            OverrideConfig(
                param_overrides=dict(existing.param_overrides),
                actuator_overrides=dict(existing.actuator_overrides),
                sim_stage=req.stage,
                sim_day_in_stage=clamped_day,
                sim_start_date=req.start_date,
                sim_start_hour=req.start_hour,
                preset_id=existing.preset_id,
            )
        )
        logger.info(
            "DT sim override applied: stage=%s, day=%d, start=%sT%02d:00",
            req.stage, clamped_day, req.start_date, req.start_hour,
        )

        return DtOverrideResponse(
            ok=True,
            applied_param="sim_params",
            applied_value=float(clamped_day),
            applied_stage=req.stage,
            message=(
                f"Simulation set to '{req.stage}' day {clamped_day},"
                f" starting {req.start_date} at {req.start_hour:02d}:00."
            ),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Preset application  (POST /api/dt/preset/{preset_id})
    # ─────────────────────────────────────────────────────────────────────────

    def apply_dt_preset(self, preset_id: str) -> DtPresetResponse:
        """Apply a named actuator preset (day-cycle | night-cycle | emergency-flush).

        Stores the preset's actuator overrides in ``OverrideConfig`` and
        records the ``preset_id``; retains any active param / sim overrides.

        Parameters
        ----------
        preset_id:
            One of ``"day-cycle"``, ``"night-cycle"``, or ``"emergency-flush"``.

        Returns
        -------
        DtPresetResponse
            Echoes the preset and its human-readable description.

        Raises
        ------
        ValueError
            If ``preset_id`` is not in ``DT_PRESET_DESCRIPTIONS``.
        """
        if preset_id not in DT_PRESET_DESCRIPTIONS:
            raise ValueError(
                f"Unknown preset '{preset_id}'. "
                f"Valid presets: {sorted(DT_PRESET_DESCRIPTIONS.keys())}"
            )

        actuator_overrides = _PRESET_ACTUATOR_OVERRIDES.get(preset_id, {})
        existing: OverrideConfig = self._store.get_override() or OverrideConfig()

        self._store.set_override(
            OverrideConfig(
                param_overrides=dict(existing.param_overrides),
                actuator_overrides=dict(actuator_overrides),
                sim_stage=existing.sim_stage,
                sim_day_in_stage=existing.sim_day_in_stage,
                sim_start_date=existing.sim_start_date,
                sim_start_hour=existing.sim_start_hour,
                preset_id=preset_id,
            )
        )
        description = DT_PRESET_DESCRIPTIONS[preset_id]
        logger.info("DT preset applied: %s", preset_id)

        return DtPresetResponse(
            ok=True,
            preset=preset_id,
            description=description,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Actuator set  (POST /api/actuators/set)
    # ─────────────────────────────────────────────────────────────────────────

    def apply_actuator_set(self, request: ActuatorSetRequest) -> ActuatorSetResponse:
        """Apply individual actuator-level overrides from ``POST /api/actuators/set``.

        Partial lists are supported — only listed actuators are updated.
        Unknown actuator IDs are silently skipped and the valid ones are
        still applied (the skip reason is logged at WARNING level).

        Levels are clamped to [0.0, 100.0].

        Parameters
        ----------
        request:
            Validated ``ActuatorSetRequest`` from the route handler.

        Returns
        -------
        ActuatorSetResponse
            ``overridden`` lists the IDs of actuators that were actually
            written (after unknown-ID filtering).
        """
        valid_ids = set(ACTUATOR_IDS)
        applied: list[str] = []
        skipped: list[str] = []

        existing: OverrideConfig = self._store.get_override() or OverrideConfig()
        merged: dict[str, float] = dict(existing.actuator_overrides)

        for entry in request.actuators:
            if entry.id not in valid_ids:
                logger.warning(
                    "apply_actuator_set: unknown actuator id '%s' — skipped.", entry.id
                )
                skipped.append(entry.id)
                continue
            merged[entry.id] = _clamp_level(entry.level)
            applied.append(entry.id)

        self._store.set_override(
            OverrideConfig(
                param_overrides=dict(existing.param_overrides),
                actuator_overrides=merged,
                sim_stage=existing.sim_stage,
                sim_day_in_stage=existing.sim_day_in_stage,
                sim_start_date=existing.sim_start_date,
                sim_start_hour=existing.sim_start_hour,
                preset_id=existing.preset_id,
            )
        )

        if skipped:
            logger.warning(
                "apply_actuator_set: %d unknown IDs skipped: %s", len(skipped), skipped
            )
        logger.info(
            "apply_actuator_set: overrode %d actuator(s): %s", len(applied), applied
        )

        return ActuatorSetResponse(ok=True, overridden=applied)

    # ─────────────────────────────────────────────────────────────────────────
    # Reset / clear override  (POST /api/dt/override/clear  or on loop restart)
    # ─────────────────────────────────────────────────────────────────────────

    def clear_override(self) -> None:
        """Remove the active override config and restore live DT-loop control.

        Called by the "Reset All" button in the ``ManualOverride`` page and by
        ``LoopService.start_loop()`` whenever a new loop session is started to
        ensure a clean slate.

        This is a no-op if no override is currently active.
        """
        current = self._store.get_override()
        if current is not None:
            self._store.clear_override()
            logger.info("Override cleared — DT loop returned to auto mode.")
        else:
            logger.debug("clear_override called but no override was active.")

    # ─────────────────────────────────────────────────────────────────────────
    # Read helpers
    # ─────────────────────────────────────────────────────────────────────────

    def is_override_active(self) -> bool:
        """Return ``True`` if any override config is currently stored."""
        return self._store.get_override() is not None

    def get_override_summary(self) -> dict[str, Any]:
        """Return a human-readable dict summarising the current override state.

        Useful for ``GET /api/dt/state`` enrichment and for debugging.
        """
        cfg = self._store.get_override()
        if cfg is None:
            return {"active": False}

        return {
            "active":            True,
            "preset_id":         cfg.preset_id,
            "sim_stage":         cfg.sim_stage,
            "sim_day_in_stage":  cfg.sim_day_in_stage,
            "param_overrides":   dict(cfg.param_overrides),
            "actuator_overrides": dict(cfg.actuator_overrides),
        }
