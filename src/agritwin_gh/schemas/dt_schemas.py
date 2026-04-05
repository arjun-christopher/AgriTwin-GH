"""
Digital-Twin state schemas — used by:
  ``GET /api/dt/state``
  ``POST /api/dt/override``
  ``POST /api/dt/preset/{preset_id}``

Canonical growth-stage names come from
``agritwin_gh.mpc.constants.GROWTH_STAGES`` (lowercase, e.g. "flowering").
Title-cased display names are in ``schemas.enums.GROWTH_STAGE_DISPLAY_NAME``.

3D-integration readiness
------------------------
``DTStateResponse`` carries **two complementary 3D-ready blocks**:

1. **Compact top-level fields** (``current_growth_stage``, ``next_growth_stage``,
   ``actuator_visual_state``, ``time_of_day``, ``timestamp``) — flat, cheap to
   consume in a game-engine bridge.
2. **``scene_context``** — the full ``SceneContext`` object with redundant copies
   of the same data plus every actuator's on/off map and level map.

The current React dashboard reads neither block; both are present for the 3D
layer to consume without any API change.  Populate both from ``RuntimeStore``
in the route handler.

Example JSON shape
------------------
::

    {
      "crop": {
        "current": "Seedling",
        "current_index": 0,
        "progress_pct": 0.0,
        "days_in_stage": 0.0,
        "stage_duration_days": 14.0,
        "next_stage": "Unripe",
        "next_in_days": 3.0,
        "stages": ["Seedling", "Early Vegetative", "Flowering Initiation",
                   "Flowering", "Unripe", "Ripe"]
      },
      "health": {
        "status": "Healthy",
        "confidence": 99.5,
        "detail": "Crop is healthy — no pathogens detected above threshold.",
        "scanned_ago": "2 mins ago",
        "disease_label": "healthy leaves",
        "risk_score": 0.02
      },
      "sensors": [...],
      "growth": { ... },
      "timestamp": "2026-04-04T09:15:00Z",
      "time_of_day": "morning",
      "current_growth_stage": "flowering",
      "next_growth_stage": "unripe",
      "actuator_visual_state": {
        "on_off": {"fan": true, "co2": false, ...},
        "levels": {"fan": 75.0, "co2": 0.0, ...},
        "mode": ""
      },
      "scene_context": { ... }
    }
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from agritwin_gh.schemas.enums import (
    ActuatorStatus,
    HealthStatus,
    TimeOfDay,
    TrendDirection,
    GROWTH_STAGE_ORDERED,
    GROWTH_STAGE_DISPLAY_NAME,
    STAGE_DURATION_DAYS,
    DT_PRESET_DESCRIPTIONS,
)


# ── 3D actuator visual state (compact, top-level) ─────────────────────────────


class ActuatorVisualState(BaseModel):
    """Compact per-actuator state for 3D scene rendering.

    Placed at the top level of ``DTStateResponse`` so that a Unity or
    WebGL bridge can read actuator state without traversing ``scene_context``.
    Keys in both dicts are the seven canonical frontend actuator IDs:
    ``fan | vent | irrigation | heater | led | co2 | fogger``.

    Notes
    -----
    - ``on_off`` is ``True`` when ``levels[id] > 0``.
    - ``mode`` is non-empty only when a DT preset is active (e.g. ``"day-cycle"``).
    """

    on_off: dict[str, bool] = Field(
        default_factory=dict,
        description="Actuator on/off booleans keyed by canonical actuator ID.",
    )
    levels: dict[str, float] = Field(
        default_factory=dict,
        description="Actuator duty-cycle levels 0–100 keyed by canonical actuator ID.",
    )
    mode: str = Field(
        default="",
        description="Actuator control mode: 'auto' (MPC-controlled) or "
                    "'manual' (operator-overridden via POST /api/actuators/set).",
    )


# ── 3D scene context (full, nested) ───────────────────────────────────────────


class SceneContext(BaseModel):
    """Full 3D environment integration block — included in every ``DTStateResponse``.

    The current React UI does not read this block.  The future 3D / Unity layer
    will consume it without requiring any schema change.  Populate from
    ``RuntimeStore`` on every ``GET /api/dt/state`` call.

    All dicts are keyed by the seven canonical actuator IDs:
    ``fan | vent | irrigation | heater | led | co2 | fogger``.
    """

    actuator_states_on_off: dict[str, bool] = Field(
        default_factory=dict,
        description="Per-actuator on/off boolean map.",
    )
    actuator_levels: dict[str, float] = Field(
        default_factory=dict,
        description="Per-actuator duty-cycle level 0–100 map.",
    )
    current_growth_stage: str = Field(
        default="",
        description="Canonical growth stage label (lowercase), e.g. 'flowering'.",
    )
    next_growth_stage: str | None = Field(
        default=None,
        description="Canonical label of the next stage, or null if at 'ripe'.",
    )
    current_timestamp: str = Field(
        default="",
        description="ISO-8601 wall-clock timestamp of the latest DT step.",
    )
    time_of_day: TimeOfDay = Field(
        default="morning",
        description="Bucketed time-of-day: morning (06–12) | afternoon (12–18) "
                    "| evening (18–21) | night (21–06).",
    )


# ── Crop info block ───────────────────────────────────────────────────────────


class CropInfo(BaseModel):
    """Current growth-stage metadata.

    Returned as ``DTStateResponse.crop``.  Matches the ``CROP`` constant consumed
    by ``HomeDashboard.jsx`` and ``DetailedInsights.jsx``.

    ``current`` uses the **title-cased display name** (e.g. ``"Flowering"``),
    not the lowercase canonical label, because the React UI renders it directly.
    All other canonical label fields use lowercase.

    Example
    -------
    ::
        {
          "current": "Seedling",
          "current_index": 0,
          "progress_pct": 0.0,
          "days_in_stage": 0.0,
          "stage_duration_days": 14.0,
          "next_stage": "Unripe",
          "next_in_days": 3.0,
          "stages": ["Seedling", "Early Vegetative", "Flowering Initiation",
                     "Flowering", "Unripe", "Ripe"]
        }
    """

    current: str = Field(
        default="",
        description="Title-cased display name of the current stage, e.g. 'Flowering'.",
    )
    current_index: int = Field(
        default=0,
        ge=0,
        le=5,
        description="Zero-based index into ``stages`` (0 = Seedling, 5 = Ripe).",
    )
    progress_pct: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Estimated percentage completion of the current stage (0–100).",
    )
    days_in_stage: float = Field(
        default=0.0,
        ge=0.0,
        description="Days elapsed since entering the current stage.",
    )
    stage_duration_days: float = Field(
        default=0.0,
        ge=0.0,
        description="Total expected duration of the current stage in days."
                    " From STAGE_DURATION_DAYS in enums.py.",
    )
    next_stage: str | None = Field(
        default=None,
        description="Title-cased display name of the next stage, or null if at 'Ripe'.",
    )
    next_in_days: float | None = Field(
        default=None,
        description="Estimated days until the next stage transition, or null if at 'Ripe'.",
    )
    stages: list[str] = Field(
        default_factory=lambda: [
            GROWTH_STAGE_DISPLAY_NAME[s] for s in GROWTH_STAGE_ORDERED
        ],
        description="Ordered list of all six title-cased stage display names.  "
                    "Used by ``CropStageTrack`` in the React UI.",
    )


# ── Crop health block ─────────────────────────────────────────────────────────


class CropHealth(BaseModel):
    """Disease / health summary for the Dashboard ``StatusCard``.

    Returned as ``DTStateResponse.health``.  Matches the ``CROP_HEALTH`` constant
    consumed by ``HomeDashboard.jsx`` and ``DetailedInsights.jsx``.

    Example
    -------
    ::
        {
          "status": "Healthy",
          "confidence": 99.5,
          "detail": "Crop is healthy — no pathogens detected above threshold.",
          "scanned_ago": "2 mins ago",
          "disease_label": "healthy leaves",
          "risk_score": 0.02
        }
    """

    status: HealthStatus = Field(
        default="Healthy",
        description="Traffic-light status: 'Healthy' | 'Warning' | 'Risk'.",
    )
    confidence: float = Field(
        default=100.0,
        ge=0.0,
        le=100.0,
        description="Disease classifier confidence percentage (0–100).",
    )
    detail: str = Field(
        default="",
        description="One-sentence human-readable summary shown under the StatusCard.",
    )
    scanned_ago: str = Field(
        default="",
        description="Human-readable recency of the last scan, e.g. '2 mins ago'.",
    )
    disease_label: str = Field(
        default="healthy leaves",
        description="Canonical disease label of the dominant classifier output.",
    )
    risk_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Composite disease risk score (0–1) forwarded from the MPC cost function.",
    )


# ── Sensor reading block ──────────────────────────────────────────────────────


class SensorReading(BaseModel):
    """One sensor metric entry in ``DTStateResponse.sensors``.

    The list covers all eight STATE_VARIABLES shown in the UI
    (``indoor_temp``, ``indoor_humidity``, ``soil_moisture``, ``co2``,
    ``light_intensity``, ``vpd``, ``leaf_wetness_proxy``, ``disease_risk_score``).
    ``growth_stage_index`` is omitted because the react UI renders it as a stage label.

    Used by:
      - ``HomeDashboard`` top metric strip (4 chips: temp / humidity / CO₂ / light).
      - ``DetailedInsights`` 8-chip summary row and 9-row indoor sensor table.

    Example
    -------
    ::
        {
          "key": "indoor_temp",
          "label": "Temperature",
          "value": 24.2,
          "unit": "°C",
          "icon_key": "Thermometer",
          "status": "ok",
          "trend": "up",
          "trend_label": "+0.4°",
          "trend_color": "text-warning",
          "range_min": 22.0,
          "range_max": 26.0,
          "optimal_range": "22–26 °C"
        }
    """

    key: str = Field(
        description="MPC STATE_VARIABLE key, e.g. 'indoor_temp'. "
                    "Matches keys in ``agritwin_gh.mpc.constants.STATE_VARIABLES``.",
    )
    label: str = Field(description="Human-readable display label, e.g. 'Temperature'.")
    value: float = Field(description="Raw sensor reading (not pre-formatted).")
    unit: str = Field(description="Physical unit string, e.g. '°C', 'ppm', '%'.")
    icon_key: str = Field(
        default="Activity",
        description="Lucide React icon name, e.g. 'Thermometer'. "
                    "See SENSOR_META in enums.py for canonical values.",
    )
    status: Literal["ok", "warning", "critical"] = Field(
        default="ok",
        description="Backend-computed status: 'ok' if within optimal range, "
                    "'warning' if outside by ≤ 20 %, 'critical' if further out.",
    )
    trend: TrendDirection = Field(
        default="stable",
        description="Direction vs. the previous DT step: 'up' | 'down' | 'stable'.",
    )
    trend_label: str = Field(
        default="",
        description="Human-readable delta string, e.g. '+0.4°' or '-2%'. "
                    "Empty string when trend is 'stable'.",
    )
    trend_color: str = Field(
        default="",
        description="Tailwind colour token for the trend chip, e.g. 'text-warning' or "
                    "'text-primary'. Set by the service layer based on direction and status.",
    )
    range_min: float = Field(
        default=0.0,
        description="Lower bound of the optimal operating range.",
    )
    range_max: float = Field(
        default=0.0,
        description="Upper bound of the optimal operating range.",
    )
    optimal_range: str = Field(
        default="",
        description="Human-readable optimal range label shown in the sensor table, "
                    "e.g. '22–26 °C'.",
    )
    delta: float = Field(
        default=0.0,
        description="Change vs. the previous DT step (new − old) for this sensor. "
                    "Sourced from DTDiagnostics.state_delta.",
    )


# ── Top-level DT state response ───────────────────────────────────────────────


class DTStateResponse(BaseModel):
    """Response for ``GET /api/dt/state``.

    Mirrors the shape returned by ``api.js → getDtState()`` in the frontend:
      ``{ crop, health, sensors, growth }``

    Additionally carries forward-compatible 3D-integration fields at the top
    level (``timestamp``, ``time_of_day``, ``current_growth_stage``,
    ``next_growth_stage``, ``actuator_visual_state``) and a full
    ``scene_context`` block.

    Implementation notes
    --------------------
    - ``growth`` embeds the full ``GrowthIntelResponse`` so that
      ``HomeDashboard`` can render the stage-history bars without a second
      API call.
    - ``sensors`` list order is defined by ``SENSOR_META`` in ``enums.py``.
    - ``timestamp`` is the ISO-8601 wall-clock of the **latest DT step**, not
      the HTTP request time.
    """

    # ── Core dashboard data ------------------------------------------------
    crop: CropInfo
    health: CropHealth
    sensors: list[SensorReading]
    growth: "GrowthIntelResponse"  # type: ignore[name-defined]  # forward ref resolved below

    # ── 3D-ready top-level summary fields ----------------------------------
    timestamp: str = Field(
        default="",
        description="ISO-8601 timestamp of the latest DT step (UTC).",
    )
    time_of_day: TimeOfDay = Field(
        default="morning",
        description="Time-of-day bucket for the 3D environment layer.",
    )
    current_growth_stage: str = Field(
        default="",
        description="Canonical lowercase growth stage label, e.g. 'flowering'.",
    )
    next_growth_stage: str | None = Field(
        default=None,
        description="Canonical lowercase label of the next stage, or null at 'ripe'.",
    )
    actuator_visual_state: ActuatorVisualState = Field(
        default_factory=ActuatorVisualState,
        description="Compact per-actuator on/off + level map for 3D rendering.",
    )

    # ── Operational mode -------------------------------------------------------
    mode: str = Field(
        default="live",
        description=(
            "Current DT operational mode: 'live' (MPC-autonomous) or 'override' "
            "(operator control active).  Changes on POST /api/dt/override/sim and "
            "resets to 'live' on DELETE /api/dt/override."
        ),
    )

    # ── Full 3D scene context (redundant but richer) -----------------------
    scene_context: SceneContext = Field(
        default_factory=SceneContext,
        description="Full 3D scene context block.  Not consumed by the React UI.",
    )


# Break the forward reference for GrowthIntelResponse.  Import is deferred to
# avoid a top-level circular dependency at module load time.
from agritwin_gh.schemas.intelligence_schemas import GrowthIntelResponse  # noqa: E402

DTStateResponse.model_rebuild()


# ── Override / preset request and response bodies ─────────────────────────────


class DtParamOverrideRequest(BaseModel):
    """Fine-grained DT parameter override — ``POST /api/dt/override``.

    Used when the caller wants to adjust a **single numeric parameter**,
    e.g. the temperature setpoint, without touching simulation stage or time.

    Example body::

        { "param": "temperature_setpoint", "value": 25.0 }

    Valid ``param`` values include any adjustable MPC setpoint or gain; the
    route handler validates against the current DT config.
    """

    param: str = Field(
        description="DT parameter key to override, e.g. 'temperature_setpoint'.",
    )
    value: float = Field(
        description="New numeric value for the parameter.",
    )


class DtSimOverrideRequest(BaseModel):
    """Simulation-parameters override — ``POST /api/dt/override`` from ``ManualOverride`` page.

    Sent by the React ``ManualOverride`` page when the operator clicks
    **"Apply Changes"** after editing the Simulation Parameters form.

    Fields mirror the form state in ``ManualOverride.jsx``:
      ``stage, dayInStage, startDate, startHour``

    Example body::

        {
          "stage": "flowering",
          "day_in_stage": 12,
          "start_date": "2026-04-04",
          "start_hour": 9
        }

    The route handler will re-initialise the DT loop with these parameters.
    ``day_in_stage`` is clamped server-side to ``STAGE_DURATION_DAYS[stage]``.
    """

    stage: str = Field(
        description="Canonical lowercase growth stage label, e.g. 'flowering'.",
    )
    day_in_stage: int = Field(
        ge=0,
        description="Day within the current stage (0-based). Clamped to stage duration.",
    )
    start_date: str = Field(
        default="",
        description="Simulation start date in ISO format YYYY-MM-DD. Defaults to today.",
    )
    start_hour: int = Field(
        default=-1,
        ge=-1,
        le=23,
        description="Simulation start hour (0–23, local wall-clock). -1 = use current hour.",
    )


class DtOverrideResponse(BaseModel):
    """Response for any ``POST /api/dt/override`` variant.

    ``ok: True`` on success.  ``applied_*`` fields echo back whatever was
    applied (param+value for single-param overrides; stage for sim overrides).
    """

    ok: bool = True
    applied_param: str = Field(
        default="",
        description="Param key that was applied (for DtParamOverrideRequest) "
                    "or 'sim_params' for DtSimOverrideRequest.",
    )
    applied_value: float = Field(
        default=0.0,
        description="Numeric value applied (for DtParamOverrideRequest only).",
    )
    applied_stage: str = Field(
        default="",
        description="Stage name applied (for DtSimOverrideRequest only).",
    )
    override_cnn_label: str | None = Field(
        default=None,
        description="Growth-stage CNN top-1 class name for the overridden stage "
                    "(e.g. 'Stage3_Flowering_Initiation').  Null if CNN skipped.",
    )
    override_cnn_confidence: float | None = Field(
        default=None,
        description="CNN top-1 confidence in [0, 1] for override_cnn_label.  "
                    "Null if CNN skipped.",
    )
    message: str = Field(
        default="",
        description="Optional human-readable confirmation message.",
    )


class DtPresetResponse(BaseModel):
    """Response for ``POST /api/dt/preset/{preset_id}``.

    ``preset`` echoes the path parameter.  ``description`` is looked up from
    ``DT_PRESET_DESCRIPTIONS`` in ``enums.py``.

    Example::

        { "ok": true, "preset": "day-cycle",
          "description": "Optimise for standard daytime growing conditions..." }
    """

    ok: bool = True
    preset: str = Field(
        default="",
        description="Preset ID that was applied, e.g. 'day-cycle'.",
    )
    description: str = Field(
        default="",
        description="Human-readable description of the preset.  "
                    "See DT_PRESET_DESCRIPTIONS in enums.py.",
    )
