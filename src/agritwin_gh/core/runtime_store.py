"""
In-memory runtime state store for the AgriTwin-GH FastAPI server.

Responsibility
--------------
``RuntimeStore`` is the single in-process state bridge between:

* The **background DT loop** (``DTService``) — which advances the greenhouse
  simulation every 5 minutes and owns heavy computation (MPC solve, sensor
  fusion, image observation).
* The **HTTP request handlers** — which return the *latest known result* to
  the React frontend instantly, without triggering any MPC work inline.

This decoupling means every ``GET /api/dt/state`` call takes < 1 ms (a
simple locked dict read), while the MPC solver can take 50–200 ms in the
background without blocking any HTTP response.

Data flow
---------
::

    ┌─────────────────────────────────────────────────────────────┐
    │  Background DT loop  (every 5 min)                          │
    │    DTService.tick()                                         │
    │      → MPCSolver.solve()             [~100 ms]             │
    │      → DigitalTwinEngine.step()                            │
    │      → store.update_from_step_result(result)    ← write    │
    └────────────────────────────┬────────────────────────────────┘
                                 │  threading.RLock
    ┌────────────────────────────▼────────────────────────────────┐
    │  RuntimeStore  (this module)                                │
    │    _state: LatestState                                      │
    │    _state.mode: "live" | "override"                        │
    │    _state.override: OverrideConfig | None                  │
    └────────────────────────────┬────────────────────────────────┘
                                 │  read (lock-free after copy)
    ┌────────────────────────────▼────────────────────────────────┐
    │  HTTP route handlers  (async, any time)                     │
    │    GET /api/dt/state                                        │
    │      → store.as_dt_state_response()     [< 1 ms]          │
    └─────────────────────────────────────────────────────────────┘

Internal state model
--------------------
The store maintains its own lightweight *domain snapshot* dataclasses rather
than exposing raw MPC dataclasses to the API layer.  Each snapshot holds only
stdlib-compatible types (float, str, dict, list, datetime), making them safe
to unit-test and easy to serialise.

Domain snapshots:

    ``ClimateSnapshot``   — indoor sensor readings + growth/disease label
    ``ActuatorSnapshot``  — actuator levels and on/off state
    ``WeatherSnapshot``   — outdoor environment summary + forecast stubs
    ``DiseaseSnapshot``   — composite disease risk and per-pathogen breakdown
    ``GrowthSnapshot``    — growth-stage intelligence (hours, probability, history)
    ``ResourceSnapshot``  — accumulated energy / water since loop start
    ``MediaSnapshot``     — latest image references (presigned URLs)

All snapshots are bundled into ``LatestState`` and atomically replaced under
``_lock`` on every ``update_latest_state()`` call.

Thread / async safety
---------------------
``threading.RLock`` protects all writes.  Reads are taken inside the lock
and immediately released — the caller gets a reference to the old
``LatestState`` object which is never mutated after assignment.

Module-level singleton
----------------------
``get_store()`` returns the process-wide ``RuntimeStore`` singleton.  Services
and route handlers always call this factory rather than importing the
singleton directly, which keeps unit testing easy.
"""

from __future__ import annotations

import datetime as _dt
import threading
from dataclasses import dataclass, field, replace as _dataclass_replace
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Internal domain snapshot types
# ─────────────────────────────────────────────────────────────────────────────
# Private to this module — not Pydantic models, not MPC dataclasses.
# Only stdlib-compatible types (float, str, dict, list, datetime).
# The ``as_*_response()`` methods on RuntimeStore convert them to API schemas.


@dataclass
class ClimateSnapshot:
    """Raw indoor sensor readings and crop context from one DT step.

    Populated by ``update_from_step_result()`` every 5 minutes.
    Source: ``GreenhouseState`` + ``DTSnapshot`` from ``dt_loop.py``.
    """

    # Indoor sensors (match STATE_VARIABLES order)
    indoor_temp: float = 0.0            # °C
    indoor_humidity: float = 0.0        # %RH
    soil_moisture: float = 0.0          # %
    co2: float = 0.0                    # ppm
    light_intensity: float = 0.0        # lux / W·m⁻²
    vpd: float = 0.0                    # kPa
    leaf_wetness_proxy: float = 0.0     # 0–1 unitless proxy
    disease_risk_score: float = 0.0     # 0–1 composite MPC penalty score

    # Growth stage  (canonical lowercase labels from mpc.constants.GROWTH_STAGES)
    growth_stage: str = "seedling"
    growth_stage_index: int = 0         # 0-based index into GROWTH_STAGES
    next_growth_stage: str | None = None

    # Stage-progress details
    days_elapsed: float = 0.0           # total days since loop start
    days_in_stage: float = 0.0          # days elapsed within the current stage
    stage_progress_pct: float = 0.0     # 0–100, derived from days_in_stage / stage_duration

    # Disease classification (from last image observation / IntelligenceService)
    disease_label: str = "healthy leaves"
    disease_confidence: float = 0.0     # 0–1 from the disease classifier

    # Loop provenance
    step_index: int = 0
    mpc_converged: bool = False
    loop_start_ts: _dt.datetime | None = None
    last_step_ts: _dt.datetime | None = None   # logical sim timestamp of this step
    timestamp: str = ""                         # ISO-8601 string of last_step_ts

    # Per-variable delta from the last DT step (new − old for each sensor)
    state_delta: dict[str, float] = field(default_factory=dict)
    """Keys match STATE_VARIABLES: indoor_temp, indoor_humidity, co2, soil_moisture,
    light_intensity, vpd, leaf_wetness_proxy.  Populated from DTDiagnostics.state_delta."""


@dataclass
class ActuatorSnapshot:
    """Actuator levels and on/off state after the last MPC solve or manual override.

    Keys use the **frontend short IDs** defined in
    ``agritwin_gh.schemas.enums.ACTUATOR_IDS``:
    ``fan | vent | irrigation | heater | led | co2 | fogger``.

    Updated by ``update_from_step_result()`` from ``ActuatorState`` (which uses
    MPC key names like ``fan_speed``), translated via
    ``enums.CONTROL_VAR_TO_ACTUATOR_ID``.

    Also updated by ``ActuatorService.apply_overrides()`` when the operator
    sends ``POST /api/actuators/set``.
    """

    levels: dict[str, float] = field(default_factory=lambda: {
        "fan": 0.0, "vent": 0.0, "irrigation": 0.0,
        "heater": 0.0, "led": 0.0, "co2": 0.0, "fogger": 0.0,
    })
    """Duty-cycle / intensity per actuator, 0–100."""

    on_off: dict[str, bool] = field(default_factory=lambda: {
        "fan": False, "vent": False, "irrigation": False,
        "heater": False, "led": False, "co2": False, "fogger": False,
    })
    """True if the actuator has a non-zero duty cycle."""

    mode: str = "auto"
    """'auto' — MPC-controlled; 'manual' — operator-overridden via API."""

    mpc_ran_this_step: bool = False
    """True if the MPC solver ran this step (not a carried-forward solution)."""


@dataclass
class WeatherSnapshot:
    """Outdoor environment summary from the last DT step.

    Core fields (``outdoor_temp``, ``outdoor_humidity``, ``solar_rad``,
    ``wind_speed``, ``condition``) are populated from ``WeatherState``
    (``agritwin_gh.mpc.state``) during ``update_from_step_result()``.

    Extended fields (``pressure``, ``uv_index``, ``dew_point``, ``wind_dir``,
    ``visibility``, ``forecast``) are set by ``WeatherService`` when a live
    weather data source is available.
    """

    outdoor_temp: float = 0.0           # °C
    outdoor_humidity: float = 0.0       # %RH
    solar_rad: float = 0.0             # W·m⁻²
    wind_speed: float = 0.0            # m·s⁻¹
    wind_dir: str = ""                  # compass bearing or label, e.g. "NE"
    pressure: float = 1013.0           # hPa
    uv_index: float = 0.0
    dew_point: float = 0.0             # °C
    condition: str = ""                 # "Clear", "Overcast", "Rain", …
    visibility: float = 10.0           # km
    forecast: list[dict[str, Any]] = field(default_factory=list)
    """Raw forecast dicts.  ``WeatherService`` converts them to ``ForecastEntry`` schemas.
    Each dict: {time, high, low, humidity, condition, icon_key}."""

    forecast_24h: dict[str, Any] = field(default_factory=dict)
    """Model 24-hr-ahead forecast from cadence_info[\"weather_24h_ahead\"].
    Keys: temp_external, humidity_external, solar_radiation, windspeed, conditions."""


@dataclass
class DiseaseSnapshot:
    """Disease intelligence from the last inference or image observation.

    ``composite_risk`` and ``dominant_label`` come directly from the DT loop
    (``GreenhouseState.disease_risk_score`` and ``DTSnapshot.disease_classification``).

    ``pathogens`` — the per-disease breakdown — is populated by
    ``IntelligenceService``, which calls the disease classifier for individual
    class probabilities.  It is empty until ``IntelligenceService`` runs.
    """

    composite_risk: float = 0.0         # 0–1 weighted mean risk
    dominant_label: str = "healthy leaves"
    pathogens: list[dict[str, Any]] = field(default_factory=list)
    """Each dict: {label, name, pathogen, risk_score, risk_24h, probability, severity}"""
    timestamp: str = ""


@dataclass
class GrowthSnapshot:
    """Growth-stage intelligence from the last DT step or classifier run.

    ``current_stage``, ``current_stage_index``, and ``next_stage`` are
    always populated from the DT loop.

    ``hours_to_next_stage``, ``transition_prob_24h``, and ``stage_history``
    are enriched by ``IntelligenceService`` (empty lists/zeros before that).
    """

    current_stage: str = "seedling"
    current_stage_index: int = 0
    next_stage: str | None = None
    hours_to_next_stage: float = 0.0    # estimated hours to stage transition
    days_to_next_stage: float = 0.0     # same in days
    transition_prob_24h: float = 0.0    # 0–1 estimated 24h transition probability
    vpd: float = 0.0                    # current VPD (mirrored for the growth panel)
    growth_score: float = 0.0           # 0–1 composite growth quality score
    growth_stage_confidence: float = 0.0  # 0–1 from the growth-stage CNN classifier
    stage_history: list[dict[str, Any]] = field(default_factory=list)
    """Each dict: {stage, stage_key, days_used, days_target, complete, status, progress_pct}"""
    timestamp: str = ""


@dataclass
class ResourceSnapshot:
    """Accumulated energy and water since the DT loop started.

    ``DTService`` calls ``store.accumulate_resources()`` after each step,
    adding the incremental usage for that 5-minute interval based on actuator
    duty cycles.

    ``as_resources_response()`` converts totals to INR costs using default
    tariff rates (overridden by ``ResourceService`` in Phase 2).

    Monthly billing: totals are reset to zero at the start of each new
    calendar month (detected in ``accumulate_resources``).
    """

    energy_kwh_total: float = 0.0       # kWh accumulated in current billing month
    water_l_total: float = 0.0          # litres accumulated in current billing month
    loop_start_ts: _dt.datetime | None = None
    billing_month: str = ""             # "YYYY-MM" of the active billing month
    actuator_energy_kwh: dict = field(default_factory=dict)  # {key: kwh, ...}


@dataclass
class MediaSnapshot:
    """Latest image references for all three media endpoints.

    Populated by ``MediaService`` after each ``ImageStreamer`` query (Phase 3).
    Until then, all URLs are empty strings and the gallery lists are empty.

    Sources are presigned MinIO URLs valid for ~15 minutes.
    """

    # GET /api/media/latest — single best image per category
    latest_stage_src: str = ""
    latest_stage_alt: str = ""
    # CNN-derived image data (local path, class label, confidence) — carried
    # forward every step until the next image-refresh cadence fires.
    latest_stage_image_key: str = ""      # relative local path from CNN
    latest_stage_label: str = ""          # CNN class name e.g. "Stage4_Flowering"
    latest_stage_confidence: float = 0.0  # CNN confidence 0–1

    latest_leaf_src: str = ""
    latest_leaf_alt: str = ""
    latest_leaf_image_key: str = ""       # relative local path from CNN
    latest_leaf_label: str = ""           # CNN class name e.g. "tomato_leaf_healthy"
    latest_leaf_confidence: float = 0.0   # CNN confidence 0–1

    # GET /api/media/stage-images — growth-stage photo gallery
    stage_images: list[dict[str, Any]] = field(default_factory=list)
    """Each dict matches StageImageEntry fields:
    {src, alt, captured_ago, stage, confidence, image_key, bucket_name}"""

    # GET /api/media/disease-scans — leaf-scan photo gallery
    disease_scans: list[dict[str, Any]] = field(default_factory=list)
    """Each dict matches LeafScanEntry fields:
    {src, alt, captured_ago, classification, risk, confidence, image_key, bucket_name}"""


# ─────────────────────────────────────────────────────────────────────────────
# Override configuration
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class OverrideConfig:
    """Active manual override state.

    Set by the service layer via ``store.set_override()`` when the operator
    sends ``POST /api/dt/override`` or ``POST /api/dt/preset/{id}``.

    Cleared via ``store.clear_override()`` when the operator resets to live mode.

    ``param_overrides``    — DT parameter overrides keyed by MPC param name,
                             e.g. ``{"temperature_setpoint": 25.0}``.
    ``actuator_overrides`` — Direct actuator level overrides keyed by frontend
                             actuator ID (0–100), e.g. ``{"fan": 80.0}``.
    ``sim_stage``          — Force the simulated growth stage (ManualOverride page).
    ``preset_id``          — Active preset name (``"day-cycle"`` etc.), if any.
    """

    param_overrides: dict[str, float] = field(default_factory=dict)
    actuator_overrides: dict[str, float] = field(default_factory=dict)
    sim_stage: str | None = None
    sim_day_in_stage: int | None = None
    sim_start_date: str | None = None
    sim_start_hour: int | None = None
    preset_id: str | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Compound state + 3D payload types
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class LatestState:
    """Complete runtime state at one point in time.

    Atomically replaced under ``RuntimeStore._lock`` on every
    ``update_latest_state()`` call — individual snapshot fields are never
    mutated in-place.  This means route handlers that hold a reference to an
    older ``LatestState`` see a fully consistent snapshot.
    """

    climate: ClimateSnapshot = field(default_factory=ClimateSnapshot)
    actuators: ActuatorSnapshot = field(default_factory=ActuatorSnapshot)
    weather: WeatherSnapshot = field(default_factory=WeatherSnapshot)
    disease: DiseaseSnapshot = field(default_factory=DiseaseSnapshot)
    growth: GrowthSnapshot = field(default_factory=GrowthSnapshot)
    resources: ResourceSnapshot = field(default_factory=ResourceSnapshot)
    media: MediaSnapshot = field(default_factory=MediaSnapshot)

    mode: str = "live"
    """'live' — DT loop is running normally; 'override' — operator control active."""

    override: OverrideConfig | None = None
    """Active override config, or None when mode == 'live'."""

    last_updated: str = ""
    """ISO-8601 UTC timestamp when this LatestState was produced."""


@dataclass
class ThreeDPayload:
    """Compact payload prepared for the future 3D environment integration layer.

    **Data preparation only — no 3D rendering logic lives here.**

    This minimal struct is returned by ``store.get_3d_payload()`` and is
    purpose-built for a future WebSocket connection that streams live state
    to a 3D scene manager (e.g., Three.js, Unity WebGL, Babylon.js).

    The 3D scene manager can read this payload to:

    * Select the correct plant mesh and activate growth-stage particle effects
      (``growth_stage``, ``growth_stage_index``).
    * Pre-load the next plant-mesh asset (``next_growth_stage``).
    * Toggle and animate actuator props (fan blades, drip irrigation, LED glow,
      CO₂ cloud, fogger mist) using ``actuators[*].on`` and ``actuators[*].level``.
    * Drive ambient lighting and sky-box selection (``time_of_day``).

    Actuator entry format::

        {"id": "fan", "on": True, "level": 75.0}

    All 7 canonical actuators are always present in ``actuators``, in the
    canonical order: fan, vent, irrigation, heater, led, co2, fogger.
    """

    timestamp: str                          # ISO-8601 UTC
    time_of_day: str                        # "morning" | "afternoon" | "evening" | "night"
    growth_stage: str                       # canonical lowercase, e.g. "flowering"
    next_growth_stage: str | None          # None at "ripe" (final stage)
    actuators: list[dict[str, Any]]        # [{id, on, level}, …] for all 7 actuators
    growth_stage_index: int = 0            # 0-based index into GROWTH_STAGES


# ─────────────────────────────────────────────────────────────────────────────
# RuntimeStore
# ─────────────────────────────────────────────────────────────────────────────


class RuntimeStore:
    """Thread-safe in-process store for the current AgriTwin-GH runtime state.

    The store is the only shared mutable state in the AgriTwin-GH server process.
    Every other component (services, route handlers) reads from or writes to it
    using the public methods below — never accessing ``_state`` directly.

    Usage — route handlers (read)::

        store = get_store()
        return store.as_dt_state_response()

    Usage — DTService (write after each DT step)::

        store = get_store()
        store.update_from_step_result(result)

    Usage — service layer (enrich a specific domain snapshot)::

        store.update_latest_state(
            disease=DiseaseSnapshot(composite_risk=0.42, ...),
        )
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._state: LatestState = LatestState(last_updated=_now_iso())

    # ─────────────────────────────────────────────────────────────────────
    # Primary write API — update_latest_state
    # ─────────────────────────────────────────────────────────────────────

    def update_latest_state(
        self,
        *,
        climate: ClimateSnapshot | None = None,
        actuators: ActuatorSnapshot | None = None,
        weather: WeatherSnapshot | None = None,
        disease: DiseaseSnapshot | None = None,
        growth: GrowthSnapshot | None = None,
        resources: ResourceSnapshot | None = None,
        media: MediaSnapshot | None = None,
    ) -> None:
        """Partial-update one or more domain snapshots atomically.

        Only keyword arguments that are not ``None`` will be replaced.  All
        other domain snapshots are carried forward unchanged.  ``last_updated``
        is always refreshed.

        This is the canonical write path.  Call it any time a domain snapshot
        becomes stale::

            # After a DT step — climate + actuators updated together:
            store.update_latest_state(
                climate=ClimateSnapshot(...),
                actuators=ActuatorSnapshot(...),
            )

            # After IntelligenceService fetches disease risks (separate call):
            store.update_latest_state(disease=DiseaseSnapshot(...))
        """
        with self._lock:
            s = self._state
            self._state = LatestState(
                climate=climate if climate is not None else s.climate,
                actuators=actuators if actuators is not None else s.actuators,
                weather=weather if weather is not None else s.weather,
                disease=disease if disease is not None else s.disease,
                growth=growth if growth is not None else s.growth,
                resources=resources if resources is not None else s.resources,
                media=media if media is not None else s.media,
                mode=s.mode,
                override=s.override,
                last_updated=_now_iso(),
            )

    def update_from_step_result(self, result: Any) -> None:  # noqa: ANN401
        """Unpack a ``DTLoopStepResult`` from the DT loop and write to the store.

        Called by ``DTService.tick()`` after every closed-loop DT iteration
        (every 5 minutes in production, every simulated step in synthetic mode).

        Derives ``ClimateSnapshot``, ``ActuatorSnapshot``, ``WeatherSnapshot``,
        ``DiseaseSnapshot``, and ``GrowthSnapshot`` from the result.
        ``ResourceSnapshot`` and ``MediaSnapshot`` are updated separately via
        ``accumulate_resources()`` and ``update_latest_state(media=...)``.

        Parameters
        ----------
        result:
            An instance of ``agritwin_gh.mpc.dt_loop.DTLoopStepResult``.
            Typed as ``Any`` to avoid a heavy import at module load time
            (the import would normally trigger loading all MPC submodules).
        """
        from agritwin_gh.schemas.enums import (
            CONTROL_VAR_TO_ACTUATOR_ID,
            DISEASE_DISPLAY_NAME,
            DISEASE_PATHOGEN,
            GROWTH_STAGE_ORDERED as GROWTH_STAGES,
            STAGE_DURATION_DAYS,
        )

        # ── 1. Climate snapshot — derived from next_state (post-step) ─────
        ns = result.next_state          # agritwin_gh.mpc.state.GreenhouseState
        step_ts: _dt.datetime | None = result.timestamp

        stage_idx = max(0, min(int(ns.growth_stage_index), len(GROWTH_STAGES) - 1))
        stage_label = GROWTH_STAGES[stage_idx]
        next_stage = GROWTH_STAGES[stage_idx + 1] if stage_idx < len(GROWTH_STAGES) - 1 else None
        stage_dur_days = STAGE_DURATION_DAYS.get(stage_label, 14.0)

        # ``days_in_stage`` comes from cadence_info (set by DTLoop.run())
        days_in_stage: float = float(result.cadence_info.get("days_in_stage", 0.0))
        progress_pct = (
            min(100.0, (days_in_stage / stage_dur_days) * 100.0)
            if stage_dur_days > 0 else 0.0
        )

        # MPC convergence — only meaningful if the solver ran this step
        mpc_converged = False
        if result.mpc_ran_this_step and result.mpc_solution is not None:
            mpc_converged = bool(getattr(result.mpc_solution, "converged", False))

        climate = ClimateSnapshot(
            indoor_temp=float(ns.indoor_temp),
            indoor_humidity=float(ns.indoor_humidity),
            soil_moisture=float(ns.soil_moisture),
            co2=float(ns.co2),
            light_intensity=float(ns.light_intensity),
            vpd=float(ns.vpd),
            leaf_wetness_proxy=float(ns.leaf_wetness_proxy),
            disease_risk_score=float(ns.disease_risk_score),
            growth_stage=stage_label,
            growth_stage_index=stage_idx,
            next_growth_stage=next_stage,
            days_elapsed=float(result.cadence_info.get("days_elapsed", 0.0)),
            days_in_stage=round(days_in_stage, 2),
            stage_progress_pct=round(progress_pct, 1),
            disease_label=result.snapshot.disease_classification or "healthy leaves",
            disease_confidence=float(
                result.snapshot.metadata.get("disease_confidence", 0.0)
            ),
            step_index=int(result.step_index),
            mpc_converged=mpc_converged,
            loop_start_ts=self._state.climate.loop_start_ts,   # preserved across steps
            last_step_ts=step_ts,
            timestamp=step_ts.isoformat() if step_ts else _now_iso(),
            state_delta=dict(result.diagnostics.state_delta),
        )

        # ── 2. Actuator snapshot — translated from ActuatorState MPC keys ─
        aa = result.action_applied          # agritwin_gh.mpc.state.ActuatorState
        levels: dict[str, float] = {
            aid: float(getattr(aa, mpc_key, 0.0))
            for mpc_key, aid in CONTROL_VAR_TO_ACTUATOR_ID.items()
        }
        # Preserve "manual" mode while an actuator override is active so that
        # GET /api/actuators/state correctly shows MANUAL status between ticks.
        override_cfg = self._state.override
        actuator_mode = (
            "manual"
            if override_cfg and override_cfg.actuator_overrides
            else "auto"
        )
        actuators = ActuatorSnapshot(
            levels=levels,
            # Round to 2 d.p. before checking > 0 so that CVXPY solver residuals
            # (e.g. 0.001) that display as "0.00" in the log don't show as ON.
            on_off={k: round(v, 2) > 0.0 for k, v in levels.items()},
            mode=actuator_mode,
            mpc_ran_this_step=bool(result.mpc_ran_this_step),
        )

        # ── 3. Weather snapshot — from WeatherState ────────────────────────
        w = result.weather_used             # agritwin_gh.mpc.state.WeatherState
        _old_w = self._state.weather        # preserve WeatherService-supplied fields
        weather = WeatherSnapshot(
            outdoor_temp=float(getattr(w, "temp_external", 0.0)),
            outdoor_humidity=float(getattr(w, "humidity_external", 0.0)),
            solar_rad=float(getattr(w, "solar_radiation", 0.0)),
            wind_speed=float(getattr(w, "windspeed", 0.0)),
            condition=str(getattr(w, "conditions", "")),
            # Carry forward extended fields set by WeatherService
            wind_dir=_old_w.wind_dir,
            pressure=_old_w.pressure,
            uv_index=_old_w.uv_index,
            dew_point=_old_w.dew_point,
            visibility=_old_w.visibility,
            forecast=list(_old_w.forecast),
            # 24h-ahead model forecast from the weather-forecast LSTM/ensemble
            forecast_24h=dict(result.cadence_info.get("weather_24h_ahead") or {}),
        )

        # ── 4. Disease snapshot — build pathogens from cadence_info if available ─
        # ``model_disease_result`` is {disease_label: {severity_24h, present, trend_24h}}
        # ``severity_24h`` (from mpc_fused) is {disease_label: float(0–1)}
        model_disease = result.cadence_info.get("model_disease_result") or {}
        sev_24h_map   = result.cadence_info.get("severity_24h") or {}

        if model_disease or sev_24h_map:
            all_labels = sorted(set(model_disease) | set(sev_24h_map))
            pathogens_list: list[dict] = []
            for d_label in all_labels:
                dis_info  = model_disease.get(d_label, {})
                raw_sev   = float(dis_info.get("severity_24h", sev_24h_map.get(d_label, 0.0)))
                risk_24h  = round(raw_sev * 100.0, 1)   # Convert 0–1 → 0–100 scale
                prob      = float(dis_info.get("probability", raw_sev))
                if risk_24h > 80:
                    sev_label = "High"
                elif risk_24h > 60:
                    sev_label = "Medium"
                else:
                    sev_label = "Low"
                pathogens_list.append({
                    "label":      d_label,
                    "name":       DISEASE_DISPLAY_NAME.get(d_label, d_label.replace("_", " ").title()),
                    "pathogen":   DISEASE_PATHOGEN.get(d_label, ""),
                    "risk_score": round(raw_sev, 3),
                    "risk_24h":   risk_24h,
                    "probability": round(prob, 3),
                    "severity":   sev_label,
                })
        else:
            pathogens_list = list(self._state.disease.pathogens)  # preserve enriched data

        disease = DiseaseSnapshot(
            composite_risk=float(ns.disease_risk_score),
            dominant_label=result.snapshot.disease_classification or "healthy leaves",
            pathogens=pathogens_list,
            timestamp=climate.timestamp,
        )

        # ── 5. Growth snapshot — basic; hours/prob set from stage_dur ─────
        #    IntelligenceService will overwrite ``stage_history`` later.
        # Preserve growth_stage_confidence from the latest image observation.
        if result.image_refresh_this_step and result.image_observation is not None:
            gs_confidence = float(result.image_observation.growth_stage_confidence)
        else:
            gs_confidence = self._state.growth.growth_stage_confidence  # carry forward
        # Use the LSTM's predicted hours_to_transition when available (it is based
        # on actual sensor trajectories).  Fall back to the DT formula otherwise.
        _mgr_for_h   = result.cadence_info.get("model_growth_result") or {}
        _lstm_htt    = _mgr_for_h.get("hours_to_transition")
        _hours_to_next = (
            float(_lstm_htt)
            if _lstm_htt is not None and float(_lstm_htt) >= 0.0
            else max(0.0, (stage_dur_days - days_in_stage) * 24.0)
        )
        growth = GrowthSnapshot(
            current_stage=stage_label,
            current_stage_index=stage_idx,
            next_stage=next_stage,
            hours_to_next_stage=_hours_to_next,
            days_to_next_stage=max(0.0, stage_dur_days - days_in_stage),
            vpd=float(ns.vpd),
            growth_stage_confidence=gs_confidence,
            stage_history=list(self._state.growth.stage_history),  # preserve enriched
            timestamp=climate.timestamp,
        )

        # ── 6. Media snapshot — carry forward CNN image observation data ──────
        # Only updated when the image-refresh cadence fires (every ~6 steps).
        # Between refreshes, the old CNN keys/labels/confidence are preserved.
        old_media = self._state.media
        if result.image_refresh_this_step and result.image_observation is not None:
            obs = result.image_observation
            media: MediaSnapshot | None = MediaSnapshot(
                latest_stage_src=old_media.latest_stage_src,
                latest_stage_alt=old_media.latest_stage_alt,
                latest_stage_image_key=obs.growth_stage_image_key or old_media.latest_stage_image_key,
                latest_stage_label=obs.growth_stage_label or old_media.latest_stage_label,
                latest_stage_confidence=(
                    obs.growth_stage_confidence
                    if obs.growth_stage_confidence > 0.0
                    else old_media.latest_stage_confidence
                ),
                latest_leaf_src=old_media.latest_leaf_src,
                latest_leaf_alt=old_media.latest_leaf_alt,
                latest_leaf_image_key=obs.disease_image_key or old_media.latest_leaf_image_key,
                latest_leaf_label=obs.disease_label or old_media.latest_leaf_label,
                latest_leaf_confidence=(
                    obs.disease_confidence
                    if obs.disease_confidence > 0.0
                    else old_media.latest_leaf_confidence
                ),
                stage_images=old_media.stage_images,
                disease_scans=old_media.disease_scans,
            )
        else:
            media = None  # no change — update_latest_state preserves existing media

        self.update_latest_state(
            climate=climate,
            actuators=actuators,
            weather=weather,
            disease=disease,
            growth=growth,
            media=media,
        )

    def set_loop_start(self, ts: _dt.datetime) -> None:
        """Record the timestamp when the DT loop was started.

        Called once from ``DTService.setup()`` during the lifespan startup.
        Persists in ``ClimateSnapshot.loop_start_ts`` and ``ResourceSnapshot``
        so that ``as_resources_response()`` can label the billing month correctly.
        """
        with self._lock:
            s = self._state
            self._state = LatestState(
                climate=_dataclass_replace(s.climate, loop_start_ts=ts),
                actuators=s.actuators,
                weather=s.weather,
                disease=s.disease,
                growth=s.growth,
                resources=_dataclass_replace(s.resources, loop_start_ts=ts),
                media=s.media,
                mode=s.mode,
                override=s.override,
                last_updated=_now_iso(),
            )

    def accumulate_resources(
        self,
        energy_kwh: float,
        water_l: float,
        actuator_energy: dict[str, float] | None = None,
    ) -> None:
        """Add incremental resource usage to the running totals.

        Called by ``DTService`` after each DT step, based on the actuator
        duty cycles applied during that 5-minute interval.

        ``ResourceService`` uses the totals in ``as_resources_response()``
        to compute INR costs.

        Monthly rollover: if the calendar month has changed since the last
        accumulation, all totals are reset to zero before adding the new step
        (so the response always reflects the *current* billing month only).
        """
        current_month = _dt.datetime.now().strftime("%Y-%m")

        with self._lock:
            s = self._state
            r = s.resources

            # ── Month rollover ─────────────────────────────────────────
            if r.billing_month and r.billing_month != current_month:
                # New calendar month → wipe all accumulators
                r = _dataclass_replace(
                    r,
                    energy_kwh_total=0.0,
                    water_l_total=0.0,
                    actuator_energy_kwh={},
                )

            # ── Per-actuator accumulation ──────────────────────────────
            new_act = dict(r.actuator_energy_kwh)
            if actuator_energy:
                for k, v in actuator_energy.items():
                    new_act[k] = new_act.get(k, 0.0) + v

            self._state = LatestState(
                climate=s.climate,
                actuators=s.actuators,
                weather=s.weather,
                disease=s.disease,
                growth=s.growth,
                resources=_dataclass_replace(
                    r,
                    energy_kwh_total=r.energy_kwh_total + energy_kwh,
                    water_l_total=r.water_l_total + water_l,
                    billing_month=current_month,
                    actuator_energy_kwh=new_act,
                ),
                media=s.media,
                mode=s.mode,
                override=s.override,
                last_updated=_now_iso(),
            )

    # ─────────────────────────────────────────────────────────────────────
    # Read API — get_latest_state
    # ─────────────────────────────────────────────────────────────────────

    def get_latest_state(self) -> LatestState:
        """Return the complete current runtime state.

        Returns the live ``LatestState`` reference.  Since the store only
        ever *replaces* ``_state`` (never mutates it in-place), the returned
        reference is safe to read outside the lock — the caller sees a
        fully consistent snapshot.

        Services that need multiple fields from the same atomic snapshot
        should call this method once and read from the returned object,
        rather than calling ``get_mode()``, ``get_override()`` etc. separately.
        """
        with self._lock:
            return self._state

    # ─────────────────────────────────────────────────────────────────────
    # Override API — set_override / get_override / clear_override
    # Called by DTService when the operator sends an override request.
    # ─────────────────────────────────────────────────────────────────────

    def set_override(self, config: OverrideConfig) -> None:
        """Enter override mode with the given configuration.

        The store transitions to ``mode = 'override'`` and stores ``config``.
        The next DT loop tick will call ``get_override()`` and apply the
        config before the MPC solve.

        When ``config.actuator_overrides`` is non-empty the ``ActuatorSnapshot``
        is updated **immediately** so that ``GET /api/actuators/state`` and
        ``GET /api/dt/state`` reflect the override without waiting for the next
        DT loop tick.  This gives the React ``ManualOverride`` page instant
        confirmation that the apply action was accepted.

        Called by ``ControlService.apply_*`` and ``ControlService.apply_preset()``.
        """
        with self._lock:
            s = self._state
            if config.actuator_overrides:
                # Merge override levels into the current snapshot immediately.
                new_levels = dict(s.actuators.levels)
                new_levels.update(config.actuator_overrides)
                new_actuators = _dataclass_replace(
                    s.actuators,
                    levels=new_levels,
                    on_off={k: v > 0.0 for k, v in new_levels.items()},
                    mode="manual",
                )
            else:
                new_actuators = s.actuators
            self._state = LatestState(
                climate=s.climate,
                actuators=new_actuators,
                weather=s.weather,
                disease=s.disease,
                growth=s.growth,
                resources=s.resources,
                media=s.media,
                mode="override",
                override=config,
                last_updated=_now_iso(),
            )

    def get_override(self) -> OverrideConfig | None:
        """Return the active ``OverrideConfig``, or ``None`` if in live mode."""
        with self._lock:
            return self._state.override

    def clear_override(self) -> None:
        """Return to live mode and discard any active override config.

        Also resets ``ActuatorSnapshot.mode`` back to ``'auto'`` so that the
        next ``GET /api/actuators/state`` reflects MPC-controlled status
        immediately, without waiting for the next DT loop tick.
        """
        with self._lock:
            s = self._state
            self._state = LatestState(
                climate=s.climate,
                actuators=_dataclass_replace(s.actuators, mode="auto"),
                weather=s.weather,
                disease=s.disease,
                growth=s.growth,
                resources=s.resources,
                media=s.media,
                mode="live",
                override=None,
                last_updated=_now_iso(),
            )

    # ─────────────────────────────────────────────────────────────────────
    # Mode API — set_mode / get_mode
    # ─────────────────────────────────────────────────────────────────────

    def set_mode(self, mode: str) -> None:
        """Set the operational mode to ``'live'`` or ``'override'``.

        Prefer ``set_override()`` / ``clear_override()`` for transitions that
        accompany a config change.  Use ``set_mode`` only when the mode label
        needs updating independently (e.g. when the DT loop auto-recovers from
        an override on stage transition).

        Raises ``ValueError`` for unknown mode strings.
        """
        if mode not in ("live", "override"):
            raise ValueError(f"Unknown mode {mode!r} — expected 'live' or 'override'.")
        with self._lock:
            s = self._state
            self._state = _dataclass_replace(s, mode=mode, last_updated=_now_iso())

    def get_mode(self) -> str:
        """Return the current operational mode: ``'live'`` or ``'override'``."""
        with self._lock:
            return self._state.mode

    # ─────────────────────────────────────────────────────────────────────
    # 3D-ready compact payload
    # ─────────────────────────────────────────────────────────────────────

    def get_3d_payload(self) -> ThreeDPayload:
        """Return a compact payload prepared for the future 3D environment layer.

        This method is **data preparation only** — no 3D rendering or animation
        logic lives here.  The returned ``ThreeDPayload`` contains exactly the
        fields a 3D scene manager needs to:

        * Swap the plant mesh when ``growth_stage`` changes.
        * Toggle actuator prop animations (fan RPM, irrigation drip, LED glow,
          CO₂ cloud opacity) via ``actuators[*].on`` and ``actuators[*].level``.
        * Set ambient lighting and sky-box from ``time_of_day``.

        Zero DB queries, zero heavy computation — derived entirely from the
        in-memory state.  Suitable for high-frequency polling from a WebGL
        render loop once WebSocket delivery is wired in Phase 5.

        Actuator entry format::

            {"id": "fan", "on": True, "level": 75.0}

        All seven actuators are always present in canonical order:
        fan, vent, irrigation, heater, led, co2, fogger.
        """
        with self._lock:
            s = self._state
            cl = s.climate
            ac = s.actuators

        ts = cl.timestamp or _now_iso()
        time_of_day = _compute_time_of_day(cl.last_step_ts)
        actuators_3d = [
            {"id": aid, "on": ac.on_off.get(aid, False), "level": ac.levels.get(aid, 0.0)}
            for aid in ("fan", "vent", "irrigation", "heater", "led", "co2", "fogger")
        ]
        return ThreeDPayload(
            timestamp=ts,
            time_of_day=time_of_day,
            growth_stage=cl.growth_stage,
            next_growth_stage=cl.next_growth_stage,
            actuators=actuators_3d,
            growth_stage_index=cl.growth_stage_index,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Schema projection methods — called by route handlers
    #
    # Each method reads from ``_state`` once under the lock (taking a local
    # reference to the snapshot), then builds the Pydantic schema outside
    # the lock.  Imports are lazy (inside the method body) to prevent
    # circular imports between the store and the schema / MPC layers.
    # ─────────────────────────────────────────────────────────────────────

    def as_dt_state_response(self):  # noqa: ANN201
        """Project store state → ``DTStateResponse``.

        Called by ``GET /api/dt/state``.
        """
        from agritwin_gh.schemas.dt_schemas import (
            ActuatorVisualState,
            CropHealth,
            CropInfo,
            DTStateResponse,
            SceneContext,
            SensorReading,
        )
        from agritwin_gh.schemas.enums import (
            GROWTH_STAGE_DISPLAY_NAME,
            GROWTH_STAGE_ORDERED,
            SENSOR_META,
            STAGE_DURATION_DAYS,
        )
        from agritwin_gh.schemas.intelligence_schemas import GrowthIntelResponse

        with self._lock:
            s = self._state

        cl = s.climate
        ac = s.actuators
        gr = s.growth
        ov = s.override           # OverrideConfig | None
        ts = cl.timestamp or _now_iso()
        time_of_day = _compute_time_of_day(cl.last_step_ts)

        # ── Resolve effective stage values (live vs override) ─────────────
        # When an override is active with a sim_stage, use it immediately so
        # GET /api/dt/state reflects the operator's intent without waiting for
        # the next DT loop tick.  This preserves correctness: the base climate
        # snapshot is never mutated — only the response projection changes.
        if ov and ov.sim_stage:
            eff_stage = ov.sim_stage
            eff_stage_idx = (
                list(GROWTH_STAGE_ORDERED).index(ov.sim_stage)
                if ov.sim_stage in GROWTH_STAGE_ORDERED else cl.growth_stage_index
            )
            eff_next_stage = (
                GROWTH_STAGE_ORDERED[eff_stage_idx + 1]
                if eff_stage_idx < len(GROWTH_STAGE_ORDERED) - 1 else None
            )
            eff_days_in_stage = float(ov.sim_day_in_stage or cl.days_in_stage)
        else:
            eff_stage = cl.growth_stage
            eff_stage_idx = cl.growth_stage_index
            eff_next_stage = cl.next_growth_stage
            eff_days_in_stage = cl.days_in_stage

        # ── CropInfo ──────────────────────────────────────────────────────
        stage_display = GROWTH_STAGE_DISPLAY_NAME.get(eff_stage, eff_stage.title())
        next_display = (
            GROWTH_STAGE_DISPLAY_NAME.get(eff_next_stage, eff_next_stage.title())
            if eff_next_stage else None
        )
        stage_dur = STAGE_DURATION_DAYS.get(eff_stage, 14.0)
        next_in_days = max(0.0, stage_dur - eff_days_in_stage)
        stages_display = [
            GROWTH_STAGE_DISPLAY_NAME.get(st, st.title()) for st in GROWTH_STAGE_ORDERED
        ]
        crop = CropInfo(
            current=stage_display,
            current_index=eff_stage_idx,
            progress_pct=min(100.0, round((eff_days_in_stage / stage_dur) * 100.0, 1) if stage_dur else 0.0),
            days_in_stage=round(eff_days_in_stage, 1),
            stage_duration_days=stage_dur,
            next_stage=next_display,
            next_in_days=round(next_in_days, 1),
            stages=stages_display,
        )

        # ── CropHealth ────────────────────────────────────────────────────
        risk = cl.disease_risk_score
        health_status = "Healthy" if risk < 0.35 else ("Warning" if risk < 0.65 else "Risk")
        health = CropHealth(
            status=health_status,
            confidence=round(cl.disease_confidence * 100.0, 1),
            detail=f"Dominant: {cl.disease_label.title()}",
            scanned_ago="",   # set by MediaService when a scan timestamp is known
            disease_label=cl.disease_label,
            risk_score=risk,
        )

        # ── SensorReadings — one per SENSOR_META entry ────────────────────
        _values: dict[str, float] = {
            "indoor_temp":        cl.indoor_temp,
            "indoor_humidity":    cl.indoor_humidity,
            "soil_moisture":      cl.soil_moisture,
            "co2":                cl.co2,
            "light_intensity":    cl.light_intensity,
            "vpd":                cl.vpd,
            "leaf_wetness_proxy": cl.leaf_wetness_proxy,
            "disease_risk_score": cl.disease_risk_score,
        }
        sensors = [
            SensorReading(
                key=str(meta["key"]),
                label=str(meta["label"]),
                value=round(_values.get(str(meta["key"]), 0.0), 2),
                unit=str(meta["unit"]),
                icon_key=str(meta["icon_key"]),
                status=_sensor_status(
                    _values.get(str(meta["key"]), 0.0),
                    float(meta["range_min"]),
                    float(meta["range_max"]),
                ),
                range_min=float(meta["range_min"]),
                range_max=float(meta["range_max"]),
                optimal_range=str(meta["optimal_range"]),
                delta=round(cl.state_delta.get(str(meta["key"]), 0.0), 3),
            )
            for meta in SENSOR_META
        ]

        # ── GrowthIntelResponse (embedded in DTStateResponse.growth) ──────
        growth = GrowthIntelResponse(
            current_stage=gr.current_stage,
            current_stage_index=gr.current_stage_index,
            next_stage=gr.next_stage,
            hours_to_next_stage=round(gr.hours_to_next_stage, 1),
            days_to_next_stage=round(gr.days_to_next_stage, 1),
            transition_prob_24h=round(gr.transition_prob_24h, 3),
            vpd=round(gr.vpd, 2),
            growth_score=round(gr.growth_score, 2),
            timestamp=ts,
            stage_history=[],   # enriched by IntelligenceService
        )

        # ── ActuatorVisualState (compact 3D block) ─────────────────────────
        actuator_visual = ActuatorVisualState(
            on_off=dict(ac.on_off),
            levels=dict(ac.levels),
            mode=ac.mode,
        )

        # ── SceneContext (full 3D block) ──────────────────────────────────
        scene = SceneContext(
            actuator_states_on_off=dict(ac.on_off),
            actuator_levels=dict(ac.levels),
            current_growth_stage=eff_stage,
            next_growth_stage=eff_next_stage,
            current_timestamp=ts,
            time_of_day=time_of_day,
        )

        return DTStateResponse(
            crop=crop,
            health=health,
            sensors=sensors,
            growth=growth,
            timestamp=ts,
            time_of_day=time_of_day,
            current_growth_stage=eff_stage,
            next_growth_stage=eff_next_stage,
            actuator_visual_state=actuator_visual,
            mode=s.mode,
            scene_context=scene,
        )

    def as_actuator_state_response(self):  # noqa: ANN201
        """Project store actuator levels → ``ActuatorStateResponse``.

        Called by ``GET /api/actuators/state``.
        """
        from agritwin_gh.schemas.actuator_schemas import ActuatorEntry, ActuatorStateResponse
        from agritwin_gh.schemas.enums import (
            ACTUATOR_COLOR,
            ACTUATOR_ICON_KEY,
            ACTUATOR_IDS,
            ACTUATOR_LABEL,
        )

        with self._lock:
            ac = self._state.actuators

        entries = []
        for aid in ACTUATOR_IDS:
            level = ac.levels.get(aid, 0.0)
            on = ac.on_off.get(aid, False)
            status = "MANUAL" if ac.mode == "manual" else ("ON" if on else "OFF")
            entries.append(ActuatorEntry(
                id=aid,
                label=ACTUATOR_LABEL.get(aid, aid),
                icon_key=ACTUATOR_ICON_KEY.get(aid, "Settings"),
                status=status,
                active=on,
                level=round(level, 1),
                value_display=f"{round(level, 1)} %",
                color=ACTUATOR_COLOR.get(aid, "surface-highest"),
                on_off=on,
            ))
        return ActuatorStateResponse(actuators=entries)

    def as_weather_response(self):  # noqa: ANN201
        """Project store weather snapshot → ``WeatherResponse``.

        Called by ``GET /api/weather/current``.  The ``forecast`` list is
        empty until ``WeatherService`` populates ``WeatherSnapshot.forecast``.
        """
        from agritwin_gh.schemas.weather_schemas import (
            ForecastEntry,
            OutdoorCurrent,
            WeatherResponse,
            WeatherStatus,
        )

        with self._lock:
            w = self._state.weather
            ts = self._state.climate.timestamp or _now_iso()

        status_label = (
            "Warning"
            if (w.outdoor_humidity > 80 or w.outdoor_temp > 38 or w.outdoor_temp < 10)
            else "Healthy"
        )
        current = OutdoorCurrent(
            temp=round(w.outdoor_temp, 1),
            humidity=round(w.outdoor_humidity, 1),
            wind_speed=round(w.wind_speed, 1),
            wind_dir=w.wind_dir,
            pressure=round(w.pressure, 1),
            uv_index=round(w.uv_index, 1),
            dew_point=round(w.dew_point, 1),
            solar_rad=round(w.solar_rad, 1),
            condition=w.condition,
            visibility=round(w.visibility, 1),
        )
        forecast_entries = [
            ForecastEntry(
                time=fc.get("time", ""),
                high=fc.get("high", 0.0),
                low=fc.get("low", 0.0),
                humidity=fc.get("humidity", 0.0),
                condition=fc.get("condition", ""),
                icon_key=fc.get("icon_key", "Sun"),
            )
            for fc in w.forecast
            if isinstance(fc, dict)
        ]
        return WeatherResponse(
            status=WeatherStatus(
                status=status_label,
                condition=w.condition,
                detail="",
                forecast_note="",
            ),
            current=current,
            forecast=forecast_entries,
            forecast_24h=dict(w.forecast_24h),
            timestamp=ts,
        )

    def as_disease_risks_response(self):  # noqa: ANN201
        """Project store disease snapshot → ``DiseaseRisksResponse``.

        Called by ``GET /api/intelligence/disease``.  The ``pathogens`` list
        is empty until ``IntelligenceService`` enriches the snapshot via
        ``update_latest_state(disease=...)``.
        """
        from agritwin_gh.schemas.intelligence_schemas import (
            DiseaseRiskEntry,
            DiseaseRisksResponse,
        )

        with self._lock:
            ds = self._state.disease

        entries = [
            DiseaseRiskEntry(
                label=p.get("label", ""),
                name=p.get("name", ""),
                pathogen=p.get("pathogen", ""),
                risk_score=float(p.get("risk_score", 0.0)),
                risk_24h=float(p.get("risk_24h", 0.0)),
                probability=float(p.get("probability", 0.0)),
                severity=p.get("severity", "Low"),
            )
            for p in ds.pathogens
            if isinstance(p, dict)
        ]
        return DiseaseRisksResponse(
            composite_risk=round(ds.composite_risk, 3),
            dominant_label=ds.dominant_label,
            timestamp=ds.timestamp,
            pathogens=entries,
        )

    def as_growth_intel_response(self):  # noqa: ANN201
        """Project store growth snapshot → ``GrowthIntelResponse``.

        Called by ``GET /api/intelligence/growth``.  The ``stage_history``
        list is empty until ``IntelligenceService`` enriches the snapshot.
        """
        from agritwin_gh.schemas.intelligence_schemas import (
            GrowthIntelResponse,
            StageHistoryEntry,
        )

        with self._lock:
            gr = self._state.growth

        history = [
            StageHistoryEntry(
                stage=h.get("stage", ""),
                stage_key=h.get("stage_key", ""),
                days_used=float(h.get("days_used", 0.0)),
                days_target=float(h.get("days_target", 0.0)),
                complete=bool(h.get("complete", False)),
                status=h.get("status", "pending"),
                progress_pct=float(h.get("progress_pct", 0.0)),
            )
            for h in gr.stage_history
            if isinstance(h, dict)
        ]
        return GrowthIntelResponse(
            current_stage=gr.current_stage,
            current_stage_index=gr.current_stage_index,
            next_stage=gr.next_stage,
            hours_to_next_stage=round(gr.hours_to_next_stage, 1),
            days_to_next_stage=round(gr.days_to_next_stage, 1),
            transition_prob_24h=round(gr.transition_prob_24h, 3),
            vpd=round(gr.vpd, 2),
            growth_score=round(gr.growth_score, 2),
            timestamp=gr.timestamp,
            stage_history=history,
        )

    def as_resources_response(self):  # noqa: ANN201
        """Project store resource totals → ``ResourcesResponse``.

        Called by ``GET /api/resources/monthly``.

        INR cost rates default to Tamil Nadu industrial tariffs.
        ``ResourceService`` will override these using ``MPCConfig`` in Phase 2.
        """
        from agritwin_gh.schemas.resource_schemas import (
            ActuatorResourceEntry,
            MonthlyCost,
            ResourceEntry,
            ResourcesResponse,
        )

        with self._lock:
            r = self._state.resources
            ts = self._state.climate.timestamp or _now_iso()

        _ENERGY_RATE_INR_PER_KWH = 7.0       # Tamil Nadu industrial tariff
        _WATER_RATE_INR_PER_1000L = 4.0       # approximate
        energy_cost = r.energy_kwh_total * _ENERGY_RATE_INR_PER_KWH
        water_cost = (r.water_l_total / 1000.0) * _WATER_RATE_INR_PER_1000L
        month_label = _dt.datetime.now().strftime("%B %Y")

        # ── Per-actuator breakdown ─────────────────────────────────────
        _ACTUATOR_LABELS: dict[str, tuple[str, bool]] = {
            # key: (display label, has_water)
            "fan_speed":      ("Ventilation Fan",   False),
            "vent_opening":   ("Vent Opening",       False),
            "heater_output":  ("Heater",             False),
            "led_intensity":  ("LED Grow Lights",    False),
            "fogger_duty":    ("Fogger / Humidifier", False),
            "co2_valve_pct":  ("CO₂ Valve",          False),
            "irrigation_qty": ("Irrigation Pump",    True),
        }
        _WATER_PER_STEP_L = 10.0  # L per 5-min step at duty=1.0

        actuator_entries: list[ActuatorResourceEntry] = []
        for key, (label, has_water) in _ACTUATOR_LABELS.items():
            act_kwh = r.actuator_energy_kwh.get(key, 0.0)
            # Water: only irrigation; back-calculate from steps accumulated
            # (water_l accrued per step = duty * 10 L; energy used = duty * 0.15kW * (5/60)h)
            act_water_l = 0.0
            if has_water and act_kwh > 0.0:
                _rated_kw = 0.15
                _step_h = 5.0 / 60.0
                # infer total duty-steps from energy, then apply water rate
                act_water_l = (act_kwh / (_rated_kw * _step_h)) * _WATER_PER_STEP_L
            act_energy_cost = act_kwh * _ENERGY_RATE_INR_PER_KWH
            act_water_cost = (act_water_l / 1000.0) * _WATER_RATE_INR_PER_1000L
            actuator_entries.append(ActuatorResourceEntry(
                key=key,
                label=label,
                energy_kwh=round(act_kwh, 5),
                water_l=round(act_water_l, 2),
                cost_inr=round(act_energy_cost + act_water_cost, 4),
            ))

        return ResourcesResponse(
            resources=[
                ResourceEntry(label="Water",  used=round(r.water_l_total, 1),  unit="L"),
                ResourceEntry(label="Energy", used=round(r.energy_kwh_total, 3), unit="kWh"),
            ],
            cost=MonthlyCost(
                month=month_label,
                energy_inr=round(energy_cost, 2),
                water_inr=round(water_cost, 2),
                total_inr=round(energy_cost + water_cost, 2),
            ),
            actuators=actuator_entries,
            timestamp=ts,
        )

    def as_latest_media_response(self):  # noqa: ANN201
        """Project store image cache → ``LatestMediaResponse``.

        Called by ``GET /api/media/latest``.
        """
        from agritwin_gh.schemas.media_schemas import ImageEntry, LatestMediaResponse

        with self._lock:
            m = self._state.media

        return LatestMediaResponse(
            stage=ImageEntry(src=m.latest_stage_src, alt=m.latest_stage_alt),
            leaf=ImageEntry(src=m.latest_leaf_src, alt=m.latest_leaf_alt),
        )

    def as_stage_images_response(self):  # noqa: ANN201
        """Project store → ``StageImagesResponse``.

        Called by ``GET /api/media/stage-images``.
        """
        from agritwin_gh.schemas.media_schemas import StageImageEntry, StageImagesResponse

        with self._lock:
            images_raw = list(self._state.media.stage_images)

        images = [
            StageImageEntry(
                src=img.get("src", ""),
                alt=img.get("alt", ""),
                captured_ago=img.get("captured_ago", ""),
                stage=img.get("stage", ""),
                confidence=float(img.get("confidence", 0.0)),
                image_key=img.get("image_key", ""),
                bucket_name=img.get("bucket_name", ""),
            )
            for img in images_raw
            if isinstance(img, dict)
        ]
        return StageImagesResponse(images=images)

    def as_disease_images_response(self):  # noqa: ANN201
        """Project store → ``DiseaseImagesResponse``.

        Called by ``GET /api/media/disease-scans``.
        """
        from agritwin_gh.schemas.media_schemas import DiseaseImagesResponse, LeafScanEntry

        with self._lock:
            scans_raw = list(self._state.media.disease_scans)

        scans = [
            LeafScanEntry(
                src=s.get("src", ""),
                alt=s.get("alt", ""),
                captured_ago=s.get("captured_ago", ""),
                classification=s.get("classification", ""),
                risk=float(s.get("risk", 0.0)),
                confidence=float(s.get("confidence", 0.0)),
                image_key=s.get("image_key", ""),
                bucket_name=s.get("bucket_name", ""),
            )
            for s in scans_raw
            if isinstance(s, dict)
        ]
        return DiseaseImagesResponse(images=scans)

    def as_system_health_response(self):  # noqa: ANN201
        """Project store metadata → ``SystemHealthResponse``.

        Called by ``GET /api/system/health``.

        Individual row ``ok`` / ``status`` / ``status_code`` values are derived
        from live store fields.  In Phase 3 the DB and MinIO rows will be
        replaced by real connectivity probes run by ``DTService``.
        """
        from agritwin_gh.schemas.system_schemas import HealthRow, SystemHealthResponse

        with self._lock:
            cl = self._state.climate

        rows = [
            HealthRow(
                label="DT Loop",
                status="Active" if cl.step_index > 0 else "Idle",
                ok=cl.step_index > 0,
                status_code="ok" if cl.step_index > 0 else "warning",
                detail=f"Step {cl.step_index}",
            ),
            HealthRow(
                label="MPC Solver",
                status="Nominal" if cl.mpc_converged else "Pending",
                ok=cl.mpc_converged,
                status_code="ok" if cl.mpc_converged else "warning",
                detail="Converged" if cl.mpc_converged else "Awaiting first solve",
            ),
            HealthRow(label="Disease Model", status="Ready",  ok=True, status_code="ok"),
            HealthRow(label="Growth Model",  status="Ready",  ok=True, status_code="ok"),
            HealthRow(label="PostgreSQL",    status="Strong", ok=True, status_code="ok"),
            HealthRow(label="MinIO",         status="Active", ok=True, status_code="ok"),
        ]

        has_error = any(r.status_code == "error" for r in rows)
        has_warning = any(r.status_code == "warning" for r in rows)
        overall = "error" if has_error else ("warning" if has_warning else "ok")

        return SystemHealthResponse(
            rows=rows,
            overall=overall,
            timestamp=cl.timestamp or _now_iso(),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Internal utility helpers
# ─────────────────────────────────────────────────────────────────────────────


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string (ending in 'Z')."""
    return _dt.datetime.utcnow().isoformat() + "Z"


def _compute_time_of_day(ts: _dt.datetime | None) -> str:
    """Bucket a timestamp into a time-of-day label.

    Buckets (wall-clock hour)::

        morning   06:00 – 11:59
        afternoon 12:00 – 17:59
        evening   18:00 – 20:59
        night     21:00 – 05:59
    """
    if ts is None:
        ts = _dt.datetime.utcnow()
    h = ts.hour
    if 6 <= h < 12:
        return "morning"
    if 12 <= h < 18:
        return "afternoon"
    if 18 <= h < 21:
        return "evening"
    return "night"


def _sensor_status(value: float, lo: float, hi: float) -> str:
    """Return a ``SensorReading.status``-compatible string.

    Returns:
      * ``'ok'``       — value is within the optimal range [lo, hi].
      * ``'warning'``  — value is mildly out of range (within 20 % of span).
      * ``'critical'`` — value is significantly outside the optimal range.

    Matches the ``Literal["ok", "warning", "critical"]`` type on
    ``SensorReading.status``.
    """
    if lo <= value <= hi:
        return "ok"
    span = hi - lo
    margin = span * 0.2 if span > 0 else 1.0
    if (lo - margin) <= value <= (hi + margin):
        return "warning"
    return "critical"


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton factory
# ─────────────────────────────────────────────────────────────────────────────

_store_instance: RuntimeStore | None = None
_store_lock = threading.Lock()


def get_store() -> RuntimeStore:
    """Return the process-wide ``RuntimeStore`` singleton.

    Lazily creates the store on the first call; returns the same instance
    on all subsequent calls.  Thread-safe via ``_store_lock``.

    Usage::

        from agritwin_gh.core.runtime_store import get_store

        store = get_store()
        return store.as_dt_state_response()
    """
    global _store_instance
    if _store_instance is None:
        with _store_lock:
            if _store_instance is None:
                _store_instance = RuntimeStore()
    return _store_instance
