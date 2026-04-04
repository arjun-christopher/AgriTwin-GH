"""
DashboardService — aggregation facade for all dashboard API endpoints.

Responsibility
--------------
``DashboardService`` is a thin aggregation layer that maps ``RuntimeStore``
domain snapshots to the Pydantic response schemas consumed by the React
frontend.  It does **not** contain any physics or ML logic — all heavy
computation happens in the ``LoopService`` background loop and the
``IntelligenceService``; this service only reads and projects.

Why a separate service if RuntimeStore already has ``as_*_response()``?
-----------------------------------------------------------------------
* Provides a single stable import point for route handlers — the routes do
  not need to know which specific store method to call.
* Adds optional cross-domain enrichment (e.g. attaching growth intel into
  the DT state response).
* Makes unit testing easier: mock ``DashboardService`` only, not
  ``RuntimeStore``.

Endpoint coverage
-----------------
* ``GET /api/dt/state``         → ``get_dt_state()``
* ``GET /api/weather/current``  → ``get_weather()``
* ``GET /api/intelligence/disease`` → ``get_disease_risks()``
* ``GET /api/intelligence/growth``  → ``get_growth_intel()``
* ``GET /api/resources/monthly``    → ``get_resources()``
* ``GET /api/system/health``        → ``get_system_health()``  (delegates to SystemService)
* ``GET /api/actuators/state``      → ``get_actuator_state()``

Mapping conventions
-------------------
All MPC-field-to-frontend-ID translations follow
``schemas.enums.CONTROL_VAR_TO_ACTUATOR_ID`` (already applied inside
``RuntimeStore.as_*_response()``).  This service does not re-translate;
it relies on the store methods to produce correctly keyed output.

Helper utilities (module-level)
--------------------------------
``time_of_day_from_ts``
    Convert a ``datetime`` (or ISO string) to the canonical
    ``"morning" | "afternoon" | "evening" | "night"`` bucket.
    Exposed for use by route handlers and other services.

``growth_stage_display_name``
    Return the title-cased display name from a canonical lowercase label.

``actuator_frontend_id``
    Translate an MPC ``CONTROL_VARIABLES`` key to its frontend short ID.
"""

from __future__ import annotations

import datetime as _dt
import logging
from typing import Any

from agritwin_gh.core.runtime_store import RuntimeStore, get_store

logger = logging.getLogger("agritwin.services.dashboard")


# ═════════════════════════════════════════════════════════════════════════════
# Module-level helper utilities
# (imported by route handlers and other services — no RuntimeStore dependency)
# ═════════════════════════════════════════════════════════════════════════════


def time_of_day_from_ts(ts: _dt.datetime | str | None = None) -> str:
    """Bucket a timestamp into one of four time-of-day labels.

    Buckets follow ``FASTAPI_INTEGRATION_PLAN.md §8`` and the
    ``TimeOfDay`` literal in ``schemas.enums``:

    =========  ========================
    Bucket     Hours (local wall-clock)
    =========  ========================
    morning    06:00 – 11:59
    afternoon  12:00 – 17:59
    evening    18:00 – 20:59
    night      21:00 – 05:59
    =========  ========================

    Parameters
    ----------
    ts:
        A ``datetime`` object, an ISO-8601 string, or ``None`` (→ ``datetime.now()``).
    """
    if ts is None:
        dt = _dt.datetime.now()
    elif isinstance(ts, str):
        try:
            dt = _dt.datetime.fromisoformat(ts.rstrip("Z"))
        except ValueError:
            dt = _dt.datetime.now()
    else:
        dt = ts

    hour = dt.hour
    if 6 <= hour < 12:
        return "morning"
    if 12 <= hour < 18:
        return "afternoon"
    if 18 <= hour < 21:
        return "evening"
    return "night"


def growth_stage_display_name(canonical_label: str) -> str:
    """Return the title-cased display name for a canonical growth-stage label.

    Examples
    --------
    >>> growth_stage_display_name("flowering initiation")
    'Flowering Initiation'
    >>> growth_stage_display_name("early vegetative")
    'Early Vegetative'
    """
    from agritwin_gh.schemas.enums import GROWTH_STAGE_DISPLAY_NAME
    return GROWTH_STAGE_DISPLAY_NAME.get(canonical_label, canonical_label.title())


def actuator_frontend_id(mpc_key: str) -> str:
    """Translate an MPC ``CONTROL_VARIABLES`` key to its frontend short ID.

    Examples
    --------
    >>> actuator_frontend_id("fan_speed")
    'fan'
    >>> actuator_frontend_id("irrigation_qty")
    'irrigation'
    """
    from agritwin_gh.schemas.enums import CONTROL_VAR_TO_ACTUATOR_ID
    return CONTROL_VAR_TO_ACTUATOR_ID.get(mpc_key, mpc_key)


def actuator_mpc_key(frontend_id: str) -> str:
    """Translate a frontend actuator short ID back to its MPC ``CONTROL_VARIABLES`` key.

    Examples
    --------
    >>> actuator_mpc_key("fan")
    'fan_speed'
    >>> actuator_mpc_key("co2")
    'co2_valve_pct'
    """
    from agritwin_gh.schemas.enums import ACTUATOR_ID_TO_CONTROL_VAR
    return ACTUATOR_ID_TO_CONTROL_VAR.get(frontend_id, frontend_id)


def growth_stage_from_index(index: int) -> str:
    """Return the canonical growth-stage label for a zero-based index.

    Clamps out-of-range indices to the nearest valid value.
    """
    from agritwin_gh.mpc.constants import GROWTH_STAGES
    idx = max(0, min(index, len(GROWTH_STAGES) - 1))
    return GROWTH_STAGES[idx]


# ═════════════════════════════════════════════════════════════════════════════
# DashboardService
# ═════════════════════════════════════════════════════════════════════════════


class DashboardService:
    """Aggregation facade mapping RuntimeStore snapshots to API response schemas.

    All methods are synchronous and complete in < 1 ms because they only read
    from the in-memory ``RuntimeStore`` — no DB or solver calls.

    Parameters
    ----------
    store:
        Singleton ``RuntimeStore`` shared with the API layer.
        ``None`` → ``get_store()`` is called lazily.
    """

    def __init__(self, store: RuntimeStore | None = None) -> None:
        self._store: RuntimeStore = store or get_store()

    # ─────────────────────────────────────────────────────────────────────────
    # DT state  (GET /api/dt/state)
    # ─────────────────────────────────────────────────────────────────────────

    def get_dt_state(self) -> Any:
        """Return a ``DTStateResponse`` for ``GET /api/dt/state``.

        Delegates to ``RuntimeStore.as_dt_state_response()`` which assembles
        the full response including ``CropInfo``, ``CropHealth``, ``sensors``,
        ``GrowthIntelResponse``, 3D-ready ``SceneContext``, and
        ``ActuatorVisualState`` from the latest in-memory domain snapshots.
        """
        return self._store.as_dt_state_response()

    # ─────────────────────────────────────────────────────────────────────────
    # Actuator state  (GET /api/actuators/state)
    # ─────────────────────────────────────────────────────────────────────────

    def get_actuator_state(self) -> Any:
        """Return an ``ActuatorStateResponse`` for ``GET /api/actuators/state``.

        Contains one ``ActuatorEntry`` per canonical actuator ID (seven total),
        including ``on_off``, ``level``, ``status``, ``value_display``, and
        ``color`` fields consumed by the React ``ManualOverride`` card grid.
        """
        return self._store.as_actuator_state_response()

    # ─────────────────────────────────────────────────────────────────────────
    # Weather  (GET /api/weather/current)
    # ─────────────────────────────────────────────────────────────────────────

    def get_weather(self) -> Any:
        """Return a ``WeatherResponse`` for ``GET /api/weather/current``.

        Includes current outdoor conditions (temp, humidity, solar, wind) and
        a ``forecast`` list of ``ForecastEntry`` objects.  When the
        ``WeatherDisturbanceForecast`` ML model is unavailable, the forecast
        list contains synthetic diurnal entries populated by ``WeatherService``.
        """
        return self._store.as_weather_response()

    # ─────────────────────────────────────────────────────────────────────────
    # Disease intelligence  (GET /api/intelligence/disease)
    # ─────────────────────────────────────────────────────────────────────────

    def get_disease_risks(self) -> Any:
        """Return a ``DiseaseRisksResponse`` for ``GET /api/intelligence/disease``.

        Builds per-pathogen ``DiseaseRiskEntry`` objects from the
        ``DiseaseSnapshot.pathogens`` list.  When ``IntelligenceService`` has
        not yet enriched the pathogens (early startup), the response falls back
        to a single aggregate entry derived from the composite disease risk score.
        """
        return self._store.as_disease_risks_response()

    # ─────────────────────────────────────────────────────────────────────────
    # Growth intelligence  (GET /api/intelligence/growth)
    # ─────────────────────────────────────────────────────────────────────────

    def get_growth_intel(self) -> Any:
        """Return a ``GrowthIntelResponse`` for ``GET /api/intelligence/growth``.

        Includes current/next stage labels, hours/days to transition, VPD,
        growth score, transition probability, and a ``stage_history`` list
        with per-stage progress data.  Enriched by ``IntelligenceService``
        when the growth-progression model is available.
        """
        return self._store.as_growth_intel_response()

    # ─────────────────────────────────────────────────────────────────────────
    # Resources  (GET /api/resources/monthly)
    # ─────────────────────────────────────────────────────────────────────────

    def get_resources(self) -> Any:
        """Return a ``ResourcesResponse`` for ``GET /api/resources/monthly``.

        Applies Tamil Nadu electricity (₹7/kWh) and water (₹4/kL) tariff rates
        to the accumulated ``energy_kwh_total`` and ``water_l_total`` from the
        DT loop.  When ``ResourceService`` is wired with ``MPCConfig`` tariffs,
        those override the defaults.
        """
        return self._store.as_resources_response()

    # ─────────────────────────────────────────────────────────────────────────
    # System health  (GET /api/system/health)
    # ─────────────────────────────────────────────────────────────────────────

    def get_system_health(self) -> Any:
        """Return a ``SystemHealthResponse`` for ``GET /api/system/health``.

        Delegates to ``RuntimeStore.as_system_health_response()`` which derives
        the six health rows from live store fields without any DB or ML calls.
        For a richer diagnostics pass, route handlers should prefer
        ``SystemService.get_health()`` which also checks subsystem connectivity.
        """
        return self._store.as_system_health_response()

    # ─────────────────────────────────────────────────────────────────────────
    # Convenience accessor — full snapshot
    # ─────────────────────────────────────────────────────────────────────────

    def get_all(self) -> dict[str, Any]:
        """Return all dashboard payloads in a single dict.

        Useful for testing and for a hypothetical ``GET /api/dashboard/all``
        bulk endpoint.  Keys match the endpoint path segments.

        .. warning::
            This calls all seven ``as_*_response()`` methods; avoid calling it
            on every request in production.
        """
        return {
            "dt_state":      self.get_dt_state(),
            "actuators":     self.get_actuator_state(),
            "weather":       self.get_weather(),
            "disease_risks": self.get_disease_risks(),
            "growth_intel":  self.get_growth_intel(),
            "resources":     self.get_resources(),
            "system_health": self.get_system_health(),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # 3D payload accessor
    # ─────────────────────────────────────────────────────────────────────────

    def get_3d_payload(self) -> Any:
        """Return a compact ``ThreeDPayload`` for the future 3D WebSocket layer.

        Contains timestamp, time_of_day, growth_stage, next_growth_stage, and
        actuator on/off+level list — everything a 3D scene manager needs to
        animate the digital twin without any further API calls.

        Zero DB queries; derived entirely from the in-memory store.
        """
        return self._store.get_3d_payload()
