"""
MonthlySnapshotService — per-month crop telemetry persisted to PostgreSQL.

Responsibility
--------------
* Maintain an in-memory accumulator (``MonthlyAccumulator``) that collects
  sensor / actuator / MPC data from every DT loop step.
* Detect calendar-month rollovers and write a completed ``monthly_snapshots``
  row to the database.
* Manage ``crop_cycles`` rows — one per main.py invocation.
* Provide ``seed_mock_data()`` for demo seeding and ``show_table()`` for
  demo display without requiring a separate tool.

Wiring
------
``LoopService.run_one_step()`` calls ``monthly_svc.ingest_step(result, ...)``
after each step when ``AGRITWIN_MONTHLY_DB=1`` is set.

Environment variable
--------------------
``AGRITWIN_MONTHLY_DB``
    Set to ``1`` to enable automatic monthly snapshot persistence.
    Any other value (or missing) disables the feature — the DT loop runs
    normally but no monthly rows are written.
    Default: ``0`` (disabled).

Schema
------
See ``database/schema/monthly_snapshots.sql`` for the exact DDL.
See ``docs/MONTHLY_SNAPSHOT_REFERENCE.md`` for full field descriptions.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("agritwin.services.monthly_snapshot")

# ── Constants ─────────────────────────────────────────────────────────────────

_ENERGY_RATE_INR_PER_KWH: float = 7.0       # Tamil Nadu industrial tariff
_WATER_RATE_INR_PER_L:    float = 4.0 / 1000.0  # ₹4/kL → per litre

_STAGE_LABELS = [
    "seedling",
    "early vegetative",
    "flowering initiation",
    "flowering",
    "unripe",
    "ripe",
]


def _stage_idx(label: str) -> int:
    """Return 0-based index for a stage label (case-insensitive, partial match)."""
    lbl = label.lower().strip()
    for i, s in enumerate(_STAGE_LABELS):
        if lbl == s or lbl.startswith(s.split()[0]):
            return i
    return 0


# Feature flag — read once at import time so no repeated env lookups in hot path.
def monthly_db_enabled() -> bool:
    """Return True when AGRITWIN_MONTHLY_DB=1 is set in the environment."""
    return os.environ.get("AGRITWIN_MONTHLY_DB", "0").strip() == "1"


# ─────────────────────────────────────────────────────────────────────────────
# In-memory accumulator
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MonthlyAccumulator:
    """Collects data for the *current* billing month.

    Reset to a fresh instance (via ``_reset_accumulator()``) at the start
    of each new calendar month.
    """

    billing_month: str = ""          # "YYYY-MM" of the month being accumulated
    month_start_ts: _dt.datetime | None = None
    month_end_ts:   _dt.datetime | None = None

    # Step counters
    total_steps:           int = 0
    mpc_solve_count:       int = 0
    image_classify_count:  int = 0

    # Growth stage tracking
    stage_at_start:     str = "seedling"
    stage_idx_start:    int = 0
    stage_at_end:       str = "seedling"
    stage_idx_end:      int = 0
    stage_transitions:  int = 0
    stage_transition_log: list[dict] = field(default_factory=list)
    _prev_stage_idx:    int = -1     # internal, not persisted

    # Sensor sums (for computing averages at flush time)
    _sum_temp:      float = 0.0
    _sum_humidity:  float = 0.0
    _sum_co2:       float = 0.0
    _sum_soil:      float = 0.0
    _sum_light:     float = 0.0
    _sum_vpd:       float = 0.0
    _sum_leaf:      float = 0.0
    _sum_risk:      float = 0.0

    # Sensor extremes
    min_temp:    float | None = None
    max_temp:    float | None = None
    min_humidity: float | None = None
    max_humidity: float | None = None
    min_co2:     float | None = None
    max_co2:     float | None = None
    min_soil:    float | None = None
    max_soil:    float | None = None
    max_risk:    float | None = None

    # Setpoint error sums
    _sum_err_temp:     float = 0.0
    _sum_err_humidity: float = 0.0
    _sum_err_soil:     float = 0.0
    _sum_err_co2:      float = 0.0
    _sum_err_light:    float = 0.0
    _sum_err_vpd:      float = 0.0
    _err_count:        int = 0

    # Resource totals
    total_energy_kwh: float = 0.0
    total_water_l:    float = 0.0

    # Per-actuator energy kWh
    act_fan_speed_kwh:    float = 0.0
    act_vent_opening_kwh: float = 0.0
    act_heater_kwh:       float = 0.0
    act_led_kwh:          float = 0.0
    act_fogger_kwh:       float = 0.0
    act_co2_valve_kwh:    float = 0.0
    act_irrigation_kwh:   float = 0.0
    act_irrigation_water_l: float = 0.0

    # MPC solve accumulation
    _sum_mpc_cost: float = 0.0
    mpc_converge_count: int = 0
    _sum_mpc_fan:   float = 0.0
    _sum_mpc_vent:  float = 0.0
    _sum_mpc_heat:  float = 0.0
    _sum_mpc_led:   float = 0.0
    _sum_mpc_co2v:  float = 0.0
    _sum_mpc_fog:   float = 0.0
    _sum_mpc_irrig: float = 0.0

    # Weather sums
    _sum_ext_temp:    float = 0.0
    _sum_ext_humidity: float = 0.0
    _sum_solar:       float = 0.0
    _sum_wind:        float = 0.0
    _weather_count:   int = 0

    # Disease peak severities
    peak_early_blight:   float = 0.0
    peak_late_blight:    float = 0.0
    peak_leaf_mold:      float = 0.0
    peak_powdery_mildew: float = 0.0
    peak_spider_mites:   float = 0.0
    disease_alert_steps: int = 0

    # AI model run counts
    growth_lstm_runs:      int = 0
    disease_lstm_runs:     int = 0
    weather_forecast_runs: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Service
# ─────────────────────────────────────────────────────────────────────────────

class MonthlySnapshotService:
    """Ingests one DT step at a time and flushes completed months to the DB.

    Parameters
    ----------
    session_factory:
        Zero-arg callable that returns a new SQLAlchemy ``Session``.  Pass
        ``None`` to run in dry-run mode (accumulates but never writes).
    cycle_label:
        Human-readable label for the current crop cycle, e.g. ``"cycle-001"``.
        Must be unique across all runs.  Typically generated by the caller as
        ``f"cycle-{next_id:03d}"``.
    """

    def __init__(
        self,
        session_factory: Any | None = None,
        cycle_label: str = "cycle-001",
    ) -> None:
        self._session_factory = session_factory
        self._cycle_label = cycle_label
        self._cycle_id: int | None = None          # DB id from crop_cycles table
        self._acc = MonthlyAccumulator()
        self._lock = threading.Lock()
        self._initialised = False

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def initialise(self) -> None:
        """Create (or reuse) the crop_cycles row for this run.

        Must be called once before ``ingest_step()``.  Safe to call multiple
        times — idempotent after first call.
        """
        if self._initialised:
            return
        with self._lock:
            if self._initialised:
                return
            if self._session_factory is not None:
                try:
                    self._cycle_id = self._ensure_cycle_row()
                    logger.info(
                        "MonthlySnapshotService: cycle_id=%d label=%s",
                        self._cycle_id,
                        self._cycle_label,
                    )
                except Exception:
                    logger.exception("MonthlySnapshotService: failed to create cycle row")
            self._initialised = True

    def ingest_step(
        self,
        result: Any,                # DTLoopStepResult
        actuator_energy: dict[str, float],
        water_l: float,
        energy_kwh: float,
        cadence_info: dict[str, Any] | None = None,
        disease_info: dict[str, Any] | None = None,
        weather_info: dict[str, Any] | None = None,
    ) -> None:
        """Accumulate one DT step into the current month bucket.

        Parameters
        ----------
        result          : DTLoopStepResult from loop_service
        actuator_energy : {actuator_key: kwh} dict computed in loop_service
        water_l         : irrigation water this step (litres)
        energy_kwh      : total energy this step (kWh)
        cadence_info    : result.cadence_info dict (has image_due, growth_due etc.)
        disease_info    : model_disease_result dict — nested format:
                          ``{"early_blight": {"severity_24h": 0.07, ...}, ...}``
        weather_info    : weather disturbance dict keys: temp_external,
                          humidity_external, solar_radiation, windspeed
        """
        if not self._initialised:
            self.initialise()

        now_ts = result.timestamp or _dt.datetime.utcnow()
        current_month = now_ts.strftime("%Y-%m")

        with self._lock:
            # ── Month rollover detection ───────────────────────────────────
            if self._acc.billing_month and self._acc.billing_month != current_month:
                # Month has changed → flush the completed month
                self._flush_month()
                self._acc = MonthlyAccumulator()

            # ── Initialise new month bucket ────────────────────────────────
            if not self._acc.billing_month:
                self._acc.billing_month = current_month
                self._acc.month_start_ts = now_ts
                _init_cs = result.current_state
                _init_idx = max(0, min(int(_init_cs.growth_stage_index), len(_STAGE_LABELS) - 1))
                _init_lbl = (
                    getattr(_init_cs, "growth_stage_label", None)
                    or _STAGE_LABELS[_init_idx]
                )
                self._acc.stage_at_start = _init_lbl
                self._acc.stage_idx_start = _init_idx
                self._acc._prev_stage_idx = _init_idx

            self._acc.month_end_ts = now_ts
            self._acc.total_steps += 1

            # ── Sensor ingestion (from current_state = INPUT line) ─────────
            cs = result.current_state
            self._acc._sum_temp     += cs.indoor_temp
            self._acc._sum_humidity += cs.indoor_humidity
            self._acc._sum_co2      += cs.co2
            self._acc._sum_soil     += cs.soil_moisture
            self._acc._sum_light    += cs.light_intensity
            self._acc._sum_vpd      += cs.vpd
            self._acc._sum_leaf     += cs.leaf_wetness_proxy
            self._acc._sum_risk     += cs.disease_risk_score

            def _minmax(cur_min, cur_max, val):
                return (
                    (val if cur_min is None else min(cur_min, val)),
                    (val if cur_max is None else max(cur_max, val)),
                )

            self._acc.min_temp,    self._acc.max_temp    = _minmax(self._acc.min_temp,    self._acc.max_temp,    cs.indoor_temp)
            self._acc.min_humidity,self._acc.max_humidity= _minmax(self._acc.min_humidity,self._acc.max_humidity,cs.indoor_humidity)
            self._acc.min_co2,     self._acc.max_co2     = _minmax(self._acc.min_co2,     self._acc.max_co2,     cs.co2)
            self._acc.min_soil,    self._acc.max_soil    = _minmax(self._acc.min_soil,     self._acc.max_soil,    cs.soil_moisture)
            self._acc.max_risk     = max(self._acc.max_risk or 0.0, cs.disease_risk_score)

            # Disease alert threshold (risk > 0.3)
            if cs.disease_risk_score > 0.3:
                self._acc.disease_alert_steps += 1

            # ── Stage transition detection ─────────────────────────────────
            cur_stage_idx = cs.growth_stage_index
            stage_lbl = (
                getattr(cs, "growth_stage_label", None)
                or _STAGE_LABELS[max(0, min(int(cur_stage_idx), len(_STAGE_LABELS) - 1))]
            )
            if (
                self._acc._prev_stage_idx >= 0
                and cur_stage_idx != self._acc._prev_stage_idx
            ):
                self._acc.stage_transitions += 1
                self._acc.stage_transition_log.append({
                    "from_stage": _STAGE_LABELS[self._acc._prev_stage_idx] if self._acc._prev_stage_idx < len(_STAGE_LABELS) else str(self._acc._prev_stage_idx),
                    "to_stage":   stage_lbl,
                    "step":       result.step_index,
                    "ts":         now_ts.isoformat(),
                })
            self._acc._prev_stage_idx = cur_stage_idx
            self._acc.stage_at_end  = stage_lbl
            self._acc.stage_idx_end = cur_stage_idx

            # ── Setpoint errors ────────────────────────────────────────────
            diag = getattr(result, "diagnostics", None)
            if diag is not None:
                try:
                    errs = (
                        getattr(diag, "setpoint_error", None)
                        or getattr(diag, "setpoint_errors", None)
                        or {}
                    )
                    if errs:
                        self._acc._sum_err_temp     += errs.get("indoor_temp", 0.0)
                        self._acc._sum_err_humidity += errs.get("indoor_humidity", 0.0)
                        self._acc._sum_err_soil     += errs.get("soil_moisture", 0.0)
                        self._acc._sum_err_co2      += errs.get("co2", 0.0)
                        self._acc._sum_err_light    += errs.get("light_intensity", 0.0)
                        self._acc._sum_err_vpd      += errs.get("vpd", 0.0)
                        self._acc._err_count += 1
                except Exception:
                    pass

            # ── Resource accounting ────────────────────────────────────────
            self._acc.total_energy_kwh  += energy_kwh
            self._acc.total_water_l     += water_l
            self._acc.act_fan_speed_kwh    += actuator_energy.get("fan_speed",      0.0)
            self._acc.act_vent_opening_kwh += actuator_energy.get("vent_opening",   0.0)
            self._acc.act_heater_kwh       += actuator_energy.get("heater_output",  0.0)
            self._acc.act_led_kwh          += actuator_energy.get("led_intensity",  0.0)
            self._acc.act_fogger_kwh       += actuator_energy.get("fogger_duty",    0.0)
            self._acc.act_co2_valve_kwh    += actuator_energy.get("co2_valve_pct",  0.0)
            self._acc.act_irrigation_kwh   += actuator_energy.get("irrigation_qty", 0.0)
            self._acc.act_irrigation_water_l += water_l

            # ── MPC solve data ─────────────────────────────────────────────
            _mpc_sol = getattr(result, "mpc_solution", None)
            if result.mpc_ran_this_step and _mpc_sol is not None:
                sol = _mpc_sol
                self._acc.mpc_solve_count  += 1
                self._acc._sum_mpc_cost    += (result.mpc_cost or 0.0)
                self._acc.mpc_converge_count += 1 if getattr(sol, "converged", False) else 0
                acts = result.action_applied
                self._acc._sum_mpc_fan   += getattr(acts, "fan_speed",      0.0)
                self._acc._sum_mpc_vent  += getattr(acts, "vent_opening",   0.0)
                self._acc._sum_mpc_heat  += getattr(acts, "heater_output",  0.0)
                self._acc._sum_mpc_led   += getattr(acts, "led_intensity",  0.0)
                self._acc._sum_mpc_co2v  += getattr(acts, "co2_valve_pct",  0.0)
                self._acc._sum_mpc_fog   += getattr(acts, "fogger_duty",    0.0)
                self._acc._sum_mpc_irrig += getattr(acts, "irrigation_qty", 0.0)

            # ── Image classify count ───────────────────────────────────────
            if result.image_refresh_this_step:
                self._acc.image_classify_count += 1

            # ── Weather ingestion ─────────────────────────────────────────
            wi = weather_info or {}
            if wi:
                self._acc._sum_ext_temp     += float(wi.get("temp_external",   0.0))
                self._acc._sum_ext_humidity += float(wi.get("humidity_external", 0.0))
                self._acc._sum_solar        += float(wi.get("solar_radiation",  0.0))
                self._acc._sum_wind         += float(wi.get("windspeed",        0.0))
                self._acc._weather_count    += 1

            # ── Disease severity peaks ─────────────────────────────────────
            # disease_info is the model_disease_result dict — nested format:
            # {"early_blight": {"severity_24h": 0.07, ...}, ...}
            di = disease_info or {}
            if di:
                def _dsev(key: str) -> float:
                    return float(di.get(key, {}).get("severity_24h", 0.0))
                self._acc.peak_early_blight   = max(self._acc.peak_early_blight,   _dsev("early_blight"))
                self._acc.peak_late_blight    = max(self._acc.peak_late_blight,    _dsev("late_blight"))
                self._acc.peak_leaf_mold      = max(self._acc.peak_leaf_mold,      _dsev("leaf_mold"))
                self._acc.peak_powdery_mildew = max(self._acc.peak_powdery_mildew, _dsev("powdery_mildew"))
                self._acc.peak_spider_mites   = max(self._acc.peak_spider_mites,   _dsev("spider_mites"))
                self._acc.disease_lstm_runs   += 1

            # ── AI run counts from cadence_info ───────────────────────────
            ci = cadence_info or {}
            if ci.get("growth_due"):
                self._acc.growth_lstm_runs += 1
            if ci.get("weather_forecast_due"):
                self._acc.weather_forecast_runs += 1

    def flush_current_month(self) -> None:
        """Force-flush the current accumulator (e.g. on shutdown or test)."""
        with self._lock:
            if self._acc.total_steps > 0:
                self._flush_month()
                self._acc = MonthlyAccumulator()

    def mark_cycle_complete(self) -> None:
        """Mark the crop_cycles row as completed (called when stage = ripe and cycle ends)."""
        if self._cycle_id is None or self._session_factory is None:
            return
        try:
            session = self._session_factory()
            try:
                from sqlalchemy import text
                session.execute(text(
                    "UPDATE crop_cycles SET completed = TRUE, ended_at = :now WHERE id = :cid"
                ), {"cid": self._cycle_id, "now": _dt.datetime.utcnow().isoformat()})
                session.commit()
                logger.info("MonthlySnapshotService: cycle %d marked complete", self._cycle_id)
            finally:
                session.close()
        except Exception:
            logger.exception("MonthlySnapshotService: failed to mark cycle complete")

    # ─────────────────────────────────────────────────────────────────────────
    # CLI / demo helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def show_table(session_factory: Any, limit: int = 20) -> None:
        """Print monthly_snapshots rows to stdout in a readable format.

        Designed for demo use::

            python -m agritwin_gh.services.monthly_snapshot_service

        Parameters
        ----------
        session_factory : zero-arg callable returning a SQLAlchemy Session
        limit           : max rows to display
        """
        try:
            from sqlalchemy import text
            session = session_factory()
            rows = session.execute(text(
                """
                SELECT
                    ms.id,
                    cc.cycle_label,
                    ms.billing_month,
                    ms.month_label,
                    ms.total_steps,
                    ms.mpc_solve_count,
                    ms.stage_at_month_start,
                    ms.stage_at_month_end,
                    ms.stage_transitions,
                    ROUND(ms.avg_indoor_temp::numeric, 2)         AS avg_temp,
                    ROUND(ms.avg_indoor_humidity::numeric, 2)     AS avg_rh,
                    ROUND(ms.total_energy_kwh::numeric, 4)        AS energy_kwh,
                    ROUND(ms.total_water_l::numeric, 2)           AS water_l,
                    ROUND(ms.total_cost_inr::numeric, 2)          AS cost_inr,
                    ms.snapshot_recorded_at
                FROM monthly_snapshots ms
                JOIN crop_cycles cc ON cc.id = ms.cycle_id
                ORDER BY ms.snapshot_recorded_at DESC
                LIMIT :lim
                """
            ), {"lim": limit}).fetchall()
            session.close()

            if not rows:
                print("[monthly_snapshots] No rows found.")
                return

            # Header
            header = (
                f"{'ID':>4}  {'Cycle':>10}  {'Month':>7}  {'Label':>12}  "
                f"{'Steps':>5}  {'MPC':>4}  {'Stage Start':>20}  {'Stage End':>20}  "
                f"{'Trans':>5}  {'AvgT':>6}  {'AvgRH':>6}  "
                f"{'kWh':>8}  {'Water_L':>8}  {'₹Cost':>8}"
            )
            print("\n" + "═" * len(header))
            print("  AgriTwin-GH — Monthly Snapshots")
            print("═" * len(header))
            print(header)
            print("─" * len(header))
            for r in rows:
                print(
                    f"{r[0]:>4}  {r[1]:>10}  {r[2]:>7}  {r[3]:>12}  "
                    f"{r[4]:>5}  {r[5]:>4}  {str(r[6]):>20}  {str(r[7]):>20}  "
                    f"{r[8]:>5}  {str(r[9]):>6}  {str(r[10]):>6}  "
                    f"{str(r[11]):>8}  {str(r[12]):>8}  {str(r[13]):>8}"
                )
            print("─" * len(header))
            print(f"  {len(rows)} row(s) shown  (LIMIT {limit})")
            print("═" * len(header) + "\n")
        except Exception as exc:
            print(f"[show_table] Error: {exc}")

    @staticmethod
    def seed_mock_data(session_factory: Any) -> None:
        """Insert 2 complete crop cycles with 3 monthly snapshots for demo.

        Idempotent — uses ON CONFLICT DO NOTHING so safe to run multiple times.

        Usage::

            python scripts/seed_monthly_mock.py
        """
        from sqlalchemy import text

        _ENERGY_RATE = 7.0
        _WATER_RATE  = 4.0 / 1000.0

        session = session_factory()
        try:
            # ── Cycle 1 — completed (Jan-Feb 2026, seedling → ripe) ───────
            session.execute(text("""
                INSERT INTO crop_cycles (cycle_label, started_at, ended_at, completed, crop_type, notes)
                VALUES ('cycle-001', '2026-01-01 06:00:00', '2026-02-28 23:59:59', TRUE, 'tomato',
                        'First demo cycle – full seedling-to-ripe run.')
                ON CONFLICT (cycle_label) DO NOTHING
            """))
            session.execute(text("SELECT id FROM crop_cycles WHERE cycle_label = 'cycle-001'"))
            row = session.execute(text("SELECT id FROM crop_cycles WHERE cycle_label = 'cycle-001'")).fetchone()
            cycle1_id = row[0] if row else None

            if cycle1_id:
                # January 2026 — seedling & early-veg stages
                _e1 = 38.45; _w1 = 1240.0; _c1 = _e1 * _ENERGY_RATE + _w1 * _WATER_RATE
                session.execute(text("""
                    INSERT INTO monthly_snapshots (
                        cycle_id, billing_month, month_label,
                        month_start_ts, month_end_ts,
                        total_steps, mpc_solve_count, image_classify_count,
                        stage_at_month_start, stage_at_month_end,
                        stage_idx_start, stage_idx_end, stage_transitions,
                        stage_transition_log,
                        avg_indoor_temp, avg_indoor_humidity, avg_co2,
                        avg_soil_moisture, avg_light_intensity, avg_vpd,
                        avg_leaf_wetness, avg_disease_risk_score,
                        min_indoor_temp, max_indoor_temp,
                        min_indoor_humidity, max_indoor_humidity,
                        min_co2, max_co2, min_soil_moisture, max_soil_moisture,
                        max_disease_risk_score,
                        avg_setpt_err_temp, avg_setpt_err_humidity,
                        avg_setpt_err_soil, avg_setpt_err_co2,
                        avg_setpt_err_light, avg_setpt_err_vpd,
                        total_energy_kwh, total_water_l, total_cost_inr,
                        act_fan_speed_kwh, act_vent_opening_kwh, act_heater_output_kwh,
                        act_led_intensity_kwh, act_fogger_duty_kwh, act_co2_valve_kwh,
                        act_irrigation_kwh,
                        act_fan_speed_inr, act_vent_opening_inr, act_heater_output_inr,
                        act_led_intensity_inr, act_fogger_duty_inr, act_co2_valve_inr,
                        act_irrigation_inr, act_irrigation_water_l,
                        avg_mpc_cost, mpc_converge_count,
                        avg_mpc_fan, avg_mpc_vent, avg_mpc_heat,
                        avg_mpc_led, avg_mpc_co2v, avg_mpc_fog, avg_mpc_irrig,
                        avg_ext_temp, avg_ext_humidity, avg_solar_radiation, avg_wind_speed,
                        peak_early_blight_sev, peak_late_blight_sev,
                        peak_leaf_mold_sev, peak_powdery_mildew_sev, peak_spider_mites_sev,
                        disease_alert_steps,
                        growth_lstm_runs, disease_lstm_runs, weather_forecast_runs
                    ) VALUES (
                        :cid, '2026-01', 'January 2026',
                        '2026-01-01 06:00:00', '2026-01-31 23:55:00',
                        8928, 446, 93,
                        'seedling', 'early vegetative',
                        0, 1, 1,
                        :stl1,
                        26.3, 71.8, 520.4, 62.1, 8840.0, 0.89, 0.18, 0.12,
                        22.1, 31.4, 64.2, 81.5, 412.0, 664.8, 54.0, 72.0,
                        0.45,
                        2.8, -3.1, -7.2, -45.6, -1840.0, 0.22,
                        :e1, :w1, :c1,
                        14.16, 1.49, 5.58, 4.46, 0.74, 0.19, 11.84,
                        99.12, 10.43, 39.06, 31.22, 5.18, 1.33, 82.88,
                        :w1, 62.4, 446,
                        0.22, 0.14, 0.09, 0.18, 0.12, 0.00, 0.48,
                        24.2, 74.1, 280.5, 14.8,
                        0.09, 0.07, 0.11, 0.31, 0.04, 22,
                        446, 446, 446
                    )
                    ON CONFLICT (cycle_id, billing_month) DO NOTHING
                """), {"cid": cycle1_id, "e1": round(_e1,4), "w1": _w1, "c1": round(_c1,2),
                       "stl1": '[{"from_stage":"seedling","to_stage":"early vegetative","step":4464,"ts":"2026-01-16T12:30:00"}]'})

                # February 2026 — flowering initiation → ripe
                _e2 = 29.82; _w2 = 720.0; _c2 = _e2 * _ENERGY_RATE + _w2 * _WATER_RATE
                session.execute(text("""
                    INSERT INTO monthly_snapshots (
                        cycle_id, billing_month, month_label,
                        month_start_ts, month_end_ts,
                        total_steps, mpc_solve_count, image_classify_count,
                        stage_at_month_start, stage_at_month_end,
                        stage_idx_start, stage_idx_end, stage_transitions,
                        stage_transition_log,
                        avg_indoor_temp, avg_indoor_humidity, avg_co2,
                        avg_soil_moisture, avg_light_intensity, avg_vpd,
                        avg_leaf_wetness, avg_disease_risk_score,
                        min_indoor_temp, max_indoor_temp,
                        min_indoor_humidity, max_indoor_humidity,
                        min_co2, max_co2, min_soil_moisture, max_soil_moisture,
                        max_disease_risk_score,
                        avg_setpt_err_temp, avg_setpt_err_humidity,
                        avg_setpt_err_soil, avg_setpt_err_co2,
                        avg_setpt_err_light, avg_setpt_err_vpd,
                        total_energy_kwh, total_water_l, total_cost_inr,
                        act_fan_speed_kwh, act_vent_opening_kwh, act_heater_output_kwh,
                        act_led_intensity_kwh, act_fogger_duty_kwh, act_co2_valve_kwh,
                        act_irrigation_kwh,
                        act_fan_speed_inr, act_vent_opening_inr, act_heater_output_inr,
                        act_led_intensity_inr, act_fogger_duty_inr, act_co2_valve_inr,
                        act_irrigation_inr, act_irrigation_water_l,
                        avg_mpc_cost, mpc_converge_count,
                        avg_mpc_fan, avg_mpc_vent, avg_mpc_heat,
                        avg_mpc_led, avg_mpc_co2v, avg_mpc_fog, avg_mpc_irrig,
                        avg_ext_temp, avg_ext_humidity, avg_solar_radiation, avg_wind_speed,
                        peak_early_blight_sev, peak_late_blight_sev,
                        peak_leaf_mold_sev, peak_powdery_mildew_sev, peak_spider_mites_sev,
                        disease_alert_steps,
                        growth_lstm_runs, disease_lstm_runs, weather_forecast_runs
                    ) VALUES (
                        :cid, '2026-02', 'February 2026',
                        '2026-02-01 00:00:00', '2026-02-28 23:55:00',
                        8064, 403, 84,
                        'flowering initiation', 'ripe',
                        2, 5, 3,
                        :stl2,
                        24.8, 68.4, 612.3, 65.0, 10200.0, 0.97, 0.15, 0.10,
                        21.0, 29.8, 61.0, 78.2, 480.0, 742.0, 58.0, 75.0,
                        0.38,
                        1.9, -1.8, -4.4, -28.2, -1200.0, 0.18,
                        :e2, :w2, :c2,
                        10.84, 1.12, 4.18, 3.36, 0.56, 0.14, 9.62,
                        75.88, 7.84, 29.26, 23.52, 3.92, 0.98, 67.34,
                        :w2, 48.2, 403,
                        0.18, 0.11, 0.07, 0.15, 0.09, 0.02, 0.38,
                        25.4, 72.8, 310.2, 16.2,
                        0.07, 0.06, 0.08, 0.22, 0.03, 14,
                        403, 403, 403
                    )
                    ON CONFLICT (cycle_id, billing_month) DO NOTHING
                """), {"cid": cycle1_id, "e2": round(_e2,4), "w2": _w2, "c2": round(_c2,2),
                       "stl2": '[{"from_stage":"flowering initiation","to_stage":"flowering","step":2688,"ts":"2026-02-10T12:00:00"},{"from_stage":"flowering","to_stage":"unripe","step":5376,"ts":"2026-02-20T00:00:00"},{"from_stage":"unripe","to_stage":"ripe","step":7392,"ts":"2026-02-27T00:00:00"}]'})

            # ── Cycle 2 — ongoing (started March 2026) ───────────────────
            session.execute(text("""
                INSERT INTO crop_cycles (cycle_label, started_at, completed, crop_type, notes)
                VALUES ('cycle-002', '2026-03-01 06:00:00', FALSE, 'tomato',
                        'Second demo cycle – currently running.')
                ON CONFLICT (cycle_label) DO NOTHING
            """))
            row2 = session.execute(text("SELECT id FROM crop_cycles WHERE cycle_label = 'cycle-002'")).fetchone()
            cycle2_id = row2[0] if row2 else None

            if cycle2_id:
                # March 2026 — seedling & early-veg stages
                _e3 = 41.10; _w3 = 1380.0; _c3 = _e3 * _ENERGY_RATE + _w3 * _WATER_RATE
                session.execute(text("""
                    INSERT INTO monthly_snapshots (
                        cycle_id, billing_month, month_label,
                        month_start_ts, month_end_ts,
                        total_steps, mpc_solve_count, image_classify_count,
                        stage_at_month_start, stage_at_month_end,
                        stage_idx_start, stage_idx_end, stage_transitions,
                        stage_transition_log,
                        avg_indoor_temp, avg_indoor_humidity, avg_co2,
                        avg_soil_moisture, avg_light_intensity, avg_vpd,
                        avg_leaf_wetness, avg_disease_risk_score,
                        min_indoor_temp, max_indoor_temp,
                        min_indoor_humidity, max_indoor_humidity,
                        min_co2, max_co2, min_soil_moisture, max_soil_moisture,
                        max_disease_risk_score,
                        avg_setpt_err_temp, avg_setpt_err_humidity,
                        avg_setpt_err_soil, avg_setpt_err_co2,
                        avg_setpt_err_light, avg_setpt_err_vpd,
                        total_energy_kwh, total_water_l, total_cost_inr,
                        act_fan_speed_kwh, act_vent_opening_kwh, act_heater_output_kwh,
                        act_led_intensity_kwh, act_fogger_duty_kwh, act_co2_valve_kwh,
                        act_irrigation_kwh,
                        act_fan_speed_inr, act_vent_opening_inr, act_heater_output_inr,
                        act_led_intensity_inr, act_fogger_duty_inr, act_co2_valve_inr,
                        act_irrigation_inr, act_irrigation_water_l,
                        avg_mpc_cost, mpc_converge_count,
                        avg_mpc_fan, avg_mpc_vent, avg_mpc_heat,
                        avg_mpc_led, avg_mpc_co2v, avg_mpc_fog, avg_mpc_irrig,
                        avg_ext_temp, avg_ext_humidity, avg_solar_radiation, avg_wind_speed,
                        peak_early_blight_sev, peak_late_blight_sev,
                        peak_leaf_mold_sev, peak_powdery_mildew_sev, peak_spider_mites_sev,
                        disease_alert_steps,
                        growth_lstm_runs, disease_lstm_runs, weather_forecast_runs
                    ) VALUES (
                        :cid, '2026-03', 'March 2026',
                        '2026-03-01 06:00:00', '2026-03-31 23:55:00',
                        8928, 446, 93,
                        'seedling', 'early vegetative',
                        0, 1, 1,
                        :stl3,
                        27.1, 73.2, 534.8, 60.8, 9120.0, 0.91, 0.17, 0.14,
                        23.0, 32.8, 65.0, 83.1, 420.0, 680.0, 52.0, 70.0,
                        0.51,
                        3.1, -2.8, -9.1, -51.2, -1980.0, 0.24,
                        :e3, :w3, :c3,
                        15.41, 1.64, 6.17, 4.93, 0.82, 0.21, 12.92,
                        107.87, 11.48, 43.19, 34.51, 5.74, 1.47, 90.44,
                        :w3, 66.8, 446,
                        0.24, 0.15, 0.10, 0.19, 0.13, 0.00, 0.51,
                        26.0, 76.5, 298.3, 17.1,
                        0.10, 0.08, 0.13, 0.36, 0.05, 26,
                        446, 446, 446
                    )
                    ON CONFLICT (cycle_id, billing_month) DO NOTHING
                """), {"cid": cycle2_id, "e3": round(_e3,4), "w3": _w3, "c3": round(_c3,2),
                       "stl3": '[{"from_stage":"seedling","to_stage":"early vegetative","step":4464,"ts":"2026-03-16T12:30:00"}]'})

            session.commit()
            print("[seed_mock_data] Done — 2 cycles, 3 monthly snapshot rows inserted.")
        except Exception as exc:
            session.rollback()
            logger.exception("seed_mock_data failed: %s", exc)
            print(f"[seed_mock_data] Error: {exc}")
        finally:
            session.close()

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _ensure_cycle_row(self) -> int:
        """Insert or retrieve the crop_cycles row, return its id."""
        from sqlalchemy import text
        session = self._session_factory()
        try:
            session.execute(text("""
                INSERT INTO crop_cycles (cycle_label, crop_type)
                VALUES (:label, 'tomato')
                ON CONFLICT (cycle_label) DO NOTHING
            """), {"label": self._cycle_label})
            session.commit()
            row = session.execute(
                text("SELECT id FROM crop_cycles WHERE cycle_label = :label"),
                {"label": self._cycle_label},
            ).fetchone()
            return row[0]
        finally:
            session.close()

    def _flush_month(self) -> None:
        """Write the completed accumulator to the DB (called inside ``_lock``)."""
        acc = self._acc

        if acc.total_steps == 0:
            return  # nothing to write

        # Compute averages
        n = acc.total_steps

        def _avg(s: float) -> float | None:
            return round(s / n, 4) if n > 0 else None

        def _avg_mpc(s: float) -> float | None:
            k = acc.mpc_solve_count
            return round(s / k, 4) if k > 0 else None

        def _avg_w(s: float) -> float | None:
            k = acc._weather_count
            return round(s / k, 4) if k > 0 else None

        def _inr(kwh: float) -> float:
            return round(kwh * _ENERGY_RATE_INR_PER_KWH, 4)

        act_irrig_inr = round(
            acc.act_irrigation_kwh * _ENERGY_RATE_INR_PER_KWH
            + acc.act_irrigation_water_l * _WATER_RATE_INR_PER_L,
            4,
        )
        total_cost = round(
            acc.total_energy_kwh * _ENERGY_RATE_INR_PER_KWH
            + acc.total_water_l * _WATER_RATE_INR_PER_L,
            2,
        )

        month_label = _dt.datetime.strptime(acc.billing_month, "%Y-%m").strftime("%B %Y")

        if self._session_factory is None:
            logger.info(
                "MonthlySnapshotService (dry-run): would flush month=%s steps=%d cost=₹%.2f",
                acc.billing_month, n, total_cost,
            )
            return

        try:
            from sqlalchemy import text
            session = self._session_factory()
            try:
                session.execute(text("""
                    INSERT INTO monthly_snapshots (
                        cycle_id, billing_month, month_label,
                        month_start_ts, month_end_ts,
                        total_steps, mpc_solve_count, image_classify_count,
                        stage_at_month_start, stage_at_month_end,
                        stage_idx_start, stage_idx_end, stage_transitions,
                        stage_transition_log,
                        avg_indoor_temp, avg_indoor_humidity, avg_co2,
                        avg_soil_moisture, avg_light_intensity, avg_vpd,
                        avg_leaf_wetness, avg_disease_risk_score,
                        min_indoor_temp, max_indoor_temp,
                        min_indoor_humidity, max_indoor_humidity,
                        min_co2, max_co2, min_soil_moisture, max_soil_moisture,
                        max_disease_risk_score,
                        avg_setpt_err_temp, avg_setpt_err_humidity,
                        avg_setpt_err_soil, avg_setpt_err_co2,
                        avg_setpt_err_light, avg_setpt_err_vpd,
                        total_energy_kwh, total_water_l, total_cost_inr,
                        act_fan_speed_kwh, act_vent_opening_kwh, act_heater_output_kwh,
                        act_led_intensity_kwh, act_fogger_duty_kwh, act_co2_valve_kwh,
                        act_irrigation_kwh,
                        act_fan_speed_inr, act_vent_opening_inr, act_heater_output_inr,
                        act_led_intensity_inr, act_fogger_duty_inr, act_co2_valve_inr,
                        act_irrigation_inr, act_irrigation_water_l,
                        avg_mpc_cost, mpc_converge_count,
                        avg_mpc_fan, avg_mpc_vent, avg_mpc_heat,
                        avg_mpc_led, avg_mpc_co2v, avg_mpc_fog, avg_mpc_irrig,
                        avg_ext_temp, avg_ext_humidity, avg_solar_radiation, avg_wind_speed,
                        peak_early_blight_sev, peak_late_blight_sev,
                        peak_leaf_mold_sev, peak_powdery_mildew_sev, peak_spider_mites_sev,
                        disease_alert_steps,
                        growth_lstm_runs, disease_lstm_runs, weather_forecast_runs
                    ) VALUES (
                        :cycle_id, :billing_month, :month_label,
                        :month_start_ts, :month_end_ts,
                        :total_steps, :mpc_solve_count, :image_classify_count,
                        :stage_start, :stage_end,
                        :idx_start, :idx_end, :stage_transitions,
                        :stage_transition_log,
                        :avg_temp, :avg_rh, :avg_co2,
                        :avg_soil, :avg_light, :avg_vpd,
                        :avg_leaf, :avg_risk,
                        :min_temp, :max_temp,
                        :min_rh, :max_rh,
                        :min_co2, :max_co2, :min_soil, :max_soil,
                        :max_risk,
                        :err_temp, :err_rh, :err_soil, :err_co2, :err_light, :err_vpd,
                        :tot_energy, :tot_water, :tot_cost,
                        :fan_kwh, :vent_kwh, :heat_kwh, :led_kwh, :fog_kwh, :co2v_kwh, :irrig_kwh,
                        :fan_inr, :vent_inr, :heat_inr, :led_inr, :fog_inr, :co2v_inr, :irrig_inr,
                        :irrig_water_l,
                        :avg_mpc_cost, :mpc_conv,
                        :mpc_fan, :mpc_vent, :mpc_heat, :mpc_led, :mpc_co2v, :mpc_fog, :mpc_irrig,
                        :ext_temp, :ext_rh, :solar, :wind,
                        :pk_eb, :pk_lb, :pk_lm, :pk_pm, :pk_sm,
                        :alert_steps,
                        :g_lstm, :d_lstm, :wf_runs
                    )
                    ON CONFLICT (cycle_id, billing_month) DO UPDATE SET
                        month_end_ts         = EXCLUDED.month_end_ts,
                        total_steps          = EXCLUDED.total_steps,
                        mpc_solve_count      = EXCLUDED.mpc_solve_count,
                        stage_at_month_end   = EXCLUDED.stage_at_month_end,
                        stage_idx_end        = EXCLUDED.stage_idx_end,
                        total_energy_kwh     = EXCLUDED.total_energy_kwh,
                        total_water_l        = EXCLUDED.total_water_l,
                        total_cost_inr       = EXCLUDED.total_cost_inr,
                        snapshot_recorded_at = :flush_ts
                """), {
                    "flush_ts": _dt.datetime.utcnow().isoformat(),
                    "cycle_id":            self._cycle_id,
                    "billing_month":       acc.billing_month,
                    "month_label":         month_label,
                    "month_start_ts":      acc.month_start_ts,
                    "month_end_ts":        acc.month_end_ts,
                    "total_steps":         acc.total_steps,
                    "mpc_solve_count":     acc.mpc_solve_count,
                    "image_classify_count":acc.image_classify_count,
                    "stage_start":         acc.stage_at_start,
                    "stage_end":           acc.stage_at_end,
                    "idx_start":           acc.stage_idx_start,
                    "idx_end":             acc.stage_idx_end,
                    "stage_transitions":   acc.stage_transitions,
                    "stage_transition_log":json.dumps(acc.stage_transition_log),
                    "avg_temp":            _avg(acc._sum_temp),
                    "avg_rh":              _avg(acc._sum_humidity),
                    "avg_co2":             _avg(acc._sum_co2),
                    "avg_soil":            _avg(acc._sum_soil),
                    "avg_light":           _avg(acc._sum_light),
                    "avg_vpd":             _avg(acc._sum_vpd),
                    "avg_leaf":            _avg(acc._sum_leaf),
                    "avg_risk":            _avg(acc._sum_risk),
                    "min_temp":            acc.min_temp,
                    "max_temp":            acc.max_temp,
                    "min_rh":              acc.min_humidity,
                    "max_rh":              acc.max_humidity,
                    "min_co2":             acc.min_co2,
                    "max_co2":             acc.max_co2,
                    "min_soil":            acc.min_soil,
                    "max_soil":            acc.max_soil,
                    "max_risk":            acc.max_risk,
                    "err_temp":            _avg(acc._sum_err_temp) if acc._err_count else None,
                    "err_rh":              _avg(acc._sum_err_humidity) if acc._err_count else None,
                    "err_soil":            _avg(acc._sum_err_soil) if acc._err_count else None,
                    "err_co2":             _avg(acc._sum_err_co2) if acc._err_count else None,
                    "err_light":           _avg(acc._sum_err_light) if acc._err_count else None,
                    "err_vpd":             _avg(acc._sum_err_vpd) if acc._err_count else None,
                    "tot_energy":          round(acc.total_energy_kwh, 6),
                    "tot_water":           round(acc.total_water_l, 4),
                    "tot_cost":            total_cost,
                    "fan_kwh":             round(acc.act_fan_speed_kwh,    6),
                    "vent_kwh":            round(acc.act_vent_opening_kwh, 6),
                    "heat_kwh":            round(acc.act_heater_kwh,       6),
                    "led_kwh":             round(acc.act_led_kwh,          6),
                    "fog_kwh":             round(acc.act_fogger_kwh,       6),
                    "co2v_kwh":            round(acc.act_co2_valve_kwh,    6),
                    "irrig_kwh":           round(acc.act_irrigation_kwh,   6),
                    "fan_inr":             _inr(acc.act_fan_speed_kwh),
                    "vent_inr":            _inr(acc.act_vent_opening_kwh),
                    "heat_inr":            _inr(acc.act_heater_kwh),
                    "led_inr":             _inr(acc.act_led_kwh),
                    "fog_inr":             _inr(acc.act_fogger_kwh),
                    "co2v_inr":            _inr(acc.act_co2_valve_kwh),
                    "irrig_inr":           act_irrig_inr,
                    "irrig_water_l":       round(acc.act_irrigation_water_l, 4),
                    "avg_mpc_cost":        _avg_mpc(acc._sum_mpc_cost),
                    "mpc_conv":            acc.mpc_converge_count,
                    "mpc_fan":             _avg_mpc(acc._sum_mpc_fan),
                    "mpc_vent":            _avg_mpc(acc._sum_mpc_vent),
                    "mpc_heat":            _avg_mpc(acc._sum_mpc_heat),
                    "mpc_led":             _avg_mpc(acc._sum_mpc_led),
                    "mpc_co2v":            _avg_mpc(acc._sum_mpc_co2v),
                    "mpc_fog":             _avg_mpc(acc._sum_mpc_fog),
                    "mpc_irrig":           _avg_mpc(acc._sum_mpc_irrig),
                    "ext_temp":            _avg_w(acc._sum_ext_temp),
                    "ext_rh":              _avg_w(acc._sum_ext_humidity),
                    "solar":               _avg_w(acc._sum_solar),
                    "wind":                _avg_w(acc._sum_wind),
                    "pk_eb":               round(acc.peak_early_blight,   4),
                    "pk_lb":               round(acc.peak_late_blight,    4),
                    "pk_lm":               round(acc.peak_leaf_mold,      4),
                    "pk_pm":               round(acc.peak_powdery_mildew, 4),
                    "pk_sm":               round(acc.peak_spider_mites,   4),
                    "alert_steps":         acc.disease_alert_steps,
                    "g_lstm":              acc.growth_lstm_runs,
                    "d_lstm":              acc.disease_lstm_runs,
                    "wf_runs":             acc.weather_forecast_runs,
                })
                session.commit()
                logger.info(
                    "MonthlySnapshotService: flushed month=%s steps=%d cost=₹%.2f",
                    acc.billing_month, acc.total_steps, total_cost,
                )
            except Exception:
                session.rollback()
                logger.exception("MonthlySnapshotService: DB flush failed")
            finally:
                session.close()
        except Exception:
            logger.exception("MonthlySnapshotService: session_factory failed during flush")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point — python -m agritwin_gh.services.monthly_snapshot_service
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3] / "src"))

    from agritwin_gh.utils.database import get_db_manager

    mgr = get_db_manager()
    mgr.create_engine_instance()

    def _sf():
        return mgr.get_session()

    cmd = sys.argv[1] if len(sys.argv) > 1 else "show"
    if cmd == "seed":
        MonthlySnapshotService.seed_mock_data(_sf)
    elif cmd == "show":
        MonthlySnapshotService.show_table(_sf)
    else:
        print(f"Usage: python -m agritwin_gh.services.monthly_snapshot_service [seed|show]")
