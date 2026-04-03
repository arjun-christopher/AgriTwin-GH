"""
``realtime_core`` — Reusable closed-loop DT + MPC orchestration engine.

All general-purpose logic for the real-time greenhouse Digital-Twin loop
lives here so it can be imported by test scripts, evaluation harnesses,
web services, or any future integration — without CLI / console / signal
dependencies.

Quick start
-----------
>>> from agritwin_gh.mpc.realtime_core import RealtimeLoop, RealtimeLoopConfig
>>> cfg = RealtimeLoopConfig(growth_stage="flowering", days_elapsed=5.0, total_steps=12)
>>> loop = RealtimeLoop(cfg, session=db_session)
>>> loop.setup()
>>> summary = loop.run()        # <- runs all 12 steps
>>> print(summary.total_cost)

Or step-by-step:

>>> loop.setup()
>>> for i in range(1, 13):
...     result = loop.step(i)
...     print(result.payload.alert_level)
"""

from __future__ import annotations

import datetime as _dt
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from sqlalchemy.orm import Session

from agritwin_gh.models.timeseries import Base, RealtimeGreenhouseStream
from agritwin_gh.mpc.config import MPCConfig, load_mpc_config
from agritwin_gh.mpc.constants import (
    GROWTH_STAGE_TO_DB,
    GROWTH_STAGES,
    DT_MINUTES,
    compute_dew_point,
    compute_leaf_wetness_proxy,
    compute_vpd,
    stage_label_to_index,
)
from agritwin_gh.mpc.digital_twin_output import DigitalTwinOutput
from agritwin_gh.mpc.disease_penalty import DiseaseRiskPenalty
from agritwin_gh.mpc.disturbance import WeatherDisturbanceForecast
from agritwin_gh.mpc.greenhouse_model import GreenhouseTransitionModel
from agritwin_gh.mpc.growth_weights import GrowthStageWeights
from agritwin_gh.mpc.image_streamer import ImageStreamer
from agritwin_gh.mpc.mpc_input_preparation import MPCInputPreparation
from agritwin_gh.mpc.mpc_solver import MPCSolution, MPCSolver
from agritwin_gh.mpc.dt_engine import DigitalTwinEngine
from agritwin_gh.mpc.dt_state import DTStepInput
from agritwin_gh.mpc.state import (
    ActuatorState,
    ControllerDecisionContext,
    DigitalTwinStepPayload,
    FusedState,
    GreenhouseState,
    WeatherState,
)
from agritwin_gh.mpc.state_fusion import StateFusion

logger = logging.getLogger("agritwin.realtime.core")


# ═════════════════════════════════════════════════════════════════════════════
# Stage constants
# ═════════════════════════════════════════════════════════════════════════════

STAGE_DURATION_HOURS: dict[str, int] = {
    "seedling":              336,   # 14 days
    "early vegetative":      480,   # 20 days
    "flowering initiation":  240,   # 10 days
    "flowering":             360,   # 15 days
    "unripe":                480,   # 20 days
    "ripe":                  240,   # 10 days
}


def compute_prior_stage_hours() -> dict[str, int]:
    """Cumulative hours before each stage (for ``days_from_cycle_start``)."""
    acc, result = 0, {}
    for s in GROWTH_STAGES:
        result[s] = acc
        acc += STAGE_DURATION_HOURS[s]
    return result


PRIOR_STAGE_HOURS: dict[str, int] = compute_prior_stage_hours()


# ═════════════════════════════════════════════════════════════════════════════
# Diurnal defaults
# ═════════════════════════════════════════════════════════════════════════════

def diurnal_temp(hour: int) -> float:
    """Sinusoidal indoor temp (°C) as a function of hour-of-day."""
    return round(22.0 + 5.0 * math.sin(math.pi * (hour - 6) / 12), 2)


def diurnal_humidity(hour: int) -> float:
    """Sinusoidal indoor RH (%) peaking at night."""
    return round(65.0 - 10.0 * math.sin(math.pi * (hour - 6) / 12), 2)


def diurnal_solar(hour: int) -> float:
    """Simple daytime solar radiation estimate (W/m²)."""
    if not (6 <= hour < 20):
        return 0.0
    return round(600.0 * math.sin(math.pi * (hour - 6) / 14), 2)


def default_soil_moisture(stage: str) -> float:
    """Stage-aware default soil moisture (%)."""
    return {
        "seedling": 70.0,
        "early vegetative": 65.0,
        "flowering initiation": 60.0,
        "flowering": 58.0,
        "unripe": 55.0,
        "ripe": 50.0,
    }.get(stage, 60.0)


# ═════════════════════════════════════════════════════════════════════════════
# Energy estimation
# ═════════════════════════════════════════════════════════════════════════════

def estimate_energy(actuators: ActuatorState) -> float:
    """Estimate kWh consumed by one 5-minute DT step."""
    dt_hours = DT_MINUTES / 60.0
    rated_kw = {
        "fan_speed":      0.75,
        "vent_opening":   0.10,
        "heater_output":  3.00,
        "led_intensity":  2.00,
        "fogger_duty":    0.20,
        "co2_valve_pct":  0.05,
        "irrigation_qty": 0.15,   # pump
    }
    total = 0.0
    for fld, kw in rated_kw.items():
        duty = getattr(actuators, fld, 0.0) or 0.0
        total += kw * duty * dt_hours
    return round(total, 6)


# ═════════════════════════════════════════════════════════════════════════════
# Realtime-aware MPCInputPreparation override
# ═════════════════════════════════════════════════════════════════════════════

class RealtimeMPCInputPreparation(MPCInputPreparation):
    """Reads the latest greenhouse row from ``realtime_greenhouse_stream``
    instead of the historical ``greenhouse_data`` table.

    All other methods (weather, growth, disease context) continue to query
    their original tables so AI models receive real historical context.
    """

    def __init__(
        self,
        session: Session,
        run_id: str,
        filter_by_run: bool = True,
    ) -> None:
        super().__init__(session)
        self._run_id = run_id
        self._filter_by_run = filter_by_run

    def get_latest_greenhouse_row(self) -> dict[str, Any] | None:
        q = self._session.query(RealtimeGreenhouseStream)
        if self._filter_by_run:
            q = q.filter(RealtimeGreenhouseStream.run_id == self._run_id)
        row = q.order_by(RealtimeGreenhouseStream.datetime.desc()).first()
        if row is None:
            return None
        return {
            "datetime": row.datetime,
            "indoor_temp": row.indoor_temp,
            "indoor_humidity": row.indoor_humidity,
            "indoor_air_velocity": row.indoor_air_velocity or 0.5,
            "indoor_co2": row.indoor_co2,
            "solarradiation": row.solarradiation,
            "day_night_flag": row.day_night_flag,
            "vpd": row.vpd,
            "dew_point": row.dew_point,
            "leaf_wetness_proxy": row.leaf_wetness_proxy,
            "soil_moisture": row.soil_moisture,
            "stage_name": GROWTH_STAGE_TO_DB.get(
                row.growth_stage or "seedling", "seedling"
            ),
        }


# ═════════════════════════════════════════════════════════════════════════════
# DB helpers
# ═════════════════════════════════════════════════════════════════════════════

def ensure_stream_table(session: Session) -> None:
    """Create ``realtime_greenhouse_stream`` if it does not already exist."""
    engine = session.get_bind()
    Base.metadata.create_all(engine, tables=[RealtimeGreenhouseStream.__table__])
    logger.info("realtime_greenhouse_stream table ensured.")


def seed_initial_state(
    session: Session,
    run_id: str,
    growth_stage: str,
    hours_in_stage: float,
    stage_progress_pct: float,
    days_from_cycle_start: float,
    start_ts: _dt.datetime,
) -> GreenhouseState:
    """Seed the stream with an initial bootstrap row.

    Priority:
    1. Most recent row from ``greenhouse_data`` (real historical readings).
    2. Diurnal-aligned synthetic defaults if the historical table is empty.

    Returns the initial ``GreenhouseState`` for the first DT step.
    """
    from agritwin_gh.models.timeseries import GreenhouseData

    latest = (
        session.query(GreenhouseData)
        .order_by(GreenhouseData.datetime.desc())
        .first()
    )

    hour = start_ts.hour

    if latest is not None:
        indoor_temp = latest.indoor_temp or diurnal_temp(hour)
        indoor_humidity = latest.indoor_humidity or diurnal_humidity(hour)
        indoor_co2 = latest.indoor_co2 or 800.0
        solar = latest.solarradiation or diurnal_solar(hour)
        vpd = latest.vpd or compute_vpd(indoor_temp, indoor_humidity)
        dew_pt = latest.dew_point or compute_dew_point(indoor_temp, indoor_humidity)
        lw = latest.leaf_wetness_proxy or compute_leaf_wetness_proxy(
            indoor_humidity, indoor_temp, dew_pt
        )
        air_vel = latest.indoor_air_velocity or 0.5
        dnf = (
            latest.day_night_flag
            if latest.day_night_flag is not None
            else int(6 <= hour < 20)
        )
    else:
        logger.warning(
            "greenhouse_data is empty — using diurnal defaults for seed."
        )
        indoor_temp = diurnal_temp(hour)
        indoor_humidity = diurnal_humidity(hour)
        indoor_co2 = 800.0
        solar = diurnal_solar(hour)
        vpd = compute_vpd(indoor_temp, indoor_humidity)
        dew_pt = compute_dew_point(indoor_temp, indoor_humidity)
        lw = compute_leaf_wetness_proxy(indoor_humidity, indoor_temp, dew_pt)
        air_vel = 0.5
        dnf = int(6 <= hour < 20)

    soil_moisture = default_soil_moisture(growth_stage)

    row = RealtimeGreenhouseStream(
        run_id=run_id,
        step_index=0,
        source="bootstrap",
        datetime=start_ts,
        indoor_temp=indoor_temp,
        indoor_humidity=indoor_humidity,
        indoor_air_velocity=air_vel,
        indoor_co2=indoor_co2,
        solarradiation=solar,
        day_night_flag=dnf,
        vpd=vpd,
        dew_point=dew_pt,
        leaf_wetness_proxy=lw,
        soil_moisture=soil_moisture,
        disease_risk_score=0.0,
        growth_stage=growth_stage,
        growth_stage_index=stage_label_to_index(growth_stage),
        disease_classification="healthy leaves",
        fan_speed=0.0,
        vent_opening=0.0,
        heater_output=0.0,
        led_intensity=0.0,
        fogger_duty=0.0,
        co2_valve_pct=0.0,
        irrigation_qty=0.0,
        mpc_ran=False,
        mpc_converged=False,
        mpc_fallback_used=False,
        step_cost=0.0,
        solve_time_ms=0.0,
        hours_in_current_stage=hours_in_stage,
        stage_progress_pct=stage_progress_pct,
        hours_to_stage_transition=float(
            STAGE_DURATION_HOURS[growth_stage] - hours_in_stage
        ),
        step_energy_kwh=0.0,
        cumulative_energy_kwh=0.0,
        cumulative_water_litres=0.0,
        alert_level="GREEN",
    )
    session.add(row)
    session.commit()

    return GreenhouseState(
        indoor_temp=indoor_temp,
        indoor_humidity=indoor_humidity,
        soil_moisture=soil_moisture,
        co2=indoor_co2,
        light_intensity=solar,
        disease_risk_score=0.0,
        growth_stage_index=stage_label_to_index(growth_stage),
        vpd=vpd,
        leaf_wetness_proxy=lw,
        timestamp=start_ts,
    )


def _f(v) -> float | None:
    """Convert numpy scalars (or any numeric) to a plain Python float."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def write_step_to_stream(
    session: Session,
    *,
    run_id: str,
    step_index: int,
    payload: DigitalTwinStepPayload,
    hours_in_stage: float,
    stage_progress_pct: float,
    cumulative_energy_kwh: float,
    cumulative_water_litres: float,
) -> None:
    """Persist one DT step to ``realtime_greenhouse_stream``."""
    obs = payload.observed_state
    act = payload.applied_actuators
    ts = payload.timestamp or _dt.datetime.utcnow()

    dew_pt = compute_dew_point(
        _f(obs.get("indoor_temp", 25.0)),
        _f(obs.get("indoor_humidity", 65.0)),
    )

    row = RealtimeGreenhouseStream(
        run_id=run_id,
        step_index=step_index,
        source="dt_sim",
        datetime=ts,
        indoor_temp=_f(obs.get("indoor_temp")),
        indoor_humidity=_f(obs.get("indoor_humidity")),
        indoor_air_velocity=0.5,
        indoor_co2=_f(obs.get("co2")),
        solarradiation=_f(obs.get("light_intensity")),
        day_night_flag=int(6 <= ts.hour < 20) if ts else 1,
        vpd=_f(obs.get("vpd")),
        dew_point=_f(dew_pt),
        leaf_wetness_proxy=_f(obs.get("leaf_wetness_proxy")),
        soil_moisture=_f(obs.get("soil_moisture")),
        disease_risk_score=_f(payload.disease_risk_score),
        growth_stage=payload.growth_stage,
        growth_stage_index=obs.get("growth_stage_index"),
        disease_classification=payload.disease_classification,
        fan_speed=_f(act.get("fan_speed")),
        vent_opening=_f(act.get("vent_opening")),
        heater_output=_f(act.get("heater_output")),
        led_intensity=_f(act.get("led_intensity")),
        fogger_duty=_f(act.get("fogger_duty")),
        co2_valve_pct=_f(act.get("co2_valve_pct")),
        irrigation_qty=_f(act.get("irrigation_qty")),
        mpc_ran=True,
        mpc_converged=bool(payload.solver_performance.get("converged", False)),
        mpc_fallback_used=bool(payload.solver_performance.get("fallback_used", False)),
        step_cost=_f(payload.step_cost),
        solve_time_ms=_f(payload.solver_performance.get("solve_time_ms", 0.0)),
        hours_in_current_stage=_f(hours_in_stage),
        stage_progress_pct=_f(stage_progress_pct),
        hours_to_stage_transition=_f(payload.hours_to_stage_transition),
        step_energy_kwh=_f(payload.cumulative_energy_kwh - cumulative_energy_kwh),
        cumulative_energy_kwh=_f(payload.cumulative_energy_kwh),
        cumulative_water_litres=_f(payload.cumulative_water_litres),
        alert_level=payload.alert_level,
    )
    session.add(row)
    session.commit()


# ═════════════════════════════════════════════════════════════════════════════
# Configuration dataclass (replaces argparse.Namespace dependency)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class RealtimeLoopConfig:
    """All parameters needed to run a closed-loop session.

    Can be constructed from CLI args, a YAML file, a web request, etc.
    """
    growth_stage: str
    days_elapsed: float
    total_steps: int
    mpc_every: int = 3
    device: str = "cpu"
    no_images: bool = False
    dry_run: bool = False
    run_id: str | None = None       # auto-generated if None
    start_ts: _dt.datetime | None = None  # defaults to now()
    auto_advance_stage: bool = False  # advance growth stage when duration elapses


# ═════════════════════════════════════════════════════════════════════════════
# Step result & run summary
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class RealtimeStepResult:
    """Returned by ``RealtimeLoop.step()``."""
    step_index: int
    payload: DigitalTwinStepPayload
    hours_in_stage: float
    stage_progress_pct: float
    wall_time_ms: float
    cumulative_energy_kwh: float
    cumulative_water_litres: float
    total_cost: float


@dataclass
class RealtimeRunSummary:
    """Returned by ``RealtimeLoop.run()``."""
    run_id: str
    growth_stage: str
    steps_run: int
    total_energy_kwh: float
    total_water_litres: float
    total_cost: float
    start_ts: _dt.datetime
    end_ts: _dt.datetime


# ═════════════════════════════════════════════════════════════════════════════
# Core orchestrator
# ═════════════════════════════════════════════════════════════════════════════

class RealtimeLoop:
    """Reusable closed-loop DT + MPC orchestrator.

    Usage::

        loop = RealtimeLoop(config, session)
        loop.setup()
        summary = loop.run(on_step=my_callback)

    Or step-by-step::

        loop.setup()
        for i in range(1, config.total_steps + 1):
            result = loop.step(i)
            ...
        summary = loop.finish()
    """

    def __init__(
        self,
        config: RealtimeLoopConfig,
        session: Session,
        *,
        disease_classifier: Callable | None = None,
        growth_classifier: Callable | None = None,
    ) -> None:
        self.config = config
        self.session = session

        # ── Resolve identity & timestamps ─────────────────────────────
        now = _dt.datetime.now()
        self.run_id: str = config.run_id or f"rt_{now:%Y%m%d_%H%M%S}"
        self.start_ts: _dt.datetime = (
            config.start_ts or now.replace(second=0, microsecond=0)
        )
        self._dt_delta = _dt.timedelta(minutes=DT_MINUTES)

        # ── Derived temporal quantities ───────────────────────────────
        self._hours_in_stage_0 = config.days_elapsed * 24.0
        self._stage_progress_0 = min(
            100.0,
            100.0 * self._hours_in_stage_0 / STAGE_DURATION_HOURS[config.growth_stage],
        )
        self._days_from_cycle_0 = (
            PRIOR_STAGE_HOURS[config.growth_stage] / 24.0 + config.days_elapsed
        )

        # ── MPC config ────────────────────────────────────────────────
        self._cfg: MPCConfig = load_mpc_config()

        # ── Sub-system wiring ─────────────────────────────────────────
        self._rt_input_prep = RealtimeMPCInputPreparation(
            session=session,
            run_id=self.run_id,
            filter_by_run=True,
        )
        image_streamer = ImageStreamer(session)
        weather_forecast = WeatherDisturbanceForecast(
            run_id=self._cfg.environment_forecast_run_id,
            device=config.device,
        )
        self._disease_penalty = DiseaseRiskPenalty(
            run_id=self._cfg.disease_progression_run_id,
        )
        self._growth_weights = GrowthStageWeights(
            stage_weight_multipliers=self._cfg.stage_weight_multipliers,
            run_id=self._cfg.growth_progression_run_id,
        )
        self._fusion = StateFusion(
            config=self._cfg,
            input_prep=self._rt_input_prep,
            weather=weather_forecast,
            disease_penalty=self._disease_penalty,
            growth_weights=self._growth_weights,
            image_streamer=image_streamer,
            disease_classifier=disease_classifier,
            growth_classifier=growth_classifier,
        )
        self._dt_model = GreenhouseTransitionModel()
        self._dt_engine = DigitalTwinEngine(dt_minutes=DT_MINUTES)
        self._solver = MPCSolver(config=self._cfg, model=self._dt_model)
        self._output = DigitalTwinOutput(run_id=self.run_id)

        # ── Mutable loop state (initialised in setup()) ───────────────
        self.current_state: GreenhouseState | None = None
        self._current_ts: _dt.datetime = self.start_ts + self._dt_delta
        self._prev_action: ActuatorState | None = None
        self._last_weather_step: dict = {}
        self._hours_in_stage: float = 0.0
        self._stage_progress: float = 0.0
        self.cumulative_energy: float = 0.0
        self.cumulative_water: float = 0.0
        self.total_cost: float = 0.0
        self.steps_run: int = 0
        self._setup_done = False

    # ─────────────────────────────────────────────────────────────────
    # setup
    # ─────────────────────────────────────────────────────────────────

    def setup(self) -> GreenhouseState:
        """Ensure DB table + seed the initial bootstrap row.

        Must be called before ``step()`` or ``run()``.
        Returns the initial ``GreenhouseState``.
        """
        if not self.config.dry_run:
            ensure_stream_table(self.session)
            self.current_state = seed_initial_state(
                session=self.session,
                run_id=self.run_id,
                growth_stage=self.config.growth_stage,
                hours_in_stage=self._hours_in_stage_0,
                stage_progress_pct=self._stage_progress_0,
                days_from_cycle_start=self._days_from_cycle_0,
                start_ts=self.start_ts,
            )
        else:
            hour = self.start_ts.hour
            self.current_state = GreenhouseState(
                indoor_temp=diurnal_temp(hour),
                indoor_humidity=diurnal_humidity(hour),
                soil_moisture=default_soil_moisture(self.config.growth_stage),
                co2=800.0,
                light_intensity=diurnal_solar(hour),
                disease_risk_score=0.0,
                growth_stage_index=stage_label_to_index(self.config.growth_stage),
                vpd=compute_vpd(diurnal_temp(hour), diurnal_humidity(hour)),
                leaf_wetness_proxy=0.0,
                timestamp=self.start_ts,
            )

        self._hours_in_stage = self._hours_in_stage_0
        self._stage_progress = self._stage_progress_0
        self._setup_done = True
        return self.current_state

    # ─────────────────────────────────────────────────────────────────
    # step
    # ─────────────────────────────────────────────────────────────────

    def step(self, step_i: int) -> RealtimeStepResult:
        """Execute a single closed-loop step.

        Parameters
        ----------
        step_i : int
            1-based step index.

        Returns
        -------
        RealtimeStepResult
        """
        if not self._setup_done:
            raise RuntimeError("Call setup() before step().")

        wall_start = time.perf_counter()
        growth_stage = self.config.growth_stage
        mpc_every = max(1, self.config.mpc_every)

        # ── Advance stage accounting ──────────────────────────────────
        self._hours_in_stage += DT_MINUTES / 60.0

        # ── Auto-advance growth stage if duration elapses ─────────────
        if self.config.auto_advance_stage:
            dur = STAGE_DURATION_HOURS.get(growth_stage)
            if dur and self._hours_in_stage >= dur:
                idx = GROWTH_STAGES.index(growth_stage)
                if idx + 1 < len(GROWTH_STAGES):
                    old_stage = growth_stage
                    growth_stage = GROWTH_STAGES[idx + 1]
                    self.config.growth_stage = growth_stage
                    self._hours_in_stage = 0.0
                    logger.info(
                        "Step %d: growth stage advanced %s -> %s",
                        step_i, old_stage, growth_stage,
                    )

        self._stage_progress = min(
            100.0,
            100.0 * self._hours_in_stage / STAGE_DURATION_HOURS[growth_stage],
        )

        run_mpc = (step_i % mpc_every == 1) or (step_i == 1)

        if run_mpc:
            payload = self._mpc_step(step_i, growth_stage)
        else:
            payload = self._hold_step(step_i, growth_stage)

        # ── Write step to PostgreSQL ──────────────────────────────────
        if not self.config.dry_run:
            write_step_to_stream(
                self.session,
                run_id=self.run_id,
                step_index=step_i,
                payload=payload,
                hours_in_stage=self._hours_in_stage,
                stage_progress_pct=self._stage_progress,
                cumulative_energy_kwh=self.cumulative_energy,
                cumulative_water_litres=self.cumulative_water,
            )

        wall_dt_ms = (time.perf_counter() - wall_start) * 1000.0
        self.steps_run += 1
        self._current_ts += self._dt_delta

        return RealtimeStepResult(
            step_index=step_i,
            payload=payload,
            hours_in_stage=self._hours_in_stage,
            stage_progress_pct=self._stage_progress,
            wall_time_ms=wall_dt_ms,
            cumulative_energy_kwh=self.cumulative_energy,
            cumulative_water_litres=self.cumulative_water,
            total_cost=self.total_cost,
        )

    # ─────────────────────────────────────────────────────────────────
    # run  (full loop with optional callbacks)
    # ─────────────────────────────────────────────────────────────────

    def run(
        self,
        *,
        on_step: Callable[[RealtimeStepResult], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> RealtimeRunSummary:
        """Run the full closed-loop for ``config.total_steps`` steps.

        Parameters
        ----------
        on_step:
            Called after each step with the :class:`RealtimeStepResult`.
            Use this to print to console, save artifacts, etc.
        should_stop:
            Checked at the start of each iteration; return *True* to exit
            the loop early (e.g. from a signal handler or UI cancel).

        Returns
        -------
        RealtimeRunSummary
        """
        if not self._setup_done:
            self.setup()

        for step_i in range(1, self.config.total_steps + 1):
            if should_stop and should_stop():
                break

            result = self.step(step_i)

            if on_step is not None:
                on_step(result)

        return self.finish()

    # ─────────────────────────────────────────────────────────────────
    # finish  (build summary)
    # ─────────────────────────────────────────────────────────────────

    def finish(self) -> RealtimeRunSummary:
        """Return a :class:`RealtimeRunSummary` for the steps run so far."""
        end_ts = self._current_ts - self._dt_delta
        return RealtimeRunSummary(
            run_id=self.run_id,
            growth_stage=self.config.growth_stage,
            steps_run=self.steps_run,
            total_energy_kwh=self.cumulative_energy,
            total_water_litres=self.cumulative_water,
            total_cost=self.total_cost,
            start_ts=self.start_ts,
            end_ts=end_ts,
        )

    # ─────────────────────────────────────────────────────────────────
    # Internal: MPC step
    # ─────────────────────────────────────────────────────────────────

    def _mpc_step(self, step_i: int, growth_stage: str) -> DigitalTwinStepPayload:
        """Fuse → solve → DT engine → format payload."""

        # 1. Fuse state
        fused: FusedState = self._fusion.fuse(timestamp=self._current_ts)

        if fused.weather_forecast:
            self._last_weather_step = fused.weather_forecast[0]

        # 2. Adaptive cost weights
        weights = self._growth_weights.get_weights(
            fused.growth_stage,
            base_weights=self._cfg.cost_weight_vector,
        )

        # 3. MPC solve
        solution: MPCSolution = self._solver.solve(
            fused=fused,
            previous_control=self._prev_action,
        )
        actuators = solution.first_action
        self._prev_action = actuators

        # 4. Resource accounting
        step_energy = estimate_energy(actuators)
        step_water = actuators.irrigation_qty
        self.cumulative_energy += step_energy
        self.cumulative_water += step_water
        self.total_cost += solution.total_cost

        # 5. DT engine for full physics + diagnostics
        weather_now = self._build_weather_state()
        dt_input = DTStepInput(
            current_state=fused.greenhouse_state,
            action=actuators,
            weather=weather_now,
            growth_stage=fused.growth_stage or growth_stage,
            disease_risk_score=fused.disease_risk_score,
            disease_classification=fused.disease_classification or "healthy leaves",
            disease_severity=dict(fused.current_severity),
            dt_minutes=DT_MINUTES,
            step_index=step_i,
            timestamp=self._current_ts,
        )
        dt_out = self._dt_engine.step(dt_input)
        predicted_next: GreenhouseState | None = dt_out.next_state

        # 6. Solver-performance snapshot
        solver_perf = {
            "converged": solution.converged,
            "fallback_used": solution.fallback_used,
            "solve_time_ms": solution.solve_time_ms,
            "n_iterations": solution.n_iterations,
            "n_function_evals": solution.n_function_evals,
            "solver_status": solution.solver_status,
        }

        # 7. Decision context
        decision_ctx = ControllerDecisionContext(
            run_id=self.run_id,
            timestamp=self._current_ts,
            step_index=step_i,
            solver_config={
                "method": self._cfg.solver_method,
                "horizon_hours": self._cfg.prediction_horizon_hours,
                "dt_minutes": self._cfg.dt_minutes,
                "max_iter": self._cfg.solver_max_iter,
            },
            cost_weights=weights,
            weather_stress_summary=self._solver.last_weather_stress_summary or {},
            disease_context_summary={
                "risk_score": fused.disease_risk_score,
                "classification": fused.disease_classification,
                "severity_24h": dict(fused.severity_24h),
            },
            constraint_tightening=self._solver.last_constraint_tightening or {},
            solver_performance=solver_perf,
            model_ids={
                "environment_forecast": self._cfg.environment_forecast_run_id,
                "disease_progression": self._cfg.disease_progression_run_id,
                "growth_progression": self._cfg.growth_progression_run_id,
            },
        )

        # 8. Format payload
        payload: DigitalTwinStepPayload = self._output.format_step(
            fused=fused,
            actuators=actuators,
            predicted_next=predicted_next,
            step_cost=solution.total_cost,
            energy_kwh=step_energy,
            water_litres=step_water,
            cost_breakdown=solution.cost_breakdown,
            solver_converged=solution.converged,
            weather_stress=self._solver.last_weather_stress_summary,
            tightened_constraints=self._solver.last_constraint_tightening,
            decision_context=decision_ctx,
            solver_performance=solver_perf,
        )

        # Propagate DT-simulated state forward
        predicted_next.timestamp = self._current_ts + self._dt_delta
        self.current_state = predicted_next

        return payload

    # ─────────────────────────────────────────────────────────────────
    # Internal: hold step (reuse last actuators, DT physics only)
    # ─────────────────────────────────────────────────────────────────

    def _hold_step(self, step_i: int, growth_stage: str) -> DigitalTwinStepPayload:
        """Non-MPC step — hold actuators, advance physics via DT engine."""

        payload = DigitalTwinStepPayload(
            timestamp=self._current_ts,
            run_id=self.run_id,
            step_index=step_i,
            observed_state=(
                self.current_state.to_dict()
                if hasattr(self.current_state, "to_dict")
                else {}
            ),
            growth_stage=growth_stage,
            disease_risk_score=self.current_state.disease_risk_score,
            applied_actuators=(
                self._prev_action.to_dict() if self._prev_action else {}
            ),
            step_cost=0.0,
            cumulative_energy_kwh=self.cumulative_energy,
            cumulative_water_litres=self.cumulative_water,
            alert_level="GREEN",
            solver_performance={
                "converged": False,
                "fallback_used": False,
                "solve_time_ms": 0.0,
            },
            hours_to_stage_transition=float(
                STAGE_DURATION_HOURS[growth_stage] - self._hours_in_stage
            ),
        )

        weather_now = self._build_weather_state()
        dt_input = DTStepInput(
            current_state=self.current_state,
            action=self._prev_action or ActuatorState(),
            weather=weather_now,
            growth_stage=growth_stage,
            disease_risk_score=self.current_state.disease_risk_score,
            disease_classification="healthy leaves",
            dt_minutes=DT_MINUTES,
            step_index=step_i,
            timestamp=self._current_ts,
        )
        dt_out = self._dt_engine.step(dt_input)
        dt_out.next_state.timestamp = self._current_ts + self._dt_delta
        self.current_state = dt_out.next_state

        return payload

    # ─────────────────────────────────────────────────────────────────
    # Internal: build weather state from last forecast
    # ─────────────────────────────────────────────────────────────────

    def _build_weather_state(self) -> WeatherState:
        return WeatherState(
            temp_external=self._last_weather_step.get("temp", 20.0),
            humidity_external=self._last_weather_step.get("humidity", 65.0),
            solar_radiation=self._last_weather_step.get("solarradiation", 0.0),
            windspeed=self._last_weather_step.get("windspeed", 0.0),
            conditions="forecast",
        )


# ═════════════════════════════════════════════════════════════════════════════
# Persistent run registry
# ═════════════════════════════════════════════════════════════════════════════

import json as _json
from pathlib import Path as _Path

_REGISTRY_DIR = _Path(__file__).resolve().parents[3] / "logs" / "realtime"
_REGISTRY_FILE = _REGISTRY_DIR / "registry.json"


class RunRegistry:
    """Index of all run folders in ``logs/realtime/`` for cross-run comparison.

    The registry is stored as a JSON array in ``logs/realtime/registry.json``.
    Each entry records the run ID, growth stage, step count, timestamps,
    resource totals, and the path to the run's artifact directory.

    Usage::

        registry = RunRegistry()
        registry.register(summary, config)
        all_runs = registry.load()
    """

    def __init__(self, registry_path: _Path | None = None) -> None:
        self._path = registry_path or _REGISTRY_FILE

    def load(self) -> list[dict]:
        """Load and return the full registry (empty list if file missing)."""
        if not self._path.exists():
            return []
        try:
            with open(self._path, encoding="utf-8") as fh:
                data = _json.load(fh)
                return data if isinstance(data, list) else []
        except (_json.JSONDecodeError, ValueError):
            return []

    def register(
        self,
        summary: RealtimeRunSummary,
        config: RealtimeLoopConfig,
        artifact_dir: _Path | str | None = None,
    ) -> dict:
        """Append a completed run to the registry and return its entry.

        Parameters
        ----------
        summary:
            The :class:`RealtimeRunSummary` returned by ``RealtimeLoop.finish()``.
        config:
            The :class:`RealtimeLoopConfig` used for the run.
        artifact_dir:
            Path to the run's artifact folder.  ``None`` → inferred from
            ``logs/realtime/<run_id>``.
        """
        entry = {
            "run_id": summary.run_id,
            "growth_stage": summary.growth_stage,
            "days_elapsed_at_start": config.days_elapsed,
            "planned_steps": config.total_steps,
            "steps_run": summary.steps_run,
            "start_ts": str(summary.start_ts),
            "end_ts": str(summary.end_ts),
            "simulated_hours": round(summary.steps_run * DT_MINUTES / 60, 2),
            "total_energy_kwh": summary.total_energy_kwh,
            "total_water_litres": summary.total_water_litres,
            "total_mpc_cost": summary.total_cost,
            "mpc_every_steps": config.mpc_every,
            "auto_advance_stage": config.auto_advance_stage,
            "images_enabled": not config.no_images,
            "device": config.device,
            "artifact_dir": str(
                artifact_dir or (_REGISTRY_DIR / summary.run_id)
            ),
        }

        entries = self.load()
        # Deduplicate by run_id (overwrite if re-registered).
        entries = [e for e in entries if e.get("run_id") != summary.run_id]
        entries.append(entry)

        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as fh:
            _json.dump(entries, fh, indent=2)

        logger.info("Registered run %s in %s", summary.run_id, self._path)
        return entry

    def get_run(self, run_id: str) -> dict | None:
        """Look up a single run by ID."""
        for entry in self.load():
            if entry.get("run_id") == run_id:
                return entry
        return None

    def list_runs(
        self,
        growth_stage: str | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        """Return runs, optionally filtered by growth stage.

        Newest first (by ``start_ts``).
        """
        entries = self.load()
        if growth_stage:
            entries = [e for e in entries if e.get("growth_stage") == growth_stage]
        entries.sort(key=lambda e: e.get("start_ts", ""), reverse=True)
        if limit:
            entries = entries[:limit]
        return entries
