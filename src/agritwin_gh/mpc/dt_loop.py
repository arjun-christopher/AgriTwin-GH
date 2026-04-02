"""
Digital Twin closed-loop orchestration.

``DTLoop`` is the multi-rate simulation loop that alternates between:

* **DT step** (every 5 min) — advance climate state, weather, diagnostics.
* **MPC solve** (every 15 min) — recompute optimal actuator commands.
* **Image observation refresh** (every 30–60 min) — placeholder for
  disease/growth image reclassification.

The loop is **simulated** — timestamps advance logically in 5-minute
increments; no wall-clock waiting.  Each iteration yields a
``DTLoopStepResult`` that downstream code can log, display, or accumulate.

Timer cadences
--------------
=========  ===============  ===============================================
Period     Steps            What happens
=========  ===============  ===============================================
5 min      every step       DT engine advances state; weather disturbance
                            updates; disease risk / derived metrics refresh.
15 min     every 3rd step   MPC solver runs; new actuator trajectory
                            replaces the last action.
30–60 min  every 6–12 step  Image observation refresh hook fires (future
                            disease/growth image reclassification).
=========  ===============  ===============================================

Event-triggered early MPC re-solve
-----------------------------------
``should_force_mpc_update()`` checks the *post-step* state against safety
thresholds:

* Relative humidity > 85 %
* Disease risk score > 0.55
* Temperature outside [12, 38] °C

If any condition is true the MPC solver runs immediately, even if the
15-minute cadence has not elapsed.

Usage
-----
>>> loop = DTLoop(growth_stage="flowering")
>>> for result in loop.run(n_steps=144):
...     print(result.step_index, result.next_state.indoor_temp)
"""

from __future__ import annotations

import datetime as _dt
import logging
from dataclasses import dataclass, field
from typing import Any, Generator

from .config import MPCConfig
from .constants import (
    DT_MINUTES,
    compute_disease_risk_score,
)
from .dt_engine import DigitalTwinEngine
from .dt_state import DTDiagnostics, DTSnapshot, DTStepInput, DTStepOutput
from .greenhouse_model import GreenhouseModelParams
from .mpc_solver import MPCSolution
from .state import ActuatorState, FusedState, GreenhouseState, WeatherState

from .dt_image_observer import ImageObserver, SyntheticImageObserver
from .dt_input_provider import DTInputProvider, ImageObservation, SyntheticInputProvider
from .dt_runtime_prep import (
    build_fused_state,
    build_mpc_solver,
    prepare_growth_stages,
    prepare_initial_state,
    prepare_weather_sequence,
)

logger = logging.getLogger(__name__)


# ── Step result container ─────────────────────────────────────────────────────


@dataclass
class DTLoopStepResult:
    """Aggregated result of one closed-loop iteration.

    Combines the DT engine output with MPC metadata so callers
    do not need to track both.
    """

    step_index: int = 0
    timestamp: _dt.datetime | None = None

    # Pre-step state (before the DT engine advances)
    current_state: GreenhouseState = field(default_factory=GreenhouseState)

    # Weather disturbance used this step
    weather_used: WeatherState = field(default_factory=WeatherState)

    # DT output
    next_state: GreenhouseState = field(default_factory=GreenhouseState)
    diagnostics: DTDiagnostics = field(default_factory=DTDiagnostics)
    snapshot: DTSnapshot = field(default_factory=DTSnapshot)

    # Actuator that was applied this step
    action_applied: ActuatorState = field(default_factory=ActuatorState)

    # MPC metadata (populated only on steps where MPC ran)
    mpc_ran_this_step: bool = False
    mpc_forced: bool = False
    mpc_solution: MPCSolution | None = None
    mpc_cost: float | None = None

    # Image refresh metadata (populated on image-refresh steps)
    image_refresh_this_step: bool = False
    image_observation: ImageObservation | None = None

    # Cadence flags (informational)
    cadence_info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation."""
        return {
            "step_index": self.step_index,
            "timestamp": str(self.timestamp) if self.timestamp else None,
            "current_state": self.current_state.to_dict(),
            "weather_used": self.weather_used.to_dict(),
            "next_state": self.next_state.to_dict(),
            "diagnostics": self.diagnostics.to_dict(),
            "action_applied": self.action_applied.to_dict(),
            "mpc_ran_this_step": self.mpc_ran_this_step,
            "mpc_forced": self.mpc_forced,
            "mpc_cost": self.mpc_cost,
            "image_refresh_this_step": self.image_refresh_this_step,
            "image_observation": (
                self.image_observation.to_dict()
                if self.image_observation else None
            ),
            "cadence_info": self.cadence_info,
        }


# ── Event-trigger thresholds ─────────────────────────────────────────────────

_FORCE_MPC_RH_THRESH: float = 85.0       # %RH
_FORCE_MPC_RISK_THRESH: float = 0.55     # disease risk score
_FORCE_MPC_TEMP_LO: float = 12.0         # °C
_FORCE_MPC_TEMP_HI: float = 38.0         # °C


def should_force_mpc_update(state: GreenhouseState) -> bool:
    """Check whether an early MPC re-solve should be triggered.

    Conditions (any one triggers):

    * Relative humidity exceeds 85 % — rapid humidification may require
      an immediate fan/vent response.
    * Disease risk score exceeds 0.55 — pathogen-favourable conditions
      warrant proactive actuator changes.
    * Temperature outside the [12, 38] °C safe band — crop safety risk.

    Parameters
    ----------
    state:
        Post-step greenhouse state to evaluate.

    Returns
    -------
    bool
        ``True`` if any threshold is breached.
    """
    if state.indoor_humidity > _FORCE_MPC_RH_THRESH:
        return True
    if state.disease_risk_score > _FORCE_MPC_RISK_THRESH:
        return True
    if state.indoor_temp < _FORCE_MPC_TEMP_LO or state.indoor_temp > _FORCE_MPC_TEMP_HI:
        return True
    return False


# ── Multi-rate cadence constants ──────────────────────────────────────────────

MPC_CADENCE_STEPS: int = 3       # every 15 min (3 × 5 min)
IMAGE_CADENCE_STEPS: int = 6     # every 30 min (6 × 5 min)


# ── Closed-loop orchestrator ─────────────────────────────────────────────────


class DTLoop:
    """Multi-rate closed-loop DT + MPC simulation.

    Parameters
    ----------
    growth_stage:
        Canonical growth-stage label (from user input).
    start_time:
        Simulation anchor time.  ``None`` → ``datetime.now()``.
    n_steps:
        Total number of 5-minute steps.  Default 288 → 24 hours.
    dt_minutes:
        Step duration.
    model_params:
        ARX transition model coefficients (default → standard).
    mpc_config:
        MPC solver configuration.  ``None`` → auto-configured for
        closed-loop execution (1 h prediction/control horizon).
    mpc_cadence_steps:
        MPC solve interval expressed as number of DT steps.
    image_cadence_steps:
        Image observation refresh interval (DT steps).
    weather_base_temp:
        Mean outdoor temperature for synthetic weather.
    weather_diurnal_amp:
        Half-range of daily temperature swing.
    """

    def __init__(
        self,
        growth_stage: str,
        start_time: _dt.datetime | None = None,
        n_steps: int = 288,
        dt_minutes: int = DT_MINUTES,
        model_params: GreenhouseModelParams | None = None,
        mpc_config: MPCConfig | None = None,
        mpc_cadence_steps: int = MPC_CADENCE_STEPS,
        image_cadence_steps: int = IMAGE_CADENCE_STEPS,
        weather_base_temp: float = 20.0,
        weather_diurnal_amp: float = 8.0,
        input_provider: DTInputProvider | None = None,
        image_observer: ImageObserver | None = None,
    ) -> None:
        # ── Validate inputs ───────────────────────────────────────
        from .constants import GROWTH_STAGES

        if growth_stage not in GROWTH_STAGES:
            raise ValueError(
                f"Invalid growth stage '{growth_stage}'. "
                f"Must be one of: {GROWTH_STAGES}"
            )
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")
        if dt_minutes < 1:
            raise ValueError(f"dt_minutes must be >= 1, got {dt_minutes}")
        if mpc_cadence_steps < 1:
            raise ValueError(f"mpc_cadence_steps must be >= 1, got {mpc_cadence_steps}")
        if image_cadence_steps < 1:
            raise ValueError(f"image_cadence_steps must be >= 1, got {image_cadence_steps}")

        self._growth_stage = growth_stage
        self._start_time = start_time or _dt.datetime.now()
        self._n_steps = n_steps
        self._dt_minutes = dt_minutes
        self._mpc_cadence = mpc_cadence_steps
        self._image_cadence = image_cadence_steps

        # ── Input provider (optional — falls back to synthetic) ───
        self._input_provider: DTInputProvider = input_provider or SyntheticInputProvider(
            growth_stage=growth_stage,
            start_time=self._start_time,
            base_temp=weather_base_temp,
            diurnal_amp=weather_diurnal_amp,
        )

        # ── Image observer (optional — falls back to synthetic) ───
        self._image_observer: ImageObserver = image_observer or SyntheticImageObserver()

        # ── Build sub-systems ─────────────────────────────────────
        self._engine = DigitalTwinEngine(
            model_params=model_params,
            dt_minutes=dt_minutes,
        )
        self._solver = build_mpc_solver(config=mpc_config)

        # ── Prepare sequences (delegated to input provider) ───────
        # Weather needs extra steps for MPC look-ahead.
        weather_extra = 24  # 2 hours of look-ahead headroom
        self._weather_seq = self._input_provider.get_weather_sequence(
            n_steps=n_steps + weather_extra,
            dt_minutes=dt_minutes,
        )
        self._stages = prepare_growth_stages(n_steps, growth_stage)

        # ── Initial state (delegated to input provider) ───────────
        self._initial_state = self._input_provider.get_initial_state()

        # ── Runtime state ─────────────────────────────────────────
        self._prev_action: ActuatorState | None = None
        self._current_action = ActuatorState()
        self._last_mpc_solution: MPCSolution | None = None

    # ── Properties ────────────────────────────────────────────────────

    @property
    def engine(self) -> DigitalTwinEngine:
        return self._engine

    @property
    def initial_state(self) -> GreenhouseState:
        return self._initial_state

    @property
    def growth_stage(self) -> str:
        return self._growth_stage

    @property
    def image_observer(self) -> ImageObserver:
        return self._image_observer

    @property
    def input_provider(self) -> DTInputProvider:
        return self._input_provider

    # ── Main loop ─────────────────────────────────────────────────────

    def run(
        self,
        n_steps: int | None = None,
    ) -> Generator[DTLoopStepResult, None, None]:
        """Execute the closed-loop simulation, yielding one result per step.

        Workflow per step
        -----------------
        1. Determine cadence: is this an MPC step?  Image-refresh step?
        2. If MPC is due (or forced), run the solver and update the action.
        3. Build a ``DTStepInput`` from current state + action + weather.
        4. Run ``DigitalTwinEngine.step()`` to get the next state.
        5. Check ``should_force_mpc_update()``; if triggered and MPC did
           NOT already run this step, re-solve and re-run the DT step.
        6. Yield ``DTLoopStepResult``.
        7. Advance state: next_state becomes current state for the next
           iteration.

        Parameters
        ----------
        n_steps:
            Override the configured step count.

        Yields
        ------
        DTLoopStepResult
        """
        total = n_steps or self._n_steps
        if total > len(self._weather_seq):
            raise RuntimeError(
                f"Requested {total} steps but weather sequence has only "
                f"{len(self._weather_seq)} entries.  Increase weather "
                f"look-ahead or reduce n_steps."
            )
        state = GreenhouseState(**self._initial_state.to_dict())

        for step in range(total):
            ts = self._start_time + _dt.timedelta(
                minutes=step * self._dt_minutes,
            )
            weather = self._weather_seq[step]
            stage = self._stages[step] if step < len(self._stages) else self._growth_stage

            # Cadence flags
            mpc_due = (step % self._mpc_cadence == 0)
            image_due = (step % self._image_cadence == 0)

            mpc_ran = False
            mpc_forced = False
            mpc_solution: MPCSolution | None = None
            image_obs: ImageObservation | None = None

            # ── MPC solve (if due) ───────────────────────────────
            if mpc_due:
                mpc_solution = self._run_mpc(state, stage, weather, step, ts)
                self._current_action = mpc_solution.first_action
                mpc_ran = True
                self._last_mpc_solution = mpc_solution

            # ── Image observation refresh ─────────────────────────
            if image_due:
                image_obs = self._image_observer.observe(
                    growth_stage=stage,
                    state=state,
                    step_index=step,
                    timestamp=ts,
                )

            # ── DT step ──────────────────────────────────────────
            dt_input = DTStepInput(
                current_state=state,
                action=self._current_action,
                weather=weather,
                growth_stage=stage,
                disease_risk_score=state.disease_risk_score,
                disease_classification=(
                    "healthy leaves" if state.disease_risk_score < 0.3
                    else "early blight"
                ),
                dt_minutes=self._dt_minutes,
                step_index=step,
                timestamp=ts,
            )
            dt_out = self._engine.step(dt_input)

            # ── Event-triggered early MPC re-solve ────────────────
            if not mpc_ran and should_force_mpc_update(dt_out.next_state):
                logger.info(
                    "Step %d: event-triggered MPC re-solve "
                    "(T=%.1f, RH=%.1f, risk=%.3f)",
                    step,
                    dt_out.next_state.indoor_temp,
                    dt_out.next_state.indoor_humidity,
                    dt_out.next_state.disease_risk_score,
                )
                mpc_solution = self._run_mpc(
                    state, stage, weather, step, ts,
                )
                self._current_action = mpc_solution.first_action
                mpc_ran = True
                mpc_forced = True
                self._last_mpc_solution = mpc_solution

                # Re-run DT step with the corrected action.
                dt_input.action = self._current_action
                dt_out = self._engine.step(dt_input)

            # ── Yield result ──────────────────────────────────────
            result = DTLoopStepResult(
                step_index=step,
                timestamp=ts,
                current_state=GreenhouseState(**state.to_dict()),
                weather_used=weather,
                next_state=dt_out.next_state,
                diagnostics=dt_out.diagnostics,
                snapshot=dt_out.snapshot,
                action_applied=self._current_action,
                mpc_ran_this_step=mpc_ran,
                mpc_forced=mpc_forced,
                mpc_solution=mpc_solution,
                mpc_cost=(
                    mpc_solution.total_cost if mpc_solution else None
                ),
                image_refresh_this_step=image_due,
                image_observation=image_obs,
                cadence_info={
                    "mpc_due": mpc_due,
                    "mpc_forced": mpc_forced,
                    "image_due": image_due,
                    "step_in_mpc_cycle": step % self._mpc_cadence,
                    "step_in_image_cycle": step % self._image_cadence,
                },
            )
            yield result

            # ── Advance state ─────────────────────────────────────
            self._prev_action = self._current_action
            state = dt_out.next_state

    # ── Private helpers ───────────────────────────────────────────────

    def _run_mpc(
        self,
        state: GreenhouseState,
        growth_stage: str,
        weather: WeatherState,
        step_index: int,
        timestamp: _dt.datetime,
    ) -> MPCSolution:
        """Build FusedState and invoke the MPC solver.

        Weather look-ahead: slices up to 12 future weather steps from the
        pre-generated sequence so the solver can anticipate diurnal shifts.
        """
        disease_risk = compute_disease_risk_score(
            temp=state.indoor_temp,
            humidity=state.indoor_humidity,
            leaf_wetness=state.leaf_wetness_proxy,
            growth_stage=growth_stage,
        )

        # Weather look-ahead (12 steps = 1 hour at 5-min dt).
        lookahead = 12
        wf = [
            w.to_dict()
            for w in self._weather_seq[step_index: step_index + lookahead]
        ]
        if not wf:
            wf = [weather.to_dict()]

        fused = build_fused_state(
            state=state,
            growth_stage=growth_stage,
            disease_risk=disease_risk,
            weather_forecast=wf,
            timestamp=timestamp,
        )

        solution = self._solver.solve(
            fused=fused,
            weather_forecast=wf,
            previous_control=self._prev_action,
        )

        logger.debug(
            "Step %d MPC: cost=%.4f converged=%s fallback=%s (%.1f ms)",
            step_index,
            solution.total_cost,
            solution.converged,
            solution.fallback_used,
            solution.solve_time_ms,
        )
        return solution
