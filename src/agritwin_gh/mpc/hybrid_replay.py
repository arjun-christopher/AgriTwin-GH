"""
Hybrid replay engine with digital-twin correction hooks.

Replays historical greenhouse data through the MPC controller while
optionally using the ``GreenhouseTransitionModel`` to generate
*model-corrected* next-state predictions.  This enables:

* **Counterfactual analysis** — "What would the MPC have done?"
* **Model validation** — compare model-predicted vs. actually-observed states.
* **Digital-twin drift detection** — quantify the correction delta.

Usage::

    from agritwin_gh.mpc.hybrid_replay import ReplayConfig, ReplayEngine

    engine = ReplayEngine(solver=solver, model=model, config=ReplayConfig())
    for step in engine.replay(historical_states, weather_series):
        print(step.correction_delta)
"""

from __future__ import annotations

import datetime as _dt
import logging
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Generator

from .greenhouse_model import GreenhouseTransitionModel
from .mpc_solver import MPCSolution, MPCSolver
from .state import ActuatorState, FusedState, GreenhouseState, WeatherState

logger = logging.getLogger(__name__)


# ── Configuration ─────────────────────────────────────────────────────────────


@dataclass
class ReplayConfig:
    """Configuration for a hybrid replay session.

    Attributes
    ----------
    use_model_predicted_start : bool
        If *True*, each step starts from the **model-predicted** state
        (closed-loop simulation).  If *False* (default), each step starts
        from the **DB-observed** state (open-loop replay).
    compute_correction : bool
        Whether to compute the delta between model-predicted and actual
        observed next state.
    replay_id : str
        Unique identifier.  Auto-generated if empty.
    """

    use_model_predicted_start: bool = False
    compute_correction: bool = True
    replay_id: str = ""

    def __post_init__(self) -> None:
        if not self.replay_id:
            self.replay_id = f"replay-{uuid.uuid4().hex[:12]}"


# ── Per-step result ───────────────────────────────────────────────────────────


@dataclass
class ReplayStep:
    """Result of one hybrid-replay step.

    Attributes
    ----------
    step_index : int
        Ordinal position in the replay sequence.
    timestamp : datetime | None
        Logical time of this step.
    observed_state : GreenhouseState
        What the DB recorded for this timestep.
    mpc_action : ActuatorState
        Control action the MPC solver would have applied.
    model_predicted_state : GreenhouseState | None
        State predicted by ``GreenhouseTransitionModel`` given
        ``observed_state`` + ``mpc_action``.
    actual_next_state : GreenhouseState | None
        What the DB recorded for the *next* timestep (ground truth).
    correction_delta : dict[str, float]
        ``actual − predicted`` per state variable (empty if correction
        is disabled or data unavailable).
    solution : MPCSolution | None
        Full solver result for deeper inspection.
    """

    step_index: int = 0
    timestamp: _dt.datetime | None = None
    observed_state: GreenhouseState = field(default_factory=GreenhouseState)
    mpc_action: ActuatorState = field(default_factory=ActuatorState)
    model_predicted_state: GreenhouseState | None = None
    actual_next_state: GreenhouseState | None = None
    correction_delta: dict[str, float] = field(default_factory=dict)
    solution: MPCSolution | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "step_index": self.step_index,
            "timestamp": str(self.timestamp) if self.timestamp else None,
            "observed_state": self.observed_state.to_dict(),
            "mpc_action": self.mpc_action.to_dict(),
            "correction_delta": self.correction_delta,
        }
        if self.model_predicted_state is not None:
            d["model_predicted_state"] = self.model_predicted_state.to_dict()
        if self.actual_next_state is not None:
            d["actual_next_state"] = self.actual_next_state.to_dict()
        return d


# ── Replay summary ────────────────────────────────────────────────────────────


@dataclass
class ReplaySummary:
    """Aggregate statistics for a completed replay session."""

    replay_id: str = ""
    total_steps: int = 0
    mean_correction: dict[str, float] = field(default_factory=dict)
    max_correction: dict[str, float] = field(default_factory=dict)
    solver_convergence_rate: float = 0.0
    mean_solve_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Hook protocol ─────────────────────────────────────────────────────────────

# Hooks are plain callables — no ABC needed.
# on_step_complete : (ReplayStep) -> None
# on_correction_applied : (dict[str, float]) -> None
OnStepComplete = Callable[["ReplayStep"], None]
OnCorrectionApplied = Callable[[dict[str, float]], None]


# ── Replay engine ─────────────────────────────────────────────────────────────


class ReplayEngine:
    """Replays a sequence of historical states through the MPC controller.

    Parameters
    ----------
    solver : MPCSolver
        The MPC solver instance (with cost function, constraints, etc.).
    model : GreenhouseTransitionModel
        Used to compute model-predicted next states.
    config : ReplayConfig
        Session-level configuration.
    on_step_complete : callable, optional
        Hook invoked after each step is processed.
    on_correction_applied : callable, optional
        Hook invoked when a correction delta is computed.
    """

    def __init__(
        self,
        solver: MPCSolver,
        model: GreenhouseTransitionModel,
        config: ReplayConfig | None = None,
        on_step_complete: OnStepComplete | None = None,
        on_correction_applied: OnCorrectionApplied | None = None,
    ) -> None:
        self._solver = solver
        self._model = model
        self._config = config or ReplayConfig()
        self._on_step = on_step_complete
        self._on_correction = on_correction_applied

    # ── Main replay loop ──────────────────────────────────────────────

    def replay(
        self,
        fused_states: list[FusedState],
        weather_series: list[WeatherState] | None = None,
    ) -> Generator[ReplayStep, None, ReplaySummary]:
        """Replay historical fused states through MPC and yield per-step results.

        Parameters
        ----------
        fused_states : list[FusedState]
            Chronologically ordered fused states (one per control step).
        weather_series : list[WeatherState], optional
            Matching weather observations (same length as *fused_states*).
            Used by the transition model for disturbance input.

        Yields
        ------
        ReplayStep
            Per-step result with correction deltas.

        Returns
        -------
        ReplaySummary
            Aggregate statistics (accessible via ``generator.send(None)``
            after the loop or via the ``replay_collect`` convenience method).
        """
        cfg = self._config
        prev_action: ActuatorState | None = None
        running_state: GreenhouseState | None = None

        converged_count = 0
        total_solve_ms = 0.0
        correction_accum: dict[str, list[float]] = {}

        for idx in range(len(fused_states)):
            fused = fused_states[idx]

            # Determine starting state for this step
            if cfg.use_model_predicted_start and running_state is not None:
                effective_fused = self._override_greenhouse_state(fused, running_state)
            else:
                effective_fused = fused

            # MPC solve
            solution = self._solver.solve(
                fused=effective_fused,
                previous_control=prev_action,
            )
            action = solution.first_action
            prev_action = action

            if solution.converged:
                converged_count += 1
            total_solve_ms += solution.solve_time_ms

            # Model-predicted next state
            weather = (
                weather_series[idx] if weather_series and idx < len(weather_series)
                else WeatherState()
            )
            predicted_next = self._model.step(
                state=effective_fused.greenhouse_state,
                action=action,
                weather=weather,
            )
            running_state = predicted_next

            # Actual next state (from next fused entry, if available)
            actual_next: GreenhouseState | None = None
            if idx + 1 < len(fused_states):
                actual_next = fused_states[idx + 1].greenhouse_state

            # Correction delta
            delta: dict[str, float] = {}
            if cfg.compute_correction and actual_next is not None and predicted_next is not None:
                pred_arr = predicted_next.to_numpy()
                actual_arr = actual_next.to_numpy()
                diff = actual_arr - pred_arr
                from .constants import STATE_VARIABLES
                delta = {
                    STATE_VARIABLES[i]: round(float(diff[i]), 4)
                    for i in range(len(STATE_VARIABLES))
                }
                for k, v in delta.items():
                    correction_accum.setdefault(k, []).append(v)

            step = ReplayStep(
                step_index=idx,
                timestamp=fused.timestamp,
                observed_state=fused.greenhouse_state,
                mpc_action=action,
                model_predicted_state=predicted_next,
                actual_next_state=actual_next,
                correction_delta=delta,
                solution=solution,
            )

            # Fire hooks
            if self._on_step:
                self._on_step(step)
            if self._on_correction and delta:
                self._on_correction(delta)

            yield step

        # Build summary
        n = len(fused_states) or 1
        mean_corr = {
            k: round(sum(vs) / len(vs), 4)
            for k, vs in correction_accum.items()
        }
        max_corr = {
            k: round(max(abs(v) for v in vs), 4)
            for k, vs in correction_accum.items()
        }
        return ReplaySummary(
            replay_id=cfg.replay_id,
            total_steps=len(fused_states),
            mean_correction=mean_corr,
            max_correction=max_corr,
            solver_convergence_rate=converged_count / n,
            mean_solve_time_ms=total_solve_ms / n,
        )

    # ── Convenience ───────────────────────────────────────────────────

    def replay_collect(
        self,
        fused_states: list[FusedState],
        weather_series: list[WeatherState] | None = None,
    ) -> tuple[list[ReplayStep], ReplaySummary]:
        """Run the full replay and return all steps + summary.

        This is a non-streaming alternative to ``replay()`` that collects
        everything into memory.
        """
        steps: list[ReplayStep] = []
        gen = self.replay(fused_states, weather_series)
        try:
            while True:
                steps.append(next(gen))
        except StopIteration as exc:
            summary = exc.value or ReplaySummary(
                replay_id=self._config.replay_id,
                total_steps=len(steps),
            )
        return steps, summary

    # ── Internal helpers ──────────────────────────────────────────────

    @staticmethod
    def _override_greenhouse_state(
        fused: FusedState,
        new_gs: GreenhouseState,
    ) -> FusedState:
        """Return a shallow copy of *fused* with the greenhouse state replaced."""
        from dataclasses import fields
        kwargs = {f.name: getattr(fused, f.name) for f in fields(fused)}
        kwargs["greenhouse_state"] = new_gs
        return FusedState(**kwargs)

    @property
    def config(self) -> ReplayConfig:
        return self._config
