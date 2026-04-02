"""
Receding-horizon MPC solver for AgriTwin-GH.

Optimises actuator commands over a finite horizon by rolling forward the
``GreenhouseTransitionModel`` and minimising the cost function assembled
by ``CostBuilder``.

**Mathematical structure**

Decision variable:

    u = [u_0, u_1, …, u_{N-1}]  ∈  R^{N_u × N}

    where N_u = 7 (CONTROL_VARIABLES) and N = control horizon steps.

Objective (single-shooting formulation):

    min_u  J(u) = Σ_{k=0}^{N-1}  ℓ(x_k, u_k, u_{k-1})  +  V_f(x_N)

    where x_{k+1} = f(x_k, u_k, d_k)   (greenhouse transition model)
          d_k                            (weather disturbance at step k)

Constraints:

    • u_lo ≤ u_k ≤ u_hi              (actuator box bounds)
    • |u_k − u_{k-1}| ≤ Δu_max       (rate-of-change limits)
    • u_k[irrigation_qty] ≥ 0         (irrigation non-negativity, via bound)
    • (optional) x_lo ≤ x_k ≤ x_hi   (environment safe-range soft penalties)

Solver:

    ``scipy.optimize.minimize(method=config.solver_method)``  (default SLSQP).
    Falls back to a single step of the ``RuleBasedController`` if the optimiser
    fails to converge.

Primary scope (this version):

    Optimises **temperature, humidity, soil moisture** tracking.
    CO2, light, disease-penalty, and growth-stage adaptive weights are
    structurally present via ``CostBuilder`` but dormant (low default weights)
    until explicitly enabled.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import minimize, OptimizeResult

from .baseline_controller import RuleBasedController
from .config import MPCConfig
from .constraints import ConstraintSet, get_default_constraints, tighten_constraints_for_disease
from .cost_function import CostBuilder, DiseaseContext
from .constants import CONTROL_VARIABLES, STATE_VARIABLES
from .greenhouse_model import GreenhouseTransitionModel
from .setpoints import get_setpoint
from .state import ActuatorState, FusedState, GreenhouseState, WeatherState
from .weather_adaptation import WeatherAdaptiveModifiers, compute_weather_adaptation

logger = logging.getLogger(__name__)

N_U = len(CONTROL_VARIABLES)   # 7
N_X = len(STATE_VARIABLES)     # 9

# Index of irrigation_qty in CONTROL_VARIABLES
_IRR_IDX = list(CONTROL_VARIABLES).index("irrigation_qty")


# ── Solution dataclass ────────────────────────────────────────────────────────


@dataclass
class MPCSolution:
    """Result of a single MPC solve call.

    Attributes
    ----------
    optimal_controls : list[ActuatorState]
        Optimal actuator sequence over the control horizon.
    predicted_states : list[GreenhouseState]
        Predicted state trajectory (length = horizon + 1 including initial).
    first_action : ActuatorState
        ``optimal_controls[0]`` — the only action actually applied
        (receding-horizon principle).
    total_cost : float
        Optimised objective value.
    cost_breakdown : dict[str, float]
        Per-component cost contributions for inspection / logging.
    solve_time_ms : float
        Wall-clock time of the ``scipy.optimize.minimize`` call.
    solver_status : str
        Human-readable solver exit status.
    converged : bool
        Whether the solver reached a feasible local optimum.
    fallback_used : bool
        Whether the baseline controller was used as a fallback.
    n_iterations : int
        Number of solver iterations.
    n_function_evals : int
        Number of objective-function evaluations.
    """

    optimal_controls: list[ActuatorState] = field(default_factory=list)
    predicted_states: list[GreenhouseState] = field(default_factory=list)
    first_action: ActuatorState = field(default_factory=ActuatorState)
    total_cost: float = 0.0
    cost_breakdown: dict[str, float] = field(default_factory=dict)
    solve_time_ms: float = 0.0
    solver_status: str = ""
    converged: bool = False
    fallback_used: bool = False
    n_iterations: int = 0
    n_function_evals: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "first_action": self.first_action.to_dict(),
            "total_cost": self.total_cost,
            "cost_breakdown": dict(self.cost_breakdown),
            "solve_time_ms": self.solve_time_ms,
            "solver_status": self.solver_status,
            "converged": self.converged,
            "fallback_used": self.fallback_used,
            "n_iterations": self.n_iterations,
            "n_function_evals": self.n_function_evals,
            "horizon_length": len(self.optimal_controls),
        }


# ── MPC Solver ────────────────────────────────────────────────────────────────


class MPCSolver:
    """Receding-horizon Model Predictive Controller.

    Parameters
    ----------
    config : MPCConfig
        Solver and cost configuration.
    model : GreenhouseTransitionModel or None
        Plant model for rollout.  Defaults to ``GreenhouseTransitionModel()``.
    constraints : ConstraintSet or None
        Actuator and environmental constraints.  If ``None`` the stage-
        independent defaults are used.
    fallback_controller : RuleBasedController or None
        Baseline controller used when the optimiser fails.  If ``None`` a
        default ``RuleBasedController`` is instantiated.
    """

    def __init__(
        self,
        config: MPCConfig,
        model: GreenhouseTransitionModel | None = None,
        constraints: ConstraintSet | None = None,
        fallback_controller: RuleBasedController | None = None,
    ) -> None:
        self._cfg = config
        self._model = model or GreenhouseTransitionModel()
        self._constraints = constraints or get_default_constraints()
        self._fallback = fallback_controller or RuleBasedController(
            constraints=self._constraints,
        )
        # Cached from last solve for runner handoff
        self._last_weather_stress: dict[str, float] | None = None
        self._last_constraint_tightening: dict[str, Any] | None = None

    # ── Public API ────────────────────────────────────────────────────

    def solve(
        self,
        fused: FusedState,
        weather_forecast: list[dict[str, Any]] | None = None,
        previous_control: ActuatorState | None = None,
        horizon_override: int | None = None,
    ) -> MPCSolution:
        """Run the MPC optimisation for the current timestep.

        Parameters
        ----------
        fused : FusedState
            Comprehensive fused state assembled by ``StateFusion``.
        weather_forecast : list of dicts, optional
            Per-step weather disturbance over the horizon.  Falls back to
            the forecast stored inside *fused* if ``None``.
        previous_control : ActuatorState, optional
            Last applied control (for rate-limit and switching-cost
            initialisation).  Zero actuators if ``None``.
        horizon_override : int, optional
            Override the control horizon from config (useful for shorter
            test horizons).

        Returns
        -------
        MPCSolution
        """
        growth_stage = fused.growth_stage or "seedling"
        N = horizon_override or self._cfg.control_horizon_steps

        # Resolve weather forecast
        weather_seq = self._prepare_weather(fused, weather_forecast, N)

        # ── Disease context ───────────────────────────────────────────
        disease_ctx = DiseaseContext.from_fused(fused)

        # ── Weather-adaptive cost modifiers ───────────────────────────
        setpoint = get_setpoint(growth_stage)
        weather_mods = compute_weather_adaptation(
            weather_seq, setpoint, N, self._cfg,
        )

        # Cache weather stress summary for runner handoff
        self._last_weather_stress = weather_mods.to_summary()

        # ── Stage-transition info ─────────────────────────────────────
        next_stage = fused.next_stage or None
        dt_minutes = self._cfg.dt_minutes
        steps_to_transition: int | None = None
        if next_stage and fused.hours_to_transition > 0:
            steps_to_transition = int(fused.hours_to_transition * 60 / dt_minutes)

        # Build cost function for the current growth stage
        cost_builder = CostBuilder(
            self._cfg,
            growth_stage,
            energy_costs=self._cfg.energy_costs or None,
            disease_context=disease_ctx,
            next_stage=next_stage,
            steps_to_transition=steps_to_transition,
        )

        # Resolve constraints (stage-aware, then tighten for disease)
        constraints = get_default_constraints(growth_stage)
        constraints = tighten_constraints_for_disease(
            constraints,
            disease_ctx.risk_score,
            rh_tightening_threshold=self._cfg.disease_rh_tightening_risk_threshold,
            rh_tightened_ceiling=self._cfg.disease_rh_tightened_ceiling,
            fogger_suppress_threshold=self._cfg.disease_fogger_suppress_risk_threshold,
            fogger_suppressed_max_duty=self._cfg.disease_fogger_suppressed_max_duty,
            severity_24h=disease_ctx.severity_24h,
        )

        # Cache constraint tightening info for runner handoff
        self._last_constraint_tightening = self._extract_tightening_info(
            constraints, disease_ctx.risk_score,
        )

        # Previous control for rate-limit and switching cost
        u_prev = (previous_control or ActuatorState()).to_numpy()

        # Initial state
        x0 = fused.greenhouse_state

        # ── Build optimisation problem ────────────────────────────────
        bounds = self._build_bounds(N, constraints)
        rate_constraints = self._build_rate_constraints(N, u_prev, constraints)

        # Warm-start from baseline controller, clipped to satisfy rate constraints
        u_init = self._warm_start(
            x0, growth_stage, fused.disease_risk_score,
            weather_seq, N,
        )
        u_init = self._clip_to_rate_limits(u_init, u_prev, constraints)

        # Objective wrapper (closure over model, cost, weather)
        eval_count = [0]

        def objective(u_flat: np.ndarray) -> float:
            eval_count[0] += 1
            return self._evaluate_objective(
                u_flat, x0, weather_seq, cost_builder, N,
                weather_modifiers=weather_mods.modifiers,
            )

        # ── Solve ─────────────────────────────────────────────────────
        t0 = time.perf_counter()
        try:
            result: OptimizeResult = minimize(
                objective,
                u_init.ravel(),
                method=self._cfg.solver_method,
                bounds=bounds,
                constraints=rate_constraints,
                options={
                    "maxiter": self._cfg.solver_max_iter,
                    "ftol": self._cfg.solver_ftol,
                    "disp": False,
                },
            )
            solve_ms = (time.perf_counter() - t0) * 1000.0
            converged = result.success
            solver_status = result.message if isinstance(result.message, str) else str(result.message)
            n_iter = getattr(result, "nit", 0)
            u_opt = result.x.reshape(N, N_U)

        except Exception as exc:
            solve_ms = (time.perf_counter() - t0) * 1000.0
            logger.warning("MPC solver exception: %s — using fallback", exc)
            return self._fallback_solution(
                x0, growth_stage, fused, weather_seq, N,
                solve_ms, str(exc), eval_count[0],
            )

        if not converged:
            # Try to salvage non-converged result if it still improves
            # over the warm-start (baseline) cost.
            use_nc = False
            if np.all(np.isfinite(result.x)):
                u_nc = result.x.reshape(N, N_U)
                u_nc = self._clip_to_rate_limits(u_nc, u_prev, constraints)
                nc_cost = objective(u_nc.ravel())
                init_cost = objective(u_init.ravel())
                if nc_cost < init_cost:
                    u_opt = u_nc
                    use_nc = True
                    logger.info(
                        "Non-converged result improves cost (%.4f → %.4f) "
                        "— using clipped result",
                        init_cost, nc_cost,
                    )
            if not use_nc:
                logger.warning(
                    "MPC solver did not converge (%s) — using fallback",
                    solver_status,
                )
                return self._fallback_solution(
                    x0, growth_stage, fused, weather_seq, N,
                    solve_ms, solver_status, eval_count[0],
                )

        # ── Unpack solution ───────────────────────────────────────────
        optimal_controls = [ActuatorState.from_numpy(u_opt[k]) for k in range(N)]
        predicted_states = self._rollout_states(x0, optimal_controls, weather_seq)

        # Cost breakdown for inspection
        breakdown = self._compute_cost_breakdown(
            x0, predicted_states, optimal_controls, cost_builder,
            weather_modifiers=weather_mods.modifiers,
        )
        breakdown["weather_temp_stress"] = round(float(np.mean(weather_mods.temp_stress)) if weather_mods.temp_stress else 0.0, 4)
        breakdown["weather_rh_stress"] = round(float(np.mean(weather_mods.rh_stress)) if weather_mods.rh_stress else 0.0, 4)
        breakdown["disease_risk_score"] = round(disease_ctx.risk_score, 4)
        breakdown["disease_severity_amplifier"] = round(disease_ctx.severity_amplifier, 4)

        logger.info(
            "MPC solved | J=%.4f | %d iter | %.1f ms | %s",
            result.fun, n_iter, solve_ms, solver_status,
        )

        return MPCSolution(
            optimal_controls=optimal_controls,
            predicted_states=[x0] + predicted_states,
            first_action=optimal_controls[0],
            total_cost=float(result.fun),
            cost_breakdown=breakdown,
            solve_time_ms=solve_ms,
            solver_status=solver_status,
            converged=True,
            fallback_used=False,
            n_iterations=n_iter,
            n_function_evals=eval_count[0],
        )

    def get_first_action(
        self,
        fused: FusedState,
        weather_forecast: list[dict[str, Any]] | None = None,
        previous_control: ActuatorState | None = None,
    ) -> ActuatorState:
        """Convenience: solve and return only the first actuator command."""
        solution = self.solve(fused, weather_forecast, previous_control)
        return solution.first_action

    # ── Solver context accessors (for runner handoff) ─────────────────

    @property
    def last_weather_stress_summary(self) -> dict[str, float] | None:
        """Weather stress summary from the most recent ``solve()`` call."""
        return self._last_weather_stress

    @property
    def last_constraint_tightening(self) -> dict[str, Any] | None:
        """Constraint tightening info from the most recent ``solve()`` call."""
        return self._last_constraint_tightening

    @staticmethod
    def _extract_tightening_info(
        constraints: ConstraintSet,
        risk_score: float,
    ) -> dict[str, Any]:
        """Build a summary dict of any constraint tightening that was applied."""
        info: dict[str, Any] = {"risk_score": round(risk_score, 4)}
        env = constraints.environmental
        if hasattr(env, "get"):
            rh_hi = env.get("indoor_humidity", (0, 100))[1]
        else:
            rh_hi = getattr(env, "indoor_humidity_max", 100.0)
        if rh_hi < 95.0:
            info["rh_ceiling"] = rh_hi
        fogger_bounds = constraints.actuator_bounds.get("fogger_duty", (0.0, 1.0))
        if fogger_bounds[1] < 1.0:
            info["fogger_suppressed"] = fogger_bounds[1]
        return info

    # ── Objective evaluation ──────────────────────────────────────────

    def _evaluate_objective(
        self,
        u_flat: np.ndarray,
        x0: GreenhouseState,
        weather_seq: list[dict[str, Any]],
        cost_builder: CostBuilder,
        N: int,
        weather_modifiers: list[np.ndarray] | None = None,
    ) -> float:
        """Single-shooting objective: roll forward and sum stage + terminal cost."""
        U = u_flat.reshape(N, N_U)
        controls = [ActuatorState.from_numpy(U[k]) for k in range(N)]
        trajectory = self._model.simulate(x0, controls, weather_seq[:N])

        # Build arrays for CostBuilder
        states_np = [x0.to_numpy()] + [s.to_numpy() for s in trajectory]
        controls_np = [U[k] for k in range(N)]

        return cost_builder.total_cost(states_np, controls_np, weather_modifiers)

    # ── Constraint construction ───────────────────────────────────────

    def _build_bounds(
        self,
        N: int,
        constraints: ConstraintSet,
    ) -> list[tuple[float, float]]:
        """Build scipy-compatible box bounds for the flattened decision variable.

        Returns a list of (lo, hi) tuples, length = N × N_U.
        """
        act_bounds = constraints.actuator_bounds
        step_bounds: list[tuple[float, float]] = []
        for name in CONTROL_VARIABLES:
            lo, hi = act_bounds.get(name, (0.0, 1.0))
            step_bounds.append((lo, hi))

        # Repeat for each horizon step
        return step_bounds * N

    def _build_rate_constraints(
        self,
        N: int,
        u_prev: np.ndarray,
        constraints: ConstraintSet,
    ) -> list[dict[str, Any]]:
        """Build linear inequality constraints enforcing rate-of-change limits.

        For each consecutive step pair (k-1, k) and each actuator j:
            u_k[j] - u_{k-1}[j] ≤  Δmax_j
            u_{k-1}[j] - u_k[j] ≤  Δmax_j   (i.e. |Δu| ≤ Δmax)

        Encoded as scipy 'ineq' constraints:  c(u) ≥ 0.
        """
        rate_limits = constraints.actuator_rate_limits
        # Add 10% slack to the constraint formulation so the warm-start
        # (clipped to exact limits) lies strictly inside the feasible
        # region, preventing SLSQP "Inequality constraints incompatible".
        _SLACK = 1.05
        rate_lo = np.array([rate_limits.get(n, (-1.0, 1.0))[0] * _SLACK for n in CONTROL_VARIABLES])
        rate_hi = np.array([rate_limits.get(n, (-1.0, 1.0))[1] * _SLACK for n in CONTROL_VARIABLES])

        def rate_constraint(u_flat: np.ndarray) -> np.ndarray:
            U = u_flat.reshape(N, N_U)
            violations = []
            for k in range(N):
                u_km1 = u_prev if k == 0 else U[k - 1]
                delta = U[k] - u_km1
                # rate_hi - delta ≥ 0  and  delta - rate_lo ≥ 0
                violations.append(rate_hi - delta)
                violations.append(delta - rate_lo)
            return np.concatenate(violations)

        return [{"type": "ineq", "fun": rate_constraint}]

    # ── Weather preparation ───────────────────────────────────────────

    def _prepare_weather(
        self,
        fused: FusedState,
        weather_forecast: list[dict[str, Any]] | None,
        N: int,
    ) -> list[dict[str, Any]]:
        """Resolve weather forecast to at least N dicts."""
        if weather_forecast and len(weather_forecast) >= N:
            return weather_forecast[:N]

        # Use fused.weather_forecast if available
        wf = weather_forecast or fused.weather_forecast or []
        if not wf:
            # Persistence: repeat current conditions
            gs = fused.greenhouse_state
            fallback = {
                "temp_external": gs.indoor_temp,
                "humidity_external": gs.indoor_humidity,
                "solar_radiation": gs.light_intensity,
            }
            wf = [fallback]

        # Pad by repeating last entry
        while len(wf) < N:
            wf.append(wf[-1])
        return wf[:N]

    # ── Warm start ────────────────────────────────────────────────────

    @staticmethod
    def _clip_to_rate_limits(
        U: np.ndarray,
        u_prev: np.ndarray,
        constraints: ConstraintSet,
    ) -> np.ndarray:
        """Clip warm-start control sequence so it satisfies rate constraints.

        Ensures the initial point passed to the optimizer is feasible with
        respect to both box bounds and rate-of-change limits, preventing
        SLSQP from starting with an infeasible iterate.
        """
        N, Nu = U.shape
        rate_limits = constraints.actuator_rate_limits
        act_bounds = constraints.actuator_bounds
        rate_lo = np.array([rate_limits.get(n, (-1.0, 1.0))[0] for n in CONTROL_VARIABLES])
        rate_hi = np.array([rate_limits.get(n, (-1.0, 1.0))[1] for n in CONTROL_VARIABLES])
        box_lo = np.array([act_bounds.get(n, (0.0, 1.0))[0] for n in CONTROL_VARIABLES])
        box_hi = np.array([act_bounds.get(n, (0.0, 1.0))[1] for n in CONTROL_VARIABLES])

        U_clipped = U.copy()
        prev = u_prev.copy()
        for k in range(N):
            # Clamp to rate limits relative to previous step
            lo = np.maximum(prev + rate_lo, box_lo)
            hi = np.minimum(prev + rate_hi, box_hi)
            U_clipped[k] = np.clip(U_clipped[k], lo, hi)
            prev = U_clipped[k]
        return U_clipped

    def _warm_start(
        self,
        x0: GreenhouseState,
        growth_stage: str,
        disease_risk: float,
        weather_seq: list[dict[str, Any]],
        N: int,
    ) -> np.ndarray:
        """Generate an initial control sequence from the baseline controller.

        Rolls the baseline forward for N steps to produce a physically
        plausible starting point for the optimiser.
        """
        baseline = RuleBasedController(constraints=self._constraints)
        state = x0
        U = np.zeros((N, N_U), dtype=np.float64)

        for k in range(N):
            payload = baseline.compute_action(
                state, growth_stage, disease_risk=disease_risk,
            )
            U[k] = payload.actuators.to_numpy()
            state = self._model.step(state, payload.actuators, weather_seq[k])

        return U

    # ── State rollout ─────────────────────────────────────────────────

    def _rollout_states(
        self,
        x0: GreenhouseState,
        controls: list[ActuatorState],
        weather_seq: list[dict[str, Any]],
    ) -> list[GreenhouseState]:
        """Roll the model forward using the given control sequence."""
        return self._model.simulate(x0, controls, weather_seq[:len(controls)])

    # ── Cost breakdown ────────────────────────────────────────────────

    def _compute_cost_breakdown(
        self,
        x0: GreenhouseState,
        trajectory: list[GreenhouseState],
        controls: list[ActuatorState],
        cost_builder: CostBuilder,
        weather_modifiers: list[np.ndarray] | None = None,
    ) -> dict[str, float]:
        """Decompose the total cost into per-component contributions.

        Provides visibility into which terms dominate the objective —
        essential for tuning.
        """
        N = len(controls)
        states_np = [x0.to_numpy()] + [s.to_numpy() for s in trajectory]
        controls_np = [c.to_numpy() for c in controls]

        total_stage = cost_builder.total_cost(
            states_np, controls_np, weather_modifiers,
        )

        # Separate terminal contribution
        terminal = 0.0
        if len(states_np) > N:
            terminal = cost_builder.terminal_cost.evaluate(states_np[N])

        stage_only = total_stage - terminal

        return {
            "stage_cost_total": round(stage_only, 4),
            "terminal_cost": round(terminal, 4),
            "total": round(total_stage, 4),
        }

    # ── Fallback ──────────────────────────────────────────────────────

    def _fallback_solution(
        self,
        x0: GreenhouseState,
        growth_stage: str,
        fused: FusedState,
        weather_seq: list[dict[str, Any]],
        N: int,
        solve_ms: float,
        status_msg: str,
        n_evals: int,
    ) -> MPCSolution:
        """Produce a solution from the baseline controller when the solver fails."""
        logger.info("Generating fallback solution from RuleBasedController")
        baseline = RuleBasedController(constraints=self._constraints)
        controls: list[ActuatorState] = []
        state = x0

        for k in range(N):
            payload = baseline.compute_action(
                state, growth_stage, disease_risk=fused.disease_risk_score,
            )
            controls.append(payload.actuators)
            state = self._model.step(state, payload.actuators, weather_seq[k])

        trajectory = self._rollout_states(x0, controls, weather_seq)

        # Evaluate cost of fallback trajectory (basic — no disease/weather enrichment)
        cost_builder = CostBuilder(self._cfg, growth_stage)
        states_np = [x0.to_numpy()] + [s.to_numpy() for s in trajectory]
        controls_np = [c.to_numpy() for c in controls]
        total_cost = cost_builder.total_cost(states_np, controls_np, None)

        return MPCSolution(
            optimal_controls=controls,
            predicted_states=[x0] + trajectory,
            first_action=controls[0] if controls else ActuatorState(),
            total_cost=total_cost,
            cost_breakdown={"fallback": total_cost},
            solve_time_ms=solve_ms,
            solver_status=f"FALLBACK: {status_msg}",
            converged=False,
            fallback_used=True,
            n_iterations=0,
            n_function_evals=n_evals,
        )

    # ── Accessors ─────────────────────────────────────────────────────

    @property
    def model(self) -> GreenhouseTransitionModel:
        return self._model

    @property
    def config(self) -> MPCConfig:
        return self._cfg
