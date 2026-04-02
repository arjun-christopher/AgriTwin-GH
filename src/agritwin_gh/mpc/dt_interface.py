"""
Digital Twin plant interface.

``DigitalTwinPlant`` is the thin, stable public interface that downstream
code (MPC runner, experiment runner, dashboard) uses to run DT simulations.
Internally it delegates all transition logic to ``DigitalTwinEngine``.

Usage
-----
>>> plant = DigitalTwinPlant()
>>> step_in = DTStepInput(
...     current_state=state,
...     action=actuators,
...     weather=weather,
...     growth_stage="flowering",
... )
>>> result = plant.step(step_in)
>>> next_state = result.next_state
"""

from __future__ import annotations

from .constants import DT_MINUTES
from .dt_engine import DigitalTwinEngine
from .dt_state import DTStepInput, DTStepOutput
from .greenhouse_model import GreenhouseModelParams, GreenhouseTransitionModel


class DigitalTwinPlant:
    """Virtual greenhouse plant model with diagnostic tracking.

    Wraps ``DigitalTwinEngine`` to provide ``step()`` and ``simulate()``
    with full diagnostics (energy, water, disease risk, effect attribution,
    state deltas, disease-environment flags).

    Parameters
    ----------
    model_params:
        Tuneable ARX model coefficients.  ``None`` → sensible defaults.
    dt_minutes:
        Step duration in minutes (default ``DT_MINUTES`` = 5).
    """

    def __init__(
        self,
        model_params: GreenhouseModelParams | None = None,
        dt_minutes: int = DT_MINUTES,
    ) -> None:
        self._engine = DigitalTwinEngine(
            model_params=model_params,
            dt_minutes=dt_minutes,
        )

    # ── Public API ────────────────────────────────────────────────────

    def step(self, step_input: DTStepInput) -> DTStepOutput:
        """Simulate one timestep forward.

        Parameters
        ----------
        step_input:
            Complete input payload for this step.

        Returns
        -------
        DTStepOutput
            Predicted next state, diagnostics, and a full snapshot.
        """
        return self._engine.step(step_input)

    def simulate(
        self,
        step_inputs: list[DTStepInput],
    ) -> list[DTStepOutput]:
        """Run multiple steps sequentially, chaining state forward.

        Each step's ``next_state`` becomes the ``current_state`` for the
        next input.  The ``current_state`` field in elements after the
        first is **overwritten** with the chained state.

        Parameters
        ----------
        step_inputs:
            Ordered list of step inputs.  ``step_inputs[0].current_state``
            is the initial state; subsequent entries need only provide
            ``action``, ``weather``, and context fields.

        Returns
        -------
        list[DTStepOutput]
            One output per input step.
        """
        return self._engine.simulate(step_inputs)

    # ── Properties ────────────────────────────────────────────────────

    @property
    def engine(self) -> DigitalTwinEngine:
        """Access the underlying DT engine."""
        return self._engine

    @property
    def model(self) -> GreenhouseTransitionModel:
        """Access the underlying ARX transition model."""
        return self._engine.model

    @property
    def dt_minutes(self) -> int:
        """Step duration in minutes."""
        return self._engine.dt_minutes
