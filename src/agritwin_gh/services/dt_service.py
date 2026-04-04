"""
DTService — orchestrates the Digital-Twin loop for the FastAPI layer.

Responsibility
--------------
This service is the primary bridge between the FastAPI route handlers and the
core DT / MPC engine.  It:

1. Manages a ``RealtimeLoop`` (or ``DTLoop`` in synthetic mode) instance.
2. Drives DT steps either from a background async coroutine or on-demand from
   tests.
3. After each step, updates ``RuntimeStore`` so HTTP handlers can read the
   latest state without running the solver inline.
4. Accepts and applies manual overrides (``apply_override``) and named
   presets (``apply_preset``).
5. Exposes ``get_current_state()`` which projects ``RuntimeStore`` into a
   ``DTStateResponse`` for the ``GET /api/dt/state`` endpoint.

Modules reused from the MPC layer
----------------------------------
* ``agritwin_gh.mpc.realtime_core.RealtimeLoop``   — production DB-backed loop.
* ``agritwin_gh.mpc.dt_loop.DTLoop``               — synthetic simulation loop.
* ``agritwin_gh.mpc.dt_runtime_prep``              — initial state + weather prep.
* ``agritwin_gh.mpc.state_fusion.StateFusion``     — assembles ``FusedState``.

Usage
-----
::

    from agritwin_gh.services.dt_service import DTService
    from agritwin_gh.core.runtime_store import RuntimeStore

    svc = DTService(session=db_session, store=store)
    svc.setup(growth_stage="flowering", days_elapsed=10.0)
    result = svc.tick()          # advance one DT step
    response = svc.get_current_state()

Implementation status
---------------------
This is a **placeholder skeleton**.  Method bodies are intentionally empty
(or minimal) until the implementation phase.  The class structure, imports,
and docstrings define the contract that the route handlers depend on.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy.orm import Session

from agritwin_gh.core.runtime_store import RuntimeStore

logger = logging.getLogger("agritwin.services.dt")


class DTService:
    """Orchestrates the DT + MPC loop for the HTTP API layer."""

    def __init__(
        self,
        session: Session | None = None,
        store: RuntimeStore | None = None,
        synthetic: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        session:
            SQLAlchemy session for DB-backed mode.  ``None`` in synthetic mode.
        store:
            Singleton ``RuntimeStore`` shared with the API layer.
        synthetic:
            If ``True`` (default), use ``DTLoop`` with synthetic weather
            (no DB required).  If ``False``, use ``RealtimeLoop`` with real
            sensor data from PostgreSQL.
        """
        self._session = session
        self._store = store or RuntimeStore()
        self._synthetic = synthetic
        self._loop: Any = None   # RealtimeLoop or DTLoop — set in setup()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def setup(
        self,
        growth_stage: str = "flowering",
        days_elapsed: float = 0.0,
        total_steps: int | None = None,
    ) -> None:
        """Initialise the DT loop and prepare the initial state.

        TODO (implementation phase):
            if self._synthetic:
                from agritwin_gh.mpc.dt_loop import DTLoop
                self._loop = DTLoop(growth_stage=growth_stage, ...)
            else:
                from agritwin_gh.mpc.realtime_core import RealtimeLoop, RealtimeLoopConfig
                cfg = RealtimeLoopConfig(growth_stage=growth_stage, days_elapsed=days_elapsed,
                                        total_steps=total_steps or 288)
                self._loop = RealtimeLoop(cfg, session=self._session)
                self._loop.setup()
        """
        logger.info(
            "DTService.setup: stage=%s, days_elapsed=%.1f, synthetic=%s",
            growth_stage, days_elapsed, self._synthetic,
        )
        # Store setup context so the loop service can read it later.
        # DTService.setup() does not write to RuntimeStore directly; the
        # LoopService.start_loop() call in lifespan.py owns that transition.

    def tick(self) -> Any:
        """Advance the DT loop by one step and update the RuntimeStore.

        Returns the ``DTLoopStepResult`` from the loop.

        TODO (implementation phase):
            result = self._loop.step(self._store.step_index + 1)
            self._store.update_from_step_result(result)
            return result
        """
        state = self._store.get_latest_state()
        logger.debug(
            "DTService.tick: step_index=%d (delegated to LoopService)",
            state.climate.step_index,
        )
        return None

    async def run_background_loop(self, n_steps: int = 288) -> None:
        """Async coroutine that drives ``n_steps`` DT steps in the background.

        Yield control between steps so the event loop can handle HTTP requests.

        TODO (implementation phase):
            import asyncio
            for i in range(n_steps):
                self.tick()
                await asyncio.sleep(0)   # yield to event loop
        """
        logger.info("DTService.run_background_loop: %d steps (not yet implemented).", n_steps)

    # ── State access ──────────────────────────────────────────────────────────

    def get_current_state(self):
        """Return a ``DTStateResponse`` from the RuntimeStore."""
        return self._store.as_dt_state_response()

    # ── Manual control ────────────────────────────────────────────────────────

    def apply_override(self, param: str, value: float) -> None:
        """Record a manual parameter override for the next DT step.

        TODO (implementation phase):
            Map ``param`` to the correct field in ``GreenhouseState`` /
            ``MPCConfig`` and apply it before the next ``tick()``.
        """
        logger.info("DTService.apply_override: %s = %s", param, value)
        # Delegate to ControlService which owns the OverrideConfig write path.
        from agritwin_gh.services.control_service import ControlService
        from agritwin_gh.schemas.dt_schemas import DtParamOverrideRequest
        ControlService(store=self._store).apply_dt_param_override(
            DtParamOverrideRequest(param=param, value=float(value))
        )

    def apply_preset(self, preset_id: str) -> None:
        """Apply a named quick-action preset.

        Presets: ``day-cycle`` | ``night-cycle`` | ``emergency-flush``

        TODO (implementation phase):
            Translate preset_id into a dict of actuator overrides and call
            ``apply_override`` for each one.
        """
        logger.info("DTService.apply_preset: %s (not yet implemented)", preset_id)
