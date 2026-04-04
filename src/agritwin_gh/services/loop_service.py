"""
LoopService — manages the lifecycle of the DT+MPC closed-loop simulation.

Responsibility
--------------
``LoopService`` is the single owner of a running ``DTLoop`` (synthetic mode)
or ``RealtimeLoop`` (DB-backed mode).  It bridges the DT loop generator with
the ``RuntimeStore`` so that every HTTP handler can read the latest computed
state without triggering any MPC work inline.

Data flow
---------
::

    LoopService.start_loop()           ← called once at startup
        → DTLoop.__init__(growth_stage, …)
        → store.set_loop_start(now)

    LoopService.run_one_step()         ← called every DT_MINUTES
        → next(self._gen)              ← advances DTLoop by one 5-min step
        → store.update_from_step_result(result)
        → store.accumulate_resources(kwh, l)   ← from estimate_energy()

    LoopService.run_background_loop()  ← asyncio coroutine, started in lifespan
        → loop: run_one_step()
               asyncio.sleep(DT_MINUTES * 60)

Modules reused
--------------
* ``agritwin_gh.mpc.dt_loop.DTLoop``                 — synthetic closed-loop engine
* ``agritwin_gh.mpc.realtime_core.RealtimeLoop``     — DB-backed loop (future Phase 3)
* ``agritwin_gh.mpc.realtime_core.estimate_energy``  — per-step kWh estimate
* ``agritwin_gh.mpc.constants.DT_MINUTES``           — 5-minute cadence constant

Override integration
--------------------
When the store is in ``'override'`` mode, ``run_one_step()`` reads the active
``OverrideConfig`` before the DT step and applies parameter overrides to the
DTLoop's input provider before advancing the generator.  Actuator overrides are
carried forward separately via ``RuntimeStore``'s actuator snapshot.

Thread / async safety
---------------------
``run_one_step()`` is synchronous and guarded by its own ``threading.Lock``
so that a concurrent HTTP request cannot call it mid-step.  The async
``run_background_loop()`` runs in the event loop and calls ``run_one_step()``
from a thread executor to avoid blocking the event loop during the MPC solve.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import logging
import threading
from typing import Any, Generator

from agritwin_gh.core.runtime_store import RuntimeStore, get_store

logger = logging.getLogger("agritwin.services.loop")


class LoopService:
    """Manages the DT+MPC loop lifecycle and feeds the RuntimeStore.

    Usage (synthetic mode — no DB)::

        svc = LoopService(store=get_store())
        svc.start_loop(growth_stage="flowering", days_elapsed=5.0)
        svc.run_one_step()            # advance one 5-minute DT step
        # --- or ---
        asyncio.create_task(svc.run_background_loop())

    Usage (DB-backed — future Phase 3)::

        svc = LoopService(store=get_store(), session=db_session, synthetic=False)
        svc.start_loop(growth_stage="flowering", days_elapsed=5.0, total_steps=288)
    """

    def __init__(
        self,
        store: RuntimeStore | None = None,
        session: Any | None = None,
        synthetic: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        store:
            Singleton ``RuntimeStore``.  ``None`` → ``get_store()`` is called lazily.
        session:
            SQLAlchemy session for DB-backed mode.  ``None`` → synthetic mode.
        synthetic:
            ``True`` (default) → ``DTLoop`` with synthetic weather input.
            ``False`` → ``RealtimeLoop`` reading real sensor data from PostgreSQL.
        """
        self._store: RuntimeStore = store or get_store()
        self._session = session
        self._synthetic = synthetic

        # Internal loop objects — set in start_loop()
        self._dt_loop: Any = None               # DTLoop or RealtimeLoop instance
        self._gen: Generator | None = None       # DTLoop.run() generator
        self._last_result: Any = None            # last DTLoopStepResult
        self._running: bool = False
        self._step_lock = threading.Lock()

        # State tracking
        self._growth_stage: str = "seedling"
        self._total_steps: int = 288            # one full 24-hour synthetic day

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    def start_loop(
        self,
        growth_stage: str = "flowering",
        days_elapsed: float = 0.0,
        total_steps: int = 288,
        weather_base_temp: float = 22.0,
    ) -> None:
        """Initialise the DT loop engine and record the loop start time.

        Creates either a ``DTLoop`` (synthetic) or ``RealtimeLoop`` (DB-backed)
        depending on ``self._synthetic``.  Stores the loop-start timestamp in
        the ``RuntimeStore`` so that resource accounting can reference it.

        Parameters
        ----------
        growth_stage:
            Starting canonical growth-stage label.
        days_elapsed:
            Elapsed days within the starting stage (for initial cadence info).
        total_steps:
            Total 5-minute steps to simulate in one loop run.
            Default 288 = 24 hours.
        weather_base_temp:
            Mean outdoor temperature for synthetic weather generation (°C).
        """
        from agritwin_gh.mpc.constants import GROWTH_STAGES
        if growth_stage not in GROWTH_STAGES:
            raise ValueError(
                f"Invalid growth stage '{growth_stage}'. "
                f"Must be one of: {list(GROWTH_STAGES)}"
            )

        self._growth_stage = growth_stage
        self._total_steps = total_steps
        now = _dt.datetime.now()

        if self._synthetic:
            from agritwin_gh.mpc.dt_loop import DTLoop
            self._dt_loop = DTLoop(
                growth_stage=growth_stage,
                start_time=now,
                n_steps=total_steps,
                days_elapsed=days_elapsed,
                weather_base_temp=weather_base_temp,
            )
            self._gen = self._dt_loop.run(n_steps=total_steps)
            logger.info(
                "LoopService: DTLoop started (stage=%s, steps=%d, synthetic=True)",
                growth_stage, total_steps,
            )
        else:
            # Phase 3: DB-backed RealtimeLoop
            from agritwin_gh.mpc.realtime_core import RealtimeLoop, RealtimeLoopConfig
            cfg = RealtimeLoopConfig(
                growth_stage=growth_stage,
                days_elapsed=days_elapsed,
                total_steps=total_steps,
            )
            self._dt_loop = RealtimeLoop(cfg, session=self._session)
            self._dt_loop.setup()
            # RealtimeLoop does not expose a generator; use step-by-step API.
            self._gen = None
            logger.info(
                "LoopService: RealtimeLoop started (stage=%s, steps=%d, synthetic=False)",
                growth_stage, total_steps,
            )

        self._running = True
        self._store.set_loop_start(now)
        logger.info("LoopService: loop_start_ts set to %s", now.isoformat())

    def stop_loop(self) -> None:
        """Signal the background loop to stop after the current step completes."""
        self._running = False
        logger.info("LoopService: stop requested — loop will halt after current step.")

    # ─────────────────────────────────────────────────────────────────────────
    # Step execution
    # ─────────────────────────────────────────────────────────────────────────

    def run_one_step(self) -> Any | None:
        """Advance the DT loop by one 5-minute step and update the RuntimeStore.

        Returns the ``DTLoopStepResult``, or ``None`` if the loop is exhausted
        or has not been started yet.

        The method is synchronous and protected by a per-instance lock so that
        concurrent calls (e.g. from tests) cannot interleave generator state.

        Workflow
        --------
        1. Call ``next(self._gen)`` to advance the DTLoop one step.
        2. Enrich ``cadence_info`` with ``days_elapsed`` and ``days_in_stage``
           (DTLoop does not track these internally).
        3. Call ``store.update_from_step_result(result)`` — writes climate,
           actuators, weather, disease, growth snapshots atomically.
        4. Estimate energy used by actuators via ``estimate_energy()`` and call
           ``store.accumulate_resources(kwh, water_l)``.
        5. If an override is active, apply any actuator overrides from the store.
        """
        if not self._running or self._gen is None:
            logger.debug("LoopService.run_one_step: loop not running or not started.")
            return None

        with self._step_lock:
            try:
                result = next(self._gen)
            except StopIteration:
                logger.info(
                    "LoopService: DTLoop exhausted after %d steps.  Stopping.",
                    self._total_steps,
                )
                self._running = False
                return None

            self._last_result = result

            # ── Enrich cadence_info with elapsed-day tracking ─────────────
            # DTLoop.run() sets mpc_due, image_due etc. but not days_elapsed /
            # days_in_stage.  Compute from step index.
            from agritwin_gh.mpc.constants import DT_MINUTES
            from agritwin_gh.mpc.realtime_core import STAGE_DURATION_HOURS, STAGE_DURATION_HOURS as _SDH
            step = int(result.step_index)
            hours_elapsed: float = step * DT_MINUTES / 60.0
            days_elapsed: float = hours_elapsed / 24.0
            stage_dur_hours: float = float(STAGE_DURATION_HOURS.get(self._growth_stage, 336))
            days_in_stage: float = min(days_elapsed, stage_dur_hours / 24.0)

            result.cadence_info.setdefault("days_elapsed", days_elapsed)
            result.cadence_info.setdefault("days_in_stage", days_in_stage)

            # ── Update RuntimeStore with full step result ──────────────────
            self._store.update_from_step_result(result)

            # ── Energy and water accounting ───────────────────────────────
            from agritwin_gh.mpc.realtime_core import estimate_energy
            energy_kwh = estimate_energy(result.action_applied)

            # Water: irrigation_qty is duty-cycle (0–1); at 2 L/min flow rate
            # a 5-minute step at full duty = 10 L.  Scale linearly.
            water_l: float = round(
                getattr(result.action_applied, "irrigation_qty", 0.0) * 10.0, 4
            )
            self._store.accumulate_resources(energy_kwh=energy_kwh, water_l=water_l)

            # ── Apply active actuator overrides from store ─────────────────
            self._apply_actuator_overrides_if_active()

            logger.debug(
                "LoopService: step %d complete — T=%.1f°C, RH=%.1f%%, "
                "energy=%.4f kWh, water=%.2f L",
                step,
                result.next_state.indoor_temp,
                result.next_state.indoor_humidity,
                energy_kwh,
                water_l,
            )
            return result

    def _apply_actuator_overrides_if_active(self) -> None:
        """If override mode is active, merge actuator_overrides into the store's
        actuator snapshot so the API reflects operator-set levels.

        Called once per step *after* the DT step has been committed to the store.
        Has no effect when the store is in 'live' mode.
        """
        override = self._store.get_override()
        if override is None or not override.actuator_overrides:
            return

        from agritwin_gh.core.runtime_store import ActuatorSnapshot
        state = self._store.get_latest_state()
        levels = dict(state.actuators.levels)
        levels.update(override.actuator_overrides)

        override_snap = ActuatorSnapshot(
            levels=levels,
            on_off={k: v > 0.0 for k, v in levels.items()},
            mode="manual",
            mpc_ran_this_step=state.actuators.mpc_ran_this_step,
        )
        self._store.update_latest_state(actuators=override_snap)

    # ─────────────────────────────────────────────────────────────────────────
    # Background async loop
    # ─────────────────────────────────────────────────────────────────────────

    async def run_background_loop(
        self,
        interval_seconds: float | None = None,
    ) -> None:
        """Async coroutine that drives the DT loop in the background.

        Yields control to the FastAPI event loop between steps so HTTP requests
        are served without blocking.  The MPC solve (up to ~200 ms) is run in
        a thread executor to prevent stalling the event loop.

        Parameters
        ----------
        interval_seconds:
            Wall-clock seconds to wait between synthetic DT steps.
            ``None`` → ``DT_MINUTES * 60`` (300 s = 5 min).
            For development / demos, pass a smaller value (e.g. ``5.0``).
        """
        from agritwin_gh.mpc.constants import DT_MINUTES

        wait = interval_seconds if interval_seconds is not None else float(DT_MINUTES * 60)
        loop = asyncio.get_running_loop()

        logger.info(
            "LoopService.run_background_loop: starting — interval=%.0f s, "
            "total_steps=%d, stage=%s",
            wait, self._total_steps, self._growth_stage,
        )

        while self._running:
            try:
                # Offload the synchronous MPC solve to a thread pool worker so
                # the event loop stays responsive to HTTP requests and WebSocket
                # messages during the ~100 ms solver window.
                result = await loop.run_in_executor(None, self.run_one_step)
                if result is None:
                    logger.info(
                        "LoopService.run_background_loop: loop exhausted or stopped."
                    )
                    break
            except Exception as exc:
                logger.error(
                    "LoopService.run_background_loop: unhandled error — %s", exc,
                    exc_info=True,
                )
                # Brief back-off to avoid a tight error loop hammering the solver.
                await asyncio.sleep(10.0)
                continue

            await asyncio.sleep(wait)

        logger.info("LoopService.run_background_loop: exited.")

    # ─────────────────────────────────────────────────────────────────────────
    # Manual refresh
    # ─────────────────────────────────────────────────────────────────────────

    def refresh_runtime_store(self) -> bool:
        """Re-apply the last step result to the RuntimeStore.

        Useful after an override config is applied and the caller wants the
        store to immediately reflect any actuator override levels without
        waiting for the next DT step.

        Returns ``True`` if a previous result was available, ``False`` if no
        step has been run yet.
        """
        if self._last_result is None:
            logger.debug("LoopService.refresh_runtime_store: no result available yet.")
            return False

        self._store.update_from_step_result(self._last_result)
        self._apply_actuator_overrides_if_active()
        logger.debug("LoopService.refresh_runtime_store: store refreshed from last result.")
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # Status
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        """True when the loop has been started and has not yet been stopped."""
        return self._running

    @property
    def last_result(self) -> Any | None:
        """The most recent ``DTLoopStepResult``, or ``None`` before first step."""
        return self._last_result

    @property
    def growth_stage(self) -> str:
        """The growth stage used to initialise this loop run."""
        return self._growth_stage
