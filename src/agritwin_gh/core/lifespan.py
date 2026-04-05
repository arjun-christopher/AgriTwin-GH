"""
FastAPI lifespan context manager — startup and shutdown hooks.

Responsibility
--------------
``lifespan`` is an async context manager passed to ``FastAPI(lifespan=...)``
in ``agritwin_gh.api.app.create_app()``.

On **startup** the sequence is:

  1. Ensure the singleton ``RuntimeStore`` is alive.
  2. Instantiate ``LoopService`` and call ``start_loop()`` — this creates the
     ``DTLoop`` engine but does *not* block FastAPI startup.
  3. Run **one initial DT step** so the store holds real data before the first
     HTTP request arrives.
  4. If ``AGRITWIN_BACKGROUND_LOOP=1``, launch an async background task that
     advances the loop every ``AGRITWIN_STEP_INTERVAL_SEC`` seconds
     (default 300 s = 5 min).  Otherwise the server runs in **step-mode**:
     the store keeps the data from the initial step until the next server
     restart or a manual ``LoopService.run_one_step()`` call.

On **shutdown**:
  1. Signal the ``LoopService`` to stop (clears the ``_running`` flag so the
     background loop exits cleanly after its current step).
  2. Cancel the ``asyncio.Task`` for the background loop and await its
     cancellation with a 5-second timeout.
  3. Reset the module-level singletons so a hot-reload works correctly.

Environment variables
---------------------
``AGRITWIN_BACKGROUND_LOOP``  — ``"0"`` to disable continuous mode (default: ``"1"`` = enabled).
``AGRITWIN_GROWTH_STAGE``     — starting growth stage (default: ``"seedling"``).
``AGRITWIN_DAYS_ELAPSED``     — elapsed days within the starting stage (default: ``"0.0"``).
``AGRITWIN_TOTAL_STEPS``      — total 5-minute steps per loop run (default: ``"25632"`` = full
                                89-day seedling→ripe cycle at 5-min resolution).
``AGRITWIN_AUTO_ADVANCE_STAGE``— ``"0"`` to pin a single stage (default: ``"1"`` = advance).
``AGRITWIN_STEP_INTERVAL_SEC``— seconds between continuous-mode steps (default: ``"300"``).
``TESTING``                   — ``"1"`` to skip the loop entirely for unit tests.

Duplicate-startup guard
-----------------------
``_STARTUP_LOCK`` (threading.Lock) and ``_loop_service_singleton`` prevent
re-entrant or duplicate initialisation when Uvicorn uses ``--reload``.
The singletons are reset to ``None`` during shutdown so a subsequent reload
reinitialises cleanly.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, AsyncGenerator

from fastapi import FastAPI

if TYPE_CHECKING:
    from agritwin_gh.services.loop_service import LoopService

logger = logging.getLogger("agritwin.lifespan")

# ── Module-level singletons (reset on shutdown) ───────────────────────────────
# These survive hot-reload within the same OS process and prevent the loop
# from being started twice if lifespan() is entered more than once.

_STARTUP_LOCK: threading.Lock = threading.Lock()
_loop_service_singleton: "LoopService | None" = None
_bg_task_singleton: asyncio.Task | None = None  # type: ignore[type-arg]


# ── Internal helpers ───────────────────────────────────────────────────────────

def _env_bool(key: str, *, default: bool = False) -> bool:
    """Read a boolean environment variable (``"1"`` → True)."""
    return os.getenv(key, "1" if default else "0") == "1"


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Async lifespan context manager for FastAPI startup and shutdown."""

    global _loop_service_singleton, _bg_task_singleton

    # ─────────────────────────────────────────────────────────────────────────
    # STARTUP
    # ─────────────────────────────────────────────────────────────────────────
    _SEP = "─" * 58
    logger.info(_SEP)
    logger.info("AgriTwin-GH API  →  startup")
    logger.info(_SEP)

    _store_ok = False
    _loop_ok = False
    _loop_mode = "disabled"
    _loop_svc: "LoopService | None" = None
    _bg_task: asyncio.Task | None = None  # type: ignore[type-arg]

    # ── Step 1: RuntimeStore ─────────────────────────────────────────────────
    try:
        from agritwin_gh.api.dependencies import get_runtime_store
        store = get_runtime_store()
        _init_state = store.get_latest_state()
        _store_ok = True
        logger.info(
            "  [store ] RuntimeStore ready   step_index=%d  mode=%s",
            _init_state.climate.step_index,
            store.get_mode(),
        )
    except Exception as exc:
        store = None  # type: ignore[assignment]
        logger.error("  [store ] RuntimeStore FAILED  — %s", exc, exc_info=True)

    # ── Step 2: LoopService  ─────────────────────────────────────────────────
    _testing = _env_bool("TESTING")
    _continuous = _env_bool("AGRITWIN_BACKGROUND_LOOP", default=True)

    if not _store_ok:
        logger.warning("  [loop  ] LoopService skipped  — RuntimeStore unavailable")
        _loop_mode = "disabled (store unavailable)"

    elif _testing:
        logger.info("  [loop  ] LoopService skipped  — TESTING=1")
        _loop_mode = "disabled (TESTING=1)"

    elif _loop_service_singleton is not None:
        # Hot-reload / duplicate lifespan entry guard
        logger.warning(
            "  [loop  ] LoopService already running  — skipping duplicate init"
        )
        _loop_svc = _loop_service_singleton
        _bg_task = _bg_task_singleton
        _loop_ok = True
        _loop_mode = "continuous (pre-existing)" if _continuous else "step-mode (pre-existing)"

    else:
        with _STARTUP_LOCK:
            _growth_stage = os.getenv("AGRITWIN_GROWTH_STAGE", "seedling")
            _days_elapsed = _env_float("AGRITWIN_DAYS_ELAPSED", 0.0)
            _total_steps = _env_int("AGRITWIN_TOTAL_STEPS", 25632)  # 2136 h × 12 steps/h
            _interval_sec = _env_float("AGRITWIN_STEP_INTERVAL_SEC", 300.0)
            _auto_advance = _env_bool("AGRITWIN_AUTO_ADVANCE_STAGE", default=True)

            # ── 2a: Instantiate and start the loop engine ────────────────
            try:
                from agritwin_gh.services.loop_service import LoopService
                _loop_svc = LoopService(store=store)
                _loop_svc.start_loop(
                    growth_stage=_growth_stage,
                    days_elapsed=_days_elapsed,
                    total_steps=_total_steps,
                    auto_advance_stage=_auto_advance,
                )
                _loop_ok = True
                logger.info(
                    "  [loop  ] LoopService started     stage=%s  days_elapsed=%.1f"
                    "  total_steps=%d  auto_advance=%s",
                    _growth_stage, _days_elapsed, _total_steps, _auto_advance,
                )
            except Exception as exc:
                logger.error(
                    "  [loop  ] LoopService FAILED to start_loop  — %s", exc,
                    exc_info=True,
                )
                _loop_ok = False

            # ── 2b: Run initial step so store has real data at boot ──────
            if _loop_ok:
                try:
                    # Run in executor to avoid blocking the event loop during
                    # the ~100 ms MPC solve at startup.
                    ev_loop = asyncio.get_running_loop()
                    result = await ev_loop.run_in_executor(
                        None, _loop_svc.run_one_step
                    )
                    if result is not None:
                        logger.info(
                            "  [loop  ] Initial step done      step=%d  T=%.1f°C  RH=%.1f%%",
                            result.step_index,
                            result.next_state.indoor_temp,
                            result.next_state.indoor_humidity,
                        )
                    else:
                        logger.warning(
                            "  [loop  ] Initial step returned None  — store holds zero-state"
                        )
                except Exception as exc:
                    logger.warning(
                        "  [loop  ] Initial step failed  — store holds zero-state  %s", exc,
                        exc_info=True,
                    )

            # ── 2c: Launch continuous background task or stay in step-mode ──
            if _loop_ok and _continuous:
                try:
                    _bg_task = asyncio.create_task(
                        _loop_svc.run_background_loop(interval_seconds=_interval_sec),
                        name="agritwin_dt_loop",
                    )
                    _loop_mode = "continuous"
                    logger.info(
                        "  [loop  ] Background task started  interval=%.0fs",
                        _interval_sec,
                    )
                except Exception as exc:
                    logger.error(
                        "  [loop  ] Background task FAILED to launch  — %s", exc,
                        exc_info=True,
                    )
                    _bg_task = None
                    _loop_mode = "step-mode (bg-task failed)"

            elif _loop_ok:
                _loop_mode = "step-mode"
                logger.info(
                    "  [loop  ] Running in step-mode  — "
                    "set AGRITWIN_BACKGROUND_LOOP=1 for continuous updates"
                )

            else:
                _loop_mode = "disabled (start_loop failed)"

            # Store singletons for duplicate-startup guard
            _loop_service_singleton = _loop_svc
            _bg_task_singleton = _bg_task

    # ── Step 3: Startup summary ──────────────────────────────────────────────
    logger.info(_SEP)
    logger.info("  Runtime store : %s", "OK" if _store_ok else "FAILED")
    logger.info("  Loop service  : %s", "OK" if _loop_ok else ("disabled" if not _store_ok or _testing else "FAILED"))
    logger.info("  Backend mode  : %s", _loop_mode)
    logger.info("  API routes    : /api/*")
    logger.info(_SEP)

    # ── Hand control to FastAPI ────────────────────────────────────────────
    yield

    # ─────────────────────────────────────────────────────────────────────────
    # SHUTDOWN
    # ─────────────────────────────────────────────────────────────────────────
    logger.info(_SEP)
    logger.info("AgriTwin-GH API  →  shutdown")
    logger.info(_SEP)

    with _STARTUP_LOCK:
        if _loop_svc is not None:
            _loop_svc.stop_loop()
            logger.info("  [loop  ] Stop signal sent.")

        if _bg_task is not None and not _bg_task.done():
            _bg_task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(_bg_task), timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            logger.info("  [loop  ] Background task cancelled.")

        _loop_service_singleton = None
        _bg_task_singleton = None

    logger.info("  Shutdown complete.")
    logger.info(_SEP)
