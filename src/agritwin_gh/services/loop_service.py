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
import logging.handlers
import os
import pathlib
import threading
from typing import Any, Generator

from agritwin_gh.core.runtime_store import RuntimeStore, get_store

logger = logging.getLogger("agritwin.services.loop")

# ── Dedicated loop-trace file logger ─────────────────────────────────────────
# Writes a structured phase log: DB → AI → MPC → DT → DB → AI → MPC → DT …
# Stored at  logs/dt_loop_YYYYMMDD.log  (one file per day, 5 MB cap).

_LOOP_LOG_DIR = pathlib.Path(__file__).resolve().parents[3] / "logs"
_loop_file_logger: logging.Logger | None = None
_loop_file_lock = threading.Lock()


class _WriteThroughRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """RotatingFileHandler with write_through=True.

    Python's default text-mode FileHandler buffers ~8 KB before flushing to
    disk.  That means log lines only appear in the file after the buffer fills
    or the process exits — making live monitoring impossible.  This subclass
    opens the underlying stream with ``write_through=True`` so every ``emit()``
    hits the OS immediately (no TextIOWrapper buffering layer).
    """

    def _open(self):
        import io as _io

        mode = self.mode          # 'a' by default
        raw  = _io.FileIO(self.baseFilename, mode=mode + "b" if "b" not in mode else mode)
        buf  = _io.BufferedWriter(raw)
        return _io.TextIOWrapper(buf, encoding=self.encoding or "utf-8", write_through=True)


def _get_loop_file_logger() -> logging.Logger:
    """Return (and lazily create) the dedicated DT-loop trace file logger."""
    global _loop_file_logger
    if _loop_file_logger is not None:
        return _loop_file_logger
    with _loop_file_lock:
        if _loop_file_logger is not None:          # double-checked
            return _loop_file_logger
        _LOOP_LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = _LOOP_LOG_DIR / f"dt_loop_{_dt.date.today():%Y%m%d}.log"
        handler = _WriteThroughRotatingFileHandler(
            log_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        handler.setFormatter(
            logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        flog = logging.getLogger("agritwin.loop_trace")
        flog.setLevel(logging.DEBUG)
        flog.addHandler(handler)
        flog.propagate = False          # keep trace out of the main console
        _loop_file_logger = flog
    return _loop_file_logger


def _phase(step: int, tag: str, msg: str) -> None:
    """Write one phase-tagged line to the loop trace log."""
    _get_loop_file_logger().info("STEP %04d | %-5s | %s", step, tag, msg)


def _time_of_day(hour: int) -> str:
    """Return a human-readable time-of-day label for a given hour (0–23)."""
    if 6 <= hour < 12:
        return "Morning"
    if 12 <= hour < 16:
        return "Afternoon"
    if 16 <= hour < 20:
        return "Evening"
    return "Night"


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

        # Monthly snapshot service (enabled when AGRITWIN_MONTHLY_DB=1)
        self._monthly_svc: Any = None

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    def start_loop(
        self,
        growth_stage: str = "seedling",
        days_elapsed: float = 0.0,
        total_steps: int = 25632,  # 2136 h × 12 steps/h = full seedling→ripe cycle
        weather_base_temp: float = 22.0,
        auto_advance_stage: bool = True,
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
            Default 288 = 24 hours.  For a full crop cycle (all 6 stages)
            use 25632 (2136 h × 12 steps/h).  For a single stage transition
            from *seedling* to *early vegetative* use at least 9792 steps.
        weather_base_temp:
            Mean outdoor temperature for synthetic weather generation (°C).
        auto_advance_stage:
            When ``True`` the loop automatically advances to the next growth
            stage once the current stage's configured duration (from
            ``STAGE_DURATION_HOURS``) has elapsed.  Requires ``total_steps``
            to cover the desired number of stage transitions.
            Default ``False`` keeps a fixed stage across the run.
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
            from agritwin_gh.mpc.dt_input_provider import CSVInputProvider
            csv_provider = CSVInputProvider(start_time=now)
            self._dt_loop = DTLoop(
                growth_stage=growth_stage,
                start_time=now,
                n_steps=total_steps,
                days_elapsed=days_elapsed,
                input_provider=csv_provider,
                auto_advance_stage=auto_advance_stage,
            )
            self._gen = self._dt_loop.run(n_steps=total_steps)
            logger.info(
                "LoopService: DTLoop started (stage=%s, steps=%d, "
                "auto_advance=%s, synthetic=True)",
                growth_stage, total_steps, auto_advance_stage,
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

        # ── Monthly snapshot service ────────────────────────────────────────
        from agritwin_gh.services.monthly_snapshot_service import (
            MonthlySnapshotService as _MSS, monthly_db_enabled as _mde,
        )
        if _mde():
            try:
                from agritwin_gh.utils.database import get_db_manager as _get_dbm
                from sqlalchemy import text as _stext
                _dbmgr = _get_dbm()
                _tmp_sess = _dbmgr.get_session()
                try:
                    _cnt = _tmp_sess.execute(
                        _stext("SELECT COUNT(*) FROM crop_cycles")
                    ).scalar() or 0
                finally:
                    _tmp_sess.close()
                _cycle_label = f"cycle-{_cnt + 1:03d}"
                self._monthly_svc = _MSS(
                    session_factory=_dbmgr.get_session,
                    cycle_label=_cycle_label,
                )
                self._monthly_svc.initialise()
                logger.info(
                    "LoopService: MonthlySnapshotService enabled (cycle=%s)",
                    _cycle_label,
                )
            except Exception:
                logger.exception(
                    "LoopService: MonthlySnapshotService init failed — monthly snapshots disabled"
                )
                self._monthly_svc = None

    def stop_loop(self) -> None:
        """Signal the background loop to stop after the current step completes."""
        self._running = False
        if self._monthly_svc is not None:
            try:
                self._monthly_svc.flush_current_month()
                logger.info("LoopService: monthly snapshot flushed on stop.")
            except Exception:
                logger.exception("LoopService: monthly flush on stop failed")
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

            # ── Growth-stage penalty: LSTM current_stage → authoritative stage ──
            # Authority priority: (1) active sim_stage override, (2) ARX physics
            # index.  When the operator has forced a specific stage, the LSTM
            # prediction is irrelevant — override it so logs and the API agree.
            # Additionally adjust hours_to_transition for any elapsed days within
            # the overridden stage (sim_day_in_stage), preventing the LSTM from
            # reporting the full stage duration when the operator skipped ahead.
            # _GS and _ov_chk are computed here (outside if _mgr:) so they are
            # available for INPUT / MPC log display regardless of LSTM presence.
            from agritwin_gh.schemas.enums import GROWTH_STAGE_ORDERED as _GS
            _ov_chk = self._store.get_override()
            _mgr = (result.cadence_info or {}).get("model_growth_result") or {}
            if _mgr:
                if _ov_chk and _ov_chk.sim_stage and _ov_chk.sim_stage in list(_GS):
                    # Override is the authority
                    _auth_stage = _ov_chk.sim_stage
                    _auth_idx   = list(_GS).index(_auth_stage)
                    _auth_next  = _GS[_auth_idx + 1] if _auth_idx < len(_GS) - 1 else None
                    _mgr["current_stage"] = _auth_stage
                    _mgr["next_stage"]    = _auth_next
                    # Adjust h_to_transition by subtracting already-elapsed days
                    if _ov_chk.sim_day_in_stage and _ov_chk.sim_day_in_stage > 0:
                        _raw_htt = _mgr.get("hours_to_transition")
                        if _raw_htt is not None:
                            _mgr["hours_to_transition"] = max(
                                0.0, _raw_htt - float(_ov_chk.sim_day_in_stage) * 24.0
                            )
                    result.cadence_info["model_growth_result"] = _mgr
                else:
                    # No sim override — ARX physics is the authority
                    _dt_idx   = max(0, min(int(result.next_state.growth_stage_index), len(_GS) - 1))
                    _dt_stage = _GS[_dt_idx]
                    if _mgr.get("current_stage") and _mgr["current_stage"] != _dt_stage:
                        logger.debug(
                            "Growth-stage penalty: LSTM=%r  DT=%r  → correcting to DT",
                            _mgr["current_stage"], _dt_stage,
                        )
                        _mgr["current_stage"] = _dt_stage
                        _mgr["next_stage"] = _GS[_dt_idx + 1] if _dt_idx < len(_GS) - 1 else None
                        result.cadence_info["model_growth_result"] = _mgr

            # ── Phase logging — DB → AI → MPC → DT ───────────────────────
            step = int(result.step_index)
            ci   = result.cadence_info or {}

            # TIME: step-start timestamp and time-of-day label
            _now = _dt.datetime.now()
            _phase(step, "TIME",
                   f"{_now:%Y-%m-%d} {_now:%A} {_now:%H:%M:%S}  "
                   f"{_time_of_day(_now.hour)} ({_now.hour:02d}:00–{(_now.hour + 1) % 24:02d}:00)")

            # INPUT: greenhouse state at the START of this step (read from CSV)
            cs = result.current_state
            # ARX physics never sees the sim_stage override so cs.growth_stage_label
            # is always the raw physical stage.  Display the overridden stage when active.
            _log_stage = _ov_chk.sim_stage if (_ov_chk and _ov_chk.sim_stage) else cs.growth_stage_label
            _log_stage_idx = (
                list(_GS).index(_ov_chk.sim_stage)
                if (_ov_chk and _ov_chk.sim_stage and _ov_chk.sim_stage in list(_GS))
                else cs.growth_stage_index
            )
            _phase(step, "INPUT",
                   f"state in  — T={cs.indoor_temp:.1f}°C  RH={cs.indoor_humidity:.1f}%  "
                   f"CO₂={cs.co2:.0f}ppm  soil={cs.soil_moisture:.1f}%  "
                   f"light={cs.light_intensity:.0f}lux  VPD={cs.vpd:.2f}kPa  "
                   f"leaf={cs.leaf_wetness_proxy:.2f}  risk={cs.disease_risk_score:.3f}  "
                   f"stage={_log_stage}({_log_stage_idx})")

            # INPUT: weather disturbance applied this step
            w = result.weather_used
            _phase(step, "INPUT",
                   f"weather in — T_ext={w.temp_external:.1f}°C  "
                   f"RH_ext={w.humidity_external:.1f}%  "
                   f"solar={w.solar_radiation:.0f}W/m²  wind={w.windspeed:.1f}km/h  "
                   f"cond={w.conditions}")

            # AI: growth-progression LSTM — current→next stage transition forecast
            # Inputs: 24-h window of T, RH, CO₂, light (current reading = window endpoint)
            gr = ci.get("model_growth_result") or {}
            if gr:
                htt = gr.get("hours_to_transition")
                htt_str = f"{htt:.1f}h" if htt is not None else "—"
                _phase(step, "AI",
                       f"growth-progression LSTM  "
                       f"in=[T={cs.indoor_temp:.1f}°C  RH={cs.indoor_humidity:.1f}%  "
                       f"CO₂={cs.co2:.0f}ppm  light={cs.light_intensity:.0f}lux  "
                       f"(24h window)] "
                       f"→ stage={gr.get('current_stage') or cs.growth_stage_label}  "
                       f"next={gr.get('next_stage') or '—'}  "
                       f"h_to_transition={htt_str}  "
                       f"within_24h={gr.get('within_24h', '—')}  "
                       f"within_48h={gr.get('within_48h', '—')}")
            elif result.mpc_ran_this_step:
                htt = ci.get("hours_to_transition")
                htt_str = f"{htt:.1f}h" if htt is not None else "—"
                _phase(step, "AI",
                       f"growth-progression LSTM  "
                       f"in=[T={cs.indoor_temp:.1f}°C  RH={cs.indoor_humidity:.1f}%  "
                       f"CO₂={cs.co2:.0f}ppm  light={cs.light_intensity:.0f}lux  "
                       f"(24h window)] "
                       f"→ stage={cs.growth_stage_label}  "
                       f"next={ci.get('growth_next_stage') or '—'}  "
                       f"h_to_transition={htt_str}  "
                       f"within_24h={ci.get('transition_within_24h', '—')}  "
                       f"within_48h={ci.get('transition_within_48h', '—')}")

            # AI: disease-progression model — per-disease severity forecast
            # Inputs: 24-h window of T, RH, VPD, leaf-wetness + disease history
            dr = ci.get("model_disease_result") or {}
            if dr:
                rows = "  ".join(
                    f"{d}=[sev_24h={v.get('severity_24h', 0):.3f}  "
                    f"present={v.get('present', False)}  "
                    f"trend={v.get('trend_24h', '—')}]"
                    for d, v in dr.items()
                )
                _phase(step, "AI",
                       f"disease-progression  "
                       f"in=[T={cs.indoor_temp:.1f}°C  RH={cs.indoor_humidity:.1f}%  "
                       f"VPD={cs.vpd:.2f}kPa  leaf={cs.leaf_wetness_proxy:.2f}  "
                       f"(24h window)] "
                       f"→ {rows}")
            elif result.mpc_ran_this_step:
                cur_sev  = ci.get("current_severity", {})
                sev_24h  = ci.get("severity_24h", {})
                sev_48h  = ci.get("severity_48h", {})
                if cur_sev or sev_24h:
                    diseases = sorted(set(cur_sev) | set(sev_24h) | set(sev_48h))
                    rows = "  ".join(
                        f"{d}=[now={cur_sev.get(d, 0):.3f}  24h={sev_24h.get(d, 0):.3f}  48h={sev_48h.get(d, 0):.3f}]"
                        for d in diseases
                    )
                else:
                    rows = f"risk={cs.disease_risk_score:.3f}  24h=—  48h=—"
                _phase(step, "AI",
                       f"disease-progression  "
                       f"in=[T={cs.indoor_temp:.1f}°C  RH={cs.indoor_humidity:.1f}%  "
                       f"VPD={cs.vpd:.2f}kPa  leaf={cs.leaf_wetness_proxy:.2f}  "
                       f"(24h window)] "
                       f"→ {rows}")

            # AI: image classifiers — growth-stage CNN + disease CNN
            # (run after both LSTMs; image folder selection uses LSTM current_stage
            #  and disease present-flags as inputs)
            if result.image_refresh_this_step and result.image_observation is not None:
                obs = result.image_observation
                _phase(step, "AI",
                       f"growth-stage CNN  in=[img:{obs.growth_stage_image_key or '—'}] "
                       f"→ {obs.growth_stage_label or '—'}  ({obs.growth_stage_confidence:.1%})")
                _phase(step, "AI",
                       f"disease CNN       in=[img:{obs.disease_image_key or '—'}] "
                       f"→ {obs.disease_label or '—'}  ({obs.disease_confidence:.1%})")
            else:
                _phase(step, "AI",
                       f"image classify — skipped  (image_due={ci.get('image_due', False)})")

            # AI: weather-forecast model (Chronos+XGBoost+LSTM ensemble)
            # Logs the 24h-ahead model prediction at every step (model runs once
            # on startup; same point forecast is shown for all steps in a run).
            w24 = ci.get("weather_24h_ahead", {})
            if w24:
                _phase(step, "AI",
                       f"weather-forecast (24h ahead)  "
                       f"T_ext={w24.get('temp_external', 0):.1f}°C  "
                       f"RH={w24.get('humidity_external', 0):.1f}%  "
                       f"solar={w24.get('solar_radiation', 0):.0f}W/m²  "
                       f"wind={w24.get('windspeed', 0):.1f}km/h  "
                       f"conditions={w24.get('conditions', '—')}")

            # MPC: CVXPY solver — logs all 9 state inputs and all 7 actuator outputs
            if result.mpc_ran_this_step:
                reason = "event" if result.mpc_forced else "cadence"
                act = result.action_applied
                sol = result.mpc_solution
                cost_str = f"{sol.total_cost:.4f}" if sol else "—"
                conv_str = ("yes" if sol and sol.converged else
                            "fallback" if sol and sol.fallback_used else "no")
                _mpc_stage_idx = (
                    list(_GS).index(_ov_chk.sim_stage)
                    if (_ov_chk and _ov_chk.sim_stage and _ov_chk.sim_stage in list(_GS))
                    else cs.growth_stage_index
                )
                _phase(step, "MPC",
                       f"solve [{reason}] in=[T={cs.indoor_temp:.1f}  RH={cs.indoor_humidity:.1f}  "
                       f"CO₂={cs.co2:.0f}  soil={cs.soil_moisture:.1f}  "
                       f"light={cs.light_intensity:.0f}  VPD={cs.vpd:.2f}  "
                       f"leaf={cs.leaf_wetness_proxy:.2f}  risk={cs.disease_risk_score:.3f}  "
                       f"stage={_mpc_stage_idx}]  "
                       f"→ fan={act.fan_speed:.2f}  vent={act.vent_opening:.2f}  "
                       f"heat={act.heater_output:.2f}  led={act.led_intensity:.2f}  "
                       f"co2v={act.co2_valve_pct:.2f}  fog={act.fogger_duty:.2f}  "
                       f"irrig={act.irrigation_qty:.2f}  "
                       f"cost={cost_str}  converged={conv_str}")
            else:
                _phase(step, "MPC",
                       f"solve — skipped  (mpc_due={ci.get('mpc_due', False)}  "
                       f"forced={result.mpc_forced})")

            # DT: ARX physics engine — all 9 next-state variables produced
            ns = result.next_state
            _phase(step, "DT",
                   f"physics → T={ns.indoor_temp:.1f}°C  RH={ns.indoor_humidity:.1f}%  "
                   f"CO₂={ns.co2:.0f}ppm  soil={ns.soil_moisture:.1f}%  "
                   f"light={ns.light_intensity:.0f}lux  VPD={ns.vpd:.2f}kPa  "
                   f"leaf={ns.leaf_wetness_proxy:.2f}  risk={ns.disease_risk_score:.3f}  "
                   f"stage={ns.growth_stage_label}({ns.growth_stage_index})")

            # DT diagnostics: state delta, resource accounting, timing
            diag = result.diagnostics
            sd = diag.state_delta
            _phase(step, "DT",
                   f"Δstate    → ΔT={sd.get('indoor_temp', 0):+.3f}°C  "
                   f"ΔRH={sd.get('indoor_humidity', 0):+.3f}%  "
                   f"ΔCO₂={sd.get('co2', 0):+.1f}ppm  "
                   f"Δsoil={sd.get('soil_moisture', 0):+.3f}%  "
                   f"Δlight={sd.get('light_intensity', 0):+.1f}lux  "
                   f"ΔVPD={sd.get('vpd', 0):+.4f}kPa  "
                   f"Δleaf={sd.get('leaf_wetness_proxy', 0):+.4f}  "
                   f"energy={diag.energy_kwh:.5f}kWh  water={diag.water_litres:.3f}L  "
                   f"compute={diag.step_compute_ms:.1f}ms")

            # DT diagnostics: signed setpoint error (actual − target) per variable
            se = diag.setpoint_error
            if se:
                sp_parts = "  ".join(f"{k}={v:+.3f}" for k, v in se.items())
                _phase(step, "DT", f"setpt_err → {sp_parts}")

            # DT diagnostics: disease-favourable environment boolean flags
            df = diag.disease_environment_flags
            if df:
                flags_active = [k for k, v in df.items() if v]
                flags_str = (
                    "  ".join(k for k in flags_active)
                    if flags_active else "all_clear"
                )
                _phase(step, "DT",
                       f"disease_env → {flags_str}  "
                       f"(risk_post={diag.disease_risk_recomputed:.3f})")

            # DT diagnostics: per-actuator effect attribution for T and RH
            ea = diag.effect_attribution
            for _var, _short in (("indoor_temp", "T"), ("indoor_humidity", "RH")):
                attrs = ea.get(_var, {})
                if attrs:
                    attrs_str = "  ".join(f"{k}={v:+.4f}" for k, v in attrs.items())
                    _phase(step, "DT", f"effects[{_short}] → {attrs_str}")

            # DT diagnostics: bounds clamping (only logged when active)
            if diag.bounds_clamped:
                _phase(step, "DT", f"bounds_clamped → {diag.bounds_clamped}")

            # ── Enrich cadence_info with elapsed-day tracking ─────────────
            # DTLoop.run() sets mpc_due, image_due etc. but not days_elapsed /
            # days_in_stage.  Compute from step index.
            from agritwin_gh.mpc.constants import DT_MINUTES
            from agritwin_gh.mpc.realtime_core import STAGE_DURATION_HOURS, STAGE_DURATION_HOURS as _SDH
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

            # Per-actuator energy for billing breakdown (same rated_kw as estimate_energy)
            _rated_kw = {
                "fan_speed":      0.75,
                "vent_opening":   0.10,
                "heater_output":  3.00,
                "led_intensity":  2.00,
                "fogger_duty":    0.20,
                "co2_valve_pct":  0.05,
                "irrigation_qty": 0.15,
            }
            _dt_h = 5.0 / 60.0
            # Mapping: act_key → frontend override ID (override levels are 0-100 scale)
            _ACT_KEY_TO_OID: dict[str, str] = {
                "fan_speed": "fan", "vent_opening": "vent", "heater_output": "heater",
                "led_intensity": "led", "fogger_duty": "fogger",
                "co2_valve_pct": "co2", "irrigation_qty": "irrigation",
            }
            _override_ref = self._store.get_override()
            _override_acts = _override_ref.actuator_overrides if _override_ref else {}
            _act_energy: dict[str, float] = {}
            for _ak, _kw in _rated_kw.items():
                _oid = _ACT_KEY_TO_OID.get(_ak)
                if result.mpc_ran_this_step:
                    # MPC ran: MPC is the authority — use its output for every actuator.
                    # This ensures actuators MPC explicitly zeroed (e.g. heater=0.00)
                    # are not still billed at the old manual-override level.
                    _lvl = getattr(result.action_applied, _ak, 0.0) or 0.0
                elif _oid and _oid in _override_acts:
                    # No MPC this step: use the operator-set override level (0-100 → 0-1)
                    _lvl = _override_acts[_oid] / 100.0
                else:
                    _lvl = getattr(result.action_applied, _ak, 0.0) or 0.0
                _act_energy[_ak] = round(_kw * _lvl * _dt_h, 6)

            self._store.accumulate_resources(
                energy_kwh=energy_kwh, water_l=water_l, actuator_energy=_act_energy
            )

            # ── Monthly snapshot ingestion ─────────────────────────────────
            if self._monthly_svc is not None:
                _w_used = result.weather_used
                _monthly_weather = {
                    "temp_external":     getattr(_w_used, "temp_external",     0.0),
                    "humidity_external": getattr(_w_used, "humidity_external", 0.0),
                    "solar_radiation":   getattr(_w_used, "solar_radiation",   0.0),
                    "windspeed":         getattr(_w_used, "windspeed",         0.0),
                } if _w_used is not None else {}
                try:
                    self._monthly_svc.ingest_step(
                        result=result,
                        actuator_energy=_act_energy,
                        water_l=water_l,
                        energy_kwh=energy_kwh,
                        cadence_info=result.cadence_info or {},
                        disease_info=(result.cadence_info or {}).get("model_disease_result") or {},
                        weather_info=_monthly_weather,
                    )
                except Exception:
                    logger.exception("LoopService: monthly_svc.ingest_step failed")

            # RES: step-end resource and cost summary
            _rs = self._store.get_latest_state().resources
            _cost_inr = (
                _rs.energy_kwh_total * 7.0          # ₹7.00 / kWh
                + (_rs.water_l_total / 1000.0) * 4.0  # ₹4.00 / kL
            )
            _phase(step, "RES",
                   f"step: energy={energy_kwh:.5f}kWh  water={water_l:.3f}L  "
                   f"\u2502  month total: energy={_rs.energy_kwh_total:.3f}kWh  "
                   f"water={_rs.water_l_total:.1f}L  cost=\u20b9{_cost_inr:.2f}")

            # RES actuator breakdown (only non-zero actuators for brevity)
            _ERATES = 7.0
            _WRATE_PER_L = 4.0 / 1000.0
            _act_parts: list[str] = []
            for _k, _kwh in _act_energy.items():
                if _kwh <= 0.0:
                    continue
                _act_cost = _kwh * _ERATES
                if _k == "irrigation_qty":
                    _act_wl = (_act_energy[_k] / (0.15 * _dt_h)) * 10.0
                    _act_cost += _act_wl * _WRATE_PER_L
                    _act_parts.append(f"{_k}={_kwh:.5f}kWh+{_act_wl:.3f}L(\u20b9{_act_cost:.3f})")
                else:
                    _act_parts.append(f"{_k}={_kwh:.5f}kWh(\u20b9{_act_cost:.3f})")
            if _act_parts:
                _phase(step, "RES", "actuators: " + "  ".join(_act_parts))

            # ── MPC authority: remove overrides that MPC explicitly zeroed ──
            if result.mpc_ran_this_step:
                self._clear_mpc_overridden_actuators(result)

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

    # Mapping: frontend override ID → MPC action_applied attribute name
    _OVERRIDE_ID_TO_ACT_ATTR: dict[str, str] = {
        "fan":        "fan_speed",
        "vent":       "vent_opening",
        "heater":     "heater_output",
        "led":        "led_intensity",
        "fogger":     "fogger_duty",
        "co2":        "co2_valve_pct",
        "irrigation": "irrigation_qty",
    }

    def _clear_mpc_overridden_actuators(self, result) -> None:  # type: ignore[type-arg]
        """After MPC runs, remove from override.actuator_overrides any actuator
        that MPC explicitly set to 0.  MPC authority overrides a manual ON toggle
        so the store — and the frontend on next poll — reflects MPC output.
        """
        override = self._store.get_override()
        if override is None or not override.actuator_overrides:
            return
        new_overrides = dict(override.actuator_overrides)
        changed = False
        for oid, attr in self._OVERRIDE_ID_TO_ACT_ATTR.items():
            if oid in new_overrides:
                mpc_level = getattr(result.action_applied, attr, None)
                override_val = override.actuator_overrides[oid]
                # Clear the override when MPC and the stored override disagree
                # on the on/off state:
                #   • MPC=0  + override>0 → MPC wants it OFF, clear the ON-override
                #   • MPC>0  + override=0 → MPC wants it ON,  clear the OFF-override
                # In both cases MPC is the authority; the override entry would
                # otherwise silently win when _apply_actuator_overrides_if_active
                # merges it back into the actuator snapshot.
                mpc_on  = mpc_level is not None and mpc_level >= 0.01
                ovr_on  = override_val > 0.0
                if mpc_level is not None and (mpc_on != ovr_on):
                    del new_overrides[oid]
                    changed = True
                    logger.info(
                        "MPC cleared actuator override: %s (was %.0f%%) → MPC=%.2f",
                        oid, override_val, mpc_level,
                    )
        if changed:
            from agritwin_gh.core.runtime_store import OverrideConfig as _OC  # noqa: PLC0415
            self._store.set_override(
                _OC(
                    param_overrides=dict(override.param_overrides),
                    actuator_overrides=new_overrides,
                    sim_stage=override.sim_stage,
                    sim_day_in_stage=override.sim_day_in_stage,
                    sim_start_date=override.sim_start_date,
                    sim_start_hour=override.sim_start_hour,
                    preset_id=override.preset_id,
                )
            )

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
            # Match the rounding used in update_from_step_result: CVXPY solver
            # residuals (e.g. 0.001) that display as "0.00" in the log must not
            # show as ON.  Override values (0–100) are unaffected by this rounding.
            on_off={k: round(v, 2) > 0.0 for k, v in levels.items()},
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
