"""
SystemService — system health diagnostic builder.

Responsibility
--------------
``SystemService`` assembles the ``SystemHealthResponse`` consumed by the
Dashboard's "System Health" panel at ``GET /api/system/health``.

It checks six logical subsystems by reading from the in-memory
``RuntimeStore`` (zero DB or solver calls).  Each subsystem maps to one
``HealthRow`` in the response.

Subsystem checks (in display order)
-------------------------------------
1. **DT Loop**       — was the last DT step recent enough?
2. **MPC Solver**    — did MPC converge on the last step that ran it?
3. **Sensor Feed**   — are sensor readings non-zero (sanity check)?
4. **Database**      — passes-through in synthetic mode; checks session in Phase 3.
5. **MinIO Storage** — did ``MediaService`` produce a presigned URL this session?
6. **AI Models**     — has the disease classifier produced pathogen breakdowns?

Severity derivation
-------------------
Each row has a ``status_code`` ∈ ``{"ok", "warning", "error"}``.  The
overall panel severity is the worst across all rows::

    "error"   if any row.status_code == "error"
    "warning" if any row.status_code == "warning"  (and none are "error")
    "ok"      otherwise

Staleness thresholds
--------------------
The staleness check for the DT Loop uses a configurable threshold
(``MAX_STEP_AGE_SEC``).  The default ``300 s`` (= 1 DT step) means the
loop shows "warning" if no step has been processed in the last 5 minutes.
A threshold of ``600 s`` (= 2 steps) triggers "error".

Design note
-----------
This service is intentionally **read-only** — it writes nothing to the
store.  It is stateless and can be instantiated per-request or once per
process.  ``DashboardService.get_system_health()`` delegates to this service
when present; otherwise it falls back to
``RuntimeStore.as_system_health_response()``.
"""

from __future__ import annotations

import datetime as _dt
import logging

from agritwin_gh.core.runtime_store import RuntimeStore, get_store
from agritwin_gh.schemas.system_schemas import HealthRow, SystemHealthResponse

logger = logging.getLogger("agritwin.services.system")

# Seconds since last step before the DT Loop row turns from "ok" to "warning".
_WARN_STEP_AGE_SEC = 300   # 1 DT step (5 min)
# Seconds since last step before "warning" escalates to "error".
_ERROR_STEP_AGE_SEC = 600  # 2 DT steps (10 min)


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class SystemService:
    """Derive system health from RuntimeStore snapshots.

    Parameters
    ----------
    store:
        Shared ``RuntimeStore`` singleton.  ``None`` → ``get_store()``.
    """

    def __init__(self, store: RuntimeStore | None = None) -> None:
        self._store: RuntimeStore = store or get_store()

    # ─────────────────────────────────────────────────────────────────────────
    # Public entry point
    # ─────────────────────────────────────────────────────────────────────────

    def get_health(self) -> SystemHealthResponse:
        """Build and return a ``SystemHealthResponse``.

        Reads from ``RuntimeStore`` only.  All six subsystem checks complete
        in microseconds.

        Returns
        -------
        SystemHealthResponse
            Six ``HealthRow`` items plus ``overall`` severity and ``timestamp``.
        """
        state = self._store.get_latest_state()
        ts = _now_iso()

        rows: list[HealthRow] = [
            self._check_dt_loop(state),
            self._check_mpc_solver(state),
            self._check_sensor_feed(state),
            self._check_database(state),
            self._check_minio(state),
            self._check_ai_models(state),
        ]

        # Determine overall severity (worst row wins)
        if any(r.status_code == "error" for r in rows):
            overall = "error"
        elif any(r.status_code == "warning" for r in rows):
            overall = "warning"
        else:
            overall = "ok"

        return SystemHealthResponse(rows=rows, overall=overall, timestamp=ts)

    # ─────────────────────────────────────────────────────────────────────────
    # Individual subsystem checks
    # ─────────────────────────────────────────────────────────────────────────

    def _check_dt_loop(self, state) -> HealthRow:
        """DT Loop — check freshness of the last simulator step.

        * ``ok``      — last step < 5 min ago.
        * ``warning`` — last step 5–10 min ago (behind schedule).
        * ``error``   — no step ever recorded OR step > 10 min ago.
        """
        last_ts: _dt.datetime | None = state.climate.last_step_ts
        if last_ts is None:
            return HealthRow(
                label="DT Loop",
                status="Not started",
                ok=False,
                status_code="error",
                detail="The digital-twin loop has not run any steps yet.",
                last_updated=_now_iso(),
            )

        # Make last_ts timezone-aware for comparison
        if last_ts.tzinfo is None:
            last_ts = last_ts.replace(tzinfo=_dt.timezone.utc)
        now = _dt.datetime.now(_dt.timezone.utc)
        age_sec = (now - last_ts).total_seconds()

        if age_sec < _WARN_STEP_AGE_SEC:
            step_idx = state.climate.step_index
            return HealthRow(
                label="DT Loop",
                status="Running",
                ok=True,
                status_code="ok",
                detail=f"Step {step_idx} completed {int(age_sec)} s ago.",
                last_updated=last_ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            )
        if age_sec < _ERROR_STEP_AGE_SEC:
            return HealthRow(
                label="DT Loop",
                status="Delayed",
                ok=False,
                status_code="warning",
                detail=f"Last step {int(age_sec // 60)} min ago — loop may be stalled.",
                last_updated=last_ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            )
        return HealthRow(
            label="DT Loop",
            status="Stalled",
            ok=False,
            status_code="error",
            detail=f"No step in {int(age_sec // 60)} min — loop has likely halted.",
            last_updated=last_ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        )

    def _check_mpc_solver(self, state) -> HealthRow:
        """MPC Solver — did the solver converge on the most recent solve step?

        * ``ok``      — solver ran and converged (``mpc_ran_this_step`` and
                         ``mpc_converged``).
        * ``warning`` — solver ran but did not converge (``mpc_converged = False``).
        * ``ok``      — solver did not run this step (normal; MPC is multi-rate).
        """
        mpc_ran = state.actuators.mpc_ran_this_step
        converged = state.climate.mpc_converged

        if not mpc_ran:
            return HealthRow(
                label="MPC Solver",
                status="Idle",
                ok=True,
                status_code="ok",
                detail="MPC solver did not run this DT step (multi-rate cadence).",
                last_updated=_now_iso(),
            )
        if converged:
            cost = getattr(state.climate, "mpc_cost", None)
            detail = (
                f"Converged; cost {cost:.2f}." if cost is not None and cost > 0
                else "Converged."
            )
            return HealthRow(
                label="MPC Solver",
                status="Nominal",
                ok=True,
                status_code="ok",
                detail=detail,
                last_updated=_now_iso(),
            )
        return HealthRow(
            label="MPC Solver",
            status="No solution",
            ok=False,
            status_code="warning",
            detail="MPC ran but did not converge — previous solution carried forward.",
            last_updated=_now_iso(),
        )

    def _check_sensor_feed(self, state) -> HealthRow:
        """Sensor Feed — sanity check that sensor readings are non-zero.

        This is a lightweight check: if indoor temperature has never left 0 °C
        the sensor feed has not yet been wired up (or the loop has not started).

        * ``ok``      — temperature is non-zero (sensors producing data).
        * ``warning`` — temperature is exactly 0 (not yet populated).
        """
        indoor_temp = state.climate.indoor_temp

        if indoor_temp != 0.0:
            return HealthRow(
                label="Sensor Feed",
                status="Nominal",
                ok=True,
                status_code="ok",
                detail=f"All sensors responding — indoor temp {indoor_temp:.1f} °C.",
                last_updated=_now_iso(),
            )
        return HealthRow(
            label="Sensor Feed",
            status="No data",
            ok=False,
            status_code="warning",
            detail="Sensor readings are all zero — DT loop may not have started.",
            last_updated=_now_iso(),
        )

    def _check_database(self, state) -> HealthRow:
        """Database connection — Synthetic mode always passes.

        In Phase 1 (synthetic simulation), no live DB session is maintained.
        When a real session is wired (Phase 3), this check should ping the
        DB and report latency.  For now this row reports a static "Synthetic"
        status to avoid false alarms.

        * ``ok``      — synthetic mode; always healthy.
        """
        return HealthRow(
            label="Database",
            status="Synthetic",
            ok=True,
            status_code="ok",
            detail="DT loop is running in synthetic mode — no live DB session required.",
            last_updated=_now_iso(),
        )

    def _check_minio(self, state) -> HealthRow:
        """MinIO Storage — did ``MediaService`` successfully presign at least one URL?

        Reads ``media.latest_stage_src`` from the store: a non-empty value
        means at least one presigned URL was produced during this session.

        * ``ok``      — at least one presigned URL exists in the store.
        * ``warning`` — both latest_stage_src and latest_leaf_src are empty
                         (MediaService hasn't run yet or MinIO is not reachable).
        """
        stage_src = state.media.latest_stage_src
        leaf_src = state.media.latest_leaf_src

        if stage_src or leaf_src:
            return HealthRow(
                label="MinIO Storage",
                status="Connected",
                ok=True,
                status_code="ok",
                detail="Pre-signed image URLs are being generated successfully.",
                last_updated=_now_iso(),
            )
        return HealthRow(
            label="MinIO Storage",
            status="No images",
            ok=False,
            status_code="warning",
            detail=(
                "No presigned image URLs in store. "
                "MediaService may not have been called yet, or MinIO is unreachable."
            ),
            last_updated=_now_iso(),
        )

    def _check_ai_models(self, state) -> HealthRow:
        """AI Models — has the disease classifier produced per-pathogen breakdowns?

        Reads ``disease.pathogens`` from the store.  A non-empty list means
        ``IntelligenceService`` has successfully run the disease classifier at
        least once this session.

        * ``ok``      — per-pathogen breakdown is populated.
        * ``warning`` — only the aggregate composite_risk is available (classifier
                         not yet run or returned no output).
        """
        pathogens = state.disease.pathogens
        composite = state.disease.composite_risk

        if pathogens:
            top_label = max(pathogens, key=lambda p: p.get("probability", 0.0), default={})
            dominant = top_label.get("label", "unknown")
            return HealthRow(
                label="AI Models",
                status="Active",
                ok=True,
                status_code="ok",
                detail=(
                    f"Disease classifier online — {len(pathogens)} class(es) scored. "
                    f"Dominant: '{dominant}'."
                ),
                last_updated=_now_iso(),
            )
        return HealthRow(
            label="AI Models",
            status="Pending" if composite == 0.0 else "Partial",
            ok=composite > 0.0,
            status_code="warning",
            detail=(
                "Per-pathogen breakdown not yet available. "
                "Intelligence service will populate this after the first inference run."
            ),
            last_updated=_now_iso(),
        )
