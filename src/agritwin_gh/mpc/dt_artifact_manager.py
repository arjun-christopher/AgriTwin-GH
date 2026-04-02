"""
Run-ID-based artifact manager for DT closed-loop simulation results.

Creates a timestamped run folder following the repo convention
(cf. ``save_evaluation_artifacts`` in ``evaluation.py``) and persists:

* **run_metadata.json** — configuration, parameters, force-MPC thresholds.
* **state_trajectory.json** — per-step current → next state + weather.
* **mpc_actions.json** — actuator commands on MPC-solve steps.
* **diagnostics.json** — energy, water, attribution, disease flags.
* **summary.json** — enriched run-level aggregates.

The manager composes with :class:`JsonFileOutputWriter` — it creates a
writer pointed at the run directory, so the existing per-step fan-out
logic works unchanged.

Usage
-----
>>> mgr = DTArtifactManager(base_dir="logs/dt_runs")
>>> writer = mgr.create_output_writer()
>>> # ... loop ...
>>> mgr.save_run_metadata({...})
>>> mgr.save_summary(summary_dict)
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import pathlib
from typing import Any

from .dt_output_writer import JsonFileOutputWriter

logger = logging.getLogger(__name__)

# Default root for DT run artifacts (relative to workspace root).
_DEFAULT_BASE_DIR = "logs/dt_runs"


class DTArtifactManager:
    """Manages a run-ID-based artifact folder for one DT simulation run.

    Parameters
    ----------
    base_dir:
        Parent directory under which the per-run folder is created.
        Defaults to ``logs/dt_runs``.
    run_id:
        Explicit run identifier.  If ``None``, auto-generates
        ``dt_run_YYYYMMDD_HHMMSS``.
    """

    def __init__(
        self,
        base_dir: str | pathlib.Path = _DEFAULT_BASE_DIR,
        run_id: str | None = None,
    ) -> None:
        self._run_id = run_id or f"dt_run_{_dt.datetime.now():%Y%m%d_%H%M%S}"
        self._run_dir = pathlib.Path(base_dir) / self._run_id
        self._run_dir.mkdir(parents=True, exist_ok=True)
        logger.info("DT artifact folder: %s", self._run_dir)

    # ── Properties ────────────────────────────────────────────────

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def run_dir(self) -> pathlib.Path:
        return self._run_dir

    # ── Factory for the per-step writer ───────────────────────────

    def create_output_writer(self) -> JsonFileOutputWriter:
        """Return a :class:`JsonFileOutputWriter` that writes into the
        run directory using ``"run"`` as the file tag.

        Resulting file names inside the run folder::

            dt_state_run.json
            dt_mpc_actions_run.json
            dt_diagnostics_run.json
            dt_summary_run.json
        """
        return JsonFileOutputWriter(
            output_dir=self._run_dir,
            run_tag="run",
        )

    # ── Metadata save ─────────────────────────────────────────────

    def save_run_metadata(self, metadata: dict[str, Any]) -> pathlib.Path:
        """Persist run configuration and parameters.

        Parameters
        ----------
        metadata:
            Arbitrary mapping — typically includes growth stage, base temp,
            hours, dt_minutes, cadences, force-MPC thresholds, CLI args, etc.

        Returns
        -------
        pathlib.Path
            Resolved path of the written file.
        """
        metadata.setdefault("run_id", self._run_id)
        return self._write_json("run_metadata.json", metadata)

    # ── Summary save ──────────────────────────────────────────────

    def save_summary(self, summary: dict[str, Any]) -> pathlib.Path:
        """Persist the enriched run-level summary.

        This is written as a separate top-level file (not inside the
        per-step writer) so it can be loaded independently for run
        comparison dashboards.
        """
        summary.setdefault("run_id", self._run_id)
        return self._write_json("summary.json", summary)

    # ── Private ───────────────────────────────────────────────────

    def _write_json(
        self,
        filename: str,
        data: Any,
    ) -> pathlib.Path:
        path = self._run_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info("Saved %s", path)
        return path.resolve()
