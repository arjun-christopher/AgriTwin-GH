"""
Top-level evaluation API and artifact persistence for AgriTwin-GH MPC.

Provides:
  • ``run_evaluation()``          — one-call API that builds configs, runs
    controllers, and returns a ``ComparisonReport``.
  • ``save_evaluation_artifacts`` — write JSON artefacts to
    ``src/agritwin_gh/mpc/mpc_results/<run_id>/``.
  • ``load_evaluation_report``    — deserialise a saved report.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .baseline_controller import RuleBasedController
from .config import MPCConfig
from .evaluation_metrics import ControllerMetricsBundle
from .experiment_runner import (
    ComparisonReport,
    ExperimentConfig,
    ExperimentRunner,
    generate_default_growth_stages,
    generate_default_weather,
    make_baseline_adapter,
    make_default_initial_state,
    make_mpc_adapter,
)
from .mpc_solver import MPCSolver
from .state import GreenhouseState, WeatherState
from .yield_proxy import YieldProxyResult, YieldProxyWeights

logger = logging.getLogger(__name__)

# Repo-standard artifact root (relative to workspace)
_ARTIFACT_ROOT = Path("src/agritwin_gh/mpc/mpc_results")


# ── Convenience: run a full evaluation with defaults ──────────────────────────


def run_evaluation(
    n_steps: int = 288,
    dt_minutes: int = 5,
    initial_state: GreenhouseState | None = None,
    weather_sequence: list[WeatherState] | None = None,
    growth_stage_sequence: list[str] | None = None,
    growth_stage: str = "flowering",
    mpc_config: MPCConfig | None = None,
    yield_weights: YieldProxyWeights | None = None,
    include_baseline: bool = True,
    include_mpc: bool = True,
    random_seed: int = 42,
    experiment_name: str = "",
) -> ComparisonReport:
    """One-call evaluation entry point.

    Sets up experiment config, registers requested controllers, runs
    the experiment, and returns a ``ComparisonReport``.

    Parameters
    ----------
    n_steps : int
        Number of simulation steps.
    initial_state : GreenhouseState, optional
        Starting state. Uses sensible defaults if ``None``.
    weather_sequence : list[WeatherState], optional
        External disturbances. Generates a diurnal cycle if ``None``.
    growth_stage_sequence : list[str], optional
        Growth stage label per step. Constant *growth_stage* if ``None``.
    mpc_config : MPCConfig, optional
        Config for the MPC solver.
    include_baseline : bool
        Register the rule-based baseline controller.
    include_mpc : bool
        Register the MPC solver.
    random_seed : int
        Seed for reproducibility.

    Returns
    -------
    ComparisonReport
    """
    _init = initial_state or make_default_initial_state()
    _weather = weather_sequence or generate_default_weather(n_steps, dt_minutes)
    _stages = growth_stage_sequence or generate_default_growth_stages(n_steps, growth_stage)

    cfg = ExperimentConfig(
        n_steps=n_steps,
        dt_minutes=dt_minutes,
        initial_state=_init,
        weather_sequence=_weather,
        growth_stage_sequence=_stages,
        random_seed=random_seed,
        yield_proxy_weights=yield_weights,
        experiment_name=experiment_name or f"eval_{_dt.datetime.now():%Y%m%d_%H%M%S}",
    )

    runner = ExperimentRunner(cfg)

    if include_baseline:
        runner.register_controller(
            "baseline",
            make_baseline_adapter(),
            controller_type="rule_based",
        )

    if include_mpc:
        _mpc_cfg = mpc_config or MPCConfig()
        runner.register_controller(
            "mpc",
            make_mpc_adapter(config=_mpc_cfg),
            controller_type="mpc_disease_aware",
        )

    report = runner.run()
    logger.info(
        "Evaluation complete: %d controllers, %d steps",
        len(report.controller_metrics),
        n_steps,
    )
    return report


# ── Artifact persistence ──────────────────────────────────────────────────────


def save_evaluation_artifacts(
    report: ComparisonReport,
    workspace_root: str | Path | None = None,
    run_id: str | None = None,
) -> Path:
    """Serialise evaluation artefacts to disk.

    Writes to ``<workspace_root>/src/agritwin_gh/mpc/mpc_results/<run_id>/``:
      • ``report_summary.json``   — summary table + improvements
      • ``full_metrics.json``     — per-controller metric bundles
      • ``yield_proxy.json``      — yield proxy breakdowns
      • ``experiment_config.json``— reproducibility metadata

    Returns the output directory path.
    """
    run_id = run_id or f"eval_{_dt.datetime.now():%Y%m%d_%H%M%S}"

    if workspace_root is None:
        workspace_root = Path.cwd()
    out_dir = Path(workspace_root) / _ARTIFACT_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Summary table
    summary = {
        "run_id": run_id,
        "generated_at": report.generated_at.isoformat(),
        "summary_table": report.summary_table(),
        "improvements": report.improvements,
    }
    _write_json(out_dir / "report_summary.json", summary)

    # 2. Full metrics per controller
    full_metrics = {
        cid: m.to_dict() for cid, m in report.controller_metrics.items()
    }
    _write_json(out_dir / "full_metrics.json", full_metrics)

    # 3. Yield proxy
    yield_data = {
        cid: yr.to_dict() for cid, yr in report.yield_results.items()
    }
    _write_json(out_dir / "yield_proxy.json", yield_data)

    # 4. Config
    _write_json(out_dir / "experiment_config.json", report.experiment_config.to_dict())

    logger.info("Artifacts saved to %s", out_dir)
    return out_dir


def load_evaluation_report(artifact_dir: str | Path) -> dict[str, Any]:
    """Load a saved evaluation report summary from disk.

    Returns
    -------
    dict
        The parsed ``report_summary.json`` content.
    """
    p = Path(artifact_dir) / "report_summary.json"
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# ── Helpers ───────────────────────────────────────────────────────────────────


class _SafeEncoder(json.JSONEncoder):
    """Handle datetime and numpy types in JSON serialisation."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, _dt.datetime):
            return obj.isoformat()
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _write_json(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, cls=_SafeEncoder, ensure_ascii=False)
