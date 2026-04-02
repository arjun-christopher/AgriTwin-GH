"""
Output writer abstraction — separate artifact streams for the DT loop.

Splits simulation results into four independent outputs:

1. **Stepwise DT state log** — per-step environmental readings.
2. **MPC actions log** — actuator commands on steps where MPC ran.
3. **DT diagnostics log** — energy, water, attribution, disease flags.
4. **Summary results** — aggregated run statistics.

The ``DTOutputWriter`` protocol defines the contract.  The concrete
``JsonFileOutputWriter`` writes each stream to a separate JSON file.
A future ``DatabaseOutputWriter`` can persist the same data to
PostgreSQL without changing the loop or logger.

Future DB migration path
------------------------
1. Create ``DatabaseOutputWriter(DTOutputWriter)`` that holds a
   SQLAlchemy session.
2. ``write_state_step`` → ``INSERT INTO dt_state_log``.
3. ``write_mpc_action`` → ``INSERT INTO dt_mpc_actions``.
4. ``write_diagnostics`` → ``INSERT INTO dt_diagnostics``.
5. ``write_summary`` → ``INSERT INTO dt_run_summary``.
6. Pass it to the CLI / runner — zero changes to loop code.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import pathlib
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ── Protocol ──────────────────────────────────────────────────────────────────


@runtime_checkable
class DTOutputWriter(Protocol):
    """Contract for persisting DT loop results.

    Each method handles one artifact stream.  Implementations decide
    *where* and *how* the data lands (JSON files, database, message
    queue, etc.).
    """

    def write_state_step(self, record: dict[str, Any]) -> None:
        """Persist one step of DT environmental state."""
        ...

    def write_mpc_action(self, record: dict[str, Any]) -> None:
        """Persist one MPC actuator decision."""
        ...

    def write_diagnostics(self, record: dict[str, Any]) -> None:
        """Persist one step of DT diagnostics."""
        ...

    def write_summary(self, summary: dict[str, Any]) -> None:
        """Persist the run-level summary."""
        ...

    def flush(self) -> dict[str, str]:
        """Flush buffered data and return a mapping of artifact names
        to their destinations (file paths, table names, etc.)."""
        ...


# ── Concrete: JSON file writer ────────────────────────────────────────────────


class JsonFileOutputWriter:
    """Writes four separate JSON artifact files.

    Parameters
    ----------
    output_dir:
        Directory where artifacts are saved.  Created if missing.
    run_tag:
        Short identifier appended to file names (e.g. timestamp).
    """

    def __init__(
        self,
        output_dir: str | pathlib.Path = "logs",
        run_tag: str = "",
    ) -> None:
        self._dir = pathlib.Path(output_dir)
        self._tag = run_tag or _dt.datetime.now().strftime("%Y%m%d_%H%M%S")

        self._state_steps: list[dict[str, Any]] = []
        self._mpc_actions: list[dict[str, Any]] = []
        self._diagnostics: list[dict[str, Any]] = []
        self._summary: dict[str, Any] | None = None

    # ── Protocol implementation ───────────────────────────────────

    def write_state_step(self, record: dict[str, Any]) -> None:
        self._state_steps.append(record)

    def write_mpc_action(self, record: dict[str, Any]) -> None:
        self._mpc_actions.append(record)

    def write_diagnostics(self, record: dict[str, Any]) -> None:
        self._diagnostics.append(record)

    def write_summary(self, summary: dict[str, Any]) -> None:
        self._summary = summary

    def flush(self) -> dict[str, str]:
        """Write all buffered data to JSON files and return paths."""
        self._dir.mkdir(parents=True, exist_ok=True)

        artifacts: dict[str, str] = {}

        artifacts["state_log"] = self._write_json(
            f"dt_state_{self._tag}.json",
            self._state_steps,
        )
        artifacts["mpc_actions"] = self._write_json(
            f"dt_mpc_actions_{self._tag}.json",
            self._mpc_actions,
        )
        artifacts["diagnostics"] = self._write_json(
            f"dt_diagnostics_{self._tag}.json",
            self._diagnostics,
        )
        if self._summary is not None:
            artifacts["summary"] = self._write_json(
                f"dt_summary_{self._tag}.json",
                self._summary,
            )

        logger.info(
            "Flushed %d state / %d MPC / %d diag records → %s",
            len(self._state_steps),
            len(self._mpc_actions),
            len(self._diagnostics),
            self._dir,
        )
        return artifacts

    # ── Private ───────────────────────────────────────────────────

    def _write_json(self, filename: str, data: Any) -> str:
        path = self._dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        return str(path.resolve())


# ── Helper: fan-out from DTLoopStepResult to writer ───────────────────────────


def fanout_step_to_writer(
    result: Any,
    writer: DTOutputWriter,
) -> None:
    """Decompose one ``DTLoopStepResult`` into the writer's four streams.

    This keeps the loop clean — it calls ``fanout_step_to_writer`` once
    per step and all routing happens here.
    """
    cs = result.current_state
    ns = result.next_state
    wx = result.weather_used
    diag = result.diagnostics
    ts_str = str(result.timestamp) if result.timestamp else None

    # 1. State step (pre-step → post-step + weather disturbance)
    writer.write_state_step({
        "step": result.step_index,
        "ts": ts_str,
        "current_state": {
            "indoor_temp": round(cs.indoor_temp, 3),
            "indoor_humidity": round(cs.indoor_humidity, 3),
            "soil_moisture": round(cs.soil_moisture, 3),
            "co2": round(cs.co2, 2),
            "light_intensity": round(cs.light_intensity, 2),
            "disease_risk_score": round(cs.disease_risk_score, 5),
        },
        "weather": {
            "temp_external": round(wx.temp_external, 3),
            "humidity_external": round(wx.humidity_external, 3),
            "solar_radiation": round(wx.solar_radiation, 2),
            "windspeed": round(wx.windspeed, 2),
        },
        "next_state": {
            "indoor_temp": round(ns.indoor_temp, 3),
            "indoor_humidity": round(ns.indoor_humidity, 3),
            "soil_moisture": round(ns.soil_moisture, 3),
            "co2": round(ns.co2, 2),
            "light_intensity": round(ns.light_intensity, 2),
            "disease_risk_score": round(ns.disease_risk_score, 5),
            "vpd": round(ns.vpd, 4),
            "leaf_wetness_proxy": round(ns.leaf_wetness_proxy, 4),
            "growth_stage_index": ns.growth_stage_index,
        },
    })

    # 2. MPC action (only on solve steps)
    if result.mpc_ran_this_step:
        act = result.action_applied
        writer.write_mpc_action({
            "step": result.step_index,
            "ts": ts_str,
            "forced": result.mpc_forced,
            "cost": round(result.mpc_cost, 4) if result.mpc_cost is not None else None,
            "converged": (
                result.mpc_solution.converged
                if result.mpc_solution else None
            ),
            "fallback_used": (
                result.mpc_solution.fallback_used
                if result.mpc_solution else None
            ),
            "fan_speed": round(act.fan_speed, 3),
            "vent_opening": round(act.vent_opening, 3),
            "irrigation_qty": round(act.irrigation_qty, 3),
            "heater_output": round(act.heater_output, 3),
            "led_intensity": round(act.led_intensity, 3),
            "co2_valve_pct": round(act.co2_valve_pct, 3),
            "fogger_duty": round(act.fogger_duty, 3),
        })

    # 3. Diagnostics (every step)
    writer.write_diagnostics({
        "step": result.step_index,
        "ts": ts_str,
        "energy_kwh": round(diag.energy_kwh, 6),
        "water_litres": round(diag.water_litres, 4),
        "disease_risk_recomputed": round(diag.disease_risk_recomputed, 5),
        "bounds_clamped": diag.bounds_clamped,
        "setpoint_error": {
            k: round(v, 4) for k, v in diag.setpoint_error.items()
        } if diag.setpoint_error else {},
        "state_delta": {
            k: round(v, 4) for k, v in diag.state_delta.items()
        } if diag.state_delta else {},
        "effect_attribution": {
            var: {src: round(val, 6) for src, val in sources.items()}
            for var, sources in diag.effect_attribution.items()
        } if diag.effect_attribution else {},
        "disease_environment_flags": diag.disease_environment_flags or {},
        "image_refresh": result.image_refresh_this_step,
    })
