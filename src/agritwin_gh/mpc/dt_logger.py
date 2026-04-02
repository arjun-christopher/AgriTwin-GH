"""
Lightweight step logger for the DT closed-loop simulation.

Accumulates ``DTLoopStepResult`` objects, prints a live progress line
every *N* steps, and writes a JSON summary at the end of the run.

Responsibilities
----------------
* **Live console output** — concise one-line status per cadence event.
* **Accumulation** — stores all results for post-run analysis.
* **JSON export** — ``save()`` writes an artifact file with per-step
  summaries and run-level aggregates.

This module is intentionally simple.  It does *not* depend on a database
or any external service — output goes to a local JSON file and ``stdout``.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import pathlib
from dataclasses import dataclass, field
from typing import Any

from .state import GreenhouseState

logger = logging.getLogger(__name__)


@dataclass
class DTLoopRunSummary:
    """Post-run aggregate statistics."""

    total_steps: int = 0
    total_mpc_solves: int = 0
    total_forced_mpc: int = 0
    total_image_refreshes: int = 0
    total_energy_kwh: float = 0.0
    total_water_litres: float = 0.0
    mean_temp: float = 0.0
    mean_humidity: float = 0.0
    mean_disease_risk: float = 0.0
    min_temp: float = 0.0
    max_temp: float = 0.0
    min_humidity: float = 0.0
    max_humidity: float = 0.0
    min_soil_moisture: float = 0.0
    max_soil_moisture: float = 0.0
    mean_setpoint_error_temp: float = 0.0
    mean_setpoint_error_humidity: float = 0.0
    start_time: _dt.datetime | None = None
    end_time: _dt.datetime | None = None
    growth_stage: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "total_mpc_solves": self.total_mpc_solves,
            "total_forced_mpc": self.total_forced_mpc,
            "total_image_refreshes": self.total_image_refreshes,
            "total_energy_kwh": round(self.total_energy_kwh, 4),
            "total_water_litres": round(self.total_water_litres, 4),
            "mean_temp": round(self.mean_temp, 2),
            "mean_humidity": round(self.mean_humidity, 2),
            "mean_disease_risk": round(self.mean_disease_risk, 4),
            "min_temp": round(self.min_temp, 2),
            "max_temp": round(self.max_temp, 2),
            "min_humidity": round(self.min_humidity, 2),
            "max_humidity": round(self.max_humidity, 2),
            "min_soil_moisture": round(self.min_soil_moisture, 2),
            "max_soil_moisture": round(self.max_soil_moisture, 2),
            "mean_setpoint_error_temp": round(self.mean_setpoint_error_temp, 4),
            "mean_setpoint_error_humidity": round(self.mean_setpoint_error_humidity, 4),
            "start_time": str(self.start_time) if self.start_time else None,
            "end_time": str(self.end_time) if self.end_time else None,
            "growth_stage": self.growth_stage,
        }


class DTLoopLogger:
    """Collects and persists DT closed-loop step results.

    Parameters
    ----------
    growth_stage:
        Stage label for annotation.
    console_every:
        Print a status line every *N* steps.  Set to 0 to disable.
    """

    def __init__(
        self,
        growth_stage: str = "",
        console_every: int = 1,
    ) -> None:
        self._growth_stage = growth_stage
        self._console_every = console_every
        self._step_records: list[dict[str, Any]] = []

        # Running accumulators
        self._total_energy = 0.0
        self._total_water = 0.0
        self._temp_sum = 0.0
        self._hum_sum = 0.0
        self._risk_sum = 0.0
        self._mpc_count = 0
        self._forced_mpc_count = 0
        self._image_count = 0
        self._first_ts: _dt.datetime | None = None
        self._last_ts: _dt.datetime | None = None

        # Min / max accumulators
        self._min_temp = float("inf")
        self._max_temp = float("-inf")
        self._min_hum = float("inf")
        self._max_hum = float("-inf")
        self._min_sm = float("inf")
        self._max_sm = float("-inf")

        # Setpoint error accumulators
        self._setpoint_error_temp_sum = 0.0
        self._setpoint_error_hum_sum = 0.0

    def log_step(self, result: Any) -> None:
        """Record one ``DTLoopStepResult``.

        Accepts the result object from ``DTLoop.run()`` and:
        1. Accumulates metrics.
        2. Stores a compact dict for later export.
        3. Prints a console line if appropriate.
        """
        step = result.step_index
        ns = result.next_state
        diag = result.diagnostics

        # Accumulators
        self._total_energy += diag.energy_kwh
        self._total_water += diag.water_litres
        self._temp_sum += ns.indoor_temp
        self._hum_sum += ns.indoor_humidity
        self._risk_sum += ns.disease_risk_score

        # Min / max
        self._min_temp = min(self._min_temp, ns.indoor_temp)
        self._max_temp = max(self._max_temp, ns.indoor_temp)
        self._min_hum = min(self._min_hum, ns.indoor_humidity)
        self._max_hum = max(self._max_hum, ns.indoor_humidity)
        self._min_sm = min(self._min_sm, ns.soil_moisture)
        self._max_sm = max(self._max_sm, ns.soil_moisture)

        # Setpoint tracking error
        sp_err = diag.setpoint_error or {}
        self._setpoint_error_temp_sum += abs(sp_err.get("indoor_temp", 0.0))
        self._setpoint_error_hum_sum += abs(sp_err.get("indoor_humidity", 0.0))

        if result.mpc_ran_this_step:
            self._mpc_count += 1
        if result.mpc_forced:
            self._forced_mpc_count += 1
        if result.image_refresh_this_step:
            self._image_count += 1

        if result.timestamp:
            if self._first_ts is None:
                self._first_ts = result.timestamp
            self._last_ts = result.timestamp

        # Compact record
        self._step_records.append({
            "step": step,
            "ts": str(result.timestamp) if result.timestamp else None,
            "T": round(ns.indoor_temp, 2),
            "RH": round(ns.indoor_humidity, 2),
            "SM": round(ns.soil_moisture, 2),
            "CO2": round(ns.co2, 1),
            "light": round(ns.light_intensity, 1),
            "risk": round(ns.disease_risk_score, 4),
            "energy": round(diag.energy_kwh, 6),
            "water": round(diag.water_litres, 4),
            "mpc": result.mpc_ran_this_step,
            "mpc_forced": result.mpc_forced,
            "img": result.image_refresh_this_step,
            "mpc_cost": (
                round(result.mpc_cost, 4) if result.mpc_cost is not None
                else None
            ),
        })

        # Console output
        if self._console_every > 0 and step % self._console_every == 0:
            self._print_step(step, ns, diag, result)

    def summary(self) -> DTLoopRunSummary:
        """Compute post-run aggregate statistics."""
        n = len(self._step_records)
        return DTLoopRunSummary(
            total_steps=n,
            total_mpc_solves=self._mpc_count,
            total_forced_mpc=self._forced_mpc_count,
            total_image_refreshes=self._image_count,
            total_energy_kwh=self._total_energy,
            total_water_litres=self._total_water,
            mean_temp=self._temp_sum / max(n, 1),
            mean_humidity=self._hum_sum / max(n, 1),
            mean_disease_risk=self._risk_sum / max(n, 1),
            min_temp=self._min_temp if n > 0 else 0.0,
            max_temp=self._max_temp if n > 0 else 0.0,
            min_humidity=self._min_hum if n > 0 else 0.0,
            max_humidity=self._max_hum if n > 0 else 0.0,
            min_soil_moisture=self._min_sm if n > 0 else 0.0,
            max_soil_moisture=self._max_sm if n > 0 else 0.0,
            mean_setpoint_error_temp=self._setpoint_error_temp_sum / max(n, 1),
            mean_setpoint_error_humidity=self._setpoint_error_hum_sum / max(n, 1),
            start_time=self._first_ts,
            end_time=self._last_ts,
            growth_stage=self._growth_stage,
        )

    def save(self, path: str | pathlib.Path) -> pathlib.Path:
        """Write the complete run log to a JSON file.

        Parameters
        ----------
        path:
            Output file path.  Parent directories are created if needed.

        Returns
        -------
        pathlib.Path
            Resolved path of the written file.
        """
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        summ = self.summary()
        payload = {
            "run_summary": summ.to_dict(),
            "steps": self._step_records,
        }

        with open(p, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

        logger.info("DT loop log saved to %s (%d steps)", p, len(self._step_records))
        return p.resolve()

    # ── Private ───────────────────────────────────────────────────────

    def _print_step(
        self,
        step: int,
        ns: GreenhouseState,
        diag: Any,
        result: Any,
    ) -> None:
        """Print a concise one-line status to stdout."""
        markers: list[str] = []
        if result.mpc_ran_this_step and not result.mpc_forced:
            markers.append("MPC")
        if result.mpc_forced:
            markers.append("MPC!")
        if result.image_refresh_this_step:
            markers.append("IMG")

        marker_str = " [" + ",".join(markers) + "]" if markers else ""

        ts_str = (
            result.timestamp.strftime("%H:%M")
            if result.timestamp else "??:??"
        )

        # Core environmental line
        line = (
            f"  step {step:4d}  {ts_str}  "
            f"T={ns.indoor_temp:5.1f}°C  "
            f"RH={ns.indoor_humidity:5.1f}%  "
            f"SM={ns.soil_moisture:5.1f}%  "
            f"CO2={ns.co2:6.1f}  "
            f"risk={ns.disease_risk_score:.3f}"
            f"{marker_str}"
        )
        print(line)

        # On MPC steps, print key actuator outputs on a second line.
        if result.mpc_ran_this_step:
            act = result.action_applied
            print(
                f"           ↳ act: fan={act.fan_speed:.2f} "
                f"heat={act.heater_output:.2f} "
                f"vent={act.vent_opening:.2f} "
                f"irr={act.irrigation_qty:.2f} "
                f"fog={act.fogger_duty:.2f}"
            )
