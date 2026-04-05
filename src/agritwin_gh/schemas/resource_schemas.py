"""
Resource consumption and cost schemas — used by ``GET /api/resources/monthly``.

Energy and water accounting formulas are documented in
``docs/MPC_COMPLETE_GUIDE.md`` §18 ("Resource Cost Calculation").

Currency: Indian Rupees (INR).  Electricity tariff and water rate are
loaded from ``agritwin_gh.mpc.config.MPCConfig``.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ResourceEntry(BaseModel):
    """Usage for one resource type in a billing period."""

    label: str = Field(description="'Water' | 'Energy'")
    used: float
    unit: str = Field(description="'L' for water, 'kWh' for energy.")


class MonthlyCost(BaseModel):
    """Cost breakdown for a calendar month."""

    month: str = Field(description="Display string, e.g. 'April 2026'.")
    energy_inr: float = Field(default=0.0, description="Energy cost in INR.")
    water_inr: float = Field(default=0.0, description="Water cost in INR.")
    total_inr: float = Field(default=0.0, description="Total resource cost in INR.")


class ActuatorResourceEntry(BaseModel):
    """Per-actuator monthly resource consumption and cost."""

    key: str = Field(description="MPC control variable key, e.g. 'fan_speed'.")
    label: str = Field(description="Display label, e.g. 'Ventilation Fan'.")
    energy_kwh: float = Field(default=0.0, description="Energy consumed this month in kWh.")
    water_l: float = Field(default=0.0, description="Water used this month in litres (irrigation only).")
    cost_inr: float = Field(default=0.0, description="Total cost for this actuator in INR.")


class ResourcesResponse(BaseModel):
    """Response for ``GET /api/resources/monthly``."""

    resources: list[ResourceEntry]
    cost: MonthlyCost
    actuators: list[ActuatorResourceEntry] = Field(
        default_factory=list,
        description="Per-actuator resource breakdown for the current billing month.",
    )
    timestamp: str = Field(
        default="",
        description="ISO-8601 timestamp when the resource totals were computed.",
    )
