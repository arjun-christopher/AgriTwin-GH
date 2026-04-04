"""
ResourceService — energy and water consumption aggregation.

Responsibility
--------------
* Accumulate per-step energy (kWh) and water (L) from the DT loop results
  stored in ``RuntimeStore``.
* Group totals by calendar month.
* Apply Tamil Nadu electricity and water tariffs from ``MPCConfig`` to
  compute INR costs.
* Return a ``ResourcesResponse`` for ``GET /api/resources/monthly``.

Cost formula reference
----------------------
See ``docs/MPC_COMPLETE_GUIDE.md`` §18 for the exact energy and water
consumption model formulas.  Brief summary:

    Energy (kWh) = Σ actuator_power_kw × dt_hours   (per step)
    Water (L)    = Σ irrigation_duty × flow_rate_lpm × dt_min   (per step)

    Cost (INR)   = energy_kwh × tariff_per_kwh + water_L / 1000 × tariff_per_kL

Tariffs (Tamil Nadu, April 2026):
    Electricity: ₹7.00 / kWh
    Water:       ₹4.00 / kL

These values are sourced from ``MPCConfig`` at runtime; the constants here
are provisional fallbacks only.
"""

from __future__ import annotations

import logging

from agritwin_gh.core.runtime_store import RuntimeStore

logger = logging.getLogger("agritwin.services.resource")

# Provisional tariff fallbacks (overridden by MPCConfig in implementation phase).
_ENERGY_RATE_INR_PER_KWH: float = 7.0
_WATER_RATE_INR_PER_KL: float = 4.0


class ResourceService:
    """Aggregates energy/water consumption and computes INR costs."""

    def __init__(self, store: RuntimeStore) -> None:
        self._store = store

    def get_monthly(self):
        """Return monthly resource consumption and cost breakdown.

        TODO (implementation phase):
            1. Read ``energy_kwh_total`` and ``water_l_total`` from the store.
            2. Read tariffs from ``load_mpc_config().resource_energy_rate`` etc.
            3. Group by calendar month if the loop has run across month boundaries.
        """
        return self._store.as_resources_response()
