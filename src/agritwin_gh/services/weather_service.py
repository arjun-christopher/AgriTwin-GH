"""
WeatherService — wraps the ARX weather disturbance forecast for the API.

Responsibility
--------------
* Retrieve the latest outdoor weather observation from ``RuntimeStore``.
* Generate a short-range hourly forecast using
  ``agritwin_gh.mpc.disturbance.WeatherDisturbanceForecast``.
* Fall back to a synthetic diurnal model (mirroring
  ``agritwin_gh.mpc.dt_runtime_prep.prepare_weather_sequence``) when the
  ML model artifact is unavailable.
* Format the result as a ``WeatherResponse`` for ``GET /api/weather/current``.

Modules reused from the MPC layer
----------------------------------
* ``agritwin_gh.mpc.disturbance.WeatherDisturbanceForecast``
* ``agritwin_gh.mpc.dt_runtime_prep.prepare_weather_sequence``
* ``agritwin_gh.mpc.state.WeatherState``
"""

from __future__ import annotations

import logging

from agritwin_gh.core.runtime_store import RuntimeStore

logger = logging.getLogger("agritwin.services.weather")

# Lucide icon keys for common weather conditions (used by ForecastEntry.icon_key).
_CONDITION_ICON_MAP: dict[str, str] = {
    "sunny": "Sun",
    "partly cloudy": "CloudSun",
    "overcast": "Cloud",
    "rainy": "CloudRain",
    "stormy": "CloudLightning",
    "foggy": "CloudFog",
}


class WeatherService:
    """Serves weather data for the API weather endpoint."""

    def __init__(self, store: RuntimeStore) -> None:
        self._store = store

    def get_current(self):
        """Return current outdoor conditions and forecast as ``WeatherResponse``.

        TODO (implementation phase):
            1. Load the environment-forecast model artifact via
               ``WeatherDisturbanceForecast``.
            2. Run the forecast for the next 6 hours.
            3. Determine weather status (Healthy / Warning / Risk) from stress
               thresholds (temp > 35 → Warning, RH > 85 → Warning, etc.).
            4. Build ``WeatherResponse`` with real forecast entries.
        """
        return self._store.as_weather_response()
