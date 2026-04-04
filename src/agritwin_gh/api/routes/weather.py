"""
Weather routes — ``/api/weather/*``.

Endpoints
---------
GET /api/weather/current → ``WeatherResponse``

Responsibility
--------------
Serves outdoor ambient conditions (current observation + hourly forecast)
from ``WeatherService``, which wraps:

  * ``agritwin_gh.mpc.disturbance.WeatherDisturbanceForecast`` — ARX-based
    forecast (uses the environment-forecast PyTorch model artifact).
  * A synthetic diurnal fallback when the DB / model is unavailable (mirrors
    ``dt_runtime_prep.prepare_weather_sequence``).

The ``WeatherStatus.status`` field drives the Dashboard weather badge colour
("Healthy" | "Warning" | "Risk").
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from agritwin_gh.api.dependencies import get_dashboard_service
from agritwin_gh.schemas.weather_schemas import WeatherResponse
from agritwin_gh.services.dashboard_service import DashboardService

logger = logging.getLogger("agritwin.api.weather")

router = APIRouter(tags=["Weather"])


@router.get("/weather/current", response_model=WeatherResponse, status_code=200)
async def get_weather_current(
    svc: DashboardService = Depends(get_dashboard_service),
) -> WeatherResponse:
    """Return outdoor conditions and short-range forecast.

    Includes current observation (temp, humidity, solar, wind) and a
    ``forecast`` list of ``ForecastEntry`` objects.  Falls back to a
    synthetic diurnal forecast when the ML model is unavailable.
    """
    try:
        return svc.get_weather()
    except Exception as exc:
        logger.error("GET /weather/current failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Weather data temporarily unavailable.")
