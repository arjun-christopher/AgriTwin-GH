"""
Weather schemas — used by ``GET /api/weather/current``.

Wraps the output of ``agritwin_gh.mpc.disturbance.WeatherDisturbanceForecast``
into a frontend-ready shape.

Frontend mock shapes
--------------------
``WEATHER_STATUS`` (``mockData.js``)::

    {
      status: 'Warning',
      condition: 'High External Humidity',
      detail: 'High outdoor humidity may increase fungal disease pressure.',
      forecast: 'Expected to clear in ~3 hrs'
    }

``OUTDOOR_CURRENT``::

    {
      temp: 18.4, humidity: 89, windSpeed: 12, windDir: 'NNE',
      pressure: 1013, uvIndex: 3, dewPoint: 16.8,
      condition: 'Overcast', visibility: 8.2
    }

``OUTDOOR_FORECAST`` (array of 6 hourly entries)::

    { time: '12:00', high: 22, low: 19, humidity: 74,
      condition: 'Partly Cloudy', iconKey: 'CloudSun' }

Field naming note
-----------------
All fields use snake_case in the Python schema.  The ``api.js`` layer
translates to camelCase for the React UI when swapping out mock data.

The ``ForecastEntry.icon_key`` field carries a Lucide icon name so the
React ``DetailedInsights`` page can display the correct weather icon via its
local ``ICON_MAP`` lookup without a backend icon-name translation service.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from agritwin_gh.schemas.enums import HealthStatus


class OutdoorCurrent(BaseModel):
    """Outdoor ambient conditions at the latest observation timestamp.

    Matches ``OUTDOOR_CURRENT`` in ``mockData.js``.  All numeric fields use
    SI or common meteorological units as documented below.

    Example::

        {
          "temp": 18.4,
          "humidity": 89.0,
          "wind_speed": 12.0,
          "wind_dir": "NNE",
          "pressure": 1013.0,
          "uv_index": 3.0,
          "dew_point": 16.8,
          "solar_rad": 220.0,
          "condition": "Overcast",
          "visibility": 8.2
        }
    """

    temp: float = Field(description="Outdoor dry-bulb temperature (°C).")
    humidity: float = Field(description="Outdoor relative humidity (%).")
    wind_speed: float = Field(default=0.0, description="Wind speed (km/h).")
    wind_dir: str = Field(
        default="",
        description="Wind direction abbreviated string, e.g. 'NNE'.",
    )
    pressure: float = Field(
        default=1013.25,
        description="Atmospheric pressure at station level (hPa / mbar).",
    )
    uv_index: float = Field(
        default=0.0,
        ge=0.0,
        description="UV index (0–11+).  From the weather forecast model output.",
    )
    dew_point: float = Field(
        default=0.0,
        description="Dew-point temperature (°C), computed from temp and humidity.",
    )
    solar_rad: float = Field(
        default=0.0,
        ge=0.0,
        description="Global horizontal solar irradiance (W/m²).",
    )
    condition: str = Field(
        default="",
        description="Human-readable condition string, e.g. 'Overcast', 'Partly Cloudy'.",
    )
    visibility: float = Field(
        default=10.0,
        ge=0.0,
        description="Horizontal visibility (km).",
    )


class ForecastEntry(BaseModel):
    """One hourly forecast slot.

    Returned as an element of ``WeatherResponse.forecast``.  Typically 6 slots
    are provided (next 6 hours).

    ``icon_key`` is resolved by the React ``DetailedInsights`` page via its
    local ``ICON_MAP``.  Common values: ``'Cloud'``, ``'CloudSun'``,
    ``'CloudRain'``, ``'Sun'``, ``'Snowflake'``.

    Example::

        {
          "time": "14:00",
          "high": 22.0,
          "low": 19.0,
          "humidity": 74.0,
          "condition": "Partly Cloudy",
          "icon_key": "CloudSun"
        }
    """

    time: str = Field(description="Display time string in HH:MM format, e.g. '14:00'.")
    high: float = Field(description="Forecast high temperature (°C).")
    low: float = Field(description="Forecast low temperature (°C).")
    humidity: float = Field(description="Forecast relative humidity (%).")
    condition: str = Field(description="Human-readable condition string.")
    icon_key: str = Field(
        default="Cloud",
        description="Lucide icon name resolved by the React DetailedInsights ICON_MAP.",
    )


class WeatherStatus(BaseModel):
    """Alert / summary status for the Dashboard weather ``StatusCard``.

    Matches ``WEATHER_STATUS`` in ``mockData.js``.

    ``condition`` is the **short condition label** shown as the card title
    (e.g. ``'High External Humidity'``).  ``detail`` is the full sentence
    shown in the card body.  ``forecast_note`` is the advisory footer text.

    Example::

        {
          "status": "Warning",
          "condition": "High External Humidity",
          "detail": "High outdoor humidity may increase fungal disease pressure.",
          "forecast_note": "Expected to clear in ~3 hrs"
        }
    """

    status: HealthStatus = Field(
        default="Healthy",
        description="Traffic-light status for the weather StatusCard.",
    )
    condition: str = Field(
        default="",
        description="Short label for the dominant weather condition, "
                    "e.g. 'High External Humidity'.  Used as the card subtitle.",
    )
    detail: str = Field(
        default="",
        description="One-sentence explanation of why the status was set.",
    )
    forecast_note: str = Field(
        default="",
        description="Short advisory note, e.g. 'Expected to clear in ~3 hrs'. "
                    "Corresponds to ``forecast`` field in the frontend mock.",
    )


class WeatherResponse(BaseModel):
    """Response for ``GET /api/weather/current``.

    Contains the three blocks consumed by the frontend:
      - ``status``  → Dashboard weather ``StatusCard`` (``WEATHER_STATUS`` mock).
      - ``current`` → ``detailedInsights`` outdoor conditions panel
                      (``OUTDOOR_CURRENT`` mock).
      - ``forecast``→ ``detailedInsights`` hourly forecast strip
                      (``OUTDOOR_FORECAST`` mock).

    Typical forecast length is 6 entries (next 6 hours at hourly resolution).

    Example::

        {
          "status": { "status": "Warning", "condition": "High External Humidity",
                      "detail": "...", "forecast_note": "..." },
          "current": { "temp": 18.4, "humidity": 89, ... },
          "forecast": [ { "time": "12:00", "high": 22, ... }, ... ],
          "timestamp": "2026-04-04T09:00:00Z"
        }
    """

    status: WeatherStatus
    current: OutdoorCurrent
    forecast: list[ForecastEntry]
    timestamp: str = Field(
        default="",
        description="ISO-8601 UTC timestamp of the observation / model run.",
    )
