"""
Shared enumeration types, canonical ID sets, and domain metadata constants.

This module is the single source of truth for all string-literal types and
reference data used across the AgriTwin-GH schema layer.  No business logic
lives here — only pure data declarations.

Import hierarchy
----------------
    enums.py  ←  all other schema modules

``enums.py`` itself imports **nothing** from the rest of the
``agritwin_gh`` package to prevent circular dependencies.
The related MPC constants live in ``agritwin_gh.mpc.constants``; values that
the schemas need are duplicated here intentionally so that the API surface can
evolve independently of the physics layer.

3D readiness
------------
``TimeOfDay``, ``ACTUATOR_IDS``, ``ACTUATOR_LABEL``, and
``ACTUATOR_ICON_KEY`` are the key constants consumed by the future 3D
environment layer.  Do not remove or rename them.
"""

from __future__ import annotations

from typing import Literal

# ── String-literal type aliases ───────────────────────────────────────────────

TimeOfDay = Literal["morning", "afternoon", "evening", "night"]
"""Time-of-day bucket used by SceneContext and DTStateResponse.

Buckets (local wall-clock):
  morning   → 06:00 – 11:59
  afternoon → 12:00 – 17:59
  evening   → 18:00 – 20:59
  night     → 21:00 – 05:59
"""

HealthStatus = Literal["Healthy", "Warning", "Risk"]
"""Three-tier traffic-light status for crop health and weather badges.
Matches the ``status`` field of ``StatusCard`` in the React UI.
"""

SeverityLevel = Literal["Low", "Medium", "High", "Critical"]
"""Disease-risk severity tier.  ``Critical`` is reserved for composite risk > 0.8."""

SystemStatus = Literal["ok", "warning", "error"]
"""Overall system health code used by ``SystemHealthResponse.overall``."""

TrendDirection = Literal["up", "down", "stable"]
"""Direction of a sensor metric trend arrow."""

ActuatorStatus = Literal["ON", "OFF", "MANUAL"]
"""Operational status of one actuator.

  ON     — duty-cycle > 0, MPC-controlled.
  OFF    — duty-cycle == 0.
  MANUAL — duty-cycle overridden by the operator via ``POST /api/actuators/set``.
"""

ActuatorId = Literal["fan", "vent", "irrigation", "heater", "led", "co2", "fogger"]
"""Seven canonical frontend-facing actuator IDs.

These are the short IDs used in API payloads and by the React UI.  They map
1-to-1 to ``CONTROL_VARIABLES`` in ``agritwin_gh.mpc.constants`` (which use
longer underscore names like ``fan_speed``).
"""

# ── Actuator canonical sets ────────────────────────────────────────────────────

ACTUATOR_IDS: tuple[str, ...] = (
    "fan",
    "vent",
    "irrigation",
    "heater",
    "led",
    "co2",
    "fogger",
)
"""Ordered tuple of all seven canonical actuator IDs.

Order matches ``CONTROL_VARIABLES`` in ``agritwin_gh.mpc.constants``.
"""

# Map: MPC CONTROL_VARIABLES key  →  frontend short ID
CONTROL_VAR_TO_ACTUATOR_ID: dict[str, str] = {
    "fan_speed":      "fan",
    "vent_opening":   "vent",
    "irrigation_qty": "irrigation",
    "heater_output":  "heater",
    "led_intensity":  "led",
    "co2_valve_pct":  "co2",
    "fogger_duty":    "fogger",
}
"""Translate MPC ``CONTROL_VARIABLES`` names to frontend actuator IDs."""

ACTUATOR_ID_TO_CONTROL_VAR: dict[str, str] = {
    v: k for k, v in CONTROL_VAR_TO_ACTUATOR_ID.items()
}
"""Reverse map: frontend actuator ID → MPC ``CONTROL_VARIABLES`` key."""

ACTUATOR_LABEL: dict[str, str] = {
    "fan":        "Ventilation Fan",
    "vent":       "Vent Opening",
    "irrigation": "Irrigation",
    "heater":     "Heater",
    "led":        "LED Intensity",
    "co2":        "CO₂ Valve",
    "fogger":     "Fogger",
}
"""Human-readable display label per actuator.  Matches labels in ``ManualOverride.jsx``."""

ACTUATOR_ICON_KEY: dict[str, str] = {
    "fan":        "Fan",
    "vent":       "Wind",
    "irrigation": "Droplets",
    "heater":     "Flame",
    "led":        "Sun",
    "co2":        "Leaf",
    "fogger":     "CloudDrizzle",
}
"""Lucide React icon name per actuator.  Values must match icons available in ``lucide-react``."""

ACTUATOR_COLOR: dict[str, str] = {
    "fan":        "primary",
    "vent":       "secondary",
    "irrigation": "secondary",
    "heater":     "warning",
    "led":        "warning",
    "co2":        "primary",
    "fogger":     "secondary",
}
"""UI badge colour token per actuator.  Tokens map to Tailwind ``--color-<token>`` CSS vars."""

# ── Growth stage metadata ─────────────────────────────────────────────────────

GROWTH_STAGE_DISPLAY_NAME: dict[str, str] = {
    "seedling":              "Seedling",
    "early vegetative":      "Early Vegetative",
    "flowering initiation":  "Flowering Initiation",
    "flowering":             "Flowering",
    "unripe":                "Unripe",
    "ripe":                  "Ripe",
}
"""Title-cased display names for canonical growth stage labels.

The backend uses lowercase canonical labels (from ``mpc.constants.GROWTH_STAGES``).
The frontend displays title-cased versions.  Use this map when serialising
to the API response.
"""

STAGE_DURATION_DAYS: dict[str, float] = {
    "seedling":              14.0,   # 336 h (source: realtime_core.py STAGE_DURATION_HOURS)
    "early vegetative":      20.0,   # 480 h
    "flowering initiation":  10.0,   # 240 h
    "flowering":             15.0,   # 360 h
    "unripe":                20.0,   # 480 h
    "ripe":                  10.0,   # 240 h
}
"""Expected duration of each growth stage in days.

Derived directly from ``realtime_core.py STAGE_DURATION_HOURS`` (÷ 24).
Used by the service layer to compute ``progress_pct`` and ``days_to_next_stage``.
"""

GROWTH_STAGE_ORDERED: tuple[str, ...] = (
    "seedling",
    "early vegetative",
    "flowering initiation",
    "flowering",
    "unripe",
    "ripe",
)
"""Canonical growth stage order, mirroring ``mpc.constants.GROWTH_STAGES``."""

# ── Disease metadata ──────────────────────────────────────────────────────────

DISEASE_PATHOGEN: dict[str, str] = {
    "powdery mildew": "L. taurica",
    "spider mites":   "T. urticae",
    "leaf mold":      "P. fulva",
    "early blight":   "A. solani",
    "late blight":    "P. infestans",
    "healthy leaves": "",
}
"""Scientific pathogen name per disease.  Empty string for 'healthy leaves'.

Used to populate ``DiseaseRiskEntry.pathogen`` without querying the DB.
"""

DISEASE_DISPLAY_NAME: dict[str, str] = {
    "powdery mildew": "Powdery Mildew",
    "spider mites":   "Spider Mites",
    "leaf mold":      "Leaf Mold",
    "early blight":   "Early Blight",
    "late blight":    "Late Blight",
    "healthy leaves": "Healthy Leaves",
}
"""Title-cased display names for canonical disease labels."""

GROWTH_CNN_DISPLAY_NAME: dict[str, str] = {
    "Stage1_Seedling":              "Seedling",
    "Stage2_Early_Vegetative":      "Early Vegetative",
    "Stage3_Flowering_Initiation":  "Flw. Init.",
    "Stage4_Flowering":             "Flowering",
    "Stage5_Unripe":                "Unripe",
    "Stage6_Ripe":                  "Ripe",
}
"""Display names for growth-stage CNN class names (from label_map.json)."""

DISEASE_CNN_DISPLAY_NAME: dict[str, str] = {
    "tomato_early_blight":   "Early Blight",
    "tomato_late_blight":    "Late Blight",
    "tomato_leaf_healthy":   "Healthy Leaves",
    "tomato_leaf_mold":      "Leaf Mold",
    "tomato_powdery_mildew": "Powdery Mildew",
    "tomato_spider_mites":   "Spider Mites",
}
"""Display names for disease CNN class names (from label_map.json)."""

# ── Sensor display metadata ────────────────────────────────────────────────────

SENSOR_META: list[dict[str, object]] = [
    {
        "key":           "indoor_temp",
        "label":         "Temperature",
        "unit":          "°C",
        "icon_key":      "Thermometer",
        "range_min":     22.0,
        "range_max":     26.0,
        "optimal_range": "22–26 °C",
    },
    {
        "key":           "indoor_humidity",
        "label":         "Humidity",
        "unit":          "%",
        "icon_key":      "Droplets",
        "range_min":     60.0,
        "range_max":     70.0,
        "optimal_range": "60–70 %",
    },
    {
        "key":           "soil_moisture",
        "label":         "Soil Moisture",
        "unit":          "%",
        "icon_key":      "Sprout",
        "range_min":     50.0,
        "range_max":     75.0,
        "optimal_range": "50–75 %",
    },
    {
        "key":           "co2",
        "label":         "CO₂",
        "unit":          "ppm",
        "icon_key":      "Wind",
        "range_min":     800.0,
        "range_max":     1000.0,
        "optimal_range": "800–1000 ppm",
    },
    {
        "key":           "light_intensity",
        "label":         "Light Intensity",
        "unit":          "lux",
        "icon_key":      "Sun",
        "range_min":     10000.0,
        "range_max":     15000.0,
        "optimal_range": "10k–15k lux",
    },
    {
        "key":           "vpd",
        "label":         "VPD",
        "unit":          "kPa",
        "icon_key":      "Activity",
        "range_min":     0.8,
        "range_max":     1.2,
        "optimal_range": "0.8–1.2 kPa",
    },
    {
        "key":           "leaf_wetness_proxy",
        "label":         "Leaf Wetness",
        "unit":          "proxy",
        "icon_key":      "CloudDrizzle",
        "range_min":     0.0,
        "range_max":     0.3,
        "optimal_range": "0.0–0.3",
    },
    {
        "key":           "disease_risk_score",
        "label":         "Disease Risk",
        "unit":          "score",
        "icon_key":      "AlertTriangle",
        "range_min":     0.0,
        "range_max":     0.3,
        "optimal_range": "< 0.3",
    },
]
"""Per-sensor display metadata for all eight STATE_VARIABLES shown in the UI.

Each entry corresponds to a key in ``agritwin_gh.mpc.constants.STATE_VARIABLES``
(excluding ``growth_stage_index``, which is not shown as a raw number).
Used by the service layer to construct ``SensorReading`` instances.
"""

# ── DT preset catalogue ────────────────────────────────────────────────────────

DT_PRESET_DESCRIPTIONS: dict[str, str] = {
    "day-cycle": (
        "Optimise for standard daytime growing conditions (06:00–21:00): "
        "elevated LED, fan at 60 %, heater reduced."
    ),
    "night-cycle": (
        "Switch to night mode: LED off, reduce fan speed, lower temperature "
        "setpoint to 19 °C, fogger standby."
    ),
    "emergency-flush": (
        "Maximum ventilation and irrigation flush: fan 100 %, vents fully open, "
        "irrigation triggered, CO₂ valve closed."
    ),
}
"""Valid DT preset IDs and their human-readable descriptions.

Used by ``POST /api/dt/preset/{preset_id}`` and ``DtPresetResponse.description``.
Keys must match path parameter values accepted by the route.
"""
