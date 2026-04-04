"""
Pydantic response and request schemas for the AgriTwin-GH FastAPI layer.

Each sub-module owns the schemas for one API resource group::

    enums               →  shared Literal types, canonical IDs, domain metadata
    dt_schemas          →  /api/dt/*
    actuator_schemas    →  /api/actuators/*
    weather_schemas     →  /api/weather/*
    intelligence_schemas→  /api/intelligence/*
    resource_schemas    →  /api/resources/*
    media_schemas       →  /api/media/*
    system_schemas      →  /api/system/*

All schemas are Pydantic v2 ``BaseModel`` subclasses.  They are deliberately
kept separate from the core MPC dataclasses (``agritwin_gh.mpc.state``) so
that the API surface can evolve independently of the physics layer.

Import hierarchy
----------------
enums  ←  all schema modules  ←  this __init__

No schema module imports from ``__init__``.  Cross-module dependencies are
only allowed in the direction: ``enums`` → other modules, and
``intelligence_schemas`` → ``dt_schemas`` (for ``GrowthIntelResponse``).

Usage
-----
For services and routes, prefer importing directly from the sub-module::

    from agritwin_gh.schemas.dt_schemas import DTStateResponse

For test utilities or external consumers that want a single import point::

    from agritwin_gh.schemas import DTStateResponse, ActuatorStateResponse
"""

# ── Enumeration types and lookup constants ────────────────────────────────────
from agritwin_gh.schemas.enums import (
    ActuatorId,
    ActuatorStatus,
    ACTUATOR_COLOR,
    ACTUATOR_ICON_KEY,
    ACTUATOR_ID_TO_CONTROL_VAR,
    ACTUATOR_IDS,
    ACTUATOR_LABEL,
    CONTROL_VAR_TO_ACTUATOR_ID,
    DISEASE_DISPLAY_NAME,
    DISEASE_PATHOGEN,
    DT_PRESET_DESCRIPTIONS,
    GROWTH_STAGE_DISPLAY_NAME,
    GROWTH_STAGE_ORDERED,
    HealthStatus,
    SENSOR_META,
    SeverityLevel,
    STAGE_DURATION_DAYS,
    SystemStatus,
    TimeOfDay,
    TrendDirection,
)

# ── Digital-twin state & override schemas ─────────────────────────────────────
from agritwin_gh.schemas.dt_schemas import (
    ActuatorVisualState,
    CropHealth,
    CropInfo,
    DtOverrideResponse,
    DtParamOverrideRequest,
    DtPresetResponse,
    DtSimOverrideRequest,
    DTStateResponse,
    SceneContext,
    SensorReading,
)

# ── Actuator control schemas ──────────────────────────────────────────────────
from agritwin_gh.schemas.actuator_schemas import (
    ActuatorEntry,
    ActuatorSetEntry,
    ActuatorSetRequest,
    ActuatorSetResponse,
    ActuatorStateResponse,
)

# ── Intelligence / ML insight schemas ────────────────────────────────────────
from agritwin_gh.schemas.intelligence_schemas import (
    DiseaseRiskEntry,
    DiseaseRisksResponse,
    GrowthIntelResponse,
    StageHistoryEntry,
)

# ── Weather schemas ───────────────────────────────────────────────────────────
from agritwin_gh.schemas.weather_schemas import (
    ForecastEntry,
    OutdoorCurrent,
    WeatherResponse,
    WeatherStatus,
)

# ── Resource accounting schemas ───────────────────────────────────────────────
from agritwin_gh.schemas.resource_schemas import (
    MonthlyCost,
    ResourceEntry,
    ResourcesResponse,
)

# ── Media / image gallery schemas ─────────────────────────────────────────────
from agritwin_gh.schemas.media_schemas import (
    DiseaseImagesResponse,
    ImageEntry,
    LatestMediaResponse,
    LeafScanEntry,
    StageImageEntry,
    StageImagesResponse,
)

# ── System health schemas ─────────────────────────────────────────────────────
from agritwin_gh.schemas.system_schemas import (
    HealthRow,
    SystemHealthResponse,
)

__all__ = [
    # enums / constants
    "TimeOfDay",
    "HealthStatus",
    "SeverityLevel",
    "SystemStatus",
    "TrendDirection",
    "ActuatorStatus",
    "ActuatorId",
    "ACTUATOR_IDS",
    "ACTUATOR_LABEL",
    "ACTUATOR_ICON_KEY",
    "ACTUATOR_COLOR",
    "ACTUATOR_ID_TO_CONTROL_VAR",
    "CONTROL_VAR_TO_ACTUATOR_ID",
    "DISEASE_DISPLAY_NAME",
    "DISEASE_PATHOGEN",
    "DT_PRESET_DESCRIPTIONS",
    "GROWTH_STAGE_DISPLAY_NAME",
    "GROWTH_STAGE_ORDERED",
    "SENSOR_META",
    "STAGE_DURATION_DAYS",
    # dt
    "ActuatorVisualState",
    "SceneContext",
    "CropInfo",
    "CropHealth",
    "SensorReading",
    "DTStateResponse",
    "DtParamOverrideRequest",
    "DtSimOverrideRequest",
    "DtOverrideResponse",
    "DtPresetResponse",
    # actuators
    "ActuatorEntry",
    "ActuatorStateResponse",
    "ActuatorSetEntry",
    "ActuatorSetRequest",
    "ActuatorSetResponse",
    # intelligence
    "DiseaseRiskEntry",
    "DiseaseRisksResponse",
    "StageHistoryEntry",
    "GrowthIntelResponse",
    # weather
    "OutdoorCurrent",
    "ForecastEntry",
    "WeatherStatus",
    "WeatherResponse",
    # resources
    "ResourceEntry",
    "MonthlyCost",
    "ResourcesResponse",
    # media
    "ImageEntry",
    "LatestMediaResponse",
    "StageImageEntry",
    "StageImagesResponse",
    "LeafScanEntry",
    "DiseaseImagesResponse",
    # system
    "HealthRow",
    "SystemHealthResponse",
]

