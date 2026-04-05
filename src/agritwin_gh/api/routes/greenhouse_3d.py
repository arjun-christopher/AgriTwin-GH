# Module-level variable to store the last POSTed greenhouse state
_last_greenhouse_state = None
"""
3D Greenhouse API routes — `/api/greenhouse-3d/*`.

This module exposes endpoints specifically for 3D environment integration.
"""


from fastapi import APIRouter, Depends, HTTPException, Body
from pydantic import BaseModel, Field
from agritwin_gh.api.dependencies import get_dashboard_service
from agritwin_gh.schemas.dt_schemas import DTStateResponse
from agritwin_gh.services.dashboard_service import DashboardService

router = APIRouter(tags=["3D Greenhouse"])


# Pydantic model for the incoming 3D greenhouse update payload
class ActuatorState(BaseModel):
    isOn: bool

class CropStageState(BaseModel):
    stage: str

class TimeOfDayState(BaseModel):
    time: str

class CropHealthState(BaseModel):
    state: str
    blinkGreen: bool = False

class GreenhouseState(BaseModel):
    fluorescentLight: ActuatorState
    heater: ActuatorState
    energyCanister: ActuatorState
    humidifier: ActuatorState
    windowFan: ActuatorState
    vent: ActuatorState
    waterTankFloor: ActuatorState
    cropStage: CropStageState
    timeOfDay: TimeOfDayState
    cropHealth: CropHealthState

class Greenhouse3DUpdate(GreenhouseState):
    fluorescentLight: dict = Field(...)
    heater: dict = Field(...)
    energyCanister: dict = Field(...)
    humidifier: dict = Field(...)
    windowFan: dict = Field(...)
    vent: dict = Field(...)
    waterTankFloor: dict = Field(...)
    cropStage: dict = Field(...)
    timeOfDay: dict = Field(...)
    cropHealth: dict = Field(...)


@router.get("/greenhouse-3d/state", response_model=GreenhouseState, status_code=200)
async def get_greenhouse_3d_state(
    svc: DashboardService = Depends(get_dashboard_service),
) -> GreenhouseState:
    """Return the last POSTed 3D greenhouse state if available, else fallback to DashboardService."""
    global _last_greenhouse_state
    if _last_greenhouse_state is not None:
        # Return the last POSTed state as GreenhouseState
        return GreenhouseState(**_last_greenhouse_state.dict())
    # Fallback to original logic
    try:
        dt_state: DTStateResponse = svc.get_dt_state()
        sc = dt_state.scene_context
        def get_actuator_on(act_name):
            mapping = {
                "fluorescentLight": "led",
                "heater": "heater",
                "energyCanister": "co2",
                "humidifier": "fogger",
                "windowFan": "fan",
                "vent": "vent",
                "waterTankFloor": "irrigation",
            }
            key = mapping[act_name]
            return ActuatorState(isOn=bool(sc.actuator_states_on_off.get(key, False)))

        crop_stage = CropStageState(stage=dt_state.current_growth_stage.title() if dt_state.current_growth_stage else "")
        time_of_day = TimeOfDayState(time=dt_state.time_of_day.title() if dt_state.time_of_day else "")
        health_map = {
            "Healthy": "Green",
            "Warning": "Yellow",
            "Risk": "Red",
        }
        backend_health = dt_state.health.status if hasattr(dt_state, "health") and hasattr(dt_state.health, "status") else "Healthy"
        mapped_health = health_map.get(str(backend_health), "Green")
        crop_health = CropHealthState(state=mapped_health)

        return GreenhouseState(
            fluorescentLight=get_actuator_on("fluorescentLight"),
            heater=get_actuator_on("heater"),
            energyCanister=get_actuator_on("energyCanister"),
            humidifier=get_actuator_on("humidifier"),
            windowFan=get_actuator_on("windowFan"),
            vent=get_actuator_on("vent"),
            waterTankFloor=get_actuator_on("waterTankFloor"),
            cropStage=crop_stage,
            timeOfDay=time_of_day,
            cropHealth=crop_health,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail="3D greenhouse state temporarily unavailable.")


# POST endpoint to receive 3D environment updates
@router.post("/greenhouse-3d/state", status_code=200)
async def post_greenhouse_3d_state(
    payload: Greenhouse3DUpdate = Body(...)
):
    """Accepts updates from the 3D greenhouse environment (Unity/WebGL, etc).
    Stores the last POSTed state for GET requests."""
    global _last_greenhouse_state
    _last_greenhouse_state = payload
    return {"ok": True, "received": payload.dict()}
