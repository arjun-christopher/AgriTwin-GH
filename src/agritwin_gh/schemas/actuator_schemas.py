"""
Actuator state schemas — used by:
  ``GET /api/actuators/state``
  ``POST /api/actuators/set``

Canonical actuator IDs (seven, matching ``CONTROL_VARIABLES`` in
``agritwin_gh.mpc.constants`` after short-ID mapping):

    fan | vent | irrigation | heater | led | co2 | fogger

MPC ``CONTROL_VARIABLES`` use longer underscore names (``fan_speed``,
``vent_opening``, …).  The mapping to frontend short IDs lives in
``schemas.enums.CONTROL_VAR_TO_ACTUATOR_ID``.

3D readiness
------------
``ActuatorEntry`` carries ``on_off`` and ``level`` so that the
``ActuatorVisualState`` block in ``DTStateResponse`` can be assembled
cheaply from a flat list without a second store query.

Example ``GET /api/actuators/state`` response
---------------------------------------------
::

    {
      "actuators": [
        {
          "id": "fan",
          "label": "Ventilation Fan",
          "icon_key": "Fan",
          "status": "ON",
          "active": true,
          "level": 75.0,
          "value_display": "75%",
          "color": "primary",
          "on_off": true
        },
        ...
      ]
    }
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from agritwin_gh.schemas.enums import (
    ActuatorId,
    ActuatorStatus,
    ACTUATOR_COLOR,
    ACTUATOR_ICON_KEY,
    ACTUATOR_IDS,
    ACTUATOR_LABEL,
)


class ActuatorEntry(BaseModel):
    """State of a single actuator at the current control step.

    Consumed by:
      - ``HomeDashboard`` ``ActuatorTile`` (shows icon + label + ON/OFF pill).
      - ``DetailedInsights`` ``ActuatorChip`` (shows icon + label + ``value_display``).
      - ``ManualOverride`` ``ActuatorCard`` (shows toggle + level slider).
    """

    id: str = Field(
        description="Canonical frontend actuator ID: "
                    "fan | vent | irrigation | heater | led | co2 | fogger.",
    )
    label: str = Field(
        description="Human-readable display name, e.g. 'Ventilation Fan'.",
    )
    icon_key: str = Field(
        default="Activity",
        description="Lucide React icon name for the UI, e.g. 'Fan'. "
                    "See ACTUATOR_ICON_KEY in enums.py.",
    )
    status: ActuatorStatus = Field(
        default="OFF",
        description="Operational status: 'ON' (MPC-controlled, duty > 0) | "
                    "'OFF' (duty == 0) | 'MANUAL' (operator override active).",
    )
    active: bool = Field(
        default=False,
        description="True when the actuator is on and producing output.",
    )
    level: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Duty-cycle level 0–100 (raw number, not pre-formatted).",
    )
    value_display: str = Field(
        default="",
        description="Pre-formatted level string for the DetailedInsights chip, "
                    "e.g. '75%', '45 L', '55%'. "
                    "Set by the service layer based on actuator type and units.",
    )
    color: str = Field(
        default="primary",
        description="UI badge colour token (maps to --color-<token> CSS var). "
                    "See ACTUATOR_COLOR in enums.py.",
    )
    # 3D scene carry-along — not rendered by the current React UI.
    on_off: bool = Field(
        default=False,
        description="True when duty-cycle > 0.  Redundant with active but provided "
                    "explicitly for the 3D scene bridge.",
    )


class ActuatorStateResponse(BaseModel):
    """Response for ``GET /api/actuators/state``.

    The ``actuators`` list always contains exactly seven entries, one per
    canonical actuator ID in the order defined by ``ACTUATOR_IDS`` in enums.py.
    """

    actuators: list[ActuatorEntry]


class ActuatorSetEntry(BaseModel):
    """One override entry in a ``POST /api/actuators/set`` body.

    ``level`` is the target duty-cycle (0 = off, 100 = full).  The route
    handler derives ``active`` and ``status`` from ``level > 0``.
    """

    id: str = Field(description="Canonical actuator ID (fan | vent | irrigation | heater | led | co2 | fogger).")
    level: float = Field(ge=0.0, le=100.0, description="Target duty-cycle 0–100.")


class ActuatorSetRequest(BaseModel):
    """Body for ``POST /api/actuators/set``.

    The ``ManualOverride`` page sends this when the operator clicks
    **"Apply Changes"** or **"Confirm Reset"**.

    Example body::

        {
          "actuators": [
            { "id": "fan", "level": 80 },
            { "id": "heater", "level": 0 }
          ]
        }

    Partial lists are supported — only the listed actuators are updated.
    The ``ManualOverride`` "Reset all OFF" button sends all seven with ``level: 0``.
    """

    actuators: list[ActuatorSetEntry]


class ActuatorSetResponse(BaseModel):
    """Response for ``POST /api/actuators/set``.

    ``overridden`` echoes the IDs of actuators that were actually changed
    (i.e. those whose level differed from the current state).
    """

    ok: bool = True
    overridden: list[str] = Field(
        default_factory=list,
        description="Canonical IDs of actuators that were updated.",
    )
