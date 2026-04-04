"""
System health schemas — used by ``GET /api/system/health``.

Health rows surface sensor-array and pipeline diagnostics for the
Dashboard's system-health panel.  Status values mirror the ``HEALTH_ROWS``
constant in ``mockData.js``.

Frontend mock shape (``HEALTH_ROWS``)::

    [
      { label: 'Sensor Array',  status: 'Nominal', ok: true  },
      { label: 'Network Link',  status: 'Strong',  ok: true  },
      { label: 'Data Pipeline', status: 'Active',  ok: true  },
      { label: 'Calibration',   status: 'Due: 6d', ok: false },
    ]

Design notes
------------
- ``status`` is a **free-text display string** (e.g. ``'Nominal'``,
  ``'Strong'``, ``'Active'``, ``'Due: 6d'``), not an enumerated type.
  This lets the backend set context-specific labels without a schema change.
- ``ok`` is a plain boolean — drives the colour of the status indicator in
  the React UI (green if ``ok`` is ``True``, amber/red otherwise).
- ``status_code`` is the backend-internal enumerated severity code used for
  programmatic logic.  The React UI does not render it directly.
- ``SystemHealthResponse.overall`` is derived from the worst ``status_code``
  across all rows and drives the panel border colour.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from agritwin_gh.schemas.enums import SystemStatus


class HealthRow(BaseModel):
    """Status for one system component or data pipeline.

    Returned as an element of ``SystemHealthResponse.rows``.

    Example::

        { "label": "Sensor Array", "status": "Nominal",
          "ok": true, "status_code": "ok",
          "detail": "All 9 sensors responding within tolerance.",
          "last_updated": "2026-04-04T09:14:00Z" }
    """

    label: str = Field(
        description="Component display name shown in the UI, e.g. 'Sensor Array'.",
    )
    status: str = Field(
        default="",
        description="Free-text display status string rendered directly in the UI, "
                    "e.g. 'Nominal', 'Strong', 'Active', 'Due: 6d'. "
                    "No fixed enum — the service layer sets context-specific labels.",
    )
    ok: bool = Field(
        default=True,
        description="True if the component is operating normally. "
                    "Drives the colour of the status indicator dot in the React UI.",
    )
    status_code: SystemStatus = Field(
        default="ok",
        description="Backend-internal severity code: 'ok' | 'warning' | 'error'. "
                    "Used to compute SystemHealthResponse.overall.",
    )
    detail: str = Field(
        default="",
        description="Optional one-line detail string shown on hover or in an "
                    "expanded panel.",
    )
    last_updated: str = Field(
        default="",
        description="ISO-8601 timestamp or human-readable recency of the last "
                    "successful check for this component.",
    )


class SystemHealthResponse(BaseModel):
    """Response for ``GET /api/system/health``.

    The ``rows`` list is ordered from most-critical to least-critical
    component for easy scanning.  The ``overall`` field reflects the worst
    ``status_code`` across all rows.

    Example::

        {
          "rows": [
            { "label": "Sensor Array",  "status": "Nominal",  "ok": true,
              "status_code": "ok",    "detail": "All sensors nominal." },
            { "label": "Network Link",  "status": "Strong",   "ok": true,
              "status_code": "ok",    "detail": "Latency < 5 ms." },
            { "label": "Data Pipeline", "status": "Active",   "ok": true,
              "status_code": "ok",    "detail": "Last write 30 s ago." },
            { "label": "Calibration",   "status": "Due: 6d",  "ok": false,
              "status_code": "warning", "detail": "Temp sensor calibration due." }
          ],
          "overall": "warning",
          "timestamp": "2026-04-04T09:15:00Z"
        }
    """

    rows: list[HealthRow]
    overall: SystemStatus = Field(
        default="ok",
        description="Worst status_code across all rows. "
                    "Drives the panel border colour in the Dashboard.",
    )
    timestamp: str = Field(
        default="",
        description="ISO-8601 timestamp of the health check run.",
    )
