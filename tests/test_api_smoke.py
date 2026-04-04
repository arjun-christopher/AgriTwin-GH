"""
API smoke tests — FastAPI + frontend integration layer.

Covers every endpoint that the React frontend consumes via api.js.
All tests run against the in-process ASGI app (no live server required)
using FastAPI's ``TestClient`` / ``httpx.AsyncClient``.

The ``TESTING=1`` env var is set module-wide so the lifespan startup hook
skips the background DT loop (which requires a live DB).

Usage
-----
Run with pytest from the repo root::

    $env:PYTHONPATH = "src"
    pytest tests/test_api_smoke.py -v

Or via uv::

    uv run pytest tests/test_api_smoke.py -v
"""

from __future__ import annotations

import os
import sys

# ── Path + TESTING guard ──────────────────────────────────────────────────────
# Must be set before importing the app so lifespan skips the DB loop.
os.environ.setdefault("TESTING", "1")
os.environ.setdefault("PYTHONPATH", "src")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from fastapi.testclient import TestClient

from agritwin_gh.api.app import create_app

# ---------------------------------------------------------------------------
# Shared client fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client() -> TestClient:
    """Create a fresh TestClient for the module.

    ``TestClient`` runs the FastAPI lifespan synchronously, so the startup
    hook fires once (and skips the DT loop because ``TESTING=1``).
    """
    app = create_app()
    with TestClient(app) as c:
        yield c


# ===========================================================================
# GET endpoints — structure + HTTP status
# ===========================================================================

class TestGetDtState:
    def test_status_ok(self, client: TestClient) -> None:
        r = client.get("/api/dt/state")
        assert r.status_code == 200, r.text

    def test_has_crop(self, client: TestClient) -> None:
        body = client.get("/api/dt/state").json()
        assert "crop" in body
        crop = body["crop"]
        assert "current" in crop
        assert "progress_pct" in crop
        assert "stages" in crop
        assert isinstance(crop["stages"], list)
        assert len(crop["stages"]) == 6   # Seedling → Ripe

    def test_has_health(self, client: TestClient) -> None:
        body = client.get("/api/dt/state").json()
        health = body["health"]
        assert "status" in health
        assert health["status"] in ("Healthy", "Warning", "Risk")
        assert "risk_score" in health

    def test_has_sensors(self, client: TestClient) -> None:
        body = client.get("/api/dt/state").json()
        sensors = body["sensors"]
        assert isinstance(sensors, list)
        assert len(sensors) >= 4           # at least temp / humidity / co2 / light
        for s in sensors:
            assert "key" in s
            assert "value" in s
            assert "unit" in s
            assert s["status"] in ("ok", "warning", "critical")

    def test_has_growth(self, client: TestClient) -> None:
        body = client.get("/api/dt/state").json()
        growth = body["growth"]
        assert "current_stage" in growth
        assert "hours_to_next_stage" in growth

    def test_mode_field(self, client: TestClient) -> None:
        body = client.get("/api/dt/state").json()
        assert "mode" in body
        assert body["mode"] in ("live", "override")

    def test_3d_fields_present(self, client: TestClient) -> None:
        body = client.get("/api/dt/state").json()
        assert "time_of_day" in body
        assert body["time_of_day"] in ("morning", "afternoon", "evening", "night")
        assert "current_growth_stage" in body
        assert "actuator_visual_state" in body
        assert "scene_context" in body
        sc = body["scene_context"]
        assert "actuator_states_on_off" in sc
        assert "actuator_levels" in sc
        assert "time_of_day" in sc


class TestGetActuatorState:
    def test_status_ok(self, client: TestClient) -> None:
        r = client.get("/api/actuators/state")
        assert r.status_code == 200, r.text

    def test_seven_actuators(self, client: TestClient) -> None:
        body = client.get("/api/actuators/state").json()
        acts = body["actuators"]
        assert isinstance(acts, list)
        assert len(acts) == 7
        ids = {a["id"] for a in acts}
        assert ids == {"fan", "vent", "irrigation", "heater", "led", "co2", "fogger"}

    def test_actuator_fields(self, client: TestClient) -> None:
        acts = client.get("/api/actuators/state").json()["actuators"]
        for a in acts:
            assert "id" in a
            assert "label" in a
            assert "active" in a
            assert "level" in a
            assert 0.0 <= float(a["level"]) <= 100.0
            assert "status" in a
            assert a["status"] in ("ON", "OFF", "MANUAL")


class TestGetWeather:
    def test_status_ok(self, client: TestClient) -> None:
        r = client.get("/api/weather/current")
        assert r.status_code == 200, r.text

    def test_has_current_and_forecast(self, client: TestClient) -> None:
        body = client.get("/api/weather/current").json()
        assert "current" in body
        assert "forecast" in body
        assert isinstance(body["forecast"], list)

    def test_current_has_temp(self, client: TestClient) -> None:
        current = client.get("/api/weather/current").json()["current"]
        assert "temp" in current
        assert "humidity" in current
        assert "solar_rad" in current


class TestGetDiseaseRisks:
    def test_status_ok(self, client: TestClient) -> None:
        r = client.get("/api/intelligence/disease")
        assert r.status_code == 200, r.text

    def test_has_composite_risk(self, client: TestClient) -> None:
        body = client.get("/api/intelligence/disease").json()
        assert "composite_risk" in body
        assert 0.0 <= float(body["composite_risk"]) <= 1.0

    def test_has_pathogens_list(self, client: TestClient) -> None:
        body = client.get("/api/intelligence/disease").json()
        assert "pathogens" in body
        assert isinstance(body["pathogens"], list)


class TestGetGrowthIntel:
    def test_status_ok(self, client: TestClient) -> None:
        r = client.get("/api/intelligence/growth")
        assert r.status_code == 200, r.text

    def test_has_stage_fields(self, client: TestClient) -> None:
        body = client.get("/api/intelligence/growth").json()
        assert "current_stage" in body
        assert "hours_to_next_stage" in body
        assert "transition_prob_24h" in body
        assert "stage_history" in body


class TestGetResources:
    def test_status_ok(self, client: TestClient) -> None:
        r = client.get("/api/resources/monthly")
        assert r.status_code == 200, r.text

    def test_has_metrics(self, client: TestClient) -> None:
        body = client.get("/api/resources/monthly").json()
        assert "metrics" in body
        metrics = body["metrics"]
        assert isinstance(metrics, list)
        for m in metrics:
            assert "name" in m
            assert "year_month" in m
            assert "used" in m
            assert "unit" in m


class TestGetMediaLatest:
    def test_status_ok(self, client: TestClient) -> None:
        r = client.get("/api/media/latest")
        assert r.status_code == 200, r.text

    def test_has_images_list(self, client: TestClient) -> None:
        body = client.get("/api/media/latest").json()
        assert "images" in body
        assert isinstance(body["images"], list)


class TestGetStageImages:
    def test_status_ok(self, client: TestClient) -> None:
        r = client.get("/api/media/stage-images")
        assert r.status_code == 200, r.text

    def test_has_stages_dict(self, client: TestClient) -> None:
        body = client.get("/api/media/stage-images").json()
        assert "stages" in body
        stages = body["stages"]
        assert isinstance(stages, dict)


class TestGetDiseaseScans:
    def test_status_ok(self, client: TestClient) -> None:
        r = client.get("/api/media/disease-scans")
        assert r.status_code == 200, r.text

    def test_has_scans_list(self, client: TestClient) -> None:
        body = client.get("/api/media/disease-scans").json()
        assert "scans" in body
        assert isinstance(body["scans"], list)


class TestGetSystemHealth:
    def test_status_ok(self, client: TestClient) -> None:
        r = client.get("/api/system/health")
        assert r.status_code == 200, r.text

    def test_has_rows(self, client: TestClient) -> None:
        body = client.get("/api/system/health").json()
        assert "rows" in body
        rows = body["rows"]
        assert isinstance(rows, list)
        assert len(rows) >= 4
        for row in rows:
            assert "label" in row
            assert "ok" in row
            assert isinstance(row["ok"], bool)


# ===========================================================================
# POST endpoints — request acceptance + response shape
# ===========================================================================

class TestPostDtSimOverride:
    _payload = {
        "stage": "flowering",
        "day_in_stage": 7,
        "start_date": "2026-04-04",
        "start_hour": 9,
    }

    def test_status_ok(self, client: TestClient) -> None:
        r = client.post("/api/dt/override/sim", json=self._payload)
        assert r.status_code == 200, r.text

    def test_response_ok_flag(self, client: TestClient) -> None:
        body = client.post("/api/dt/override/sim", json=self._payload).json()
        assert body["ok"] is True

    def test_applied_stage_echoed(self, client: TestClient) -> None:
        body = client.post("/api/dt/override/sim", json=self._payload).json()
        assert body["applied_stage"] == "flowering"

    def test_dt_state_reflects_override(self, client: TestClient) -> None:
        """After applying override, GET /dt/state must report mode='override'
        and show the overridden stage in crop.current.
        """
        client.post("/api/dt/override/sim", json=self._payload)
        state = client.get("/api/dt/state").json()
        assert state["mode"] == "override"
        # crop.current is the title-cased display name
        assert "lowering" in state["crop"]["current"].lower()  # "Flowering"
        assert state["current_growth_stage"] == "flowering"
        assert state["crop"]["days_in_stage"] == 7.0

    def test_invalid_stage_returns_422(self, client: TestClient) -> None:
        r = client.post("/api/dt/override/sim", json={
            "stage": "not-a-real-stage",
            "day_in_stage": 1,
            "start_date": "2026-04-04",
            "start_hour": 0,
        })
        assert r.status_code in (422, 400), r.text


class TestPostDtParamOverride:
    def test_status_ok(self, client: TestClient) -> None:
        r = client.post("/api/dt/override", json={
            "param": "temperature_setpoint",
            "value": 24.5,
        })
        assert r.status_code == 200, r.text

    def test_response_structure(self, client: TestClient) -> None:
        body = client.post("/api/dt/override", json={
            "param": "co2_target",
            "value": 900.0,
        }).json()
        assert body["ok"] is True
        assert body["applied_param"] == "co2_target"
        assert float(body["applied_value"]) == pytest.approx(900.0)


class TestDeleteDtOverride:
    def test_clears_override_returns_ok(self, client: TestClient) -> None:
        # First apply an override
        client.post("/api/dt/override/sim", json={
            "stage": "seedling",
            "day_in_stage": 3,
            "start_date": "2026-04-04",
            "start_hour": 6,
        })
        # Now clear it
        r = client.delete("/api/dt/override")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["ok"] is True

    def test_dt_state_reverts_to_live(self, client: TestClient) -> None:
        # Re-apply then clear
        client.post("/api/dt/override/sim", json={
            "stage": "seedling",
            "day_in_stage": 2,
            "start_date": "2026-04-04",
            "start_hour": 7,
        })
        client.delete("/api/dt/override")
        state = client.get("/api/dt/state").json()
        assert state["mode"] == "live"


class TestPostActuatorSet:
    _payload = {
        "actuators": [
            {"id": "fan",  "level": 80.0},
            {"id": "led",  "level": 60.0},
            {"id": "vent", "level": 40.0},
        ]
    }

    def test_status_ok(self, client: TestClient) -> None:
        r = client.post("/api/actuators/set", json=self._payload)
        assert r.status_code == 200, r.text

    def test_response_ok_flag(self, client: TestClient) -> None:
        body = client.post("/api/actuators/set", json=self._payload).json()
        assert body["ok"] is True

    def test_overridden_list_has_valid_ids(self, client: TestClient) -> None:
        body = client.post("/api/actuators/set", json=self._payload).json()
        valid = {"fan", "vent", "irrigation", "heater", "led", "co2", "fogger"}
        for aid in body.get("overridden", []):
            assert aid in valid

    def test_unknown_id_does_not_crash(self, client: TestClient) -> None:
        """Unknown actuator IDs are silently skipped — see ControlService doc."""
        r = client.post("/api/actuators/set", json={
            "actuators": [
                {"id": "nonexistent_actuator", "level": 50.0},
                {"id": "fan", "level": 30.0},
            ]
        })
        assert r.status_code == 200, r.text
        body = r.json()
        assert "fan" in body["overridden"]
        assert "nonexistent_actuator" not in body.get("overridden", [])

    def test_level_clamped(self, client: TestClient) -> None:
        """Levels > 100 must be clamped to 100 by ControlService."""
        r = client.post("/api/actuators/set", json={
            "actuators": [{"id": "fan", "level": 999.0}]
        })
        assert r.status_code == 200, r.text

    def test_actuator_state_reflects_set(self, client: TestClient) -> None:
        """After POST /api/actuators/set, GET /api/actuators/state must show
        the updated level and MANUAL status.
        """
        client.post("/api/actuators/set", json={
            "actuators": [{"id": "heater", "level": 55.0}]
        })
        acts = client.get("/api/actuators/state").json()["actuators"]
        heater = next(a for a in acts if a["id"] == "heater")
        assert float(heater["level"]) == pytest.approx(55.0)
        assert heater["status"] == "MANUAL"


class TestPostDtPreset:
    @pytest.mark.parametrize("preset_id", ["day-cycle", "night-cycle", "emergency-flush"])
    def test_known_presets_ok(self, client: TestClient, preset_id: str) -> None:
        r = client.post(f"/api/dt/preset/{preset_id}")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["ok"] is True
        assert body["preset"] == preset_id
        assert "description" in body

    def test_unknown_preset_returns_error(self, client: TestClient) -> None:
        r = client.post("/api/dt/preset/not-a-preset")
        assert r.status_code in (400, 404, 422), r.text
