"""
Smoke test for /api/greenhouse-3d/state endpoint with all valid input combinations.

- Iterates through all valid values for each key in the 3D greenhouse JSON schema.
- Waits for user input (ENTER) before sending the next test case, allowing step-by-step observation.
- Prints the request and response for each test case.

Usage:
    $env:PYTHONPATH = "src"
    pytest tests/smoke_test_greenhouse_3d.py -s
"""


import os
import sys
import itertools
import subprocess
import time

import pytest
from fastapi.testclient import TestClient


# Dynamically locate the src directory (repo root/src)
def find_src_path():
    here = os.path.abspath(os.path.dirname(__file__))
    root = here
    while True:
        if os.path.isdir(os.path.join(root, "src")):
            return os.path.join(root, "src")
        parent = os.path.dirname(root)
        if parent == root:
            break
        root = parent
    raise RuntimeError("Could not find 'src' directory from {}".format(here))

src_path = find_src_path()
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from agritwin_gh.api.app import create_app

# Ensure TESTING mode and correct import path
os.environ.setdefault("TESTING", "1")
os.environ.setdefault("PYTHONPATH", "src")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Valid values for each field
ACTUATORS = [
    ("fluorescentLight", [True, False]),
    ("heater", [True, False]),
    ("energyCanister", [True, False]),
    ("humidifier", [True, False]),
    ("windowFan", [True, False]),
    ("vent", [True, False]),
    ("waterTankFloor", [True, False]),
]
CROP_HEALTH = ["Green", "Yellow", "Red"]
TIME_OF_DAY = ["Morning", "Afternoon", "Evening", "Night"]
CROP_STAGE = ["Seedling", "Vegetative", "Flowering Initiation", "Flowering", "Unripe", "Ripe"]


# Helper to check if the server is up
import requests
def wait_for_server(url, timeout=15):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url)
            if r.status_code in (200, 404, 405):
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False

@pytest.fixture(scope="module")
def ensure_server():
    # Start FastAPI server if not already running
    server_proc = None
    try:
        try:
            # Try to connect first
            requests.get("http://127.0.0.1:8000/docs", timeout=2)
        except Exception:
            # Not running, so start it
            env = os.environ.copy()
            env["AGRITWIN_NO_FRONTEND"] = "1"
            server_proc = subprocess.Popen([
                sys.executable, "main.py"
            ], cwd=os.path.dirname(os.path.dirname(__file__)), env=env)
            assert wait_for_server("http://127.0.0.1:8000/docs"), "FastAPI server did not start in time"
        yield
    finally:
        if server_proc:
            server_proc.terminate()
            server_proc.wait()

@pytest.fixture(scope="module")
def client(ensure_server):
    # Use real HTTP requests to the running server
    class RealClient:
        def post(self, url, json):
            return requests.post(f"http://127.0.0.1:8000{url}", json=json)
    return RealClient()

def build_payload(actuator_states, crop_health, time_of_day, crop_stage):
    payload = {k: {"isOn": v} for k, v in actuator_states.items()}
    payload["cropHealth"] = {"state": crop_health}
    payload["timeOfDay"] = {"time": time_of_day}
    payload["cropStage"] = {"stage": crop_stage}
    return payload

# Generate all combinations (one value per actuator, health, time, stage)
ALL_COMBINATIONS = list(itertools.product(
    *[vals for _, vals in ACTUATORS],
    CROP_HEALTH,
    TIME_OF_DAY,
    CROP_STAGE,
))

@pytest.mark.parametrize("combination", ALL_COMBINATIONS)
def test_greenhouse_3d_state(client, combination):
    # Map combination to actuator states and other fields
    actuator_keys = [k for k, _ in ACTUATORS]
    actuator_states = dict(zip(actuator_keys, combination[:len(ACTUATORS)]))
    crop_health = combination[len(ACTUATORS)]
    time_of_day = combination[len(ACTUATORS)+1]
    crop_stage = combination[len(ACTUATORS)+2]
    payload = build_payload(actuator_states, crop_health, time_of_day, crop_stage)

    print("\n--- Sending payload:")
    print(payload)
    input("Press ENTER to send request...")
    response = client.post("/api/greenhouse-3d/state", json=payload)
    print("Response status:", response.status_code)
    print("Response body:", response.json())
    assert response.status_code == 200
    # Optionally, add more assertions about the response structure/content
