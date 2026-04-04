#!/usr/bin/env python3
"""
AgriTwin-GH — API endpoint sampler.

Hits every key endpoint and pretty-prints a condensed JSON sample.
Useful for quick sanity checks without running pytest.

Usage (backend must be running on port 8000)::

    python scripts/sample_api_responses.py

Override the base URL::

    $env:AGRITWIN_API_URL = "http://localhost:8000"
    python scripts/sample_api_responses.py

The script prints a one-line summary per endpoint and an indented excerpt
of the response body.  It exits with code 1 if any request fails.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
import urllib.error


BASE_URL = os.environ.get("AGRITWIN_API_URL", "http://localhost:8000").rstrip("/")
PASS_COUNT = 0
FAIL_COUNT = 0


# ── helpers ───────────────────────────────────────────────────────────────────

def _green(s: str) -> str:
    return f"\033[32m{s}\033[0m"

def _red(s: str) -> str:
    return f"\033[31m{s}\033[0m"

def _yellow(s: str) -> str:
    return f"\033[33m{s}\033[0m"

def _bold(s: str) -> str:
    return f"\033[1m{s}\033[0m"

def _excerpt(data: object, max_keys: int = 5) -> str:
    """Return a short human-readable excerpt of a parsed JSON value."""
    if isinstance(data, dict):
        keys = list(data.keys())[:max_keys]
        snippet = {k: data[k] for k in keys}
        extra = len(data) - max_keys
        tail = f" … +{extra} more" if extra > 0 else ""
        return json.dumps(snippet, indent=2)[:400] + tail
    if isinstance(data, list):
        preview = data[:2]
        extra = len(data) - 2
        tail = f"\n  … and {extra} more" if extra > 0 else ""
        return json.dumps(preview, indent=2)[:400] + tail
    return str(data)[:200]


def _request(method: str, path: str, body: dict | None = None) -> tuple[int, object]:
    """Make an HTTP request and return (status_code, parsed_json).

    Raises ``urllib.error.URLError`` / ``urllib.error.HTTPError`` on failure
    but returns (error_status, error_body) after printing the error.
    """
    url = f"{BASE_URL}{path}"
    data_bytes: bytes | None = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data_bytes = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data_bytes, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode(errors="replace")[:300]
        return exc.code, {"error": body_text}
    except urllib.error.URLError as exc:
        return 0, {"error": str(exc.reason)}


def probe(
    label: str,
    method: str,
    path: str,
    body: dict | None = None,
    expected_status: int = 200,
) -> object:
    """Run one request, print result, update pass/fail counters, return the body."""
    global PASS_COUNT, FAIL_COUNT
    t0 = time.monotonic()
    status, data = _request(method, path, body)
    elapsed_ms = (time.monotonic() - t0) * 1000

    ok = status == expected_status
    badge = _green("PASS") if ok else _red("FAIL")
    status_str = _green(str(status)) if ok else _red(str(status))
    print(f"\n{_bold(label)}")
    print(f"  {badge}  {method} {path}  →  HTTP {status_str}  ({elapsed_ms:.0f} ms)")
    if ok:
        PASS_COUNT += 1
        print(_excerpt(data))
    else:
        FAIL_COUNT += 1
        print(_red("  Response: ") + str(data)[:300])
    return data


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(_bold(f"\nAgriTwin-GH API Sampler — {BASE_URL}"))
    print("=" * 60)

    # ── GET endpoints ─────────────────────────────────────────────────────
    dt_state = probe("GET /api/dt/state", "GET", "/api/dt/state")
    probe("GET /api/actuators/state", "GET", "/api/actuators/state")
    probe("GET /api/weather/current", "GET", "/api/weather/current")
    probe("GET /api/intelligence/disease", "GET", "/api/intelligence/disease")
    probe("GET /api/intelligence/growth", "GET", "/api/intelligence/growth")
    probe("GET /api/resources/monthly", "GET", "/api/resources/monthly")
    probe("GET /api/system/health", "GET", "/api/system/health")
    probe("GET /api/media/latest", "GET", "/api/media/latest")

    # ── POST: param override ──────────────────────────────────────────────
    probe(
        "POST /api/dt/override  (param)",
        "POST",
        "/api/dt/override",
        body={"param": "temperature_setpoint", "value": 24.5},
    )

    # ── POST: sim override → verify mode change ───────────────────────────
    current_stage = "flowering"
    if isinstance(dt_state, dict):
        raw = dt_state.get("current_growth_stage", "flowering")
        current_stage = raw if raw else "flowering"

    probe(
        "POST /api/dt/override/sim  (stage override)",
        "POST",
        "/api/dt/override/sim",
        body={
            "stage":        current_stage,
            "day_in_stage": 5,
            "start_date":   "2026-04-04",
            "start_hour":   9,
        },
    )

    state_after = probe(
        "GET /api/dt/state  (verify mode=override)",
        "GET",
        "/api/dt/state",
        expected_status=200,
    )
    if isinstance(state_after, dict):
        mode = state_after.get("mode", "?")
        mode_display = _green(mode) if mode == "override" else _red(mode)
        print(f"  mode field: {mode_display}")

    # ── POST: actuator set ────────────────────────────────────────────────
    probe(
        "POST /api/actuators/set",
        "POST",
        "/api/actuators/set",
        body={"actuators": [
            {"id": "fan",  "level": 75.0},
            {"id": "led",  "level": 60.0},
            {"id": "vent", "level": 40.0},
        ]},
    )

    # ── POST: preset ──────────────────────────────────────────────────────
    probe(
        "POST /api/dt/preset/day-cycle",
        "POST",
        "/api/dt/preset/day-cycle",
    )

    # ── DELETE: clear override ────────────────────────────────────────────
    probe(
        "DELETE /api/dt/override  (clear → mode=live)",
        "DELETE",
        "/api/dt/override",
    )

    state_cleared = probe(
        "GET /api/dt/state  (verify mode=live after clear)",
        "GET",
        "/api/dt/state",
        expected_status=200,
    )
    if isinstance(state_cleared, dict):
        mode = state_cleared.get("mode", "?")
        mode_display = _green(mode) if mode == "live" else _red(mode)
        print(f"  mode field: {mode_display}")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    total = PASS_COUNT + FAIL_COUNT
    summary = f"  Result: {PASS_COUNT}/{total} passed"
    print(_green(summary) if FAIL_COUNT == 0 else _red(summary))
    if FAIL_COUNT > 0:
        print(_yellow("  Check that `python main.py` is running on port 8000."))
        sys.exit(1)


if __name__ == "__main__":
    main()
