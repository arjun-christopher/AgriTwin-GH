"""
Shared utility functions for the MPC package.
"""

from __future__ import annotations

from pathlib import Path

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
_ARTIFACTS_DIR = _MODELS_DIR / "artifacts"


def discover_latest_artifact(prefix: str) -> Path | None:
    """Return the newest artifact directory whose name starts with *prefix*.

    Scans ``src/agritwin_gh/models/artifacts/`` for directories matching
    *prefix* and returns the lexicographically last one (newest by timestamp
    convention).  Returns ``None`` if no match is found or the artifacts
    directory does not exist.
    """
    if not _ARTIFACTS_DIR.is_dir():
        return None
    candidates = sorted(
        (d for d in _ARTIFACTS_DIR.iterdir() if d.is_dir() and d.name.startswith(prefix)),
        key=lambda p: p.name,
        reverse=True,
    )
    return candidates[0] if candidates else None
