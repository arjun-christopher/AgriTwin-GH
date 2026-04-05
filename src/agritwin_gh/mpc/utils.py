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


def load_keras_model(model_path: Path, compile: bool = False) -> "Any":  # noqa: A002
    """Load a Keras ``.keras`` model with cross-version compatibility shims.

    Keras models saved with versions that serialise ``quantization_config``
    inside Dense layer configs will fail to load on versions that no longer
    accept that argument.  This helper monkey-patches ``Dense.from_config``
    to silently drop that key before delegating to the original method, then
    restores the original after loading.
    """
    import tensorflow as tf  # noqa: PLC0415

    _Dense = tf.keras.layers.Dense
    _orig_from_config = _Dense.from_config.__func__

    @classmethod  # type: ignore[misc]
    def _patched_from_config(cls, config):
        config = dict(config)
        config.pop("quantization_config", None)
        return _orig_from_config(cls, config)

    _Dense.from_config = _patched_from_config
    try:
        model = tf.keras.models.load_model(model_path, compile=compile)
    finally:
        _Dense.from_config = classmethod(_orig_from_config)

    return model
