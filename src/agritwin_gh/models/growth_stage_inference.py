"""
growth_stage_inference.py
──────────────────────────────────────────────────────────────────────────────
Standalone inference module for the Tomato Growth Stage classifier.
Trained with Task-1 of the AgriTwin-GH pipeline.

Usage
-----
>>> from agritwin_gh.models.growth_stage_inference import predict_growth_stage
>>> result = predict_growth_stage("path/to/leaf.jpg")
>>> print(result["class_name"], result["confidence"])

Or with raw bytes:
>>> with open("leaf.jpg", "rb") as f:
...     result = predict_growth_stage(f.read())

With explicit paths (e.g. for serving from a container):
>>> result = predict_growth_stage(
...     img_bytes,
...     model_path="models/growth_stage_20260301_120000.keras",
...     label_map_path="models/artifacts/growth_stage_20260301_120000/label_map.json",
... )
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import numpy as np

# Lazy TensorFlow import ─ keeps module-side-effects minimal
_tf    = None
_keras = None


def _import_tf():
    global _tf, _keras
    if _tf is None:
        import tensorflow as tf   # noqa: PLC0415
        import keras              # noqa: PLC0415
        _tf    = tf
        _keras = keras
    return _tf, _keras


# ── Backbone preprocessing map (must mirror training CONFIG) ──────────────────

def _get_preprocess_fn(backbone: str):
    """Return the Keras preprocessing function for a given backbone name."""
    _import_tf()
    from keras.applications import (  # noqa: PLC0415
        efficientnet,
        efficientnet_v2,
        mobilenet_v2,
        resnet_v2,
        inception_v3,
    )
    _map = {
        "EfficientNetB0" : efficientnet.preprocess_input,
        "EfficientNetB3" : efficientnet.preprocess_input,
        "EfficientNetV2S": efficientnet_v2.preprocess_input,
        "MobileNetV2"    : mobilenet_v2.preprocess_input,
        "ResNet50V2"     : resnet_v2.preprocess_input,
        "InceptionV3"    : inception_v3.preprocess_input,
    }
    return _map.get(backbone, efficientnet.preprocess_input)


# ── Module-level model/label-map cache ───────────────────────────────────────

_CACHE: dict = {}


def _load_artifacts(model_path: Path, label_map_path: Path):
    """Load and cache the Keras model + label map (thread-safe for read-only use)."""
    cache_key = (str(model_path), str(label_map_path))
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    tf, keras = _import_tf()

    model = keras.models.load_model(str(model_path))

    with open(label_map_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)

    idx_to_label = {int(k): v for k, v in label_map["idx_to_label"].items()}
    label_names  = label_map["label_names"]
    backbone     = label_map.get("backbone", "EfficientNetB3")
    image_size   = tuple(label_map.get("image_size", [300, 300]))

    entry = (model, idx_to_label, label_names, backbone, image_size)
    _CACHE[cache_key] = entry
    return entry


# ── Low-level image preprocessing ────────────────────────────────────────────

def _preprocess_image(
    image_bytes_or_path: Union[str, Path, bytes, bytearray],
    image_size: tuple[int, int],
    preprocess_fn,
):
    """
    Load an image from a path or bytes, resize, normalise, and add batch dim.

    Returns:
        tf.Tensor of shape [1, H, W, 3] with dtype float32.
    """
    tf, _ = _import_tf()

    if isinstance(image_bytes_or_path, (str, Path)):
        raw = tf.io.read_file(str(image_bytes_or_path))
        img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    elif isinstance(image_bytes_or_path, (bytes, bytearray)):
        img = tf.image.decode_image(
            tf.constant(bytes(image_bytes_or_path)),
            channels=3,
            expand_animations=False,
        )
    else:
        raise TypeError(
            f"image_bytes_or_path must be str | Path | bytes | bytearray; "
            f"got {type(image_bytes_or_path).__name__}"
        )

    img = tf.image.resize(img, image_size)           # [H, W, 3]
    img = tf.cast(img, tf.float32)
    img = preprocess_fn(img)                          # backbone normalisation
    return tf.expand_dims(img, axis=0)                # [1, H, W, 3]


# ── Public API ────────────────────────────────────────────────────────────────

def predict_growth_stage(
    image_bytes_or_path: Union[str, Path, bytes, bytearray],
    *,
    model_path     : Union[str, Path, None] = None,
    label_map_path : Union[str, Path, None] = None,
    k              : int  = 3,
    tta            : bool = False,
    tta_steps      : int  = 5,
) -> dict:
    """
    Classify a tomato plant image into one of the six growth stages.

    Parameters
    ----------
    image_bytes_or_path
        Filesystem path (str or Path) **or** raw image bytes.
    model_path
        Path to the ``.keras`` model file.
        If omitted, the latest ``growth_stage_*.keras`` in this module's
        directory is used.
    label_map_path
        Path to ``label_map.json``.
        If omitted, inferred as ``artifacts/<run_id>/label_map.json``
        relative to this module's directory.
    k
        Number of top predictions included in the ``topk`` list.
    tta
        Enable Test-Time Augmentation (small random rotations averaged).
    tta_steps
        Number of TTA forward passes to average (only used when ``tta=True``).

    Returns
    -------
    dict
        ``class_name``  – top-1 class label (e.g. ``"Stage3_Flowering_Initiation"``)
        ``confidence``  – top-1 softmax probability in [0, 1]
        ``probs``       – full {class_name: probability} mapping over all classes
        ``topk``        – [(class_name, probability)] sorted descending, length ``k``
    """
    tf, _ = _import_tf()

    # ── Resolve artifact paths ────────────────────────────────────────────────
    HERE = Path(__file__).resolve().parent

    if model_path is None:
        candidates = sorted(HERE.glob("growth_stage_*.keras"))
        if not candidates:
            raise FileNotFoundError(
                f"No growth_stage_*.keras found in {HERE}. "
                "Supply model_path= explicitly."
            )
        model_path = candidates[-1]      # latest alphabetically == latest run

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if label_map_path is None:
        run_id         = model_path.stem   # e.g. growth_stage_20260301_120000
        label_map_path = HERE / "artifacts" / run_id / "label_map.json"

    label_map_path = Path(label_map_path)
    if not label_map_path.exists():
        raise FileNotFoundError(f"label_map.json not found: {label_map_path}")

    # ── Load (cached after first call) ────────────────────────────────────────
    model, idx_to_label, label_names, backbone, image_size = _load_artifacts(
        model_path, label_map_path
    )
    preprocess_fn = _get_preprocess_fn(backbone)

    # ── Preprocess image ──────────────────────────────────────────────────────
    img_tensor = _preprocess_image(image_bytes_or_path, image_size, preprocess_fn)

    # ── Inference ─────────────────────────────────────────────────────────────
    if tta and tta_steps > 1:
        import keras.layers as kl           # noqa: PLC0415
        rot_layer   = kl.RandomRotation(factor=0.08)
        accumulated = None
        for _ in range(tta_steps):
            aug      = rot_layer(img_tensor, training=True)
            step_out = model(aug, training=False).numpy()
            accumulated = step_out if accumulated is None else accumulated + step_out
        probs_array = (accumulated / tta_steps)[0]
    else:
        probs_array = model(img_tensor, training=False).numpy()[0]

    # ── Build result dict ─────────────────────────────────────────────────────
    top1_idx   = int(np.argmax(probs_array))
    class_name = idx_to_label[top1_idx]
    confidence = float(probs_array[top1_idx])

    probs_dict = {
        idx_to_label[i]: float(probs_array[i])
        for i in range(len(label_names))
    }
    topk = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)[:k]

    return {
        "class_name": class_name,
        "confidence": confidence,
        "probs"     : probs_dict,
        "topk"      : topk,
    }
