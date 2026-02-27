"""
classify_generated_leaf.py
──────────────────────────
Classifies tomato leaf images in a folder using the trained EfficientNetB0
model.  For every image found, prints the filename and predicted disease class.

Usage
─────
    python scripts/classify_generated_leaf.py
    (Enter the folder path when prompted)
"""

from __future__ import annotations

import os
import pathlib
import sys
from typing import Tuple

# ── resolve repo root (works from any cwd) ───────────────────────────────────
_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent   # …/scripts/
_REPO_ROOT  = _SCRIPT_DIR.parent                        # …/AgriTwin-GH/

# ─────────────────────────────────────────────────────────────────────────────
# LABEL MAP
# ─────────────────────────────────────────────────────────────────────────────
LABEL_DISPLAY = {
    "tomato_early_blight"   : "Early Blight",
    "tomato_late_blight"    : "Late Blight",
    "tomato_leaf_mold"      : "Leaf Mold",
    "tomato_powdery_mildew" : "Powdery Mildew",
    "tomato_spider_mites"   : "Spider Mites",
    "tomato_leaf_healthy"   : "Healthy",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# ─────────────────────────────────────────────────────────────────────────────
# FIND MODEL ARTIFACTS
# ─────────────────────────────────────────────────────────────────────────────

def _find_latest_artifacts() -> Tuple[pathlib.Path, pathlib.Path]:
    """
    Locate the most-recently-trained .keras model and its label_map.json.
    Looks in src/agritwin_gh/models/ for *.keras files and the matching
    artifacts/<run_id>/label_map.json.
    """
    models_dir    = _REPO_ROOT / "src" / "agritwin_gh" / "models"
    artifacts_dir = models_dir / "artifacts"

    keras_files = sorted(models_dir.glob("*_best.keras"), reverse=True)
    if not keras_files:
        keras_files = sorted(models_dir.glob("*.keras"), reverse=True)
    if not keras_files:
        raise FileNotFoundError(
            f"No .keras model files found in {models_dir}.\n"
            "Train the model first by running the training notebook."
        )

    model_path = keras_files[0]

    stem   = model_path.stem
    run_id = stem.replace("_best", "")

    label_map_path = artifacts_dir / run_id / "label_map.json"
    if not label_map_path.exists():
        candidates = sorted(artifacts_dir.rglob("label_map.json"), reverse=True)
        if not candidates:
            raise FileNotFoundError(
                f"label_map.json not found under {artifacts_dir}."
            )
        label_map_path = candidates[0]

    return model_path, label_map_path

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print()
    print("=" * 60)
    print("  AgriTwin-GH  ·  Tomato Disease Classifier")
    print("  Batch Folder Classification")
    print("=" * 60)
    print()

    # ── Folder input ─────────────────────────────────────────────────────────
    folder_input = input("Enter image folder path: ").strip().strip('"').strip("'")
    folder = pathlib.Path(folder_input)

    if not folder.exists():
        print(f"\n[ERROR] Folder not found: {folder}")
        sys.exit(1)
    if not folder.is_dir():
        print(f"\n[ERROR] Path is not a directory: {folder}")
        sys.exit(1)

    # ── Collect images ───────────────────────────────────────────────────────
    images = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        print(f"\n[INFO] No images found in: {folder}")
        print(f"       Supported formats: {', '.join(sorted(IMAGE_EXTENSIONS))}")
        sys.exit(0)

    print(f"\nFound {len(images)} image(s) in: {folder}\n")

    # ── Load model ───────────────────────────────────────────────────────────
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    src_path = str(_REPO_ROOT / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    from agritwin_gh.models.inference import load_inference_assets, predict_image

    print("Loading model ...", end=" ", flush=True)
    model_path, label_map_path = _find_latest_artifacts()
    inference_model, loaded_label_map = load_inference_assets(model_path, label_map_path)
    run_id = label_map_path.parent.name
    print(f"done  (run {run_id})\n")

    # ── Classify each image ──────────────────────────────────────────────────
    print(f"{'#':<4}  {'File Name':<40}  {'Predicted Class':<22}  {'Confidence':>10}")
    print("-" * 82)

    for idx, img_path in enumerate(images, 1):
        try:
            result = predict_image(
                image_input      = img_path,
                inference_model  = inference_model,
                loaded_label_map = loaded_label_map,
                top_k            = 1,
            )
            pred_label   = result["class_name"]
            pred_display = LABEL_DISPLAY.get(pred_label, pred_label)
            confidence   = result["confidence"]
            conf_str     = f"{confidence:.1%}"
        except Exception as e:
            pred_display = "ERROR"
            conf_str     = str(e)[:20]

        print(f"{idx:<4}  {img_path.name:<40}  {pred_display:<22}  {conf_str:>10}")

    print("-" * 82)
    print(f"\nDone. Classified {len(images)} image(s).\n")


if __name__ == "__main__":
    main()