"""
classify_input_leaf.py
──────────────────────
Two modes of operation:

  1. Folder classification  – classifies every image in a chosen folder.
  2. AI-generated leaf      – generates a synthetic tomato leaf image via
                              Stable Diffusion for a user-specified disease,
                              displays it, classifies it with the trained
                              EfficientNetB0 model, then discards the image
                              (nothing is written to disk).

Usage
─────
    python scripts/classify_input_leaf.py
    (Choose a mode when prompted)
"""

from __future__ import annotations

import io
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

# User-facing disease names → Stable Diffusion prompt fragment
DISEASE_PROMPTS = {
    "1": ("Early Blight",    "early blight disease, dark concentric ring lesions on leaves"),
    "2": ("Late Blight",     "late blight disease, water-soaked dark patches on leaves"),
    "3": ("Leaf Mold",       "leaf mold disease, yellow spots and olive-green mold on leaves"),
    "4": ("Powdery Mildew",  "powdery mildew disease, white powdery coating on leaves"),
    "5": ("Spider Mites",    "spider mite infestation, tiny yellow stippling and webbing on leaves"),
    "6": ("Healthy",         "healthy green leaf, no disease, vibrant and fresh"),
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

    keras_files = sorted(models_dir.glob("disease_*_best.keras"), reverse=True)
    if not keras_files:
        keras_files = sorted(models_dir.glob("disease_*.keras"), reverse=True)
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
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _load_model():
    """Load inference model and label map once; return (inference_model, label_map)."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    src_path = str(_REPO_ROOT / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    from agritwin_gh.models.disease_inference import load_inference_assets

    print("Loading classifier model ...", end=" ", flush=True)
    model_path, label_map_path = _find_latest_artifacts()
    inference_model, loaded_label_map = load_inference_assets(model_path, label_map_path)
    run_id = label_map_path.parent.name
    print(f"done  (run {run_id})\n")
    return inference_model, loaded_label_map


# ─────────────────────────────────────────────────────────────────────────────
# MODE 1 – FOLDER CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def classify_folder() -> None:
    folder_input = input("Enter image folder path: ").strip().strip('"').strip("'")
    folder = pathlib.Path(folder_input)

    if not folder.exists():
        print(f"\n[ERROR] Folder not found: {folder}")
        sys.exit(1)
    if not folder.is_dir():
        print(f"\n[ERROR] Path is not a directory: {folder}")
        sys.exit(1)

    images = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        print(f"\n[INFO] No images found in: {folder}")
        print(f"       Supported formats: {', '.join(sorted(IMAGE_EXTENSIONS))}")
        sys.exit(0)

    print(f"\nFound {len(images)} image(s) in: {folder}\n")

    inference_model, loaded_label_map = _load_model()  # also sets up sys.path

    from agritwin_gh.models.disease_inference import predict_image

    print(f"{'#':<4}  {'File Name':<40}  {'Predicted Class':<22}  {'Confidence':>10}")
    print("-" * 82)

    for idx, img_path in enumerate(images, 1):
        try:
            result       = predict_image(
                image_input      = img_path,
                inference_model  = inference_model,
                loaded_label_map = loaded_label_map,
                top_k            = 1,
            )
            pred_label   = result["class_name"]
            pred_display = LABEL_DISPLAY.get(pred_label, pred_label)
            conf_str     = f"{result['confidence']:.1%}"
        except Exception as e:
            pred_display = "ERROR"
            conf_str     = str(e)[:20]

        print(f"{idx:<4}  {img_path.name:<40}  {pred_display:<22}  {conf_str:>10}")

    print("-" * 82)
    print(f"\nDone. Classified {len(images)} image(s).\n")


# ─────────────────────────────────────────────────────────────────────────────
# MODE 2 – AI-GENERATE THEN CLASSIFY (no image saved to disk)
# ─────────────────────────────────────────────────────────────────────────────

def generate_and_classify() -> None:
    print("\nAvailable disease options:")
    for key, (name, _) in DISEASE_PROMPTS.items():
        print(f"  [{key}] {name}")

    choice = input("\nEnter option number: ").strip()
    if choice not in DISEASE_PROMPTS:
        print("[ERROR] Invalid choice.")
        sys.exit(1)

    disease_name, disease_desc = DISEASE_PROMPTS[choice]

    # ── Install / import diffusers stack ─────────────────────────────────────
    try:
        import torch
        from diffusers import StableDiffusionPipeline
        import matplotlib
        matplotlib.use("TkAgg" if sys.platform == "win32" else "Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[INFO] Required packages not found. Installing ...")
        import subprocess
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "diffusers", "transformers", "accelerate", "torch",
            "--quiet",
        ])
        import torch
        from diffusers import StableDiffusionPipeline
        import matplotlib
        matplotlib.use("TkAgg" if sys.platform == "win32" else "Agg")
        import matplotlib.pyplot as plt

    # ── Load Stable Diffusion pipeline ───────────────────────────────────────
    model_id = "runwayml/stable-diffusion-v1-5"
    print(f"\nLoading Stable Diffusion pipeline  ({model_id}) ...")
    print("(First run downloads ~4 GB; subsequent runs use cache)\n")

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        print("Using GPU (CUDA) for generation.\n")
    else:
        print("No GPU detected – running on CPU (may be slow).\n")

    # ── Build prompt and generate ─────────────────────────────────────────────
    prompt = (
        f"Ultra realistic high resolution close-up photograph of a tomato leaf "
        f"with {disease_desc}, inside a greenhouse, natural lighting, "
        f"professional agricultural photography, 4k, sharp focus"
    )
    negative_prompt = (
        "blurry, cartoon, illustration, painting, low quality, watermark, text"
    )

    print(f"Generating synthetic tomato leaf image  [{disease_name}] ...", flush=True)
    with torch.inference_mode():
        pil_image = pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
        ).images[0]

    # ── Display the generated image ───────────────────────────────────────────
    plt.figure(figsize=(6, 6))
    plt.imshow(pil_image)
    plt.axis("off")
    plt.title(f"Generated leaf – {disease_name}", fontsize=13, pad=10)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.5)

    # ── Convert PIL → bytes (no disk write) ──────────────────────────────────
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    image_bytes = buf.getvalue()
    buf.close()
    del pil_image  # release memory; image was never written to disk

    # ── Load classifier and predict ───────────────────────────────────────────
    inference_model, loaded_label_map = _load_model()  # also sets up sys.path

    from agritwin_gh.models.disease_inference import predict_image

    print("Classifying generated image ...\n")
    result = predict_image(
        image_input      = image_bytes,
        inference_model  = inference_model,
        loaded_label_map = loaded_label_map,
        top_k            = 3,
    )

    pred_label   = result["class_name"]
    pred_display = LABEL_DISPLAY.get(pred_label, pred_label)
    confidence   = result["confidence"]

    print("=" * 50)
    print(f"  Generated disease : {disease_name}")
    print(f"  Predicted class   : {pred_display}")
    print(f"  Confidence        : {confidence:.1%}")
    print()
    print("  Top-3 predictions:")
    for label, prob in result["topk"]:
        display = LABEL_DISPLAY.get(label, label)
        bar = "█" * int(prob * 30)
        print(f"    {display:<22}  {prob:.1%}  {bar}")
    print("=" * 50)
    print("\n[INFO] Generated image was NOT saved to disk.\n")

    # Keep plot open until user closes it
    plt.show(block=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print()
    print("=" * 60)
    print("  AgriTwin-GH  ·  Tomato Disease Classifier")
    print("=" * 60)
    print()
    print("  [1] Classify images from a folder")
    print("  [2] Generate a synthetic leaf image with AI and classify it")
    print()

    mode = input("Choose mode [1/2]: ").strip()

    if mode == "1":
        print()
        classify_folder()
    elif mode == "2":
        print()
        generate_and_classify()
    else:
        print("[ERROR] Invalid choice. Enter 1 or 2.")
        sys.exit(1)


if __name__ == "__main__":
    main()