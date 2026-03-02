"""
classify_growth_stage_input.py
──────────────────────────────
Two modes of operation:

  1. Folder classification  – classifies every image in a chosen folder.
  2. AI-generated plant     – generates a synthetic tomato plant image via
                              Stable Diffusion for a user-specified growth
                              stage, displays it, classifies it with the
                              trained EfficientNetB3 model, then discards
                              the image (nothing is written to disk).

Usage
─────
    python scripts/classify_growth_stage_input.py
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
    "Stage1_Seedling"              : "Stage 1 – Seedling",
    "Stage2_Early_Vegetative"      : "Stage 2 – Early Vegetative",
    "Stage3_Flowering_Initiation"  : "Stage 3 – Flowering Initiation",
    "Stage4_Flowering"             : "Stage 4 – Flowering",
    "Stage5_Unripe"                : "Stage 5 – Unripe",
    "Stage6_Ripe"                  : "Stage 6 – Ripe",
}

# User-facing stage names → Stable Diffusion prompt fragment
STAGE_PROMPTS = {
    "1": ("Stage 1 – Seedling",
          "young tomato seedling, two cotyledons with first true leaves just emerging, "
          "tiny fragile sprout in a greenhouse"),
    "2": ("Stage 2 – Early Vegetative",
          "early vegetative tomato plant, small leafy stem, multiple true leaves, "
          "compact green plant in a greenhouse"),
    "3": ("Stage 3 – Flowering Initiation",
          "tomato plant beginning to flower, small yellow flower buds forming, "
          "green foliage with emerging blossoms in a greenhouse"),
    "4": ("Stage 4 – Flowering",
          "tomato plant in full flower, bright yellow star-shaped open blossoms, "
          "lush green leaves, row of flowers in a greenhouse"),
    "5": ("Stage 5 – Unripe",
          "tomato plant with small green unripe tomatoes on the vine, "
          "firm green fruit clusters in a greenhouse"),
    "6": ("Stage 6 – Ripe",
          "tomato plant with ripe red tomatoes on the vine, "
          "bright red fully grown fruit clusters in a greenhouse"),
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# ─────────────────────────────────────────────────────────────────────────────
# FIND MODEL ARTIFACTS
# ─────────────────────────────────────────────────────────────────────────────

def _find_latest_artifacts() -> Tuple[pathlib.Path, pathlib.Path]:
    """
    Locate the most-recently-trained .keras model and its label_map.json.
    Looks in src/agritwin_gh/models/ for growth_stage_*.keras files and the
    matching artifacts/<run_id>/label_map.json.
    """
    models_dir    = _REPO_ROOT / "src" / "agritwin_gh" / "models"
    artifacts_dir = models_dir / "artifacts"

    keras_files = sorted(models_dir.glob("growth_stage_*_best.keras"), reverse=True)
    if not keras_files:
        keras_files = sorted(models_dir.glob("growth_stage_*.keras"), reverse=True)
    if not keras_files:
        raise FileNotFoundError(
            f"No growth stage .keras model files found in {models_dir}.\n"
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

def _resolve_model_paths() -> Tuple[pathlib.Path, pathlib.Path]:
    """
    Resolve and return (model_path, label_map_path); also ensures the src
    package is on sys.path.  The returned paths are passed explicitly to
    predict_growth_stage() so the module-level cache warms up on the first
    call and subsequent calls are instant.
    """
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    src_path = str(_REPO_ROOT / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    print("Locating growth stage classifier model ...", end=" ", flush=True)
    model_path, label_map_path = _find_latest_artifacts()
    run_id = label_map_path.parent.name
    print(f"done  (run {run_id})\n")
    return model_path, label_map_path


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

    model_path, label_map_path = _resolve_model_paths()  # also sets up sys.path

    from agritwin_gh.models.growth_stage_inference import predict_growth_stage

    # Warm up the model cache with the first image so the loading message
    # appears before the table header.
    print("Loading model weights ...", end=" ", flush=True)
    _warmup = predict_growth_stage(
        images[0],
        model_path     = model_path,
        label_map_path = label_map_path,
        k              = 1,
    )
    print("done\n")

    print(f"{'#':<4}  {'File Name':<40}  {'Predicted Stage':<36}  {'Confidence':>10}")
    print("-" * 96)

    for idx, img_path in enumerate(images, 1):
        try:
            result       = predict_growth_stage(
                image_bytes_or_path = img_path,
                model_path          = model_path,
                label_map_path      = label_map_path,
                k                   = 1,
            )
            pred_label   = result["class_name"]
            pred_display = LABEL_DISPLAY.get(pred_label, pred_label)
            conf_str     = f"{result['confidence']:.1%}"
        except Exception as e:
            pred_display = "ERROR"
            conf_str     = str(e)[:20]

        print(f"{idx:<4}  {img_path.name:<40}  {pred_display:<36}  {conf_str:>10}")

    print("-" * 96)
    print(f"\nDone. Classified {len(images)} image(s).\n")


# ─────────────────────────────────────────────────────────────────────────────
# MODE 2 – AI-GENERATE THEN CLASSIFY (no image saved to disk)
# ─────────────────────────────────────────────────────────────────────────────

def generate_and_classify() -> None:
    print("\nAvailable growth stage options:")
    for key, (name, _) in STAGE_PROMPTS.items():
        print(f"  [{key}] {name}")

    choice = input("\nEnter option number: ").strip()
    if choice not in STAGE_PROMPTS:
        print("[ERROR] Invalid choice.")
        sys.exit(1)

    stage_name, stage_desc = STAGE_PROMPTS[choice]

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
        f"Ultra realistic high resolution close-up photograph of a tomato plant "
        f"showing {stage_desc}, natural lighting, "
        f"professional agricultural photography, 4k, sharp focus"
    )
    negative_prompt = (
        "blurry, cartoon, illustration, painting, low quality, watermark, text"
    )

    print(f"Generating synthetic tomato plant image  [{stage_name}] ...", flush=True)
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
    plt.title(f"Generated plant – {stage_name}", fontsize=13, pad=10)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.5)

    # ── Convert PIL → bytes (no disk write) ──────────────────────────────────
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    image_bytes = buf.getvalue()
    buf.close()
    del pil_image  # release memory; image was never written to disk

    # ── Resolve model paths and predict ──────────────────────────────────────
    model_path, label_map_path = _resolve_model_paths()  # also sets up sys.path

    from agritwin_gh.models.growth_stage_inference import predict_growth_stage

    print("Classifying generated image ...\n")
    result = predict_growth_stage(
        image_bytes_or_path = image_bytes,
        model_path          = model_path,
        label_map_path      = label_map_path,
        k                   = 3,
    )

    pred_label   = result["class_name"]
    pred_display = LABEL_DISPLAY.get(pred_label, pred_label)
    confidence   = result["confidence"]

    print("=" * 50)
    print(f"  Generated stage   : {stage_name}")
    print(f"  Predicted stage   : {pred_display}")
    print(f"  Confidence        : {confidence:.1%}")
    print()
    print("  Top-3 predictions:")
    for label, prob in result["topk"]:
        display = LABEL_DISPLAY.get(label, label)
        bar = "█" * int(prob * 30)
        print(f"    {display:<36}  {prob:.1%}  {bar}")
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
    print("  AgriTwin-GH  ·  Tomato Growth Stage Classifier")
    print("=" * 60)
    print()
    print("  [1] Classify images from a folder")
    print("  [2] Generate a synthetic plant image with AI and classify it")
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
