"""
AgriTwin-GH — Interactive Project Setup Script
================================================
Handles:
  1. Install uv package manager (if missing)
  2. Initialise project with uv (if not already)
  3. Create / verify virtual environment
  4. Sync dependencies with uv
  5. (Optional) Download Kaggle dataset, with interactive token setup

Run:
    python setup.py
"""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIR     = PROJECT_ROOT / ".venv"
UV_LOCK      = PROJECT_ROOT / "uv.lock"
PYPROJECT    = PROJECT_ROOT / "pyproject.toml"
REQUIREMENTS = PROJECT_ROOT / "requirements.txt"
KAGGLE_JSON  = Path.home() / ".kaggle" / "kaggle.json"

KAGGLE_DATASET = "arjunchristopher/tomato-greenhouse-environment-growth-and-disease"

# Every directory that must exist for the dataset to be considered complete.
# Even one missing entry triggers the download prompt.
EXPECTED_DATASET_DIRS: list[str] = [
    "data/external/Tomato Diseases/Tomato_Early_Blight",
    "data/external/Tomato Diseases/Tomato_Late_Blight",
    "data/external/Tomato Diseases/Tomato_Leaf_Mold",
    "data/external/Tomato Diseases/Tomato_Powdery_Mildew",
    "data/external/Tomato Diseases/Tomato_Septoria_Leaf_Spot",
    "data/external/Tomato Diseases/Tomato_Spider_Mites",
    "data/external/Tomato Growth Stages/Stage1_Seedling",
    "data/external/Tomato Growth Stages/Stage2_Early_Vegetative",
    "data/external/Tomato Growth Stages/Stage3_Flowering_Initiation",
    "data/external/Tomato Growth Stages/Stage4_Flowering",
    "data/external/Tomato Growth Stages/Stage5_Unripe",
    "data/external/Tomato Growth Stages/Stage6_Ripe",
    "data/external/Tomato Healthy Leaves",
    "data/external/Weather Data",
    "data/processed/Greenhouse Indoor Conditions",
]

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "setup.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("setup")

IS_WINDOWS = platform.system() == "Windows"

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _banner(text: str) -> None:
    line = "─" * 60
    log.info(line)
    log.info(f"  {text}")
    log.info(line)


def _ok(msg: str)   -> None: log.info(f"  ✓  {msg}")
def _warn(msg: str) -> None: log.warning(f"  ⚠  {msg}")
def _err(msg: str)  -> None: log.error(f"  ✗  {msg}")


def _run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
    capture: bool = False,
) -> subprocess.CompletedProcess:
    log.debug("Running: %s", " ".join(str(c) for c in cmd))
    return subprocess.run(
        cmd,
        cwd=str(cwd or PROJECT_ROOT),
        check=check,
        capture_output=capture,
        text=True,
    )


def _ask(prompt: str, *, default: str = "") -> str:
    try:
        answer = input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    return answer or default


def _confirm(prompt: str, *, default: bool = True) -> bool:
    hint   = "[Y/n]" if default else "[y/N]"
    answer = _ask(f"{prompt} {hint}: ", default="y" if default else "n")
    return answer.lower() in {"y", "yes"}


# ──────────────────────────────────────────────────────────────────────────────
# Step 1 — Install uv
# ──────────────────────────────────────────────────────────────────────────────

def _uv_executable() -> str | None:
    """Return the path to the uv binary if it is on PATH, else None."""
    return shutil.which("uv")


def install_uv() -> str:
    """
    Ensure uv is installed and return its executable path/command.

    Install priority:
      Windows → official PowerShell installer  →  fallback: pip install uv
      Unix    → official curl installer        →  fallback: pip install uv
    """
    _banner("Step 1 of 5 — uv package manager")

    uv = _uv_executable()
    if uv:
        result  = _run([uv, "--version"], capture=True, check=False)
        version = result.stdout.strip() if result.returncode == 0 else "unknown"
        _ok(f"uv already installed  ({version})")
        return uv

    log.info("  uv not found — installing …")
    installed = False

    if IS_WINDOWS:
        ps_cmd = (
            'powershell -ExecutionPolicy ByPass -NoProfile -Command '
            '"irm https://astral.sh/uv/install.ps1 | iex"'
        )
        try:
            log.info("  → Using official Windows installer (PowerShell) …")
            subprocess.run(ps_cmd, shell=True, check=True)
            installed = True
        except subprocess.CalledProcessError:
            _warn("PowerShell installer failed — falling back to pip …")
    else:
        try:
            log.info("  → Using official Unix installer (curl) …")
            subprocess.run(
                "curl -LsSf https://astral.sh/uv/install.sh | sh",
                shell=True, check=True,
            )
            installed = True
        except subprocess.CalledProcessError:
            _warn("curl installer failed — falling back to pip …")

    if not installed:
        log.info("  → Installing uv via pip …")
        _run([sys.executable, "-m", "pip", "install", "--quiet", "uv"])

    # Refresh PATH so the newly installed binary is discoverable
    extra_paths: list[str] = []
    if IS_WINDOWS:
        extra_paths = [
            str(Path.home() / ".cargo"   / "bin"),
            str(Path.home() / ".local"   / "bin"),
            str(Path(os.environ.get("APPDATA",      "")) / "uv" / "bin"),
            str(Path(os.environ.get("USERPROFILE",  "")) / ".local" / "bin"),
        ]
    else:
        extra_paths = [
            str(Path.home() / ".local" / "bin"),
            str(Path.home() / ".cargo" / "bin"),
        ]

    os.environ["PATH"] = os.pathsep.join(extra_paths) + os.pathsep + os.environ["PATH"]

    uv = _uv_executable()
    if not uv:
        # Last resort — check if importable as a module
        probe = subprocess.run(
            [sys.executable, "-m", "uv", "--version"],
            capture_output=True, text=True, check=False,
        )
        if probe.returncode == 0:
            # Return a pseudo-command that callers can split and pass to subprocess
            return f"{sys.executable} -m uv"
        _err("Could not locate uv after installation.")
        _err("Please install it manually: https://docs.astral.sh/uv/getting-started/installation/")
        sys.exit(1)

    result  = _run([uv, "--version"], capture=True, check=False)
    version = result.stdout.strip()
    _ok(f"uv installed  ({version})")
    return uv


# ──────────────────────────────────────────────────────────────────────────────
# Step 2 — Initialise project with uv
# ──────────────────────────────────────────────────────────────────────────────

def init_project(uv: str) -> None:
    _banner("Step 2 of 5 — Project initialisation")

    if UV_LOCK.exists():
        _ok("Project already initialised (uv.lock present)")
        return

    if not PYPROJECT.exists():
        log.info("  pyproject.toml not found — running `uv init` …")
        _run([uv, "init", "--no-workspace"])
        _ok("Project initialised with uv")
    else:
        # pyproject.toml exists but no lock file yet — generate it
        log.info("  pyproject.toml found — generating uv.lock …")
        _run([uv, "lock"])
        _ok("Lock file (uv.lock) generated")


# ──────────────────────────────────────────────────────────────────────────────
# Step 3 — Virtual environment
# ──────────────────────────────────────────────────────────────────────────────

def create_venv(uv: str) -> None:
    _banner("Step 3 of 5 — Virtual environment")

    python_bin = (
        VENV_DIR / "Scripts" / "python.exe"
        if IS_WINDOWS
        else VENV_DIR / "bin" / "python"
    )

    if VENV_DIR.exists() and python_bin.exists():
        _ok(f"Virtual environment already exists  ({VENV_DIR})")
        return

    log.info(f"  Creating virtual environment at {VENV_DIR} …")
    _run([uv, "venv", str(VENV_DIR)])
    _ok("Virtual environment created")


# ──────────────────────────────────────────────────────────────────────────────
# Step 4 — Sync / install dependencies
# ──────────────────────────────────────────────────────────────────────────────

def sync_dependencies(uv: str) -> None:
    _banner("Step 4 of 5 — Dependency installation")

    if PYPROJECT.exists():
        log.info("  Running `uv sync`  (resolves & installs all project dependencies) …")
        log.info("  This may take several minutes on first run — please be patient.")
        try:
            _run([uv, "sync"])
            _ok("Dependencies synced via `uv sync`")
            return
        except subprocess.CalledProcessError:
            _warn("`uv sync` failed — falling back to requirements.txt …")

    if REQUIREMENTS.exists():
        log.info("  Installing from requirements.txt …")
        _run([uv, "pip", "install", "-r", str(REQUIREMENTS)])
        _ok("Dependencies installed from requirements.txt")
        return

    _warn("Neither pyproject.toml nor requirements.txt found — skipping.")


# ──────────────────────────────────────────────────────────────────────────────
# Step 5 — Kaggle dataset (optional)
# ──────────────────────────────────────────────────────────────────────────────

def _missing_dataset_dirs() -> list[str]:
    """Return every expected dataset directory that does not yet exist on disk."""
    return [
        rel for rel in EXPECTED_DATASET_DIRS
        if not (PROJECT_ROOT / rel).exists()
    ]


def _setup_kaggle_credentials() -> bool:
    """
    Interactive wizard to create ~/.kaggle/kaggle.json.
    Returns True when valid credentials are in place.
    """
    if KAGGLE_JSON.exists():
        _ok(f"Kaggle credentials detected at {KAGGLE_JSON}")
        return True

    print()
    print("  ┌──────────────────────────────────────────────────────┐")
    print("  │               Kaggle API Token Setup                  │")
    print("  └──────────────────────────────────────────────────────┘")
    print(textwrap.dedent(f"""
    To download the dataset you need a Kaggle API token.

    How to get your token:
      1. Log in at  https://www.kaggle.com
      2. Click your profile picture  →  Settings
      3. Scroll to the "API" section  →  click "Create New Token"
      4. A file called  kaggle.json  will be downloaded automatically.

    You can either:
      (A) Enter your Kaggle username and API key here (they will be saved
          automatically to {KAGGLE_JSON})
      (B) Manually place kaggle.json at the path above and re-run setup.
    """))

    choice = _ask("  Enter choice (A / B) [A]: ", default="A").upper()

    if choice == "B":
        _warn(f"Please place kaggle.json at {KAGGLE_JSON} and re-run setup.")
        return False

    # ── Option A: collect credentials interactively ──
    print()
    username = _ask("  Kaggle username : ")
    api_key  = _ask("  Kaggle API key  : ")

    if not username or not api_key:
        _err("Username and API key cannot be empty.")
        return False

    KAGGLE_JSON.parent.mkdir(parents=True, exist_ok=True)
    KAGGLE_JSON.write_text(
        json.dumps({"username": username, "key": api_key}, indent=2),
        encoding="utf-8",
    )

    # Restrict file permissions on non-Windows (Kaggle library requires 600)
    if not IS_WINDOWS:
        KAGGLE_JSON.chmod(stat.S_IRUSR | stat.S_IWUSR)

    _ok(f"Kaggle credentials saved to {KAGGLE_JSON}")
    return True


def _run_download_script() -> None:
    """Run scripts/download_kaggle_dataset.py inside the project venv."""
    venv_python = (
        VENV_DIR / "Scripts" / "python.exe"
        if IS_WINDOWS
        else VENV_DIR / "bin" / "python"
    )
    python_exe     = str(venv_python) if venv_python.exists() else sys.executable
    download_script = PROJECT_ROOT / "scripts" / "download_kaggle_dataset.py"

    if not download_script.exists():
        _err(f"Download script not found: {download_script}")
        _err("Cannot proceed with dataset download.")
        return

    log.info("  Launching dataset download script …")
    log.info("  Large download — this can take a while depending on your connection.")
    try:
        _run([python_exe, str(download_script)])
        _ok("Dataset download complete")
    except subprocess.CalledProcessError as exc:
        _err(f"Download script exited with code {exc.returncode}.")
        _err("Check logs/kaggle_download.log for details.")


def handle_kaggle_dataset() -> None:
    _banner("Step 5 of 5 — Kaggle dataset  (optional)")

    missing = _missing_dataset_dirs()

    if not missing:
        _ok("All expected dataset directories are present — nothing to download.")
        return

    print()
    _warn(
        f"{len(missing)} of {len(EXPECTED_DATASET_DIRS)} "
        "expected dataset directories are missing:"
    )
    for d in missing:
        print(f"      • {d}")
    print()

    if not _confirm("  Download the Kaggle dataset now?", default=True):
        log.info("  Dataset download skipped.")
        log.info(
            "  You can download it later with:\n"
            "      python scripts/download_kaggle_dataset.py"
        )
        return

    if not _setup_kaggle_credentials():
        _warn("Kaggle credentials not configured — skipping dataset download.")
        _warn(
            f"Place kaggle.json at {KAGGLE_JSON} and re-run:\n"
            "      python setup.py"
        )
        return

    _run_download_script()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print()
    print("=" * 62)
    print("   AgriTwin-GH  —  Project Setup")
    print("=" * 62)
    print()

    uv = install_uv()       # Step 1
    init_project(uv)        # Step 2
    create_venv(uv)         # Step 3
    sync_dependencies(uv)   # Step 4
    handle_kaggle_dataset() # Step 5

    print()
    log.info("=" * 62)
    log.info("  Setup complete!")
    log.info("")
    log.info("  Activate the virtual environment:")
    if IS_WINDOWS:
        log.info("      .venv\\Scripts\\activate")
    else:
        log.info("      source .venv/bin/activate")
    log.info("")
    log.info("  Or run scripts directly with uv:")
    log.info("      uv run python main.py")
    log.info("=" * 62)
    print()


if __name__ == "__main__":
    main()
