"""
AgriTwin-GH — Interactive Project Setup Script
================================================
Handles:
  1. Install uv package manager (if missing)
  2. Initialise project with uv (if not already)
  3. Create / verify virtual environment
  4. Sync dependencies with uv
  5. Create .env from .env.example (with optional credential prompts)
  6. Create config/settings.local.yaml from example
  7. Create required project directories
  8. Install frontend Node.js dependencies (npm install)
  9. (Optional) Download Kaggle dataset, with interactive token setup

At the end a formatted manual-steps guide is printed for anything
that cannot be automated (PostgreSQL, MinIO, API keys, etc.).

Run:
    python setup.py
"""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import secrets
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

# Frontend directory
FRONTEND_DIR = PROJECT_ROOT / "src" / "agritwin_gh" / "frontend"

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
    _banner("Step 1 of 9 — uv package manager")

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
    _banner("Step 2 of 9 — Project initialisation")

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
    _banner("Step 3 of 9 — Virtual environment")

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
    _banner("Step 4 of 9 — Dependency installation")

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


# ──────────────────────────────────────────────────────────────────────────────
# Step 5 — Environment file (.env)
# ──────────────────────────────────────────────────────────────────────────────

def _extract_env_value(content: str, key: str) -> str:
    """Return the current value for *key* in .env file content, or ''."""
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith(f"{key}="):
            return stripped[len(f"{key}="):]
    return ""


def _set_env_value(content: str, key: str, value: str) -> str:
    """Replace *key*'s value in-place; append the line if the key is absent."""
    pattern = rf"^{re.escape(key)}=.*$"
    replacement = f"{key}={value}"
    new_content, count = re.subn(pattern, replacement, content, flags=re.MULTILINE)
    if count == 0:
        new_content = content.rstrip("\n") + f"\n{key}={value}\n"
    return new_content


def create_env_file() -> None:
    _banner("Step 5 of 9 — Environment file (.env)")

    env_file    = PROJECT_ROOT / ".env"
    env_example = PROJECT_ROOT / ".env.example"

    if env_file.exists():
        _ok(".env already exists — skipping copy step")
    else:
        if not env_example.exists():
            _warn(".env.example not found — cannot create .env automatically")
            _warn("Create the file manually before running the application.")
            return
        shutil.copy(env_example, env_file)
        _ok(".env created from .env.example")

    # ── Frontend .env.local ────────────────────────────────────────────────
    fe_env = FRONTEND_DIR / ".env.local"
    if not fe_env.exists():
        if FRONTEND_DIR.exists():
            fe_env.write_text("VITE_API_BASE_URL=http://localhost:8000\n", encoding="utf-8")
            _ok("src/agritwin_gh/frontend/.env.local created  (VITE_API_BASE_URL=http://localhost:8000)")
        else:
            _warn("Frontend directory not found — skipping .env.local creation")
    else:
        _ok("src/agritwin_gh/frontend/.env.local already exists")

    # ── Optional interactive credential fill-in ────────────────────────────
    print()
    print("  The .env file needs real credentials for PostgreSQL, MinIO, etc.")
    if not _confirm("  Configure key credentials interactively now?", default=False):
        log.info("  Skipping credential setup — edit .env manually before running the app.")
        return

    env_content = env_file.read_text(encoding="utf-8")

    fields: list[tuple[str, str, str]] = [
        ("DB_USER",     "PostgreSQL username",        "agritwin_user"),
        ("DB_PASSWORD", "PostgreSQL password",        ""),
        ("DB_NAME",     "PostgreSQL database name",   "agritwin_db"),
        ("DB_HOST",     "PostgreSQL host",            "localhost"),
        ("DB_PORT",     "PostgreSQL port",            "5432"),
        ("SECRET_KEY",  "App secret key (blank → auto-generate)", ""),
    ]

    for key, label, default in fields:
        current = _extract_env_value(env_content, key)
        display_default = default or current or "required"
        value = _ask(f"    {label} [{display_default}]: ", default=default or current)

        if key == "SECRET_KEY" and not value:
            value = secrets.token_hex(32)
            log.info("    → AUTO-GENERATED SECRET_KEY")

        if value:
            env_content = _set_env_value(env_content, key, value)

    env_file.write_text(env_content, encoding="utf-8")
    _ok(".env credentials updated")


# ──────────────────────────────────────────────────────────────────────────────
# Step 6 — Local configuration file (settings.local.yaml)
# ──────────────────────────────────────────────────────────────────────────────

def create_local_config() -> None:
    _banner("Step 6 of 9 — Local configuration  (config/settings.local.yaml)")

    local_cfg   = PROJECT_ROOT / "config" / "settings.local.yaml"
    example_cfg = PROJECT_ROOT / "config" / "settings.local.example.yaml"

    if local_cfg.exists():
        _ok("config/settings.local.yaml already exists — skipping")
        return

    if not example_cfg.exists():
        _warn("config/settings.local.example.yaml not found — skipping")
        return

    shutil.copy(example_cfg, local_cfg)
    _ok("config/settings.local.yaml created from settings.local.example.yaml")
    log.info("  Edit config/settings.local.yaml to customise your local environment.")


# ──────────────────────────────────────────────────────────────────────────────
# Step 7 — Required project directories
# ──────────────────────────────────────────────────────────────────────────────

REQUIRED_DIRS: list[str] = [
    "data/raw",
    "data/external",
    "data/processed",
    "logs",
    "logs/dt_runs",
    "logs/realtime",
]


def create_required_dirs() -> None:
    _banner("Step 7 of 9 — Required project directories")

    created: list[str] = []
    for rel in REQUIRED_DIRS:
        d = PROJECT_ROOT / rel
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)
            created.append(rel)

    if created:
        for d in created:
            _ok(f"Created  {d}/")
    else:
        _ok("All required directories already exist")


# ──────────────────────────────────────────────────────────────────────────────
# Step 8 — Frontend Node.js dependencies
# ──────────────────────────────────────────────────────────────────────────────

def install_frontend_deps() -> None:
    _banner("Step 8 of 9 — Frontend dependencies  (npm install)")

    if not FRONTEND_DIR.exists():
        _warn(f"Frontend directory not found: {FRONTEND_DIR}")
        _warn("Skipping npm install — run it manually inside the frontend folder.")
        return

    node = shutil.which("node")
    npm  = shutil.which("npm")

    if not node or not npm:
        _warn("Node.js / npm not found on PATH.")
        _warn("Install Node.js (LTS) from https://nodejs.org and re-run setup,")
        _warn("or manually run:  cd src/agritwin_gh/frontend && npm install")
        return

    node_ver = subprocess.run(
        [node, "--version"], capture_output=True, text=True, check=False
    ).stdout.strip()
    npm_ver  = subprocess.run(
        [npm,  "--version"], capture_output=True, text=True, check=False
    ).stdout.strip()
    _ok(f"Node {node_ver}  /  npm {npm_ver}")

    node_modules = FRONTEND_DIR / "node_modules"
    if node_modules.exists():
        _ok("node_modules already present — skipping npm install")
        return

    log.info("  Running npm install in frontend directory …")
    try:
        _run([npm, "install"], cwd=FRONTEND_DIR)
        _ok("Frontend npm packages installed")
    except subprocess.CalledProcessError as exc:
        _err(f"npm install failed (exit {exc.returncode}).")
        _err(f"Run manually:  cd {FRONTEND_DIR}  &&  npm install")


# ──────────────────────────────────────────────────────────────────────────────
# Manual setup guide  (printed at the end)
# ──────────────────────────────────────────────────────────────────────────────

def print_manual_setup_guide() -> None:
    """
    Print a clean, structured guide for every step that cannot be
    automated and must be performed by the developer manually.
    """
    W = 70
    sep   = "─" * W
    thick = "═" * W

    def header(text: str) -> None:
        print(f"\n  {thick}")
        print(f"  {'MANUAL SETUP REQUIRED':^{W}}")
        print(f"  {thick}")
        print(f"  {text}")
        print(f"  {thick}\n")

    def section(num: int, title: str, purpose: str, steps: list[str]) -> None:
        print(f"  {sep}")
        print(f"  [{num}]  {title}")
        print(f"  {sep}")
        print(f"  PURPOSE : {purpose}")
        print()
        for step in steps:
            # Wrap long lines neatly
            lines = textwrap.wrap(step, width=W - 10)
            print(f"    • {lines[0]}")
            for cont in lines[1:]:
                print(f"      {cont}")
        print()

    print()
    header("The items below cannot be installed / configured automatically.")

    section(
        1,
        "PostgreSQL (+ TimescaleDB extension)",
        "Persistent time-series storage for sensor readings, DT states, "
        "monthly snapshots, and realtime stream data.",
        [
            "Install PostgreSQL 15+ from https://www.postgresql.org/download/",
            "Install TimescaleDB extension (optional but recommended): "
            "https://docs.timescale.com/self-hosted/latest/install/",
            "Create database and user:",
            "    psql -U postgres",
            "    CREATE USER agritwin_user WITH PASSWORD 'your_password';",
            "    CREATE DATABASE agritwin_db OWNER agritwin_user;",
            "    \\q",
            "Apply the database schema files in order:",
            "    psql -U agritwin_user -d agritwin_db "
            "-f database/schema/timeseries_data.sql",
            "    psql -U agritwin_user -d agritwin_db "
            "-f database/schema/image_metadata.sql",
            "    psql -U agritwin_user -d agritwin_db "
            "-f database/schema/monthly_snapshots.sql",
            "Set DB_USER, DB_PASSWORD, DB_NAME, DB_HOST, DB_PORT in .env",
        ],
    )

    section(
        2,
        "MinIO Object Storage",
        "Stores tomato disease / growth-stage images used by the ML pipeline "
        "and accessible via the FastAPI image endpoint.",
        [
            "Option A — Docker (recommended for local dev):",
            "    docker run -d --name minio -p 9000:9000 -p 9001:9001 \\",
            "      -e MINIO_ROOT_USER=minioadmin \\",
            "      -e MINIO_ROOT_PASSWORD=minioadmin123 \\",
            "      minio/minio server /data --console-address ':9001'",
            "Option B — Native binary: https://min.io/docs/minio/linux/index.html",
            "After starting MinIO, open the console at http://localhost:9001 "
            "and create the bucket  agritwin-images  (or let the upload script "
            "create it automatically).",
            "Set MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, "
            "MINIO_SECURE in .env",
            "Then run:  python scripts/upload_images_to_minio.py",
            "Verify :   python scripts/verify_image_upload.py",
        ],
    )

    section(
        3,
        "Node.js / npm  (if not already installed)",
        "Required to run the React + Vite frontend dashboard.",
        [
            "Download Node.js LTS from https://nodejs.org/",
            "After installing, run the following to set up the frontend:",
            "    cd src/agritwin_gh/frontend",
            "    npm install",
            "    npm run dev          # development server on http://localhost:5173",
            "    npm run build        # production build",
            "The Vite dev server is launched automatically by  python main.py  "
            "unless AGRITWIN_NO_FRONTEND=1 is set in .env",
        ],
    )

    section(
        4,
        "Kaggle API credentials",
        "Required to download the tomato greenhouse dataset "
        "(disease images, growth-stage images, weather CSV files).",
        [
            "1. Log in at https://www.kaggle.com",
            "2. Profile picture → Settings → API → Create New Token",
            "3. A file called kaggle.json is downloaded automatically.",
            "4. Place it at ~/.kaggle/kaggle.json "
            "(Linux/macOS: chmod 600 ~/.kaggle/kaggle.json)",
            "5. Then re-run:  python setup.py   (or run the download script "
            "directly:  python scripts/download_kaggle_dataset.py)",
        ],
    )

    section(
        5,
        "Weather / Sensor API keys  (optional)",
        "Needed if you want live weather data ingestion or real sensor feeds "
        "instead of the synthetic / CSV-based simulation.",
        [
            "Obtain a free API key from your chosen weather provider "
            "(e.g. Visual Crossing, OpenWeatherMap, Tomorrow.io).",
            "Set WEATHER_API_KEY=<your_key> in .env",
            "For real sensor hardware, set SENSOR_API_KEY=<your_key> in .env",
        ],
    )

    section(
        6,
        "Application SECRET_KEY",
        "Used by FastAPI for cryptographic signing (sessions, tokens). "
        "Must be a long, random string — never commit it to version control.",
        [
            "Generate a secure key with Python:",
            "    python -c \"import secrets; print(secrets.token_hex(32))\"",
            "Set SECRET_KEY=<generated_value> in .env",
            "Note: setup.py can auto-generate this for you if you run it with "
            "the interactive credential prompt (answer Y when asked).",
        ],
    )

    section(
        7,
        "CUDA / GPU drivers  (optional — for model training)",
        "PyTorch, TensorFlow, and Keras model training runs on CPU by default. "
        "A CUDA-capable GPU significantly speeds up training.",
        [
            "Install NVIDIA drivers: https://www.nvidia.com/Download/index.aspx",
            "Install CUDA Toolkit (12.x recommended for PyTorch 2.x): "
            "https://developer.nvidia.com/cuda-downloads",
            "Install cuDNN: https://developer.nvidia.com/cudnn",
            "Verify GPU availability inside the venv:",
            "    python -c \"import torch; print(torch.cuda.is_available())\"",
            "    python -c \"import tensorflow as tf; "
            "print(tf.config.list_physical_devices('GPU'))\"",
        ],
    )

    section(
        8,
        "Load time-series data into PostgreSQL  (after schema is applied)",
        "Populates the database with the historical CSV data so the "
        "DT loop, MPC, and dashboard have real data to work with.",
        [
            "Ensure PostgreSQL is running and .env credentials are correct.",
            "Run the loader script:",
            "    python scripts/load_timeseries_to_postgres.py",
            "Seed monthly mock snapshots (optional):",
            "    python scripts/seed_monthly_mock.py",
            "Verify the image storage system:",
            "    python scripts/verify_setup.py",
        ],
    )

    print(f"  {thick}")
    print(f"  {'Once all of the above are done, start the application with:':{W}}")
    print(f"  {'':4}uv run python main.py")
    print(f"  {'or run backend only:':4}")
    print(f"  {'':4}uvicorn main:app --reload --host 0.0.0.0 --port 8000")
    print(f"  {thick}\n")


def handle_kaggle_dataset() -> None:
    _banner("Step 9 of 9 — Kaggle dataset  (optional)")

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

    uv = install_uv()           # Step 1
    init_project(uv)            # Step 2
    create_venv(uv)             # Step 3
    sync_dependencies(uv)       # Step 4
    create_env_file()           # Step 5
    create_local_config()       # Step 6
    create_required_dirs()      # Step 7
    install_frontend_deps()     # Step 8
    handle_kaggle_dataset()     # Step 9

    print()
    log.info("=" * 62)
    log.info("  Automated setup complete!")
    log.info("")
    log.info("  Activate the virtual environment:")
    if IS_WINDOWS:
        log.info("      .venv\\Scripts\\activate")
    else:
        log.info("      source .venv/bin/activate")
    log.info("")
    log.info("  Or run directly with uv:")
    log.info("      uv run python main.py")
    log.info("=" * 62)
    print()

    print_manual_setup_guide()


if __name__ == "__main__":
    main()
