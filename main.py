"""
AgriTwin-GH — FastAPI application entry point.

Start the server
----------------
Development (auto-reload)::

    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Production::

    uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1

Or run this file directly (starts both backend + frontend dev server)::

    python main.py

Environment variables
---------------------
AGRITWIN_BACKGROUND_LOOP
    Set to ``0`` to disable the background DT simulation loop.
    Default: ``1`` (enabled — advances one step every 5 minutes).

AGRITWIN_NO_FRONTEND
    Set to ``1`` to skip launching the Vite dev server (e.g. in CI or
    when running ``uvicorn main:app`` directly).
    Default: ``0`` (Vite is launched alongside the backend).

AGRITWIN_MONTHLY_DB
    Set to ``1`` to enable automatic monthly snapshot persistence.
    At each calendar-month boundary the DT loop flushes aggregated sensor,
    actuator, MPC, and resource data into the ``monthly_snapshots`` table
    and creates / reuses a ``crop_cycles`` row for the current run.
    Requires the database schema in ``database/schema/monthly_snapshots.sql``
    to be applied first (see ``docs/MONTHLY_SNAPSHOT_REFERENCE.md``).
    Default: ``0`` (disabled — the DT loop runs normally without DB writes).
"""

import sys
import os
import subprocess
import threading
import signal
from pathlib import Path

# Add src/ to Python path so imports resolve correctly
sys.path.insert(0, str(Path(__file__).parent / "src"))

import uvicorn

from agritwin_gh.api.app import create_app

# Module-level ``app`` so ``uvicorn main:app`` works without calling main().
app = create_app()

# Absolute path to the Vite frontend directory
_FRONTEND_DIR = Path(__file__).parent / "src" / "agritwin_gh" / "frontend"


def _start_vite() -> subprocess.Popen:
    """Launch ``npm run dev`` in the frontend directory as a subprocess.

    stdout/stderr are forwarded to the current process so Vite output appears
    inline with the FastAPI logs.

    Returns the ``Popen`` handle so the caller can terminate it on shutdown.
    """
    npm = "npm.cmd" if sys.platform == "win32" else "npm"
    proc = subprocess.Popen(
        [npm, "run", "dev"],
        cwd=str(_FRONTEND_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    def _pipe_output() -> None:
        for line in proc.stdout:
            print(f"[vite] {line}", end="", flush=True)

    threading.Thread(target=_pipe_output, daemon=True).start()
    return proc


def _print_log_paths() -> None:
    """Print paths to all log files that will be written for this run."""
    from datetime import date
    import os

    logs_dir = Path(__file__).parent / "logs"
    today = date.today().strftime("%Y%m%d")

    log_files = {
        "DT loop trace": logs_dir / f"dt_loop_{today}.log",
        "Uvicorn access": logs_dir / "access.log",
        "Uvicorn error":  logs_dir / "error.log",
    }

    print()
    print("=" * 60)
    print("  Log files for this run")
    print("=" * 60)
    for name, path in log_files.items():
        abs_path = path.resolve()
        # VS Code clickable URI — file:///... opens the file in the editor
        print(f"  {name:<20} {abs_path}")
        print(f"    Open in editor: {abs_path.as_uri()}")
    print("=" * 60)
    print()


def main() -> None:
    """Start the Vite dev server and then the Uvicorn server.

    Ctrl-C shuts down both processes cleanly.
    """
    no_frontend = os.environ.get("AGRITWIN_NO_FRONTEND", "0").strip() == "1"

    _print_log_paths()

    vite_proc: subprocess.Popen | None = None

    if not no_frontend:
        print("[main] Starting Vite dev server on http://localhost:5173 …")
        vite_proc = _start_vite()

        # Ensure Vite is killed if the backend exits or Ctrl-C is pressed
        def _shutdown(signum, frame):
            if vite_proc and vite_proc.poll() is None:
                vite_proc.terminate()
            sys.exit(0)

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

    print("[main] Starting FastAPI backend on http://localhost:8000 …")
    print("[main] Unity WebGL greenhouse available at http://localhost:8000/greenhouse-3d/")
    try:
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="info",
        )
    finally:
        if vite_proc and vite_proc.poll() is None:
            print("[main] Shutting down Vite …")
            vite_proc.terminate()


if __name__ == "__main__":
    main()
