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
    Set to ``1`` to enable the background DT simulation loop.
    Default: ``0`` (disabled — steps driven manually or by tests).

AGRITWIN_NO_FRONTEND
    Set to ``1`` to skip launching the Vite dev server (e.g. in CI or
    when running ``uvicorn main:app`` directly).
    Default: ``0`` (Vite is launched alongside the backend).
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


def main() -> None:
    """Start the Vite dev server and then the Uvicorn server.

    Ctrl-C shuts down both processes cleanly.
    """
    no_frontend = os.environ.get("AGRITWIN_NO_FRONTEND", "0").strip() == "1"

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
