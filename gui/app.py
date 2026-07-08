#!/usr/bin/env python3
"""Entry point for the rover training control center.

Run with:

    python gui/app.py

Serves the modular browser GUI locally and opens it. Uses only the Python
standard library for the server; all dataset/model logic lives in ``ml/``.
"""

from __future__ import annotations

import argparse
import sys
import threading
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml import dataset_utils as du  # noqa: E402
from gui import server  # noqa: E402
import gui.api  # noqa: E402,F401 - importing registers all /api routes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the rover training control center GUI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    du.ensure_core_dirs()
    if not du.DATASET_YAML.exists():
        du.write_dataset_yaml()

    httpd, port = server.build_server(args.host, args.port)
    url = f"http://{args.host}:{port}"
    print(f"Rover control center running at {url}")
    print("Press Ctrl+C to stop.")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
