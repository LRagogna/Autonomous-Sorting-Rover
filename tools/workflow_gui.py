#!/usr/bin/env python3
"""Compatibility shim.

The training GUI was rebuilt as a modular control center under ``gui/``. This
file used to hold the whole single-file app; it now just launches the new
entry point so old commands and muscle memory keep working:

    python tools/workflow_gui.py      ->  python gui/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from gui.app import main  # noqa: E402


if __name__ == "__main__":
    print("Note: tools/workflow_gui.py now launches the new control center (gui/app.py).")
    main()
