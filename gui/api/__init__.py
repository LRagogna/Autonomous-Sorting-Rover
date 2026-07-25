"""API route modules. Importing each one registers its @route handlers.

Phase 1 tabs: core, upload, dataset, review, train.
Phase 2 tabs (detect, retrain, deploy, maintenance) are imported when present.
"""

from gui.api import core, upload, dataset, review, train  # noqa: F401

# Phase 2 modules are optional so Phase 1 works before they are written.
for _name in ("detect", "retrain", "deploy", "maintenance"):
    try:
        __import__(f"gui.api.{_name}")
    except ModuleNotFoundError:
        pass
