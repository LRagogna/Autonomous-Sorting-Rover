"""Modular local-browser control center for the rover training workflow.

The GUI is a small standard-library HTTP server (no web framework, no new
dependencies). It serves a static single-page app from ``gui/web/`` and exposes
a ``/api/*`` surface whose handlers live in ``gui/api/``. All dataset and model
logic is delegated to the ``ml/`` package, so this layer stays thin.
"""
