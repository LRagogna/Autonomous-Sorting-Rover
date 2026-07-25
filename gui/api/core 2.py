"""Core API: global state + job polling."""

from __future__ import annotations

from gui import jobs
from gui import state as state_module
from gui.server import route


@route("GET", "/api/state")
def get_state(req):
    return state_module.build_state()


@route("GET", "/api/job")
def get_job(req):
    return {
        "current": jobs.public_job(jobs.current_job()),
        "history": [jobs.public_job(job) for job in jobs.job_history()],
    }
