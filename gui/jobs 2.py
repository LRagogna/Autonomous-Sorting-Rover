"""Background job runner for long workflow steps (process, train).

Only one workflow job runs at a time. Output is captured line by line so the
browser can poll it and show a live log + progress. This is lifted from the
original single-file GUI and kept deliberately simple.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
import uuid

from ml import dataset_utils as du


_job_lock = threading.Lock()
_current_job: dict | None = None
_job_history: list[dict] = []


def current_job() -> dict | None:
    return _current_job


def job_history() -> list[dict]:
    return list(_job_history)


def public_job(job: dict | None) -> dict | None:
    """Return a JSON-safe copy of a job (log trimmed to the recent tail)."""
    if job is None:
        return None
    return {
        "id": job["id"],
        "kind": job["kind"],
        "status": job["status"],
        "startedAt": job["started_at"],
        "finishedAt": job.get("finished_at"),
        "returnCode": job.get("return_code"),
        "summary": job.get("summary"),
        "log": job["log"][-400:],
    }


def _run(job: dict, command: list[str], env: dict[str, str] | None) -> None:
    try:
        job["log"].append(f"$ {' '.join(command)}")
        merged = os.environ.copy()
        if env:
            merged.update(env)
        # Make child Python output unbuffered so the live log updates promptly.
        merged.setdefault("PYTHONUNBUFFERED", "1")
        process = subprocess.Popen(
            command, cwd=du.PROJECT_ROOT, env=merged,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            line = line.rstrip()
            job["log"].append(line)
            _maybe_capture_summary(job, line)
        code = process.wait()
        job["return_code"] = code
        job["status"] = "succeeded" if code == 0 else "failed"
        job["log"].append(f"Finished with exit code {code}.")
    except Exception as error:  # noqa: BLE001 - surface unexpected failures in the log
        job["status"] = "failed"
        job["return_code"] = 1
        job["log"].append(f"Error: {error}")
    finally:
        job["finished_at"] = time.time()


def _maybe_capture_summary(job: dict, line: str) -> None:
    """Pull structured summary lines the ml scripts print into job['summary']."""
    if line.startswith("PROCESS_SUMMARY "):
        summary: dict[str, int] = {}
        for token in line.split()[1:]:
            if "=" in token:
                key, value = token.split("=", 1)
                try:
                    summary[key] = int(value)
                except ValueError:
                    summary[key] = value
        job["summary"] = {"type": "process", **summary}
    elif line.startswith("TRAIN_RESULT "):
        summary = {}
        for token in line.split()[1:]:
            if "=" in token:
                key, value = token.split("=", 1)
                summary[key] = value
        job["summary"] = {"type": "train", **summary}


def start_job(kind: str, command: list[str], env: dict[str, str] | None = None) -> dict:
    """Start a background job; raises if one is already running."""
    global _current_job
    with _job_lock:
        if _current_job is not None and _current_job["status"] == "running":
            raise RuntimeError(f"{_current_job['kind']} is already running.")
        job = {
            "id": uuid.uuid4().hex[:10],
            "kind": kind,
            "status": "running",
            "started_at": time.time(),
            "log": [],
            "summary": None,
        }
        _current_job = job
        _job_history.insert(0, job)
        del _job_history[8:]
    threading.Thread(target=_run, args=(job, command, env), daemon=True).start()
    return job
