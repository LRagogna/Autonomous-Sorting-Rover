"""Standard-library HTTP server: routing, static files, and request helpers.

The ``/api/*`` surface is registered by the modules in ``gui/api/`` using the
``@route`` decorator. Everything else is served as a static file from
``gui/web/``. Handlers return a Python object (sent as JSON), a ``(status, obj)``
tuple, or ``None`` when they have already written the response themselves (used
for streaming images and video).
"""

from __future__ import annotations

import json
import mimetypes
import re
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ml import dataset_utils as du


WEB_DIR = Path(__file__).resolve().parent / "web"


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
class Router:
    """Tiny path router with single-segment ``{name}`` captures."""

    def __init__(self) -> None:
        self._routes: list[tuple[str, list[str], object]] = []

    def add(self, method: str, pattern: str, handler) -> None:
        self._routes.append((method.upper(), pattern.strip("/").split("/"), handler))

    def match(self, method: str, path: str):
        segments = path.strip("/").split("/")
        for route_method, pattern, handler in self._routes:
            if route_method != method or len(pattern) != len(segments):
                continue
            params: dict[str, str] = {}
            ok = True
            for pat, seg in zip(pattern, segments):
                if pat.startswith("{") and pat.endswith("}"):
                    params[pat[1:-1]] = urllib.parse.unquote(seg)
                elif pat != seg:
                    ok = False
                    break
            if ok:
                return handler, params
        return None, None


router = Router()


def route(method: str, pattern: str):
    """Decorator that registers a handler on the shared router."""
    def decorator(fn):
        router.add(method, pattern, fn)
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Request wrapper passed to every handler
# ---------------------------------------------------------------------------
class Request:
    def __init__(self, handler: BaseHTTPRequestHandler, method: str, path: str,
                 query: dict, params: dict) -> None:
        self.handler = handler
        self.method = method
        self.path = path
        self.query = query
        self.params = params
        self._body: bytes | None = None

    def q(self, key: str, default: str = "") -> str:
        return self.query.get(key, [default])[0]

    def body(self) -> bytes:
        if self._body is None:
            length = int(self.handler.headers.get("Content-Length", "0"))
            self._body = self.handler.rfile.read(length)
        return self._body

    def json(self) -> dict:
        raw = self.body() or b"{}"
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}

    def multipart(self):
        return parse_multipart(self.body(), self.handler.headers.get("Content-Type", ""))


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------
def _write(handler: BaseHTTPRequestHandler, status: int, content_type: str, body: bytes,
           extra_headers: dict | None = None) -> None:
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    for key, value in (extra_headers or {}).items():
        handler.send_header(key, value)
    handler.end_headers()
    handler.wfile.write(body)


def send_json(handler, payload, status: int = 200) -> None:
    _write(handler, status, "application/json", json.dumps(payload).encode("utf-8"))


def send_image(handler, path: Path) -> None:
    """Serve one image file (used for review frames and previews)."""
    if not path.exists() or path.suffix.lower() not in du.IMAGE_EXTENSIONS:
        send_json(handler, {"error": "Not found"}, 404)
        return
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    _write(handler, 200, content_type, path.read_bytes(),
           {"Cache-Control": "no-store"})


def serve_range(handler, path: Path) -> None:
    """Stream a (possibly large) file honoring HTTP Range requests."""
    if not path.exists() or not path.is_file():
        send_json(handler, {"error": "Not found"}, 404)
        return
    file_size = path.stat().st_size
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    start, end, status = 0, file_size - 1, 200
    range_header = handler.headers.get("Range")
    if range_header:
        match = re.match(r"bytes=(\d+)-(\d*)", range_header.strip())
        if match:
            start = int(match.group(1))
            end = int(match.group(2)) if match.group(2) else file_size - 1
            end = min(end, file_size - 1)
            if start > end or start >= file_size:
                handler.send_response(416)
                handler.send_header("Content-Range", f"bytes */{file_size}")
                handler.end_headers()
                return
            status = 206
    length = end - start + 1
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Accept-Ranges", "bytes")
    if status == 206:
        handler.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
    handler.send_header("Content-Length", str(length))
    handler.end_headers()
    try:
        with path.open("rb") as source:
            source.seek(start)
            remaining = length
            while remaining > 0:
                chunk = source.read(min(65536, remaining))
                if not chunk:
                    break
                handler.wfile.write(chunk)
                remaining -= len(chunk)
    except (BrokenPipeError, ConnectionResetError):
        pass  # browser closed the connection (common when scrubbing video)


def serve_static(handler, path: str) -> None:
    """Serve a file from gui/web/, defaulting '/' to index.html."""
    relative = "index.html" if path in ("", "/") else path.lstrip("/")
    target = (WEB_DIR / relative).resolve()
    if WEB_DIR not in target.parents and target != WEB_DIR or not target.is_file():
        send_json(handler, {"error": "Not found"}, 404)
        return
    content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
    charset = "; charset=utf-8" if content_type.startswith(("text/", "application/javascript")) else ""
    _write(handler, 200, content_type + charset, target.read_bytes())


# ---------------------------------------------------------------------------
# Multipart form parsing (browser upload)
# ---------------------------------------------------------------------------
def _parse_content_disposition(value: str) -> dict:
    result: dict[str, str] = {}
    for chunk in value.split(";"):
        chunk = chunk.strip()
        if "=" in chunk:
            key, raw = chunk.split("=", 1)
            result[key.strip().lower()] = raw.strip().strip('"')
    return result


def parse_multipart(body: bytes, content_type: str) -> tuple[dict, list[dict]]:
    match = re.search(r"boundary=(.+)", content_type)
    if not match:
        raise ValueError("Upload request is missing a multipart boundary.")
    boundary = match.group(1).strip().strip('"').encode()
    fields: dict[str, str] = {}
    files: list[dict] = []
    for part in body.split(b"--" + boundary):
        part = part.strip(b"\r\n")
        if not part or part == b"--":
            continue
        if part.endswith(b"--"):
            part = part[:-2]
        if b"\r\n\r\n" not in part:
            continue
        header_blob, data = part.split(b"\r\n\r\n", 1)
        headers: dict[str, str] = {}
        for raw_line in header_blob.decode("utf-8", "replace").split("\r\n"):
            if ":" in raw_line:
                key, value = raw_line.split(":", 1)
                headers[key.strip().lower()] = value.strip()
        disposition = _parse_content_disposition(headers.get("content-disposition", ""))
        name = disposition.get("name")
        if not name:
            continue
        if data.endswith(b"\r\n"):
            data = data[:-2]
        if disposition.get("filename"):
            files.append({"field": name, "filename": disposition["filename"], "content": data})
        else:
            fields[name] = data.decode("utf-8", "replace")
    return fields, files


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
class WorkflowHandler(BaseHTTPRequestHandler):
    server_version = "RoverControlCenter/2.0"

    def log_message(self, fmt: str, *args) -> None:  # noqa: A002 - quieter logging
        print(f"[gui] {self.address_string()} - {fmt % args}")

    def _dispatch(self, method: str) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)
        handler, params = router.match(method, path)
        if handler is None:
            if method == "GET":
                serve_static(self, path)
            else:
                send_json(self, {"error": "Not found"}, 404)
            return
        try:
            result = handler(Request(self, method, path, query, params))
            if result is None:
                return  # handler wrote its own response (streaming)
            if isinstance(result, tuple) and len(result) == 2:
                send_json(self, result[1], result[0])
            else:
                send_json(self, result)
        except Exception as error:  # noqa: BLE001 - report handler errors to the UI
            send_json(self, {"error": str(error)}, 400)

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")


def build_server(host: str, port: int) -> tuple[ThreadingHTTPServer, int]:
    """Bind a server, trying nearby ports if the first is busy."""
    last_error: OSError | None = None
    for candidate in range(port, port + 20):
        try:
            return ThreadingHTTPServer((host, candidate), WorkflowHandler), candidate
        except OSError as error:
            last_error = error
    assert last_error is not None
    raise last_error
