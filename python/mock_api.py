"""Local HarborLink API used in the lesson.

The server reads data/raw/api_invoices.json. It requires no credentials and
works offline. Start it from the project root:

    python python/mock_api.py

Or start it in the background from a Python session:

    from python.mock_api import start_background_server
    start_background_server()
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
API_DATA = ROOT / "data" / "raw" / "api_invoices.json"
PANEL_API_DATA = ROOT / "data" / "raw" / "analysis" / "api_invoices.json"
PREFERRED_PORT = 8765

_server: ThreadingHTTPServer | None = None
_thread: Thread | None = None


class HarborLinkHandler(BaseHTTPRequestHandler):
    """Serve a small read-only invoice API."""

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        path = urlparse(self.path).path.rstrip("/") or "/"
        if path == "/health":
            self._send_json({"status": "ok", "service": "harborlink-mock"})
            return
        if path == "/invoices":
            payload = json.loads(API_DATA.read_text(encoding="utf-8"))
            self._send_json(payload)
            return
        if path == "/panel/invoices":
            payload = json.loads(PANEL_API_DATA.read_text(encoding="utf-8"))
            self._send_json(payload)
            return
        self._send_json({"error": "Not found", "path": path}, status=404)

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        return

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


def _existing_harborlink_url(host: str, port: int) -> str | None:
    """Return the base URL if a HarborLink mock is already listening."""
    try:
        with urlopen(f"http://{host}:{port}/health", timeout=0.5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (URLError, TimeoutError, OSError, json.JSONDecodeError, ValueError):
        return None
    if payload.get("service") == "harborlink-mock":
        return f"http://{host}:{port}"
    return None


def start_background_server(host: str = "127.0.0.1", port: int = PREFERRED_PORT) -> str:
    """Start the mock API once and return its base URL.

    A previous Quarto render, notebook kernel or terminal session may already
    be using the preferred port. In that case we reuse the running HarborLink
    server, or bind an unused port if something else is occupying it.
    """
    global _server, _thread
    if _server is not None:
        return f"http://{host}:{_server.server_address[1]}"

    already_running = _existing_harborlink_url(host, port)
    if already_running is not None:
        return already_running

    try:
        _server = ReusableThreadingHTTPServer((host, port), HarborLinkHandler)
    except OSError:
        _server = ReusableThreadingHTTPServer((host, 0), HarborLinkHandler)

    _thread = Thread(target=_server.serve_forever, daemon=True)
    _thread.start()
    return f"http://{host}:{_server.server_address[1]}"


def stop_background_server() -> None:
    global _server, _thread
    if _server is not None:
        _server.shutdown()
        _server.server_close()
        _server = None
        _thread = None


def main() -> None:
    url = start_background_server()
    print(f"HarborLink mock API listening at {url}/invoices")
    print("Press Ctrl+C to stop.")
    try:
        if _thread is not None:
            _thread.join()
    except KeyboardInterrupt:
        stop_background_server()


if __name__ == "__main__":
    main()
