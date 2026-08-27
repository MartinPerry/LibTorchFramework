"""Serve the dashboard and expose JSON files from a local data directory."""

from __future__ import annotations

import argparse
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent


class DashboardHandler(SimpleHTTPRequestHandler):
    data_dir = ROOT / "data"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self) -> None:
        if urlparse(self.path).path != "/api/metrics":
            return super().do_GET()

        files: list[dict] = []
        errors: list[dict] = []
        if self.data_dir.is_dir():
            for path in sorted(self.data_dir.rglob("*.json")):
                try:
                    with path.open("r", encoding="utf-8") as source:
                        data = json.load(source)
                    files.append({
                        "name": path.name,
                        "path": path.relative_to(ROOT).as_posix(),
                        "data": data,
                    })
                except (OSError, UnicodeError, json.JSONDecodeError) as error:
                    errors.append({"name": path.name, "error": str(error)})

        body = json.dumps({"files": files, "errors": errors}, separators=(",", ":")).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PyTorch metrics dashboard.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--data", type=Path, default=ROOT / "data", help="Folder containing metric JSON files")
    args = parser.parse_args()

    DashboardHandler.data_dir = args.data.resolve()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), DashboardHandler)
    print(f"Dashboard: http://127.0.0.1:{args.port}")
    print(f"Metrics:   {DashboardHandler.data_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
