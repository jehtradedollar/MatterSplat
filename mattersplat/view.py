"""Serve the web viewer for a trained splat on localhost and open it in the browser.

    python -m mattersplat view <splat.ply | splat.splat> [--port 8770] [--ply] [--no-browser]

The viewer (viewer/index.html, @mkkellogg/gaussian-splats-3d from jsDelivr, so the first load
needs internet) is served at /, the splat's folder at /data/. It loads splat.splat when it sits
next to the .ply (made on the fly if missing; --ply loads the full PLY with view-dependent colour)
and sweeps.json for the sweep list. The server sends COOP/COEP headers so the sorting worker can
use shared memory, and accepts POST /screenshot?name=x (written to <splat dir>/screenshots/x.jpg).
"""

from __future__ import annotations

import argparse
import base64
import http.server
import os
import re
import sys
import threading
import webbrowser
from functools import partial

from .splat_convert import ply_to_splat

VIEWER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "viewer")


class Handler(http.server.SimpleHTTPRequestHandler):
    data_dir = "."

    def translate_path(self, path):
        path = path.split("?", 1)[0]
        if path.startswith("/data/"):
            rel = path[len("/data/"):]
            return os.path.join(self.data_dir, *[p for p in rel.split("/") if p not in ("", ".", "..")])
        return super().translate_path(path)

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def do_POST(self):
        if not self.path.startswith("/screenshot"):
            self.send_error(404)
            return
        name = re.sub(r"[^A-Za-z0-9_.-]", "_", self.path.split("name=", 1)[-1] or "shot")
        body = self.rfile.read(int(self.headers.get("Content-Length", 0))).decode()
        out_dir = os.path.join(self.data_dir, "screenshots")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, os.path.splitext(name)[0] + ".jpg")
        with open(path, "wb") as fh:
            fh.write(base64.b64decode(body.split(",", 1)[1]))
        self.send_response(200)
        self.end_headers()
        self.wfile.write(path.encode())

    def log_message(self, fmt, *args):
        if "/screenshot" in str(args[0] if args else "") or (args and str(args[1])[:1] in "45"):
            super().log_message(fmt, *args)


def serve(splat: str, port: int = 8770, use_ply: bool = False, browser: bool = True):
    splat = os.path.abspath(splat)
    data_dir = os.path.dirname(splat)
    name = os.path.basename(splat)
    if name.endswith(".ply") and not use_ply:
        compact = os.path.splitext(splat)[0] + ".splat"
        if not os.path.exists(compact) or os.path.getmtime(compact) < os.path.getmtime(splat):
            print(f"Converting {name} -> {os.path.basename(compact)}")
            ply_to_splat(splat, compact)
        name = os.path.basename(compact)
    query = f"?file=data/{name}" + ("&sh=2" if name.endswith(".ply") else "")
    handler = partial(type("H", (Handler,), {"data_dir": data_dir}), directory=VIEWER_DIR)
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    url = f"http://127.0.0.1:{port}/{query}"
    print(f"Viewer: {url}  (Ctrl+C to stop)")
    if browser:
        threading.Timer(0.5, webbrowser.open, (url,)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("splat")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--ply", action="store_true", help="load the full .ply (view-dependent colour, slower)")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args(argv)
    serve(args.splat, args.port, args.ply, not args.no_browser)


if __name__ == "__main__":
    sys.exit(main())
