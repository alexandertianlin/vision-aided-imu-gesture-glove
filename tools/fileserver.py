#!/usr/bin/env python3
"""
HTTP File Server — zero-config read/write file sharing for LAN.
No admin rights required. Run on host A, access from host B via HTTP.

Usage:
    python fileserver.py [port]
"""

import argparse
import json
import os
import shutil
import sys
import urllib.parse
from datetime import datetime
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path


class FileShareHandler(SimpleHTTPRequestHandler):
    """HTTP file server with directory listing, file download, upload, delete, and rename."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _send_json(self, status, data):
        body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, status, html):
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _resolve_path(self, raw_path):
        quoted = urllib.parse.unquote(raw_path)
        quoted = quoted.split("?")[0]
        try:
            resolved = (self.directory / quoted.lstrip("/")).resolve()
        except (ValueError, OSError):
            return None
        try:
            resolved.relative_to(self.directory)
        except ValueError:
            return None
        return resolved

    def _render_listing(self, path, rel_path):
        entries = []
        try:
            for entry in sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
                is_dir = entry.is_dir()
                size = ""
                if entry.is_file():
                    size = entry.stat().st_size
                entries.append({
                    "name": entry.name + ("/" if is_dir else ""),
                    "is_dir": is_dir,
                    "size": size,
                    "mtime": datetime.fromtimestamp(entry.stat().st_mtime).isoformat(),
                })
        except PermissionError:
            return self._send_html(403, "<h1>403 Forbidden</h1>")

        parent = ""
        if rel_path:
            parent_parts = Path(rel_path).parent.parts
            parent = "/" + "/".join(parent_parts) + "/" if parent_parts else "/"

        rows = []
        if parent is not None and rel_path:
            rows.append(f'<tr><td><a href="{parent}">..</a></td><td></td><td></td><td></td></tr>')
        for e in entries:
            url_path = "/" + str(Path(rel_path) / e["name"]) if rel_path else "/" + e["name"]
            icon_class = "folder" if e["is_dir"] else "file"

            size_str = ""
            if e["size"] != "" and not e["is_dir"]:
                s = e["size"]
                for unit in ["B", "KB", "MB", "GB"]:
                    if s < 1024:
                        size_str = f"{s:.1f} {unit}" if unit != "B" else f"{s} B"
                        break
                    s /= 1024

            rows.append(f'<tr><td><a href="{url_path}" class="{icon_class}">{e["name"]}</a></td><td>{size_str}</td><td>{e["mtime"]}</td></tr>')

        upload_html = '''
        <div class="upload-section">
            <div class="upload-row">
                <input type="file" id="fileInput" multiple />
                <button onclick="uploadFiles()">Upload</button>
            </div>
        </div>
        <script>
        async function uploadFiles() {
            const input = document.getElementById('fileInput');
            const files = input.files;
            if (!files.length) return;
            const currentPath = window.location.pathname;
            for (const file of files) {
                const resp = await fetch(currentPath + file.name, { method: 'PUT', body: file });
                if (!resp.ok) { alert('Failed: ' + file.name); }
            }
            location.reload();
        }
        </script>
        '''

        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="utf-8"/>
<title>Index of /{rel_path}</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 20px; background: #f5f5f5; }}
h1 {{ font-size: 1.2rem; color: #555; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ text-align: left; padding: 6px 12px; border-bottom: 1px solid #ddd; }}
th {{ color: #888; font-weight: 600; user-select: none; }}
a {{ color: #1565c0; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
a.folder {{ font-weight: 600; }}
button {{ background: #1976d2; color: #fff; border: none; padding: 4px 16px; cursor: pointer; border-radius: 4px; }}
button:hover {{ background: #1565c0; }}
.upload-section {{ margin: 20px 0; padding: 12px; background: #e8eaf6; border-radius: 6px; }}
.upload-row {{ display: flex; gap: 8px; align-items: center; }}
td:nth-child(2) {{ color: #777; font-variant-numeric: tabular-nums; }}
td:nth-child(3) {{ color: #999; font-size: 0.85rem; }}
</style>
</head>
<body>
<h1>Index of /{rel_path}</h1>
{upload_html}
<table><thead><tr><th>Name</th><th>Size</th><th>Modified</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table>
</body>
</html>"""
        return self._send_html(200, html)

    def _list_json(self, path, rel_path):
        entries = []
        try:
            for entry in sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
                entries.append({
                    "name": entry.name,
                    "is_dir": entry.is_dir(),
                    "size": entry.stat().st_size if entry.is_file() else 0,
                    "mtime": datetime.fromtimestamp(entry.stat().st_mtime).isoformat(),
                })
        except PermissionError:
            return self._send_json(403, {"error": "Permission denied"})
        return self._send_json(200, {"path": "/" + rel_path, "entries": entries})

    def do_GET(self):
        path = self._resolve_path(self.path)
        if path is None:
            return self._send_json(404, {"error": "Not found"})
        rel = str(path.relative_to(self.directory)).replace("\\", "/")
        if path.is_dir():
            accept = self.headers.get("Accept", "")
            if "application/json" in accept:
                return self._list_json(path, rel)
            return self._render_listing(path, rel)
        return super().do_GET()

    def do_PUT(self):
        path = self._resolve_path(self.path)
        if path is None:
            return self._send_json(400, {"error": "Invalid path"})
        path.parent.mkdir(parents=True, exist_ok=True)
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b""
        try:
            path.write_bytes(body)
            rel = str(path.relative_to(self.directory)).replace("\\", "/")
            self._send_json(200, {"status": "ok", "path": "/" + rel, "size": len(body)})
        except (PermissionError, OSError) as e:
            self._send_json(500, {"error": str(e)})

    def do_DELETE(self):
        path = self._resolve_path(self.path)
        if path is None:
            return self._send_json(400, {"error": "Invalid path"})
        if not path.exists():
            return self._send_json(404, {"error": "Not found"})
        try:
            if path.is_dir():
                path.rmdir()
            else:
                path.unlink()
            rel = str(path.relative_to(self.directory)).replace("\\", "/")
            self._send_json(200, {"status": "deleted", "path": "/" + rel})
        except (PermissionError, OSError) as e:
            self._send_json(500, {"error": str(e)})

    def do_MOVE(self):
        src = self._resolve_path(self.path)
        dest_header = self.headers.get("X-Destination", "").strip()
        if src is None or not dest_header:
            return self._send_json(400, {"error": "Missing source or destination"})
        dest = self._resolve_path(dest_header)
        if dest is None:
            return self._send_json(400, {"error": "Invalid destination"})
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest))
            rel = str(dest.relative_to(self.directory)).replace("\\", "/")
            self._send_json(200, {"status": "moved", "path": "/" + rel})
        except (PermissionError, OSError) as e:
            self._send_json(500, {"error": str(e)})

    def do_POST(self):
        method = self.headers.get("X-HTTP-Method-Override", "").upper()
        if method == "MOVE":
            return self.do_MOVE()
        if method == "DELETE":
            return self.do_DELETE()
        if method == "PUT":
            return self.do_PUT()
        return self._send_json(405, {"error": "Use X-HTTP-Method-Override header or proper HTTP method"})


def main():
    parser = argparse.ArgumentParser(description="LAN file server")
    parser.add_argument("port", nargs="?", type=int, default=8080, help="Port (default: 8080)")
    parser.add_argument("-d", "--directory", default=".", help="Root directory to serve")
    args = parser.parse_args()

    serve_dir = Path(args.directory).resolve()
    if not serve_dir.is_dir():
        print(f"Error: {serve_dir} is not a valid directory", file=sys.stderr)
        sys.exit(1)

    handler = partial(FileShareHandler, directory=serve_dir)
    server = HTTPServer(("0.0.0.0", args.port), handler)

    print(f"\n  {'='*50}")
    print(f"  File server started")
    print(f"  {'='*50}")
    print(f"  Local:   http://localhost:{args.port}/")
    print(f"  LAN:     http://192.168.2.9:{args.port}/")
    print(f"  Root:    {serve_dir}")
    print(f"  {'='*50}")
    print(f"  Methods: GET (download), PUT (upload), DELETE, MOVE")
    print(f"  Ctrl+C  to stop\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped")
        server.server_close()


if __name__ == "__main__":
    main()
