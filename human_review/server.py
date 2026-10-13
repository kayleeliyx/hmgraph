"""Small HTTP server for a local, dependency-free review interface."""
from __future__ import annotations

import json
import mimetypes
import re
import secrets
import threading
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from store import ConflictError, LabelStore, export_labels, progress, required_fields

ROOT = Path(__file__).resolve().parent


def video_review_points(catalog, video_id, labels, model=None):
    """Human decisions are point events at sampled queries, never time spans."""
    if model and model not in catalog.models:
        raise ValueError("Unknown model.")
    index = {(r["id"], r["model"], r["field"]): r for r in labels}
    counts = {key: 0 for key in ("correct", "partial", "incorrect", "uncertain", "unreviewed")}
    points = []
    for identity, reference in catalog.references.items():
        if reference["video_id"] != video_id:
            continue
        for name in catalog.models:
            if model and name != model or identity not in catalog.records[name]:
                continue
            for field in required_fields(reference):
                label = index.get((identity, name, field))
                decision = label["label"] if label else "unreviewed"
                counts[decision] += 1
                points.append({"id": identity, "task": reference["task"], "time_sec": reference["query_time_sec"],
                               "model": name, "field": field, "label": decision,
                               "flags": label["flags"] if label else {}, "notes": label["notes"] if label else ""})
    return {"counts": counts, "eligible_count": len(points), "reviewed_count": len(points) - counts["unreviewed"],
            "points": points, "source": "human_labels_only", "unit": "query_model_field_decisions"}


def byte_range(header, size):
    """Return inclusive bounds for a single HTTP range; reject malformed ranges."""
    if not header:
        return 0, size - 1
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", header.strip())
    if not match or not any(match.groups()) or size <= 0:
        raise ValueError("Invalid byte range")
    start, end = match.groups()
    if not start:
        suffix = int(end)
        if suffix <= 0:
            raise ValueError("Invalid suffix")
        return max(0, size - suffix), size - 1
    start = int(start)
    end = min(int(end), size - 1) if end else size - 1
    if start >= size or start > end:
        raise ValueError("Range outside file")
    return start, end


class ReviewServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, catalog, store):
        self.catalog, self.store = catalog, store
        self.review_token = secrets.token_urlsafe(32)
        self._timeline = None
        self._timeline_lock = threading.Lock()
        super().__init__(address, ReviewHandler)

    def timeline(self):
        if self._timeline is None:
            with self._timeline_lock:
                if self._timeline is None:
                    from timeline_data import TimelineCatalog
                    self._timeline = TimelineCatalog(self.catalog)
        return self._timeline


class ReviewHandler(BaseHTTPRequestHandler):
    server_version = "ActGroundReview/1"

    def log_message(self, fmt, *args):
        # Keep media requests and normal browsing quiet.
        if args and str(args[1] if len(args) > 1 else "").startswith(("4", "5")):
            super().log_message(fmt, *args)

    def common_headers(self):
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "same-origin")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' blob:; media-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'")

    def respond(self, code, data, content_type="application/json; charset=utf-8", filename=None):
        if content_type.startswith("application/json"):
            data = json.dumps(data, ensure_ascii=False, allow_nan=False).encode()
        elif isinstance(data, str):
            data = data.encode()
        self.send_response(code)
        self.common_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        try:
            self.get()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except KeyError:
            self.respond(404, {"error": "Example or media not found."})
        except ConflictError as exc:
            self.respond(409, {"error": str(exc)})
        except ValueError as exc:
            self.respond(400, {"error": str(exc)})
        except Exception:
            traceback.print_exc()
            self.respond(500, {"error": "Could not read this record. See the server terminal."})

    def get(self):
        url = urlsplit(self.path)
        query = parse_qs(url.query)
        reviewer = query.get("reviewer", ["Reviewer"])[0].strip()
        if not reviewer or len(reviewer) > 100:
            raise ValueError("Enter a reviewer name of 1–100 characters.")
        catalog, store = self.server.catalog, self.server.store
        if url.path == "/api/bootstrap":
            self.respond(200, dict(catalog.bootstrap(), csrf_token=self.server.review_token))
        elif url.path == "/api/timeline/videos":
            self.respond(200, {"videos": self.server.timeline().list_videos()})
        elif url.path == "/api/timeline/video":
            identity = query.get("id", [""])[0]
            model = query.get("model", [""])[0] or None
            detail = dict(self.server.timeline().get_video(identity))
            labels = store.current(catalog, reviewer)
            detail["human_review"] = video_review_points(catalog, identity, labels, model)
            self.respond(200, detail)
        elif url.path == "/api/examples":
            summary = progress(catalog, store.current(catalog, reviewer))
            examples = [dict(row, review_state=summary["examples"].get(row["id"], {})) for row in catalog.list_examples()]
            self.respond(200, {"examples": examples, "progress": {k: v for k, v in summary.items() if k != "examples"}})
        elif url.path == "/api/example":
            identity = query.get("id", [""])[0]
            detail = dict(catalog.get_example(identity))
            detail["labels"] = store.current(catalog, reviewer, identity)
            self.respond(200, detail)
        elif url.path == "/api/progress":
            self.respond(200, progress(catalog, store.current(catalog, reviewer)))
        elif url.path == "/api/export":
            format_name = query.get("format", ["jsonl"])[0]
            content_type, filename, data = export_labels(store.current(catalog, reviewer), format_name)
            self.respond(200, data, content_type, filename)
        elif url.path.startswith("/media/"):
            opaque_id = unquote(url.path[len("/media/"):])
            if not re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", opaque_id):
                raise KeyError(opaque_id)
            self.send_file(catalog.media[opaque_id], media=True)
        else:
            files = {"/": "index.html", "/index.html": "index.html", "/app.css": "app.css", "/app.js": "app.js",
                     "/static/app.css": "app.css", "/static/app.js": "app.js", "/timeline": "timeline.html",
                     "/timeline/": "timeline.html", "/static/timeline.css": "timeline.css", "/static/timeline.js": "timeline.js"}
            if url.path not in files:
                self.respond(404, {"error": "Not found."})
                return
            self.send_file(ROOT / "static" / files[url.path])

    def do_POST(self):
        try:
            if urlsplit(self.path).path != "/api/labels":
                self.respond(404, {"error": "Not found."})
                return
            origin = self.headers.get("Origin")
            if origin and urlsplit(origin).netloc != self.headers.get("Host"):
                self.respond(403, {"error": "Cross-origin label writes are blocked."})
                return
            if not secrets.compare_digest(self.headers.get("X-Review-Token", ""), self.server.review_token):
                self.respond(403, {"error": "Review session expired. Reload the page."})
                return
            if "application/json" not in self.headers.get("Content-Type", ""):
                self.respond(415, {"error": "Labels must be sent as JSON."})
                return
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 65536:
                self.respond(413, {"error": "Invalid label size."})
                return
            payload = json.loads(self.rfile.read(length))
            expected = payload.get("expected_revision") if isinstance(payload, dict) else None
            label = self.server.store.save(payload, self.server.catalog, expected)
            self.respond(200, {"saved": label, "current": True})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except ConflictError as exc:
            self.respond(409, {"error": str(exc)})
        except (ValueError, TypeError) as exc:
            self.respond(400, {"error": str(exc)})
        except Exception:
            traceback.print_exc()
            self.respond(500, {"error": "Label was not saved. See the server terminal."})

    def send_file(self, path, media=False):
        path = Path(path)
        if not path.is_file():
            self.respond(404, {"error": "Media is unavailable at its configured path."})
            return
        size = path.stat().st_size
        range_header = self.headers.get("Range") if media else None
        try:
            start, end = byte_range(range_header, size)
        except ValueError:
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.common_headers()
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(206 if range_header else 200)
        self.common_headers()
        self.send_header("Content-Type", mimetypes.guess_type(str(path))[0] or "application/octet-stream")
        self.send_header("Content-Length", str(max(0, end - start + 1)))
        if media:
            self.send_header("Accept-Ranges", "bytes")
        if range_header:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if self.command == "HEAD":
            return
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = handle.read(min(256 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


def serve(config_path, host="127.0.0.1", port=8765):
    from data import Catalog
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    state = Path(config.get("state_dir", "state"))
    if not state.is_absolute():
        state = config_path.parent / state
    catalog = Catalog(config_path)
    server = ReviewServer((host, port), catalog, LabelStore(state / "labels.sqlite3"))
    address = f"http://{host}:{server.server_address[1]}"
    print(f"ActGround Review: {address}\nExamples: {len(catalog.references):,} | Models: {len(catalog.models)}\nLabels: {state / 'labels.sqlite3'}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nReview server stopped. Labels are saved.", flush=True)
    finally:
        server.server_close()
