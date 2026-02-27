"""Flask web server for video summarizer batch processing UI."""

import collections
import json
import logging
import os
import queue
import sys
import threading
import time
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory


# ── Circular Buffer Log Handler ───────────────────────────────────
class BufferLogHandler(logging.Handler):
    """Captures log records into a fixed-size deque with a monotonic sequence."""

    def __init__(self, maxlen: int = 5000):
        super().__init__()
        self.buffer: collections.deque = collections.deque(maxlen=maxlen)
        self.seq: int = 0
        self._lock = threading.Lock()

    def emit(self, record):
        try:
            msg = self.format(record)
            with self._lock:
                self.seq += 1
                self.buffer.append(msg)
        except Exception:
            self.handleError(record)

    def get_since(self, since_seq: int) -> tuple[list[str], int]:
        """Return (new_lines, current_seq) for lines added after since_seq."""
        with self._lock:
            seq = self.seq
            new_count = seq - since_seq
            if new_count <= 0:
                return [], seq
            buf = list(self.buffer)
            new_count = min(new_count, len(buf))
            return buf[-new_count:], seq

# ── Path Setup ────────────────────────────────────────────────────
# Add src/ to sys.path so we can import services and utils
# (same pattern as src/main.py)
_project_root = Path(__file__).resolve().parent.parent
_src_dir = _project_root / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from utils.config_loader import ConfigLoader
from web.queue_manager import QueueManager
from web.worker import ProcessingWorker

# ── Video File Extensions ─────────────────────────────────────────
VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm",
    ".m4v", ".mpg", ".mpeg", ".3gp", ".ts",
}


def create_app(config_path: str = "config/processing.yaml") -> Flask:
    """Create and configure the Flask application."""

    app = Flask(
        __name__,
        static_folder=str(Path(__file__).parent / "static"),
        static_url_path="/static",
    )

    # Load processing config
    config = ConfigLoader.load_config(config_path)

    # Logging — attach circular buffer handler to root logger
    log_handler = BufferLogHandler(maxlen=5000)
    log_handler.setLevel(logging.DEBUG)
    log_handler.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-5s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    ))
    logging.getLogger().addHandler(log_handler)

    # Queue manager
    queue_dir = str(_project_root / "queue")
    qm = QueueManager(queue_dir)
    qm.recover_interrupted_batches()

    # SSE subscribers
    sse_subscribers: list[queue.Queue] = []
    sse_lock = threading.Lock()

    def broadcast_sse(event_type: str, data: dict):
        """Push an SSE event to all connected browsers."""
        payload = json.dumps({"type": event_type, **data})
        dead = []
        with sse_lock:
            for q in sse_subscribers:
                try:
                    q.put_nowait(("event", event_type, payload))
                except queue.Full:
                    dead.append(q)
            for q in dead:
                sse_subscribers.remove(q)

    # Processing worker
    worker = ProcessingWorker(qm, config, broadcast_sse)

    # ── SPA Serving ───────────────────────────────────────────────

    @app.route("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    # ── Config ────────────────────────────────────────────────────

    @app.route("/api/config")
    def api_config():
        """Return relevant config defaults for the UI."""
        return jsonify({
            "output_dir": str(config.get("paths", {}).get("output_dir", "output")),
            "input_dir": str(config.get("paths", {}).get("input_dir", "input")),
        })

    # ── File Browser ──────────────────────────────────────────────

    @app.route("/api/browse")
    def browse():
        """Browse directories and list video files.

        Query params:
          path  – directory to list (default: project input dir)
        """
        raw_path = request.args.get("path", "")
        if not raw_path:
            raw_path = str(config.get("paths", {}).get("input_dir", "input"))

        target = Path(raw_path).resolve()

        if not target.is_dir():
            return jsonify({"error": f"Not a directory: {target}"}), 400

        entries = []
        try:
            for item in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
                if item.name.startswith("."):
                    continue
                entry = {
                    "name": item.name,
                    "path": str(item),
                    "is_dir": item.is_dir(),
                }
                if not item.is_dir():
                    entry["is_video"] = item.suffix.lower() in VIDEO_EXTENSIONS
                    try:
                        entry["size_mb"] = round(item.stat().st_size / (1024 * 1024), 1)
                    except OSError:
                        entry["size_mb"] = 0
                entries.append(entry)
        except PermissionError:
            return jsonify({"error": f"Permission denied: {target}"}), 403

        return jsonify({
            "path": str(target),
            "parent": str(target.parent) if target.parent != target else None,
            "entries": entries,
        })

    @app.route("/api/browse/videos")
    def browse_videos():
        """Recursively find all video files under a directory.

        Query params:
          path  – directory to scan recursively
        """
        raw_path = request.args.get("path", "")
        if not raw_path:
            return jsonify({"error": "path is required"}), 400

        target = Path(raw_path).resolve()
        if not target.is_dir():
            return jsonify({"error": f"Not a directory: {target}"}), 400

        videos = []
        try:
            for item in sorted(target.rglob("*")):
                if item.is_file() and item.suffix.lower() in VIDEO_EXTENSIONS:
                    try:
                        size_mb = round(item.stat().st_size / (1024 * 1024), 1)
                    except OSError:
                        size_mb = 0
                    videos.append({"path": str(item), "filename": item.name, "size_mb": size_mb})
        except PermissionError:
            return jsonify({"error": f"Permission denied: {target}"}), 403

        return jsonify({"path": str(target), "videos": videos})

    # ── Queue API ─────────────────────────────────────────────────

    @app.route("/api/queue/create", methods=["POST"])
    def queue_create():
        """Create a new batch from selected video files."""
        data = request.get_json(force=True)
        files = data.get("files", [])
        output_dir = data.get("output_dir", str(config.get("paths", {}).get("output_dir", "output")))

        if not files:
            return jsonify({"error": "No files provided"}), 400

        # Validate all files exist and are videos
        valid_files = []
        for f in files:
            p = Path(f)
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
                valid_files.append(str(p))

        if not valid_files:
            return jsonify({"error": "No valid video files found"}), 400

        manifest = qm.create_batch(valid_files, output_dir)
        broadcast_sse("batch_created", {"batch_id": manifest["batch_id"]})
        return jsonify(manifest), 201

    @app.route("/api/queue/list")
    def queue_list():
        return jsonify(qm.list_batches())

    @app.route("/api/queue/<batch_id>")
    def queue_detail(batch_id):
        manifest = qm.get_batch(batch_id)
        if not manifest:
            return jsonify({"error": "Batch not found"}), 404
        return jsonify(manifest)

    @app.route("/api/queue/<batch_id>/start", methods=["POST"])
    def queue_start(batch_id):
        manifest = qm.get_batch(batch_id)
        if not manifest:
            return jsonify({"error": "Batch not found"}), 404

        if worker.is_running():
            return jsonify({"error": "A batch is already being processed"}), 409

        worker.start(batch_id)
        return jsonify({"status": "started", "batch_id": batch_id})

    @app.route("/api/queue/<batch_id>/pause", methods=["POST"])
    def queue_pause(batch_id):
        if worker.current_batch_id != batch_id:
            return jsonify({"error": "This batch is not currently processing"}), 400
        worker.pause()
        return jsonify({"status": "pausing", "batch_id": batch_id})

    @app.route("/api/queue/<batch_id>/retry-failed", methods=["POST"])
    def queue_retry(batch_id):
        manifest = qm.get_batch(batch_id)
        if not manifest:
            return jsonify({"error": "Batch not found"}), 404
        manifest = qm.retry_failed(batch_id)
        broadcast_sse("batch_updated", {"batch_id": batch_id})
        return jsonify(manifest)

    @app.route("/api/queue/<batch_id>", methods=["DELETE"])
    def queue_delete(batch_id):
        if worker.is_running() and worker.current_batch_id == batch_id:
            return jsonify({"error": "Cannot delete a batch that is currently processing"}), 409
        if qm.delete_batch(batch_id):
            broadcast_sse("batch_deleted", {"batch_id": batch_id})
            return jsonify({"status": "deleted"})
        return jsonify({"error": "Batch not found"}), 404

    # ── Logs ──────────────────────────────────────────────────────

    @app.route("/api/logs")
    def api_logs():
        """Return log lines added since a given sequence number.

        Query params:
          since – sequence number (default 0, returns all buffered lines)
        """
        since_seq = request.args.get("since", 0, type=int)
        lines, seq = log_handler.get_since(since_seq)
        return jsonify({"logs": lines, "seq": seq})

    # ── SSE Stream ────────────────────────────────────────────────

    @app.route("/api/events")
    def sse_stream():
        """Server-Sent Events endpoint for real-time updates."""
        q = queue.Queue(maxsize=256)
        with sse_lock:
            sse_subscribers.append(q)

        def generate():
            try:
                while True:
                    try:
                        msg = q.get(timeout=30)
                        if msg is None:
                            break
                        _, event_type, payload = msg
                        yield f"event: {event_type}\ndata: {payload}\n\n"
                    except queue.Empty:
                        # Heartbeat
                        yield ": heartbeat\n\n"
            finally:
                with sse_lock:
                    if q in sse_subscribers:
                        sse_subscribers.remove(q)

        return Response(
            generate(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    return app
