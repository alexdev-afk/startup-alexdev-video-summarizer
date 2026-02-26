"""Manifest-based batch queue manager for video processing."""

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


class QueueManager:
    """Manages batch processing queues with persistent manifest files.

    Each batch lives in queue/batch_NNN/ with:
      - manifest.json: batch metadata and per-video status
      - results.jsonl: append-only processing results
    """

    def __init__(self, queue_dir: str):
        self.queue_dir = Path(queue_dir)
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # ── Batch Counter (self-healing) ──────────────────────────────

    def _next_batch_id(self) -> str:
        """Generate next batch ID with self-healing counter."""
        counter_file = self.queue_dir / "batch_counter.txt"
        max_id = 0

        # Read persisted counter
        if counter_file.exists():
            try:
                max_id = int(counter_file.read_text().strip())
            except (ValueError, OSError):
                pass

        # Scan directories to self-heal if counter is stale
        for entry in self.queue_dir.iterdir():
            if entry.is_dir() and entry.name.startswith("batch_"):
                try:
                    num = int(entry.name.split("_")[1])
                    max_id = max(max_id, num)
                except (ValueError, IndexError):
                    pass

        next_id = max_id + 1
        counter_file.write_text(str(next_id))
        return f"{next_id:03d}"

    # ── Manifest I/O ─────────────────────────────────────────────

    def _batch_dir(self, batch_id: str) -> Path:
        return self.queue_dir / f"batch_{batch_id}"

    def _manifest_path(self, batch_id: str) -> Path:
        return self._batch_dir(batch_id) / "manifest.json"

    def _results_path(self, batch_id: str) -> Path:
        return self._batch_dir(batch_id) / "results.jsonl"

    def _load_manifest(self, batch_id: str) -> dict:
        path = self._manifest_path(batch_id)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_manifest(self, batch_id: str, manifest: dict):
        """Atomic write: write to .tmp then rename."""
        path = self._manifest_path(batch_id)
        tmp_path = path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        # On Windows, remove target first if it exists
        if path.exists():
            path.unlink()
        tmp_path.rename(path)

    # ── Batch CRUD ────────────────────────────────────────────────

    def create_batch(self, video_paths: list[str], output_dir: str) -> dict:
        """Create a new batch from a list of video file paths."""
        with self._lock:
            batch_id = self._next_batch_id()
            batch_dir = self._batch_dir(batch_id)
            batch_dir.mkdir(parents=True, exist_ok=True)

            videos = []
            for vp in video_paths:
                p = Path(vp)
                try:
                    size_mb = round(p.stat().st_size / (1024 * 1024), 1)
                except OSError:
                    size_mb = 0.0
                videos.append({
                    "path": str(p),
                    "filename": p.name,
                    "size_mb": size_mb,
                    "status": "pending",
                    "current_stage": None,
                    "started_at": None,
                    "completed_at": None,
                    "processing_time": None,
                    "output_path": None,
                    "error": None,
                })

            manifest = {
                "batch_id": batch_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "status": "pending",
                "output_dir": output_dir,
                "total_videos": len(videos),
                "completed_count": 0,
                "failed_count": 0,
                "videos": videos,
            }
            self._save_manifest(batch_id, manifest)
            return manifest

    def get_batch(self, batch_id: str) -> dict | None:
        """Get a single batch manifest."""
        with self._lock:
            try:
                return self._load_manifest(batch_id)
            except (FileNotFoundError, json.JSONDecodeError):
                return None

    def list_batches(self) -> list[dict]:
        """List all batches (summary view, without full video details)."""
        batches = []
        with self._lock:
            for entry in sorted(self.queue_dir.iterdir()):
                if entry.is_dir() and entry.name.startswith("batch_"):
                    try:
                        manifest = json.loads(
                            (entry / "manifest.json").read_text(encoding="utf-8")
                        )
                        # Return summary without full video list
                        batches.append({
                            "batch_id": manifest["batch_id"],
                            "created_at": manifest["created_at"],
                            "status": manifest["status"],
                            "total_videos": manifest["total_videos"],
                            "completed_count": manifest.get("completed_count", 0),
                            "failed_count": manifest.get("failed_count", 0),
                        })
                    except (FileNotFoundError, json.JSONDecodeError, KeyError):
                        continue
        return batches

    def delete_batch(self, batch_id: str) -> bool:
        """Delete a batch directory and all its contents."""
        import shutil
        with self._lock:
            batch_dir = self._batch_dir(batch_id)
            if batch_dir.exists():
                shutil.rmtree(batch_dir)
                return True
            return False

    # ── Video Status Updates ──────────────────────────────────────

    def update_video_status(
        self, batch_id: str, video_index: int, **updates
    ) -> dict:
        """Update a specific video's status within a batch."""
        with self._lock:
            manifest = self._load_manifest(batch_id)
            video = manifest["videos"][video_index]
            video.update(updates)

            # Recalculate counts
            manifest["completed_count"] = sum(
                1 for v in manifest["videos"] if v["status"] == "completed"
            )
            manifest["failed_count"] = sum(
                1 for v in manifest["videos"] if v["status"] == "failed"
            )
            self._save_manifest(batch_id, manifest)
            return manifest

    def update_batch_status(self, batch_id: str, status: str) -> dict:
        """Update the overall batch status."""
        with self._lock:
            manifest = self._load_manifest(batch_id)
            manifest["status"] = status
            self._save_manifest(batch_id, manifest)
            return manifest

    def retry_failed(self, batch_id: str) -> dict:
        """Reset all failed videos to pending for retry."""
        with self._lock:
            manifest = self._load_manifest(batch_id)
            for video in manifest["videos"]:
                if video["status"] == "failed":
                    video["status"] = "pending"
                    video["current_stage"] = None
                    video["started_at"] = None
                    video["completed_at"] = None
                    video["processing_time"] = None
                    video["error"] = None
            manifest["failed_count"] = 0
            manifest["status"] = "pending"
            self._save_manifest(batch_id, manifest)
            return manifest

    # ── Results JSONL ─────────────────────────────────────────────

    def append_result(self, batch_id: str, result: dict):
        """Append a processing result to the batch's results.jsonl."""
        with self._lock:
            results_path = self._results_path(batch_id)
            with open(results_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    # ── Startup Recovery ──────────────────────────────────────────

    def recover_interrupted_batches(self):
        """On startup: reset any 'processing' batches to 'paused'.

        Also reset any 'in_progress' videos to 'pending'.
        """
        with self._lock:
            for entry in self.queue_dir.iterdir():
                if entry.is_dir() and entry.name.startswith("batch_"):
                    try:
                        manifest = json.loads(
                            (entry / "manifest.json").read_text(encoding="utf-8")
                        )
                        if manifest.get("status") == "processing":
                            manifest["status"] = "paused"
                            for video in manifest["videos"]:
                                if video["status"] == "in_progress":
                                    video["status"] = "pending"
                                    video["current_stage"] = None
                            batch_id = manifest["batch_id"]
                            self._save_manifest(batch_id, manifest)
                    except (FileNotFoundError, json.JSONDecodeError, KeyError):
                        continue
