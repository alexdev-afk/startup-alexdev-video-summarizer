"""Background processing worker that wraps the orchestrator pipeline."""

import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional


class ProcessingWorker:
    """Background worker that processes batches of videos.

    Uses the same VideoProcessingOrchestrator and process_video_with_progress()
    method as the CLI, routing progress callbacks to SSE instead of Rich console.
    """

    CIRCUIT_BREAKER_LIMIT = 3

    def __init__(self, queue_manager, config: dict, broadcast: Callable):
        self.queue_manager = queue_manager
        self.config = config
        self.broadcast = broadcast  # broadcast(event_type, data_dict)

        self._thread: Optional[threading.Thread] = None
        self._pause_event = threading.Event()  # set = paused
        self._current_batch_id: Optional[str] = None
        self._consecutive_failures = 0

    # ── Public API ────────────────────────────────────────────────

    def start(self, batch_id: str):
        """Start or resume processing a batch."""
        if self._thread and self._thread.is_alive():
            return  # already running

        self._pause_event.clear()
        self._current_batch_id = batch_id
        self._consecutive_failures = 0

        # Reset any in_progress videos back to pending (from prior interrupted run)
        manifest = self.queue_manager.get_batch(batch_id)
        if manifest:
            for idx, video in enumerate(manifest["videos"]):
                if video["status"] == "in_progress":
                    self.queue_manager.update_video_status(
                        batch_id, idx,
                        status="pending",
                        current_stage=None,
                    )

        self.queue_manager.update_batch_status(batch_id, "processing")

        self._thread = threading.Thread(
            target=self._run, args=(batch_id,), daemon=True
        )
        self._thread.start()
        self.broadcast("batch_started", {"batch_id": batch_id})

    def pause(self):
        """Signal worker to pause after current video finishes."""
        self._pause_event.set()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def current_batch_id(self) -> Optional[str]:
        return self._current_batch_id

    # ── Background Thread ─────────────────────────────────────────

    def _run(self, batch_id: str):
        """Main worker loop — iterates through pending videos in the batch."""
        from services.orchestrator import VideoProcessingOrchestrator

        manifest = self.queue_manager.get_batch(batch_id)
        if not manifest:
            return

        output_dir = manifest.get("output_dir", "output")

        # Create orchestrator (loads models once per batch)
        orchestrator = VideoProcessingOrchestrator(self.config, output_dir)
        orchestrator.allow_service_logging = True  # let logs flow to file + web UI

        try:
            for idx, video in enumerate(manifest["videos"]):
                # Skip already-completed videos (resume support)
                if video["status"] == "completed":
                    continue

                # Check pause signal between videos
                if self._pause_event.is_set():
                    self.queue_manager.update_batch_status(batch_id, "paused")
                    self.broadcast("batch_paused", {"batch_id": batch_id})
                    return

                # Check circuit breaker
                if self._consecutive_failures >= self.CIRCUIT_BREAKER_LIMIT:
                    self.queue_manager.update_batch_status(batch_id, "paused")
                    self.broadcast("circuit_breaker", {
                        "batch_id": batch_id,
                        "consecutive_failures": self._consecutive_failures,
                    })
                    return

                # Mark video as in_progress
                video_path = Path(video["path"])
                started_at = datetime.now(timezone.utc).isoformat()
                self.queue_manager.update_video_status(
                    batch_id, idx,
                    status="in_progress",
                    current_stage="initializing",
                    started_at=started_at,
                )
                self.broadcast("video_started", {
                    "batch_id": batch_id,
                    "video_index": idx,
                    "filename": video["filename"],
                })

                # Progress callback → SSE
                def make_progress_cb(vid_idx, filename):
                    def progress_cb(stage, data):
                        self.queue_manager.update_video_status(
                            batch_id, vid_idx, current_stage=stage
                        )
                        # Sanitize data: convert Path objects to strings for JSON
                        safe_data = {
                            k: str(v) if isinstance(v, Path) else v
                            for k, v in data.items()
                        } if isinstance(data, dict) else data
                        self.broadcast("progress", {
                            "batch_id": batch_id,
                            "video_index": vid_idx,
                            "filename": filename,
                            "stage": stage,
                            "data": safe_data,
                        })
                    return progress_cb

                # Process video
                start_time = time.time()
                result = orchestrator.process_video_with_progress(
                    video_path,
                    make_progress_cb(idx, video["filename"]),
                )
                processing_time = round(time.time() - start_time, 1)

                if result.success:
                    self._consecutive_failures = 0
                    completed_at = datetime.now(timezone.utc).isoformat()
                    self.queue_manager.update_video_status(
                        batch_id, idx,
                        status="completed",
                        current_stage=None,
                        completed_at=completed_at,
                        processing_time=processing_time,
                        output_path=str(result.knowledge_file) if result.knowledge_file else None,
                    )
                    self.queue_manager.append_result(batch_id, {
                        "filename": video["filename"],
                        "success": True,
                        "processing_time": processing_time,
                        "output_path": str(result.knowledge_file) if result.knowledge_file else None,
                        "completed_at": completed_at,
                    })
                    self.broadcast("video_completed", {
                        "batch_id": batch_id,
                        "video_index": idx,
                        "filename": video["filename"],
                        "processing_time": processing_time,
                    })
                else:
                    self._consecutive_failures += 1
                    self.queue_manager.update_video_status(
                        batch_id, idx,
                        status="failed",
                        current_stage=None,
                        processing_time=processing_time,
                        error=result.error,
                    )
                    self.queue_manager.append_result(batch_id, {
                        "filename": video["filename"],
                        "success": False,
                        "processing_time": processing_time,
                        "error": result.error,
                        "failed_at": datetime.now(timezone.utc).isoformat(),
                    })
                    self.broadcast("video_failed", {
                        "batch_id": batch_id,
                        "video_index": idx,
                        "filename": video["filename"],
                        "error": result.error,
                    })

                # Re-read manifest to get updated counts
                manifest = self.queue_manager.get_batch(batch_id)

            # All videos processed
            self.queue_manager.update_batch_status(batch_id, "completed")
            self.broadcast("batch_completed", {"batch_id": batch_id})

        except Exception as e:
            self.queue_manager.update_batch_status(batch_id, "paused")
            self.broadcast("batch_paused", {
                "batch_id": batch_id,
                "error": str(e),
            })
