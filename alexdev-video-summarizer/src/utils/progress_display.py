"""
Rich-based progress display for video processing pipeline.

Provides real-time visualization of processing progress across all tools.
"""

import time
from typing import Dict, Any
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID
from rich.text import Text
from rich.live import Live

from utils.logger import get_logger

logger = get_logger(__name__)


class ProgressDisplay:
    """Rich-based progress display for processing pipeline"""
    
    def __init__(self, console: Console):
        """
        Initialize progress display
        
        Args:
            console: Rich Console instance
        """
        self.console = console
        self.current_video = ""
        self.current_stage = ""
        self.pipeline_status = {}
        self.start_time = time.time()
        
    def show_video_header(self, video_num: int, total_videos: int, video_name: str):
        """Show header for current video being processed"""
        self.current_video = video_name
        self.pipeline_status = {}
        
        header = Panel(
            f"[VIDEO] [bold cyan]Processing Video {video_num}/{total_videos}[/bold cyan]\n"
            f"[FILE] {video_name}",
            title="Current Video",
            border_style="cyan"
        )
        self.console.print(header)
        
    def update_pipeline_progress(self, stage: str, data: Dict[str, Any]):
        """
        Update pipeline progress display
        
        Args:
            stage: Current processing stage
            data: Stage-specific progress data
        """
        self.current_stage = stage
        self.pipeline_status[stage] = data
        
        # Create pipeline status table
        pipeline_table = self._create_pipeline_table()
        
        # Clear previous output and show updated table
        self.console.print("\r", end="")
        self.console.print(pipeline_table)
        
    def _create_pipeline_table(self) -> Table:
        """Create pipeline status table"""
        table = Table(title="[PIPELINE] Processing Pipeline Status")
        table.add_column("Stage", style="bold")
        table.add_column("Tool", style="cyan")
        table.add_column("Status", style="bold")
        table.add_column("Details")
        
        # Define pipeline stages in order
        stages = [
            ('initializing', 'Setup', '⚙️'),
            ('ffmpeg', 'FFmpeg', '🎞️'),
            ('scene_detection', 'PySceneDetect', '[SCENE]'),
            ('scene_processing', 'Scene Analysis', '[ANALYSIS]'),
            ('audio_pipeline', 'Audio Pipeline', '🎵'),
            ('video_gpu_pipeline', 'Video GPU Pipeline', '🖥️'),
            ('video_cpu_pipeline', 'Video CPU Pipeline', '👁️'),
            ('knowledge_generation', 'Knowledge Base', '📚')
        ]
        
        for stage_key, tool_name, icon in stages:
            status_data = self.pipeline_status.get(stage_key, {})
            status, details = self._format_stage_status(stage_key, status_data)
            
            table.add_row(
                f"{icon} {stage_key.replace('_', ' ').title()}",
                tool_name,
                status,
                details
            )
            
        return table
        
    def _format_stage_status(self, stage: str, data: Dict[str, Any]) -> tuple:
        """Format status and details for a stage"""
        if not data:
            return "[dim]Pending[/dim]", ""
            
        stage_status = data.get('stage', 'unknown')
        
        if stage_status == 'starting':
            return "[yellow]⏳ Running[/yellow]", "Starting..."
        elif stage_status == 'completed':
            return "[green][DONE] Complete[/green]", self._get_completion_details(stage, data)
        elif stage_status == 'error':
            return "[red][FAIL] Failed[/red]", data.get('error', 'Unknown error')
        else:
            return "[blue][WORK] Processing[/blue]", self._get_progress_details(stage, data)
            
    def _get_completion_details(self, stage: str, data: Dict[str, Any]) -> str:
        """Get completion details for a stage"""
        if stage == 'scene_detection':
            return f"{data.get('scene_count', 0)} scenes detected"
        elif stage == 'scene_processing':
            scene = data.get('scene', 0)
            total = data.get('total_scenes', 0)
            return f"Scene {scene}/{total} processed"
        elif stage == 'audio_pipeline':
            return "Whisper → LibROSA → pyAudioAnalysis complete"
        elif stage == 'video_gpu_pipeline':
            return "YOLO → EasyOCR complete"
        elif stage == 'video_cpu_pipeline':
            return "OpenCV complete"
        elif stage == 'knowledge_generation':
            file_name = data.get('file', 'knowledge_base.md')
            return f"Created: {file_name}"
        else:
            return "Complete"
            
    def _get_progress_details(self, stage: str, data: Dict[str, Any]) -> str:
        """Get progress details for a stage"""
        if stage == 'scene_processing':
            scene = data.get('scene', 0)
            total = data.get('total_scenes', 0)
            return f"Processing scene {scene}/{total}"
        else:
            return "In progress..."
            
    def show_video_success(self, video_name: str, processing_time: float):
        """Show successful video completion"""
        success_panel = Panel(
            f"[SUCCESS] [bold green]SUCCESS[/bold green]\n"
            f"[FILE] {video_name}\n"
            f"⏱️  Processing time: {processing_time:.1f} seconds\n"
            f"📄 Knowledge base created",
            title="Video Processed",
            border_style="green"
        )
        self.console.print(success_panel)
        
    def show_video_failure(self, video_name: str, error: str):
        """Show video processing failure"""
        failure_panel = Panel(
            f"[FAILED] [bold red]FAILED[/bold red]\n"
            f"[FILE] {video_name}\n"
            f"[ERROR] Error: {error}\n"
            f"[CONTINUE] Continuing with next video...",
            title="Processing Failed",
            border_style="red"
        )
        self.console.print(failure_panel)
        
    def show_circuit_breaker_activation(self, current_video: int, total_videos: int):
        """Show circuit breaker activation"""
        remaining = total_videos - current_video + 1
        
        circuit_breaker_panel = Panel(
            f"🚨 [bold red]CIRCUIT BREAKER ACTIVATED[/bold red]\n"
            f"⚠️  Too many consecutive failures detected\n"
            f"[STOP] Stopping batch processing\n"
            f"[STATS] Videos processed: {current_video - 1}/{total_videos}\n"
            f"⏸️  Videos remaining: {remaining}\n\n"
            f"🔧 Check system resources and try again",
            title="Batch Processing Aborted",
            border_style="red"
        )
        self.console.print(circuit_breaker_panel)