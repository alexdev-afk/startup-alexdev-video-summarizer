"""
CLI interface for video processing pipeline.

Implements 3-screen workflow:
1. Launch: Show discovered videos and confirmation
2. Processing: Real-time progress with pipeline visualization  
3. Complete: Results summary and Claude synthesis handoff
"""

import os
import time
from pathlib import Path
from typing import List, Dict, Any

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.text import Text

from services.orchestrator import VideoProcessingOrchestrator
from utils.video_discovery import VideoDiscovery
from utils.progress_display import ProgressDisplay


class VideoProcessorCLI:
    """Main CLI interface for video processing"""
    
    def __init__(self, input_dir: str, output_dir: str, config: Dict[str, Any], dry_run: bool = False):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.config = config
        self.dry_run = dry_run
        
        self.console = Console()
        self.orchestrator = VideoProcessingOrchestrator(config)
        
        # Ensure directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def run(self):
        """Main CLI workflow: Launch → Processing → Complete"""
        
        # Screen 1: Launch
        videos = self.show_launch_screen()
        if not videos:
            return
            
        if self.dry_run:
            self.show_dry_run_results(videos)
            return
            
        # Screen 2: Processing
        results = self.show_processing_screen(videos)
        
        # Screen 3: Complete
        self.show_completion_screen(results)
        
    def show_launch_screen(self) -> List[Path]:
        """Screen 1: Display discovered videos and get confirmation"""
        
        self.console.clear()
        self.console.print("\n🎬 [bold blue]alexdev-video-summarizer[/bold blue]")
        self.console.print("Scene-based institutional knowledge extraction\n")
        
        # Discover videos
        discovery = VideoDiscovery(self.input_dir)
        videos = discovery.find_videos()
        
        if not videos:
            self.console.print(f"❌ No videos found in: {self.input_dir}")
            self.console.print("   Supported formats: MP4, AVI, MOV, MKV, WebM")
            return []
            
        # Display video table
        table = Table(title="📁 Discovered Videos")
        table.add_column("File", style="cyan")
        table.add_column("Size", justify="right")
        table.add_column("Duration", justify="right")
        
        total_size = 0
        estimated_time = 0
        
        for video in videos:
            size_mb = video.stat().st_size / (1024 * 1024)
            total_size += size_mb
            estimated_time += 10  # 10 minutes average per video
            
            table.add_row(
                video.name,
                f"{size_mb:.1f} MB",
                "~10 min"  # Placeholder - could add actual duration detection
            )
            
        self.console.print(table)
        
        # Processing summary
        summary_panel = Panel(
            f"📊 [bold]Processing Summary[/bold]\n"
            f"• Videos: {len(videos)}\n"
            f"• Total Size: {total_size:.1f} MB\n"
            f"• Estimated Time: {estimated_time} minutes ({estimated_time/60:.1f} hours)\n"
            f"• Output Location: {self.output_dir}",
            title="Summary",
            border_style="green"
        )
        self.console.print(summary_panel)
        
        # User confirmation
        self.console.print("\n🚀 [bold]Ready to process videos?[/bold]")
        self.console.print("   ENTER: Start processing")
        self.console.print("   Q: Quit")
        
        while True:
            key = input().strip().lower()
            if key == '' or key == 'y':
                return videos
            elif key == 'q' or key == 'n':
                self.console.print("👋 Processing cancelled")
                return []
            else:
                self.console.print("Please press ENTER to continue or Q to quit")
                
    def show_processing_screen(self, videos: List[Path]) -> Dict[str, Any]:
        """Screen 2: Real-time processing with pipeline visualization"""
        
        self.console.clear()
        
        # Initialize progress display
        progress_display = ProgressDisplay(self.console)
        results = {
            'successful': [],
            'failed': [],
            'total_time': 0
        }
        
        start_time = time.time()
        
        for i, video in enumerate(videos, 1):
            video_start = time.time()
            
            try:
                # Show current video header
                progress_display.show_video_header(i, len(videos), video.name)
                
                # Process video with real-time updates
                result = self.orchestrator.process_video_with_progress(
                    video, progress_display.update_pipeline_progress
                )
                
                if result.success:
                    results['successful'].append((video.name, result))
                    progress_display.show_video_success(video.name, time.time() - video_start)
                else:
                    results['failed'].append((video.name, result))
                    progress_display.show_video_failure(video.name, result.error)
                    
            except KeyboardInterrupt:
                raise
            except Exception as e:
                results['failed'].append((video.name, str(e)))
                progress_display.show_video_failure(video.name, str(e))
                
                # Check circuit breaker
                if self.orchestrator.should_abort_batch():
                    progress_display.show_circuit_breaker_activation(i, len(videos))
                    break
                    
        results['total_time'] = time.time() - start_time
        return results
        
    def show_completion_screen(self, results: Dict[str, Any]):
        """Screen 3: Results summary and Claude synthesis handoff"""
        
        self.console.clear()
        
        successful_count = len(results['successful'])
        failed_count = len(results['failed'])
        total_count = successful_count + failed_count
        
        # Results header
        if successful_count > 0:
            self.console.print("🎉 [bold green]PROCESSING COMPLETE[/bold green]")
        else:
            self.console.print("⚠️  [bold yellow]PROCESSING COMPLETED WITH ISSUES[/bold yellow]")
            
        # Results summary table
        summary_table = Table(title="📊 Processing Results")
        summary_table.add_column("Status", style="bold")
        summary_table.add_column("Count", justify="right")
        summary_table.add_column("Percentage", justify="right")
        
        summary_table.add_row(
            "✅ Successful", 
            str(successful_count),
            f"{(successful_count/total_count)*100:.1f}%" if total_count > 0 else "0%"
        )
        summary_table.add_row(
            "❌ Failed", 
            str(failed_count),
            f"{(failed_count/total_count)*100:.1f}%" if total_count > 0 else "0%"
        )
        summary_table.add_row(
            "⏱️ Total Time",
            f"{results['total_time']/60:.1f} min",
            ""
        )
        
        self.console.print(summary_table)
        
        # Show successful outputs
        if successful_count > 0:
            output_panel = Panel(
                f"📁 [bold]Knowledge Base Files Created:[/bold]\n"
                + "\n".join([f"• output/{name}.md" for name, _ in results['successful']]),
                title="Output Files",
                border_style="green"
            )
            self.console.print(output_panel)
            
        # Show failed videos if any
        if failed_count > 0:
            failed_panel = Panel(
                "❌ [bold]Failed Videos:[/bold]\n"
                + "\n".join([f"• {name}: {error}" for name, error in results['failed']]),
                title="Processing Errors",
                border_style="red"
            )
            self.console.print(failed_panel)
            
        # Claude synthesis handoff message
        if successful_count > 0:
            claude_panel = Panel(
                "🧠 [bold cyan]READY FOR CLAUDE SYNTHESIS[/bold cyan]\n\n"
                "Next Steps:\n"
                f"1. Processed video data is ready in build/ directory\n"
                f"2. {successful_count} video(s) ready for knowledge synthesis\n"
                f"3. Use Claude to create TodoWrite tasks for video synthesis\n"
                f"4. Each video will become a comprehensive .md knowledge base\n\n"
                "📋 Claude: Please create synthesis todos for each processed video",
                title="Claude Integration Ready",
                border_style="cyan"
            )
            self.console.print(claude_panel)
            
        self.console.print(f"\n📁 Output directory: {self.output_dir}")
        
    def show_dry_run_results(self, videos: List[Path]):
        """Show what would be processed without executing"""
        
        self.console.print("\n🔍 [bold yellow]DRY RUN MODE[/bold yellow]")
        self.console.print("The following videos would be processed:\n")
        
        for i, video in enumerate(videos, 1):
            self.console.print(f"{i:2d}. {video.name}")
            
        self.console.print(f"\n📊 Total: {len(videos)} videos")
        self.console.print(f"⏱️  Estimated time: {len(videos) * 10} minutes")
        self.console.print(f"📁 Output: {self.output_dir}")
        self.console.print("\nRun without --dry-run to begin processing.")