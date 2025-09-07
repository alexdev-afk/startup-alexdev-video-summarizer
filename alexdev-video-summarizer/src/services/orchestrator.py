"""
Video Processing Orchestrator

Coordinates the complete video processing pipeline:
FFmpeg → PySceneDetect → Dual Pipeline Processing → Knowledge Generation

Implements fail-fast per video and circuit breaker for batch processing.
"""

import time
import gc
import logging
import contextlib
from pathlib import Path
from typing import Dict, Any, Callable, Optional
from dataclasses import dataclass
from io import StringIO
import sys

from services.ffmpeg_service import FFmpegService
from services.scene_detection_service import SceneDetectionService
from services.demucs_service import DemucsService
from services.demucs_audio_coordinator import DemucsAudioCoordinatorService
from services.internvl3_timeline_service import InternVL3TimelineService
from services.vid2seq_timeline_service import Vid2SeqTimelineService
from services.knowledge_generator import KnowledgeGenerator
from utils.processing_context import VideoProcessingContext
from utils.circuit_breaker import CircuitBreaker
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ProcessingResult:
    """Result of video processing"""
    video_path: Path
    success: bool
    knowledge_file: Optional[Path] = None
    error: Optional[str] = None
    processing_time: float = 0.0
    scenes_processed: int = 0


class VideoProcessingOrchestrator:
    """Main orchestrator for video processing pipeline"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize services
        self.ffmpeg_service = FFmpegService(config)
        self.scene_service = SceneDetectionService(config)
        self.demucs_service = DemucsService(config)
        self.audio_coordinator = DemucsAudioCoordinatorService(config)
        
        # Video processing services
        self.video_service = InternVL3TimelineService(config)
        self.vid2seq_service = Vid2SeqTimelineService(config)
        
        self.knowledge_generator = KnowledgeGenerator(config)
        
        # Circuit breaker for batch processing
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=config.get('circuit_breaker_threshold', 3)
        )
        
    @contextlib.contextmanager
    def _suppress_service_logging(self):
        """Suppress ALL logging and output from services during processing to keep CLI clean"""
        # Suppress ALL loggers by setting root logger to CRITICAL
        root_logger = logging.getLogger()
        old_root_level = root_logger.level
        
        # Get snapshot of all existing loggers and their levels
        old_levels = {}
        logger_names = list(logging.Logger.manager.loggerDict.keys())  # Create snapshot
        for name in logger_names:
            logger_obj = logging.getLogger(name)
            old_levels[name] = logger_obj.level
        
        try:
            # Silence ALL loggers
            root_logger.setLevel(logging.CRITICAL)
            for name in logger_names:
                logging.getLogger(name).setLevel(logging.CRITICAL)
                
            # Completely suppress stdout/stderr from services
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = StringIO()
            sys.stderr = StringIO()
            
            yield
            
        finally:
            # Restore stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
            # Restore ALL logger levels
            root_logger.setLevel(old_root_level)
            for name, level in old_levels.items():
                logging.getLogger(name).setLevel(level)
        
    def process_video_with_progress(self, video_path: Path, progress_callback: Callable) -> ProcessingResult:
        """
        Process single video with progress updates
        
        Args:
            video_path: Path to video file
            progress_callback: Function to call with progress updates
            
        Returns:
            ProcessingResult with success/failure status
        """
        start_time = time.time()
        
        try:
            # Initialize processing context
            context = VideoProcessingContext(video_path)
            progress_callback('initializing', {'video': video_path.name})
            
            # Step 1: FFmpeg Foundation (CRITICAL)
            progress_callback('ffmpeg', {'stage': 'starting', 'tool': 'FFmpeg'})
            with self._suppress_service_logging():
                context.audio_path, context.processed_video_path = self.ffmpeg_service.extract_streams(video_path)
            
            if not context.validate_ffmpeg_output():
                raise Exception("FFmpeg audio/video extraction failed")
            progress_callback('ffmpeg', {'stage': 'completed', 'tool': 'FFmpeg'})
            
            # Step 2: Scene Detection (CRITICAL)
            progress_callback('scene_detection', {'stage': 'starting', 'tool': 'PySceneDetect'})
            with self._suppress_service_logging():
                context.scene_data = self.scene_service.analyze_video_scenes(context.processed_video_path)
            
            if context.scene_data['scene_count'] == 0:
                raise Exception("No scenes detected in video")
            progress_callback('scene_detection', {
                'stage': 'completed', 
                'tool': 'PySceneDetect',
                'scene_count': context.scene_data['scene_count']
            })
            
            # Step 2.5: Demucs Audio Separation (BREAKTHROUGH)
            progress_callback('demucs_separation', {'stage': 'starting', 'tool': 'Demucs'})
            with self._suppress_service_logging():
                context.vocals_path, context.no_vocals_path = self.demucs_service.separate_audio(context)
            
            if not context.vocals_path.exists() or not context.no_vocals_path.exists():
                raise Exception("Demucs audio separation failed")
            progress_callback('demucs_separation', {
                'stage': 'completed',
                'tool': 'Demucs',
                'vocals_file': context.vocals_path.name,
                'instrumentals_file': context.no_vocals_path.name
            })
            
            # Step 3: Demucs Audio Processing (BREAKTHROUGH)
            progress_callback('audio_processing', {'stage': 'starting', 'approach': 'demucs_separated'})
            with self._suppress_service_logging():
                audio_timelines = self.audio_coordinator.process_all_audio_timelines(context)
            
            # Save individual timeline files
            self.audio_coordinator.save_individual_timelines(audio_timelines, context)
            
            # BREAKTHROUGH: Create combined timeline with simple merger (no 692-line heuristics!)
            combined_timeline = self.audio_coordinator.create_combined_timeline(audio_timelines, context)
            
            progress_callback('audio_processing', {
                'stage': 'completed',
                'approach': 'demucs_separated',
                'individual_timelines': len(audio_timelines),
                'combined_events': len(combined_timeline.events) if combined_timeline else 0,
                'sources': ['whisper_voice', 'librosa_music', 'pyaudio_music', 'pyaudio_voice'],
                'breakthrough': 'NO_HEURISTIC_FILTERING_NEEDED'
            })
            
            # Step 4: Frame Extraction (FIXED)
            progress_callback('frame_extraction', {'stage': 'starting'})
            # Extract representative frames for each scene (3 frames per scene)
            with self._suppress_service_logging():
                frame_extraction_result = self.scene_service.coordinate_frame_extraction(
                    context.processed_video_path, 
                    context.scene_data['boundaries'], 
                    self.ffmpeg_service
                )
            context.scene_data.update(frame_extraction_result)
            frame_count = frame_extraction_result.get('total_frames_extracted', 0)
            progress_callback('frame_extraction', {
                'stage': 'completed',
                'frames_extracted': frame_count
            })
            
            # Step 5A: InternVL3 Frame-Based Video Processing
            progress_callback('video_processing', {
                'stage': 'starting', 
                'processing_mode': 'internvl3_frames'
            })
            
            # Process all extracted frames with InternVL3
            with self._suppress_service_logging():
                video_timeline = self.video_service.generate_and_save(str(context.processed_video_path), None)
            
            progress_callback('video_processing', {
                'stage': 'internvl3_completed',
                'events_generated': len(video_timeline.events) if video_timeline else 0,
                'total_duration': video_timeline.total_duration if video_timeline else 0
            })
            
            # Cleanup video service to free VRAM
            if hasattr(self.video_service, 'cleanup'):
                self.video_service.cleanup()
                
            # Step 5B: Vid2Seq Dense Video Captioning
            progress_callback('video_processing', {
                'stage': 'vid2seq_starting',
                'processing_mode': 'vid2seq_dense_captioning'
            })
            
            # Process video with Vid2Seq for dense captioning
            with self._suppress_service_logging():
                vid2seq_timeline = self.vid2seq_service.generate_and_save(str(context.processed_video_path), None)
            
            progress_callback('video_processing', {
                'stage': 'completed',
                'vid2seq_events_generated': len(vid2seq_timeline.events) if vid2seq_timeline else 0,
                'internvl3_events_generated': len(video_timeline.events) if video_timeline else 0
            })
            
            # Cleanup Vid2Seq service to free VRAM
            if hasattr(self.vid2seq_service, 'cleanup'):
                self.vid2seq_service.cleanup()
            
            # Store video analysis results in context
            if video_timeline:
                context.video_analysis_results = {
                    'timeline': video_timeline,
                    'events': video_timeline.events,
                    'total_duration': video_timeline.total_duration,
                    'processing_mode': 'internvl3_frames'
                }
            else:
                raise Exception("InternVL3 video processing failed - no timeline generated")
            
            # Step 4: Knowledge Base Generation
            progress_callback('knowledge_generation', {'stage': 'starting'})
            
            # Generate combined knowledge file from all available timeline sources
            audio_timeline_path = context.build_directory / "audio_timelines" / "combined_audio_timeline.json"
            video_timeline_path = context.build_directory / "video_timelines" 
            
            # Collect all available video timeline sources
            video_timeline_sources = {}
            
            # Look for InternVL3 timeline files
            contextual_files = list(video_timeline_path.glob("*_contextual.json"))
            noncontextual_files = list(video_timeline_path.glob("*_noncontextual.json"))
            legacy_files = list(video_timeline_path.glob("video_timeline.json"))
            
            if contextual_files:
                latest_contextual = max(contextual_files, key=lambda f: f.stat().st_mtime)
                video_timeline_sources['contextual_vlm'] = latest_contextual
                
            if noncontextual_files:
                latest_noncontextual = max(noncontextual_files, key=lambda f: f.stat().st_mtime)
                video_timeline_sources['noncontextual_vlm'] = latest_noncontextual
                
            if legacy_files and not contextual_files and not noncontextual_files:
                latest_legacy = max(legacy_files, key=lambda f: f.stat().st_mtime)
                video_timeline_sources['internvl3'] = latest_legacy
            
            # Look for Vid2Seq timeline files
            vid2seq_files = list(video_timeline_path.glob("vid2seq_timeline.json"))
            if vid2seq_files:
                latest_vid2seq = max(vid2seq_files, key=lambda f: f.stat().st_mtime)
                video_timeline_sources['vid2seq'] = latest_vid2seq
            
            # Generate single comprehensive knowledge file
            knowledge_file = None
            
            if video_timeline_sources:
                output_path = Path("output") / f"{context.video_name}_knowledge.md"
                output_path.parent.mkdir(exist_ok=True)
                
                with self._suppress_service_logging():
                    self.knowledge_generator.generate_combined_knowledge(
                        audio_timeline_path, video_timeline_sources, context.video_name, output_path
                    )
                knowledge_file = output_path
                
                sources_summary = ", ".join(video_timeline_sources.keys())
                logger.info(f"Generated comprehensive knowledge file: {output_path} (sources: {sources_summary})")
            else:
                logger.warning("No video timeline sources found for knowledge generation")
            
            # REMOVED: Old separate file generation logic - now using single combined file
            if contextual_files:
                latest_contextual = max(contextual_files, key=lambda f: f.stat().st_mtime)
                output_path = Path("output") / f"{context.video_name}_knowledge_contextualvlm.md"
                output_path.parent.mkdir(exist_ok=True)
                
                with self._suppress_service_logging():
                    self.knowledge_generator.generate_timeline_from_files(
                        audio_timeline_path, latest_contextual, context.video_name, output_path
                    )
                knowledge_file = output_path
                logger.info(f"Generated contextual knowledge: {output_path}")
            
            # Process noncontextual version if available  
            if noncontextual_files:
                latest_noncontextual = max(noncontextual_files, key=lambda f: f.stat().st_mtime)
                output_path = Path("output") / f"{context.video_name}_knowledge_noncontextualvlm.md"
                output_path.parent.mkdir(exist_ok=True)
                
                with self._suppress_service_logging():
                    self.knowledge_generator.generate_timeline_from_files(
                        audio_timeline_path, latest_noncontextual, context.video_name, output_path
                    )
                knowledge_file = output_path  # Keep most recent as primary
                logger.info(f"Generated noncontextual knowledge: {output_path}")
            
            # Process vid2seq version if available
            if vid2seq_files:
                latest_vid2seq = max(vid2seq_files, key=lambda f: f.stat().st_mtime)
                output_path = Path("output") / f"{context.video_name}_knowledge_vid2seq.md"
                output_path.parent.mkdir(exist_ok=True)
                
                with self._suppress_service_logging():
                    self.knowledge_generator.generate_timeline_from_files(
                        audio_timeline_path, latest_vid2seq, context.video_name, output_path
                    )
                knowledge_file = output_path  # Keep most recent as primary
                logger.info(f"Generated vid2seq knowledge: {output_path}")
            
            # Fallback to legacy timeline files if no modern files found
            if not contextual_files and not noncontextual_files and not vid2seq_files:
                legacy_timeline_files = list(video_timeline_path.glob("video_timeline.json"))
                if legacy_timeline_files:
                    latest_timeline = max(legacy_timeline_files, key=lambda f: f.stat().st_mtime)
                    output_path = Path("output") / f"{context.video_name}_knowledge.md"
                    output_path.parent.mkdir(exist_ok=True)
                    
                    with self._suppress_service_logging():
                        self.knowledge_generator.generate_timeline_from_files(
                            audio_timeline_path, latest_timeline, context.video_name, output_path
                        )
                    knowledge_file = output_path
                    logger.info(f"Generated legacy knowledge: {output_path}")
                else:
                    logger.warning("No video timeline found for knowledge generation")
                
            progress_callback('knowledge_generation', {'stage': 'completed', 'file': knowledge_file})
            
            # Success
            processing_time = time.time() - start_time
            self.circuit_breaker.record_success()
            
            return ProcessingResult(
                video_path=video_path,
                success=True,
                knowledge_file=knowledge_file,
                processing_time=processing_time,
                scenes_processed=context.scene_data['scene_count'] if context.scene_data else 0
            )
            
        except KeyboardInterrupt:
            raise
        except Exception as e:
            # Fail-fast: Any error fails entire video
            self.cleanup_failed_video(video_path, context if 'context' in locals() else None)
            processing_time = time.time() - start_time
            
            # Record failure for circuit breaker
            self.circuit_breaker.record_failure()
            
            progress_callback('error', {'error': str(e), 'video': video_path.name})
            
            return ProcessingResult(
                video_path=video_path,
                success=False,
                error=str(e),
                processing_time=processing_time
            )
            
    def cleanup_failed_video(self, video_path: Path, context: Optional[VideoProcessingContext]):
        """Clean up artifacts from failed video processing"""
        if context:
            context.cleanup_artifacts()
            
        # Force garbage collection to free GPU memory
        gc.collect()
        
    def should_abort_batch(self) -> bool:
        """Check if circuit breaker should abort batch processing"""
        return self.circuit_breaker.should_trip()
        
    def get_batch_statistics(self) -> Dict[str, int]:
        """Get current batch processing statistics"""
        return {
            'consecutive_failures': self.circuit_breaker.consecutive_failures,
            'total_processed': self.circuit_breaker.total_processed,
            'total_failures': self.circuit_breaker.total_failures,
            'failure_threshold': self.circuit_breaker.failure_threshold
        }