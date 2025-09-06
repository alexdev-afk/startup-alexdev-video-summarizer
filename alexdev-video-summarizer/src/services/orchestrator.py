"""
Video Processing Orchestrator

Coordinates the complete video processing pipeline:
FFmpeg → PySceneDetect → Dual Pipeline Processing → Knowledge Generation

Implements fail-fast per video and circuit breaker for batch processing.
"""

import time
import gc
from pathlib import Path
from typing import Dict, Any, Callable, Optional
from dataclasses import dataclass

from services.ffmpeg_service import FFmpegService
from services.scene_detection_service import SceneDetectionService
from services.demucs_service import DemucsService
from services.demucs_audio_coordinator import DemucsAudioCoordinatorService
from services.internvl3_timeline_service import InternVL3TimelineService
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
        
        # Current video processing service
        self.video_service = InternVL3TimelineService(config)
        
        self.knowledge_generator = KnowledgeGenerator(config)
        
        # Circuit breaker for batch processing
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=config.get('circuit_breaker_threshold', 3)
        )
        
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
            context.audio_path, context.processed_video_path = self.ffmpeg_service.extract_streams(video_path)
            
            if not context.validate_ffmpeg_output():
                raise Exception("FFmpeg audio/video extraction failed")
            progress_callback('ffmpeg', {'stage': 'completed', 'tool': 'FFmpeg'})
            
            # Step 2: Scene Detection (CRITICAL)
            progress_callback('scene_detection', {'stage': 'starting', 'tool': 'PySceneDetect'})
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
            
            # Step 5: InternVL3 Frame-Based Video Processing
            progress_callback('video_processing', {
                'stage': 'starting', 
                'processing_mode': 'internvl3_frames'
            })
            
            # Process all extracted frames with InternVL3
            video_timeline = self.video_service.generate_and_save(str(context.processed_video_path), None)
            
            progress_callback('video_processing', {
                'stage': 'completed',
                'events_generated': len(video_timeline.events) if video_timeline else 0,
                'total_duration': video_timeline.total_duration if video_timeline else 0
            })
            
            # Cleanup video service to free VRAM
            if hasattr(self.video_service, 'cleanup'):
                self.video_service.cleanup()
            
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
            
            # Find the timeline files for knowledge generation
            audio_timeline_path = context.build_directory / "audio_timelines" / "combined_audio_timeline.json"
            video_timeline_path = context.build_directory / "video_timelines" 
            
            # Find the most recent video timeline file
            video_timeline_files = list(video_timeline_path.glob("*_timeline.json"))
            if video_timeline_files:
                latest_video_timeline = max(video_timeline_files, key=lambda f: f.stat().st_mtime)
                output_path = Path("output") / f"{context.video_name}_knowledge.md"
                output_path.parent.mkdir(exist_ok=True)
                
                self.knowledge_generator.generate_timeline_from_files(
                    audio_timeline_path, latest_video_timeline, context.video_name, output_path
                )
                knowledge_file = output_path
            else:
                logger.warning("No video timeline found for knowledge generation")
                knowledge_file = None
                
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