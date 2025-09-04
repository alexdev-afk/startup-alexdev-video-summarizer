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
from services.audio_pipeline import AudioPipelineController
from services.gpu_pipeline import VideoGPUPipelineController
from services.cpu_pipeline import VideoCPUPipelineController
from services.knowledge_generator import KnowledgeBaseGenerator
from utils.processing_context import VideoProcessingContext
from utils.circuit_breaker import CircuitBreaker


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
        self.audio_pipeline = AudioPipelineController(config)
        self.video_gpu_pipeline = VideoGPUPipelineController(config)
        self.video_cpu_pipeline = VideoCPUPipelineController(config)
        self.knowledge_generator = KnowledgeBaseGenerator(config)
        
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
            context.audio_path, context.video_path = self.ffmpeg_service.extract_streams(video_path)
            
            if not context.validate_ffmpeg_output():
                raise Exception("FFmpeg audio/video extraction failed")
            progress_callback('ffmpeg', {'stage': 'completed', 'tool': 'FFmpeg'})
            
            # Step 2: Scene Detection (CRITICAL)
            progress_callback('scene_detection', {'stage': 'starting', 'tool': 'PySceneDetect'})
            context.scene_data = self.scene_service.analyze_video_scenes(context.video_path)
            
            if context.scene_data['scene_count'] == 0:
                raise Exception("No scenes detected in video")
            progress_callback('scene_detection', {
                'stage': 'completed', 
                'tool': 'PySceneDetect',
                'scene_count': context.scene_data['scene_count']
            })
            
            # Step 2.5: Scene Splitting Coordination (Optional - for advanced processing)
            progress_callback('scene_splitting', {'stage': 'starting'})
            scene_files = self.scene_service.coordinate_scene_splitting(
                video_path, 
                context.scene_data['boundaries'], 
                self.ffmpeg_service
            )
            context.scene_data['scene_files'] = scene_files
            progress_callback('scene_splitting', {
                'stage': 'completed',
                'files_created': len(scene_files)
            })
            
            # Step 3: Per-Scene Processing (3 pipelines - FAIL ON ANY TOOL FAILURE)
            for i, scene in enumerate(context.scene_data['scenes'], 1):
                progress_callback('scene_processing', {
                    'stage': 'starting',
                    'scene': i,
                    'total_scenes': len(context.scene_data['scenes'])
                })
                
                # Audio Pipeline: Whisper → LibROSA → pyAudioAnalysis
                audio_results = self.audio_pipeline.process_scene(scene, context)
                progress_callback('audio_pipeline', {
                    'stage': 'completed',
                    'scene': i,
                    'results': audio_results
                })
                
                # Video GPU Pipeline: YOLO → EasyOCR
                video_gpu_results = self.video_gpu_pipeline.process_scene(scene, context)
                progress_callback('video_gpu_pipeline', {
                    'stage': 'completed',
                    'scene': i,
                    'results': video_gpu_results
                })
                
                # Video CPU Pipeline: OpenCV
                video_cpu_results = self.video_cpu_pipeline.process_scene(scene, context)
                progress_callback('video_cpu_pipeline', {
                    'stage': 'completed',
                    'scene': i,
                    'results': video_cpu_results
                })
                
                # Store combined results from all 3 pipelines
                context.store_scene_analysis(scene['scene_id'], audio_results, video_gpu_results, video_cpu_results)
                
                progress_callback('scene_processing', {
                    'stage': 'completed',
                    'scene': i,
                    'total_scenes': len(context.scene_data['scenes'])
                })
            
            # Step 4: Knowledge Base Generation
            progress_callback('knowledge_generation', {'stage': 'starting'})
            knowledge_file = self.knowledge_generator.generate_video_knowledge_base(
                video_path.stem, context.get_all_analysis()
            )
            progress_callback('knowledge_generation', {'stage': 'completed', 'file': knowledge_file})
            
            # Success
            processing_time = time.time() - start_time
            self.circuit_breaker.record_success()
            
            return ProcessingResult(
                video_path=video_path,
                success=True,
                knowledge_file=knowledge_file,
                processing_time=processing_time,
                scenes_processed=len(context.scene_data['scenes'])
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