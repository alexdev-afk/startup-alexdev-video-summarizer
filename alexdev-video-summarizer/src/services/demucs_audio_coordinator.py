"""
Demucs Audio Coordinator Service

Coordinates audio services with separated tracks from Demucs:
- Whisper: audio.wav (perfect transcription)  
- LibROSA: no_vocals.wav (pure music analysis)
- PyAudio music: no_vocals.wav (genre, energy)
- PyAudio voice: vocals.wav (emotion, speech features)

This eliminates the need for complex heuristic filtering in timeline merger.
"""

import time
from pathlib import Path
from typing import Dict, Any, List
from utils.logger import get_logger
from utils.processing_context import VideoProcessingContext

# Audio timeline services
from services.enhanced_whisper_timeline_service import EnhancedWhisperTimelineService
from services.librosa_timeline_service import LibROSATimelineService
from services.pyaudio_timeline_service import PyAudioTimelineService
from services.audio_timeline_merger_service import AudioTimelineMergerService

logger = get_logger(__name__)


class DemucsAudioCoordinatorError(Exception):
    """Demucs audio coordination error"""
    pass


class DemucsAudioCoordinatorService:
    """Coordinates audio services with Demucs-separated tracks"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Demucs audio coordinator"""
        self.config = config
        
        # Initialize audio services
        self.whisper_service = EnhancedWhisperTimelineService(config)
        self.librosa_service = LibROSATimelineService(config)
        self.pyaudio_service = PyAudioTimelineService(config)
        
        # BREAKTHROUGH: Updated merger with simple approach (no 692-line heuristics!)
        self.timeline_merger = AudioTimelineMergerService(config)
        
        logger.info("DemucsAudioCoordinatorService initialized with SIMPLE merger (no heuristics needed)")
    
    def process_all_audio_timelines(self, context: VideoProcessingContext) -> List[Any]:
        """
        Process all audio services with appropriate separated tracks
        
        Args:
            context: VideoProcessingContext with vocals_path and no_vocals_path
            
        Returns:
            List of enhanced timelines from all services
        """
        start_time = time.time()
        timelines = []
        
        # Verify Demucs outputs exist
        if not context.vocals_path or not context.vocals_path.exists():
            raise DemucsAudioCoordinatorError(f"Vocals track not found: {context.vocals_path}")
        
        if not context.no_vocals_path or not context.no_vocals_path.exists():
            raise DemucsAudioCoordinatorError(f"Instrumentals track not found: {context.no_vocals_path}")
        
        # Original audio file (for Whisper)
        original_audio = context.get_audio_file_path()
        if not original_audio.exists():
            raise DemucsAudioCoordinatorError(f"Original audio not found: {original_audio}")
        
        try:
            # Get optimization configs
            optimization_config = self.config.get('audio_optimization', {})
            
            # 1. Whisper Timeline (using original audio for perfect transcription)
            logger.info("Processing Whisper with original audio for perfect transcription...")
            whisper_timeline = self.whisper_service.generate_and_save(str(original_audio), source_tag="whisper_voice")
            
            timelines.append(whisper_timeline)
            logger.info(f"Whisper timeline: {len(whisper_timeline.events)} events, {len(whisper_timeline.spans)} spans")
            
            # 2. LibROSA Timeline (using no_vocals.wav for pure music analysis - OPTIMIZED)
            logger.info("Processing LibROSA with instrumentals track for optimized music analysis...")
            librosa_config = optimization_config.get('librosa_music', {})
            librosa_timeline = self.librosa_service.generate_and_save(str(context.no_vocals_path), source_tag="librosa_music", optimization=librosa_config)
            
            timelines.append(librosa_timeline)
            logger.info(f"LibROSA timeline: {len(librosa_timeline.events)} events, {len(librosa_timeline.spans)} spans")
            
            # 3. PyAudio Music: no_vocals.wav (genre, energy - OPTIMIZED)
            logger.info("Processing PyAudio with instrumentals track for optimized music features...")
            pyaudio_music_config = optimization_config.get('pyaudio_music', {})
            pyaudio_music_timeline = self.pyaudio_service.generate_and_save(str(context.no_vocals_path), source_tag="pyaudio_music", optimization=pyaudio_music_config)
            
            timelines.append(pyaudio_music_timeline)
            logger.info(f"PyAudio music timeline: {len(pyaudio_music_timeline.events)} events, {len(pyaudio_music_timeline.spans)} spans")
            
            # PyAudio Voice: vocals.wav (emotion, speech features - OPTIMIZED)
            logger.info("Processing PyAudio with vocals track for optimized voice features...")
            pyaudio_voice_config = optimization_config.get('pyaudio_voice', {})
            pyaudio_voice_timeline = self.pyaudio_service.generate_and_save(str(context.vocals_path), source_tag="pyaudio_voice", optimization=pyaudio_voice_config)
            
            timelines.append(pyaudio_voice_timeline)
            logger.info(f"PyAudio voice timeline: {len(pyaudio_voice_timeline.events)} events, {len(pyaudio_voice_timeline.spans)} spans")
            
            processing_time = time.time() - start_time
            logger.info(f"Demucs audio coordination complete: 4 timelines in {processing_time:.2f}s")
            logger.info("Source traceability: whisper_voice, librosa_music, pyaudio_music, pyaudio_voice")
            
            return timelines
            
        except Exception as e:
            # Cleanup models on error
            self._cleanup_services()
            logger.error(f"Demucs audio coordination failed: {e}")
            raise DemucsAudioCoordinatorError(f"Audio coordination failed: {str(e)}") from e
        finally:
            # Always cleanup after processing to free VRAM
            self._cleanup_services()
    
    def save_individual_timelines(self, timelines: List[Any], context: VideoProcessingContext):
        """
        Individual timeline files are already saved by each service with proper source tags.
        This method is now redundant since services save their own files with source-tagged names.
        """
        logger.info("Individual timelines already saved by services with proper source tagging:")
        for timeline in timelines:
            logger.info(f"  - Timeline with sources: {timeline.sources_used}")
    
    def create_combined_timeline(self, timelines: List[Any], context: VideoProcessingContext) -> Any:
        """
        Create combined audio timeline using SIMPLE merger (BREAKTHROUGH)
        
        Args:
            timelines: List of individual audio timelines from clean Demucs sources
            context: VideoProcessingContext for output path
            
        Returns:
            Combined timeline with chronologically ordered events
        """
        # Create output path
        timelines_dir = context.build_directory / "audio_timelines" 
        combined_path = timelines_dir / "combined_audio_timeline.json"
        
        logger.info("BREAKTHROUGH: Creating combined timeline with simple chronological merger...")
        
        # Use updated merger with simple approach (no complex heuristics!)
        combined_timeline = self.timeline_merger.create_combined_audio_timeline(
            timelines=timelines,
            output_path=str(combined_path)
        )
        
        logger.info(f"BREAKTHROUGH COMPLETE: {len(combined_timeline.events)} clean events, no complex filtering needed!")
        return combined_timeline
    
    def _cleanup_services(self):
        """Cleanup all audio service models to free VRAM"""
        try:
            logger.info("Cleaning up audio service models to free VRAM...")
            
            # Cleanup Whisper service (biggest VRAM user)
            if hasattr(self.whisper_service, 'cleanup'):
                self.whisper_service.cleanup()
            elif hasattr(self.whisper_service, 'unload_model'):
                self.whisper_service.unload_model()
                
            # Cleanup LibROSA service
            if hasattr(self.librosa_service, 'cleanup'):
                self.librosa_service.cleanup()
                
            # Cleanup PyAudio service  
            if hasattr(self.pyaudio_service, 'cleanup'):
                self.pyaudio_service.cleanup()
                
            logger.info("Audio service model cleanup complete - VRAM freed")
            
        except Exception as e:
            logger.warning(f"Audio service cleanup warning: {e}")
    
    def cleanup(self):
        """External cleanup method"""
        self._cleanup_services()