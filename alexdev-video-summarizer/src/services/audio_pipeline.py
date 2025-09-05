"""
Audio Pipeline Controller - Updated to use real timeline services.

Handles Whisper transcription → LibROSA timeline → pyAudio timeline → Timeline merger.
Generates event-based timelines instead of fake interpretive analysis.
"""

import time
from pathlib import Path
from typing import Dict, Any

from services.whisper_service import WhisperService, WhisperError
from services.librosa_timeline_service import LibROSATimelineService, LibROSATimelineError
from services.pyaudio_timeline_service import PyAudioTimelineService, PyAudioTimelineError
from services.timeline_merger_service import TimelineMergerService
from utils.logger import get_logger

logger = get_logger(__name__)


class AudioPipelineController:
    """Audio pipeline controller using real timeline services - fake analysis removed"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize audio pipeline controller with real timeline services
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.audio_config = config.get('audio_pipeline', {})
        
        # Initialize real timeline services (not fake analysis)
        self.whisper_service = WhisperService(config)
        self.librosa_timeline_service = LibROSATimelineService(config)
        self.pyaudio_timeline_service = PyAudioTimelineService(config)
        self.timeline_merger = TimelineMergerService(config)
        
        logger.info("Audio timeline pipeline controller initialized - fake analysis removed")
    
    def process_scene(self, scene: Dict[str, Any], context) -> Dict[str, Any]:
        """
        Process audio through real timeline services - generates event-based timelines
        
        Pipeline: Whisper → LibROSA Timeline → pyAudio Timeline → Timeline Merger
        
        Args:
            scene: Scene boundary data (ignored - timeline services process full audio)
            context: Video processing context
            
        Returns:
            Timeline-based audio analysis results (not fake interpretive analysis)
        """
        logger.info(f"Processing audio through timeline pipeline (not fake analysis)")
        
        try:
            if not (hasattr(context, 'audio_path') and context.audio_path):
                logger.warning(f"No audio file available for timeline processing")
                return {'error': 'No audio file available', 'timeline': []}
            
            # Step 1: Whisper transcription (kept as is - already real)
            whisper_result = self.whisper_service.transcribe_audio(context.audio_path)
            
            # Step 2: Generate LibROSA timeline (real music events)
            librosa_timeline = self.librosa_timeline_service.generate_timeline(context.audio_path)
            
            # Step 3: Generate pyAudioAnalysis timeline (real audio events)  
            pyaudio_timeline = self.pyaudio_timeline_service.generate_timeline(context.audio_path)
            
            # Step 4: Merge timelines chronologically
            merged_timeline = self.timeline_merger.merge_timelines([
                whisper_result,  # Already in timeline format
                librosa_timeline,
                pyaudio_timeline
            ])
            
            logger.info(f"Timeline pipeline complete - {len(merged_timeline.timeline)} events generated")
            
            return {
                'processing_type': 'event_based_timeline',
                'merged_timeline': merged_timeline,
                'source_timelines': {
                    'whisper': whisper_result,
                    'librosa': librosa_timeline,
                    'pyaudio': pyaudio_timeline
                },
                'note': 'Real timeline events - fake interpretive analysis removed'
            }
            
        except Exception as e:
            logger.error(f"Timeline pipeline failed: {e}")
            return {
                'processing_type': 'timeline_error',
                'error': str(e),
                'timeline': [],
                'note': 'Timeline processing failed - no fake analysis fallback'
            }
    
    # REMOVED: Old fallback methods for fake analysis services
    
