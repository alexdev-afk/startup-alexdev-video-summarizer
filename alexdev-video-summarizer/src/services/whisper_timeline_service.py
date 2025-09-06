"""
Whisper Timeline Service

Converts WhisperService output into ServiceTimeline format for integration with
TimelineMergerService. Produces word-level events, VAD span events, and maintains
chronological ordering for institutional knowledge extraction.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import json

from utils.logger import get_logger
from utils.timeline_schema import ServiceTimeline, TimelineEvent, TimelineSpan
from .whisper_service import WhisperService, WhisperError

logger = get_logger(__name__)


class WhisperTimelineService:
    """
    Timeline wrapper for WhisperService that converts transcription output
    to ServiceTimeline format with word-level events and VAD span events
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Whisper timeline service"""
        self.config = config
        self.whisper_service = WhisperService(config)
        
        # Timeline generation settings
        timeline_config = config.get('whisper_timeline', {})
        self.generate_word_events = timeline_config.get('generate_word_events', True)
        self.generate_vad_spans = timeline_config.get('generate_vad_spans', True)  
        self.generate_speaker_events = timeline_config.get('generate_speaker_events', True)
        self.word_confidence_threshold = timeline_config.get('word_confidence_threshold', 0.3)
        
        logger.info(f"Whisper timeline service initialized - word_events: {self.generate_word_events}, vad_spans: {self.generate_vad_spans}")
    
    def generate_timeline(self, audio_path: Path, scene_info: Optional[Dict] = None) -> ServiceTimeline:
        """
        Generate ServiceTimeline from audio file using WhisperService
        
        Args:
            audio_path: Path to audio.wav file
            scene_info: Optional scene boundary information
            
        Returns:
            ServiceTimeline object with events and spans
            
        Raises:
            WhisperError: If transcription fails
        """
        audio_path = Path(audio_path) if isinstance(audio_path, str) else audio_path
        logger.info(f"Generating Whisper timeline: {audio_path.name}")
        
        try:
            # Get transcription from WhisperService
            whisper_result = self.whisper_service.transcribe_audio(audio_path, scene_info)
            
            # Get audio duration from whisper result
            total_duration = 0.0
            if whisper_result.get('segments'):
                last_segment = max(whisper_result['segments'], key=lambda s: s.get('end', 0))
                total_duration = last_segment.get('end', 0)
            
            # Create ServiceTimeline
            timeline = ServiceTimeline(
                source="whisper",
                audio_file=audio_path.name,
                total_duration=total_duration
            )
            
            # Convert segments to timeline events and spans
            self._add_word_events(timeline, whisper_result['segments'])
            self._add_vad_spans(timeline, whisper_result['segments'])
            self._add_speaker_events(timeline, whisper_result['segments'])
            
            logger.info(f"Timeline generation complete: {len(timeline.events)} events, {len(timeline.spans)} spans")
            return timeline
            
        except Exception as e:
            logger.error(f"Timeline generation failed: {e}")
            raise WhisperError(f"WhisperTimelineService failed: {str(e)}") from e
    
    def _add_word_events(self, timeline: ServiceTimeline, segments: List[Dict[str, Any]]):
        """Add word-level timeline events from Whisper segments"""
        if not self.generate_word_events:
            return
        
        for segment in segments:
            words = segment.get('words', [])
            if not words:
                continue
                
            for word_data in words:
                word_text = word_data.get('word', '').strip()
                if not word_text:
                    continue
                    
                word_confidence = word_data.get('confidence', word_data.get('probability', 1.0))
                
                # Skip low confidence words if threshold is set
                if word_confidence < self.word_confidence_threshold:
                    continue
                
                # Create word event
                word_event = TimelineEvent(
                    timestamp=word_data.get('start', 0),
                    description=f"Word: '{word_text}'",
                    category="speech",
                    source="whisper",
                    confidence=word_confidence,
                    details={
                        "word": word_text,
                        "end_time": word_data.get('end', 0),
                        "duration": word_data.get('end', 0) - word_data.get('start', 0),
                        "speaker": segment.get('speaker', 'Unknown')
                    }
                )
                timeline.add_event(word_event)
    
    def _add_vad_spans(self, timeline: ServiceTimeline, segments: List[Dict[str, Any]]):
        """Add VAD segment spans (natural speech segments)"""
        if not self.generate_vad_spans:
            return
        
        for segment in segments:
            start_time = segment.get('start', 0)
            end_time = segment.get('end', 0)
            text = segment.get('text', '').strip()
            speaker = segment.get('speaker', 'Unknown')
            confidence = segment.get('confidence', 1.0)
            
            if not text or start_time >= end_time:
                continue
            
            # Determine span description based on content length
            word_count = len(text.split())
            if word_count <= 3:
                description = f"{speaker}: '{text}'"
            else:
                # Truncate long text for description
                truncated_text = ' '.join(text.split()[:8])
                if word_count > 8:
                    truncated_text += "..."
                description = f"{speaker}: '{truncated_text}'"
            
            # Create VAD span
            vad_span = TimelineSpan(
                start=start_time,
                end=end_time,
                description=description,
                category="speech", 
                source="whisper",
                confidence=confidence,
                details={
                    "full_text": text,
                    "speaker": speaker,
                    "word_count": word_count,
                    "vad_chunk_id": segment.get('vad_chunk_id', 0),
                    "language": "auto-detected"  # Would be from whisper_result['language']
                }
            )
            timeline.add_span(vad_span)
    
    def _add_speaker_events(self, timeline: ServiceTimeline, segments: List[Dict[str, Any]]):
        """Add speaker change events"""
        if not self.generate_speaker_events or len(segments) <= 1:
            return
        
        previous_speaker = None
        for segment in segments:
            current_speaker = segment.get('speaker', 'Unknown')
            start_time = segment.get('start', 0)
            
            # Detect speaker changes
            if previous_speaker is not None and current_speaker != previous_speaker:
                speaker_event = TimelineEvent(
                    timestamp=start_time,
                    description=f"Speaker change: {previous_speaker} → {current_speaker}",
                    category="speech",
                    source="whisper",
                    confidence=0.85,  # Speaker diarization confidence
                    details={
                        "previous_speaker": previous_speaker,
                        "new_speaker": current_speaker,
                        "change_type": "speaker_transition"
                    }
                )
                timeline.add_event(speaker_event)
            
            previous_speaker = current_speaker
    
    def generate_and_save(self, audio_path: Path, output_path: str, scene_info: Optional[Dict] = None) -> ServiceTimeline:
        """Generate timeline and save to file"""
        timeline = self.generate_timeline(audio_path, scene_info)
        timeline.save_to_file(output_path)
        logger.info(f"Whisper timeline saved to: {output_path}")
        return timeline
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the timeline service"""
        whisper_info = self.whisper_service.get_model_info()
        return {
            **whisper_info,
            'timeline_features': {
                'word_level_events': self.generate_word_events,
                'vad_span_events': self.generate_vad_spans,
                'speaker_change_events': self.generate_speaker_events,
                'word_confidence_threshold': self.word_confidence_threshold
            },
            'service_type': 'whisper_timeline_service'
        }
    
    def cleanup(self):
        """Cleanup resources"""
        self.whisper_service.unload_model()
        logger.debug("Whisper timeline service cleanup complete")


# Export for easy importing
__all__ = ['WhisperTimelineService']