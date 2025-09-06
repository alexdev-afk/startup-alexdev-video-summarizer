"""
Enhanced Whisper Timeline Service

Clean implementation using enhanced timeline schema with proper file organization:
- Saves intermediate files in audio_analysis/ (raw whisper output)
- Saves timeline files in audio_timelines/ (processed timeline format)
- Avoids duplicate/invalid data
- Clean hierarchical structure
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import json

from utils.logger import get_logger
from utils.enhanced_timeline_schema import (
    EnhancedTimeline, TimelineSpan, TimelineEvent,
    create_speech_span, create_word_event, create_speaker_change_event
)
from .whisper_service import WhisperService, WhisperError

logger = get_logger(__name__)


class EnhancedWhisperTimelineService:
    """
    Enhanced Whisper Timeline Service using clean hierarchical schema
    
    Responsibilities:
    - Generate speech spans (Whisper's primary expertise)
    - Generate word events within spans  
    - Detect speaker changes
    - Save intermediate files for auditing
    - Avoid generating invalid cross-domain data
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize enhanced Whisper timeline service"""
        self.config = config
        self.whisper_service = WhisperService(config)
        
        # Timeline generation settings
        timeline_config = config.get('whisper_timeline', {})
        self.generate_word_events = timeline_config.get('generate_word_events', True)
        self.generate_speaker_events = timeline_config.get('generate_speaker_events', True)
        self.word_confidence_threshold = timeline_config.get('word_confidence_threshold', 0.3)
        
        logger.info(f"Enhanced Whisper timeline service initialized")
    
    def generate_timeline(self, audio_path: Path, scene_info: Optional[Dict] = None) -> EnhancedTimeline:
        """
        Generate clean hierarchical timeline from audio file
        
        Args:
            audio_path: Path to audio.wav file
            scene_info: Optional scene boundary information
            
        Returns:
            EnhancedTimeline object with clean structure
            
        Raises:
            WhisperError: If transcription fails
        """
        audio_path = Path(audio_path) if isinstance(audio_path, str) else audio_path
        logger.info(f"Generating enhanced Whisper timeline: {audio_path.name}")
        
        try:
            # Get transcription from WhisperService
            whisper_result = self.whisper_service.transcribe_audio(audio_path, scene_info)
            
            # Save intermediate analysis file for auditing
            self._save_intermediate_analysis(Path(audio_path), whisper_result)
            
            # Extract global data
            total_duration = self._extract_duration(whisper_result)
            full_transcript = whisper_result.get('transcript', '')
            speakers = whisper_result.get('speakers', [])
            language = whisper_result.get('language', 'auto-detected')
            language_confidence = whisper_result.get('language_probability', None)
            
            # Create enhanced timeline
            timeline = EnhancedTimeline(
                audio_file=audio_path.name,
                total_duration=total_duration
            )
            
            # Set global data
            timeline.set_global_data(
                transcript=full_transcript,
                speakers=speakers,
                language=language,
                language_confidence=language_confidence
            )
            
            # Convert segments to speech spans (Whisper's expertise)
            segments = whisper_result.get('segments', [])
            self._add_speech_spans(timeline, segments)
            
            # Add processing notes for auditing
            timeline.add_processing_note(f"Processed {len(segments)} VAD segments")
            timeline.add_processing_note(f"Generated {len(timeline.spans)} speech spans")
            
            logger.info(f"Enhanced timeline complete: {len(timeline.spans)} spans, {sum(len(s.events) for s in timeline.spans)} events")
            return timeline
            
        except Exception as e:
            logger.error(f"Enhanced timeline generation failed: {e}")
            raise WhisperError(f"EnhancedWhisperTimelineService failed: {str(e)}") from e
    
    def _extract_duration(self, whisper_result: Dict[str, Any]) -> float:
        """Extract total duration from whisper result"""
        segments = whisper_result.get('segments', [])
        if segments:
            last_segment = max(segments, key=lambda s: s.get('end', 0))
            return last_segment.get('end', 0)
        return 0.0
    
    def _add_speech_spans(self, timeline: EnhancedTimeline, segments: List[Dict[str, Any]]):
        """Add speech spans with nested word events - Whisper's primary expertise"""
        
        previous_speaker = None
        
        for i, segment in enumerate(segments):
            is_first_span = (i == 0)
            is_last_span = (i == len(segments) - 1)
            start_time = segment.get('start', 0)
            end_time = segment.get('end', 0)
            text = segment.get('text', '').strip()
            speaker = segment.get('speaker', 'Unknown')
            confidence = segment.get('confidence', 1.0)
            
            if not text or start_time >= end_time:
                continue
            
            # Create speech span (Whisper's primary output)
            speech_span = create_speech_span(start_time, end_time, text, speaker, confidence)
            
            # Add word events within this span
            if self.generate_word_events:
                self._add_word_events_to_span(speech_span, segment, is_first_span, is_last_span)
            
            # Add speaker change event if detected
            if (self.generate_speaker_events and 
                previous_speaker is not None and 
                speaker != previous_speaker):
                
                speaker_event = create_speaker_change_event(start_time, previous_speaker, speaker)
                speech_span.add_event(speaker_event, is_first_span=is_first_span, is_last_span=is_last_span)
            
            timeline.add_span(speech_span)
            previous_speaker = speaker
    
    def _add_word_events_to_span(self, span: TimelineSpan, segment: Dict[str, Any], is_first_span: bool = False, is_last_span: bool = False):
        """Add word events within a speech span"""
        words = segment.get('words', [])
        if not words:
            return
        
        speaker = segment.get('speaker', 'Unknown')
        
        for word_data in words:
            word_text = word_data.get('word', '').strip()
            if not word_text:
                continue
                
            word_confidence = word_data.get('confidence', word_data.get('probability', 1.0))
            word_timestamp = word_data.get('start', 0)
            
            # Skip low confidence words
            if word_confidence < self.word_confidence_threshold:
                continue
            
            # Create word event
            word_event = create_word_event(word_timestamp, word_text, word_confidence, speaker)
            
            try:
                span.add_event(word_event, is_first_span=is_first_span, is_last_span=is_last_span)
            except ValueError as e:
                # Word timestamp outside span range - log but continue
                logger.debug(f"Word event outside span range: {e}")
                continue
    
    def _save_intermediate_analysis(self, audio_path: Path, whisper_result: Dict[str, Any]):
        """Save intermediate Whisper analysis for auditing"""
        try:
            # Save in audio_analysis directory for raw analysis data
            analysis_dir = audio_path.parent / "audio_analysis"
            analysis_dir.mkdir(exist_ok=True)
            
            # Add metadata to analysis result
            timestamp = whisper_result.get('processing_time', 0)
            analysis_with_metadata = {
                **whisper_result,
                'analysis_type': 'whisper_transcription',
                'input_file': str(audio_path),
                'service_version': 'enhanced_whisper_timeline_v1.0.0',
                'processing_pipeline': {
                    'vad_stage': 'silero_vad',
                    'transcription_stage': 'whisper',
                    'diarization_stage': 'pyannote',
                    'output_format': 'enhanced_timeline'
                }
            }
            
            # Save raw analysis
            analysis_file = analysis_dir / "whisper_analysis.json" 
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(analysis_with_metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Intermediate analysis saved: {analysis_file}")
            
        except Exception as e:
            logger.error(f"Failed to save intermediate analysis: {e}")
            # Don't raise - this is supplementary
    
    def generate_and_save(self, audio_path: Path, scene_info: Optional[Dict] = None) -> EnhancedTimeline:
        """Generate timeline and save to proper directories"""
        timeline = self.generate_timeline(audio_path, scene_info)
        
        # Create directory structure
        build_dir = Path(audio_path).parent
        timelines_dir = build_dir / "audio_timelines"
        timelines_dir.mkdir(exist_ok=True)
        
        # Save main enhanced timeline
        main_timeline_path = timelines_dir / "whisper_timeline.json"
        timeline.save_to_file(str(main_timeline_path))
        
        # Save intermediate files for auditing
        timeline.save_intermediate_files(timelines_dir)
        
        logger.info(f"Enhanced timeline saved: {main_timeline_path}")
        logger.info(f"Intermediate files saved in: {timelines_dir}")
        
        return timeline
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the enhanced timeline service"""
        whisper_info = self.whisper_service.get_model_info()
        return {
            **whisper_info,
            'enhanced_features': {
                'clean_hierarchical_structure': True,
                'speech_spans_only': True,  # No invalid cross-domain data
                'nested_word_events': self.generate_word_events,
                'speaker_change_detection': self.generate_speaker_events,
                'intermediate_file_auditing': True,
                'proper_directory_structure': True
            },
            'service_type': 'enhanced_whisper_timeline_service',
            'output_directories': {
                'raw_analysis': 'audio_analysis/',
                'timeline_data': 'audio_timelines/',
                'intermediate_auditing': 'audio_timelines/*_intermediate.json'
            }
        }
    
    def cleanup(self):
        """Cleanup resources"""
        self.whisper_service.unload_model()
        logger.debug("Enhanced Whisper timeline service cleanup complete")


# Export for easy importing
__all__ = ['EnhancedWhisperTimelineService']