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
        Generate clean hierarchical timeline from audio file (for backward compatibility)
        
        Args:
            audio_path: Path to audio.wav file
            scene_info: Optional scene boundary information
            
        Returns:
            EnhancedTimeline object with clean structure
            
        Raises:
            WhisperError: If transcription fails
        """
        # Delegate to generate_and_save without saving
        return self._generate_timeline_internal(audio_path, scene_info)
    
    def _generate_timeline_internal(self, audio_path: Path, scene_info: Optional[Dict] = None) -> EnhancedTimeline:
        """Internal method for timeline generation"""
        audio_path = Path(audio_path) if isinstance(audio_path, str) else audio_path
        logger.info(f"Generating enhanced Whisper timeline: {audio_path.name}")
        
        try:
            # Get transcription from WhisperService
            whisper_result = self.whisper_service.transcribe_audio(audio_path, scene_info)
            
            # Create enhanced timeline first
            timeline = self._create_enhanced_timeline(whisper_result, audio_path, \"whisper\")
            
            return timeline
            
        except Exception as e:
            logger.error(f"Enhanced timeline generation failed: {e}")
            raise WhisperError(f"Enhanced timeline generation failed: {str(e)}") from e
    
    def _create_enhanced_timeline(self, whisper_result: Dict[str, Any], audio_path: Path, source_tag: str) -> EnhancedTimeline:
        """Create enhanced timeline from whisper result"""
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
        self._add_speech_spans(timeline, segments, source_tag)
        
        # Add processing notes for auditing
        timeline.add_processing_note(f"Processed {len(segments)} Whisper segments from VAD regions")
        timeline.add_processing_note(f"VAD reconstruction: {len(timeline.spans)} timeline spans (1:1 with VAD regions)")
        timeline.add_processing_note(f"Using WhisperWithVAD approach for optimal transcription quality")
        
        logger.info(f"Enhanced timeline complete: {len(timeline.spans)} spans, {sum(len(s.events) for s in timeline.spans)} events")
        return timeline
    
    def _extract_duration(self, whisper_result: Dict[str, Any]) -> float:
        """Extract total duration from whisper result"""
        segments = whisper_result.get('segments', [])
        if segments:
            last_segment = max(segments, key=lambda s: s.get('end', 0))
            return last_segment.get('end', 0)
        return 0.0
    
    def _add_speech_spans(self, timeline: EnhancedTimeline, segments: List[Dict[str, Any]], source_tag: str):
        """
        Add speech spans with VAD region reconstruction
        
        Groups multiple Whisper segments back into single VAD regions for proper granularity.
        This follows WhisperWithVAD approach: accept Whisper's internal segmentation but 
        reconstruct the original VAD boundaries for timeline structure.
        """
        
        assert source_tag, \"source_tag is required\"\n        \n        # Group segments by VAD chunk ID to maintain 1:1 VAD region to timeline span mapping
        vad_groups = {}
        for segment in segments:
            vad_chunk_id = segment.get('vad_chunk_id', 0)
            if vad_chunk_id not in vad_groups:
                vad_groups[vad_chunk_id] = []
            vad_groups[vad_chunk_id].append(segment)
        
        logger.debug(f"Reconstructing {len(segments)} Whisper segments into {len(vad_groups)} VAD regions")
        
        previous_speaker = None
        
        # Process each VAD group as one timeline span (WhisperWithVAD approach)
        for chunk_id in sorted(vad_groups.keys()):
            chunk_segments = vad_groups[chunk_id]
            is_first_span = (chunk_id == min(vad_groups.keys()))
            is_last_span = (chunk_id == max(vad_groups.keys()))
            
            # Combine all segments in this VAD region like WhisperWithVAD does
            combined_text_parts = []
            all_words = []
            min_start = float('inf')
            max_end = float('-inf')
            weighted_confidence_sum = 0.0
            total_duration = 0.0
            speaker_durations = {}
            
            for segment in chunk_segments:
                start_time = segment.get('start', 0)
                end_time = segment.get('end', 0)
                text = segment.get('text', '').strip()
                speaker = segment.get('speaker', 'Unknown')
                confidence = segment.get('confidence', 1.0)
                words = segment.get('words', [])
                
                if not text or start_time >= end_time:
                    continue
                    
                combined_text_parts.append(text)
                all_words.extend(words)
                min_start = min(min_start, start_time)
                max_end = max(max_end, end_time)
                
                # Weight confidence by segment duration for accuracy
                segment_duration = end_time - start_time
                weighted_confidence_sum += confidence * segment_duration
                total_duration += segment_duration
                
                # Track speaker duration to determine primary speaker for VAD region
                speaker_durations[speaker] = speaker_durations.get(speaker, 0) + segment_duration
            
            if not combined_text_parts:
                continue
                
            # Determine primary speaker (most speaking time in this VAD region)
            primary_speaker = max(speaker_durations, key=speaker_durations.get) if speaker_durations else 'Unknown'
            
            # Calculate weighted average confidence
            final_confidence = weighted_confidence_sum / total_duration if total_duration > 0 else 1.0
            
            # Combine text with proper spacing (preserve natural flow)
            combined_text = ' '.join(combined_text_parts).strip()
            
            # Create speech span for the entire VAD region 
            speech_span = create_speech_span(min_start, max_end, combined_text, primary_speaker, final_confidence, source_tag)
            
            # Add all word events within this span
            if self.generate_word_events:
                for word_data in all_words:
                    word_text = word_data.get('word', '').strip()
                    if not word_text:
                        continue
                        
                    word_confidence = word_data.get('confidence', word_data.get('probability', 1.0))
                    word_timestamp = word_data.get('start', 0)
                    
                    # Skip low confidence words
                    if word_confidence < self.word_confidence_threshold:
                        continue
                    
                    # Use primary speaker for all words in this VAD region for consistency
                    word_event = create_word_event(word_timestamp, word_text, word_confidence, primary_speaker, source_tag)
                    
                    try:
                        speech_span.add_event(word_event, is_first_span=is_first_span, is_last_span=is_last_span)
                    except ValueError:
                        # Word timestamp outside span range - log but continue
                        logger.debug(f"Word event outside VAD span range: {word_text} at {word_timestamp:.2f}s")
                        continue
            
            # Add speaker change event if detected between VAD regions
            if (self.generate_speaker_events and 
                previous_speaker is not None and 
                primary_speaker != previous_speaker):
                
                speaker_event = create_speaker_change_event(min_start, previous_speaker, primary_speaker, source_tag)
                speech_span.add_event(speaker_event, is_first_span=is_first_span, is_last_span=is_last_span)
            
            timeline.add_span(speech_span)
            previous_speaker = primary_speaker
            
        logger.debug(f"VAD reconstruction complete: {len(vad_groups)} VAD regions -> {len(timeline.spans)} timeline spans")
    
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
            word_event = create_word_event(word_timestamp, word_text, word_confidence, speaker, source_tag)
            
            try:
                span.add_event(word_event, is_first_span=is_first_span, is_last_span=is_last_span)
            except ValueError as e:
                # Word timestamp outside span range - log but continue
                logger.debug(f"Word event outside span range: {e}")
                continue
    
    def _save_intermediate_analysis(self, audio_path: Path, whisper_result: Dict[str, Any], timeline: Optional[EnhancedTimeline] = None, source_tag: Optional[str] = None):
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
            
            # Add timeline event data if available for auditing
            if timeline:
                analysis_with_metadata.update({
                    'generated_timeline_events': [
                        {
                            "timestamp": event.timestamp,
                            "description": event.description,
                            "event_type": event.description.lower().replace(" ", "_"),
                            "source": event.source,
                            "confidence": event.confidence,
                            "details": event.details
                        } for event in timeline.events
                    ],
                    'generated_timeline_spans': [
                        {
                            "start": span.start,
                            "end": span.end,
                            "description": span.description,
                            "source": span.source,
                            "confidence": span.confidence,
                            "details": span.details,
                            "events_count": len(span.events)
                        } for span in timeline.spans
                    ]
                })
            
            # Save raw analysis with source tag
            analysis_filename = f"{source_tag or 'whisper'}_analysis.json"
            analysis_file = analysis_dir / analysis_filename 
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(analysis_with_metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Intermediate analysis saved: {analysis_file}")
            
        except Exception as e:
            logger.error(f"Failed to save intermediate analysis: {e}")
            # Don't raise - this is supplementary
    
    def generate_and_save(self, audio_path: Path, scene_info: Optional[Dict] = None, source_tag: Optional[str] = None) -> EnhancedTimeline:
        """Generate timeline and save to proper directories"""
        audio_path = Path(audio_path) if isinstance(audio_path, str) else audio_path
        logger.info(f"Generating enhanced Whisper timeline: {audio_path.name}")
        
        try:
            # Get transcription from WhisperService
            whisper_result = self.whisper_service.transcribe_audio(audio_path, scene_info)
            
            # Create enhanced timeline with source tag
            assert source_tag, "source_tag is required for timeline generation"
            timeline = self._create_enhanced_timeline(whisper_result, audio_path, source_tag)
            
            # Save intermediate analysis file for auditing with timeline data
            self._save_intermediate_analysis(Path(audio_path), whisper_result, timeline, source_tag)
            
        except Exception as e:
            logger.error(f"Enhanced timeline generation failed: {e}")
            raise WhisperError(f"Enhanced timeline generation failed: {str(e)}") from e
        
        # Apply source tag if provided
        if source_tag:
            # Replace default "whisper" with the specific tag
            if "whisper" in timeline.sources_used:
                timeline.sources_used = [source_tag] + [s for s in timeline.sources_used if s != "whisper"]
            else:
                timeline.sources_used.append(source_tag)
        
        # Create directory structure
        build_dir = Path(audio_path).parent
        timelines_dir = build_dir / "audio_timelines"
        timelines_dir.mkdir(exist_ok=True)
        
        # Save main enhanced timeline with source tag
        timeline_filename = f"{source_tag or 'whisper'}_timeline.json"
        main_timeline_path = timelines_dir / timeline_filename
        timeline.save_to_file(str(main_timeline_path))
        
        # Skip redundant enhanced_timeline.json - main timeline file is sufficient
        
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