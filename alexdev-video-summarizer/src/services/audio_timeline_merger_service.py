"""
Audio Timeline Merger Service

DEMUCS BREAKTHROUGH: Combines audio timeline objects from clean separated sources
into a single chronological timeline with no complex heuristic filtering needed.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from utils.logger import get_logger
from utils.enhanced_timeline_schema import EnhancedTimeline, TimelineEvent, TimelineSpan

logger = get_logger(__name__)


class AudioTimelineMergerError(Exception):
    """Audio timeline merger processing error"""
    pass


class AudioTimelineMergerService:
    """Service to merge audio timelines from clean Demucs-separated sources"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize audio timeline merger service"""
        self.config = config
        self.merger_config = config.get('timeline_merger', {})
        
        # Conflict resolution settings
        self.priority_order = self.merger_config.get('priority_order', ['whisper', 'librosa', 'pyaudio'])
        self.confidence_threshold = self.merger_config.get('confidence_threshold', 0.5)
        self.time_precision = self.merger_config.get('time_precision', 3)  # decimal places
        self.overlap_tolerance = self.merger_config.get('overlap_tolerance', 0.1)  # seconds
        
        logger.info(f"Audio timeline merger initialized - DEMUCS breakthrough approach")
    
    def create_combined_audio_timeline(
        self,
        timelines: List[EnhancedTimeline],
        output_path: Optional[str] = None
    ) -> EnhancedTimeline:
        """
        DEMUCS BREAKTHROUGH: Create combined timeline by simple chronological merging
        
        With Demucs separated audio sources, no complex filtering needed:
        - whisper_voice: Clean transcription
        - librosa_music: Pure music analysis
        - pyaudio_music: Music features 
        - pyaudio_voice: Voice features
        
        Args:
            timelines: List of EnhancedTimeline objects from clean Demucs sources
            output_path: Optional path to save the combined timeline
            
        Returns:
            Combined timeline with chronologically ordered clean events
        """
        start_time = time.time()
        
        try:
            if not timelines:
                raise AudioTimelineMergerError("No timelines provided for merging")
            
            # Use first timeline as base
            base_timeline = timelines[0]
            
            # Create combined timeline
            combined_timeline = EnhancedTimeline(
                audio_file=base_timeline.audio_file,
                total_duration=base_timeline.total_duration
            )
            
            # Add source traceability
            combined_timeline.sources_used.append("demucs_combined")
            
            # Collect sources from all timelines
            unique_sources = set()
            for timeline in timelines:
                unique_sources.update(timeline.sources_used)
            combined_timeline.sources_used.extend(list(unique_sources))
            
            # Merge timeline metadata (speakers, etc.)
            self._merge_timeline_metadata_simple(combined_timeline, timelines)
            
            # DEMUCS BREAKTHROUGH: Just collect all events and sort chronologically
            all_events = []
            
            for timeline in timelines:
                # Add only orphan events (events outside any span) - maintain data structure integrity
                for event in timeline.events:
                    all_events.append(event)
                    logger.debug(f"Added orphan event from {timeline.sources_used}: {event.description} at {event.timestamp}s")
                
                # NOTE: Do NOT extract events from spans - this corrupts data structure
                # Each service maintains independent spans with their events nested inside
                # Spans are added as complete units below
            
            # Sort events chronologically
            all_events.sort(key=lambda e: e.timestamp)
            combined_timeline.events = all_events
            
            # Collect all spans as complete units - maintain service independence
            all_spans = []
            for timeline in timelines:
                for span in timeline.spans:
                    all_spans.append(span)  # Keep span with its nested events intact
                    logger.debug(f"Added intact span from {timeline.sources_used}: {span.description} from {span.start}s to {span.end}s")
            
            # Sort spans chronologically
            all_spans.sort(key=lambda s: s.start)
            combined_timeline.spans = all_spans
            
            # Add processing notes
            combined_timeline.processing_notes.append("DEMUCS BREAKTHROUGH: Simple chronological merger - no heuristic filtering needed")
            combined_timeline.processing_notes.append(f"Clean sources: {list(unique_sources)}")
            combined_timeline.processing_notes.append(f"Total events: {len(all_events)} chronologically ordered")
            combined_timeline.processing_notes.append(f"Total spans: {len(all_spans)} chronologically ordered")
            
            processing_time = time.time() - start_time
            logger.info(f"DEMUCS BREAKTHROUGH: Simple merge completed in {processing_time:.2f}s")
            logger.info(f"Result: {len(all_events)} clean events, {len(all_spans)} spans (no complex filtering needed)")
            logger.info("FIXED: Whisper timeline events now properly included from spans")
            
            # Save if output path provided
            if output_path:
                combined_timeline.save_to_file(output_path)
                logger.info(f"Combined timeline saved to: {output_path}")
            
            return combined_timeline
            
        except Exception as e:
            logger.error(f"Audio timeline merge failed: {e}")
            raise AudioTimelineMergerError(f"Timeline merge failed: {e}")
    
    def _merge_timelines_data(self, timelines: List[EnhancedTimeline]) -> None:
        """Merge data from all timelines into the first one"""
        if not timelines:
            return
            
        base_timeline = timelines[0]
        
        # Merge speakers from all timelines
        all_speakers = set(base_timeline.speakers)
        for timeline in timelines[1:]:
            all_speakers.update(timeline.speakers)
        base_timeline.speakers = list(all_speakers)
        
        # Merge transcripts (Whisper takes priority)
        for timeline in timelines:
            if timeline.full_transcript and "whisper" in timeline.sources_used:
                base_timeline.full_transcript = timeline.full_transcript
                base_timeline.language = timeline.language
                base_timeline.language_confidence = timeline.language_confidence
                break
    
    def merge_timeline_files(
        self,
        timeline_paths: List[str],
        output_path: Optional[str] = None
    ) -> EnhancedTimeline:
        """
        Load and merge timeline files
        
        Args:
            timeline_paths: List of paths to EnhancedTimeline JSON files
            output_path: Optional path to save the merged timeline
            
        Returns:
            EnhancedTimeline object with combined content
        """
        try:
            # Load timelines from files
            timelines = []
            for path in timeline_paths:
                if Path(path).exists():
                    with open(path, 'r', encoding='utf-8') as f:
                        timeline_data = json.load(f)
                    
                    # Create EnhancedTimeline object from loaded data
                    timeline = self._reconstruct_enhanced_timeline(timeline_data)
                    timelines.append(timeline)
                    logger.info(f"Loaded timeline from: {path}")
                else:
                    logger.warning(f"Timeline file not found: {path}")
            
            if not timelines:
                raise EnhancedTimelineMergerError("No valid timeline files found")
            
            return self.merge_enhanced_timelines(timelines, output_path)
            
        except Exception as e:
            logger.error(f"Failed to merge timeline files: {e}")
            raise EnhancedTimelineMergerError(f"File merge failed: {e}")
    
    def _reconstruct_enhanced_timeline(self, timeline_data: Dict[str, Any]) -> EnhancedTimeline:
        """Reconstruct EnhancedTimeline object from loaded JSON data"""
        # Create base timeline
        timeline = EnhancedTimeline(
            audio_file=timeline_data.get('audio_file', ''),
            total_duration=timeline_data.get('total_duration', 0.0)
        )
        
        # Restore metadata
        if 'sources_used' in timeline_data:
            timeline.sources_used = timeline_data['sources_used']
        if 'processing_notes' in timeline_data:
            timeline.processing_notes = timeline_data['processing_notes']
        
        # Add events
        for event_data in timeline_data.get('events', []):
            event = TimelineEvent(
                timestamp=event_data['timestamp'],
                description=event_data['description'],
                source=event_data['source'],
                confidence=event_data.get('confidence', 1.0),
                details=event_data.get('details', {})
            )
            timeline.add_event(event)
        
        # Add spans
        for span_data in timeline_data.get('spans', []):
            span = TimelineSpan(
                start=span_data['start'],
                end=span_data['end'],
                description=span_data['description'],
                source=span_data['source'],
                confidence=span_data.get('confidence', 1.0),
                details=span_data.get('details', {})
            )
            timeline.add_span(span)
        
        return timeline
    
    def _create_unified_librosa_pyaudio_spans(
        self, 
        librosa_timelines: List[Any], 
        pyaudio_timelines: List[Any]
    ) -> List[Any]:
        """
        Create unified spans from LibROSA and PyAudio timelines with same boundaries
        
        Args:
            librosa_timelines: List of LibROSA timelines
            pyaudio_timelines: List of PyAudio timelines
            
        Returns:
            List of unified spans with events from both services
        """
        # Get LibROSA spans (these define the boundaries)
        librosa_spans = []
        for timeline in librosa_timelines:
            librosa_spans.extend(timeline.spans)
        
        # Get PyAudio spans (should have same boundaries)
        pyaudio_spans = []
        for timeline in pyaudio_timelines:
            pyaudio_spans.extend(timeline.spans)
        
        # If no LibROSA spans, use PyAudio spans as base
        if not librosa_spans and pyaudio_spans:
            base_spans = pyaudio_spans
        elif librosa_spans:
            base_spans = librosa_spans
        else:
            return []
        
        unified_spans = []
        
        # Create unified spans based on boundaries
        sorted_base_spans = sorted(base_spans, key=lambda s: s.start)
        for i, base_span in enumerate(sorted_base_spans):
            is_first_span = (i == 0)
            is_last_span = (i == len(sorted_base_spans) - 1)
            
            # Create new unified span with librosa/pyaudio label
            unified_span = TimelineSpan(
                start=base_span.start,
                end=base_span.end,
                description=f"Audio Analysis Segment ({base_span.end - base_span.start:.1f}s)",
                source="librosa/pyaudio",
                confidence=base_span.confidence,
                details={
                    "span_group": "librosa/pyaudio",
                    "duration": base_span.end - base_span.start
                }
            )
            
            # Find LibROSA standalone events that fall within this unified span
            for timeline in librosa_timelines:
                for event in timeline.events:
                    if unified_span.start <= event.timestamp <= unified_span.end:
                        try:
                            unified_span.add_event(event, is_first_span=is_first_span, is_last_span=is_last_span)
                            logger.debug(f"Added LibROSA event {event.timestamp} to unified span [{unified_span.start}, {unified_span.end}]")
                        except ValueError:
                            logger.debug(f"LibROSA event {event.timestamp} rejected by unified span {unified_span.start}-{unified_span.end}")
            
            # Find PyAudio standalone events that fall within this unified span
            for timeline in pyaudio_timelines:
                for event in timeline.events:
                    if unified_span.start <= event.timestamp <= unified_span.end:
                        try:
                            unified_span.add_event(event, is_first_span=is_first_span, is_last_span=is_last_span)
                            logger.debug(f"Added PyAudio event {event.timestamp} to unified span [{unified_span.start}, {unified_span.end}]")
                        except ValueError:
                            logger.debug(f"PyAudio event {event.timestamp} rejected by unified span {unified_span.start}-{unified_span.end}")
            
            unified_spans.append(unified_span)
        
        logger.debug(f"Created {len(unified_spans)} unified LibROSA/PyAudio spans")
        return unified_spans
    
    def _filter_librosa_speech_artifacts(self, combined_timeline):
        """
        Filter out LibROSA events that correlate with speech segments.
        
        Cross-service correlation filtering that removes musical events detected
        during speech, which are typically speech artifacts rather than genuine
        musical events.
        """
        try:
            # Get speech segments from Whisper events
            speech_segments = self._extract_speech_segments(combined_timeline)
            
            if not speech_segments:
                logger.debug("No speech segments found - skipping LibROSA speech artifact filtering")
                return
            
            # Filter LibROSA events
            original_events = combined_timeline.events.copy()
            librosa_events = [e for e in original_events if e.source == "librosa"]
            other_events = [e for e in original_events if e.source != "librosa"]
            
            if not librosa_events:
                logger.debug("No LibROSA events found - skipping speech artifact filtering")
                return
            
            # Analyze each LibROSA event for speech correlation
            legitimate_events = []
            filtered_artifacts = []
            
            for event in librosa_events:
                if self._is_speech_artifact(event, speech_segments, combined_timeline):
                    filtered_artifacts.append(event)
                    logger.debug(f"Filtered LibROSA speech artifact: {event.timestamp:.2f}s - {event.description}")
                else:
                    legitimate_events.append(event)
            
            # Update master timeline with filtered LibROSA events
            combined_timeline.events = other_events + legitimate_events
            
            logger.info(f"LibROSA speech artifact filtering: {len(librosa_events)} -> {len(legitimate_events)} events "
                       f"({len(filtered_artifacts)} speech artifacts removed)")
            
        except Exception as e:
            logger.warning(f"LibROSA speech artifact filtering failed: {e}")
    
    def _extract_speech_segments(self, combined_timeline) -> List[tuple]:
        """Extract speech segment timing from Whisper events in the master timeline."""
        speech_segments = []
        
        try:
            # Look for Whisper events that indicate speech timing
            whisper_events = [e for e in combined_timeline.events if e.source == "whisper"]
            
            if not whisper_events:
                logger.debug("No Whisper events found for speech segment extraction")
                return speech_segments
            
            # Group consecutive Whisper events into speech segments
            current_segment_start = None
            current_segment_end = None
            
            for event in sorted(whisper_events, key=lambda e: e.timestamp):
                if current_segment_start is None:
                    current_segment_start = event.timestamp
                    current_segment_end = event.timestamp
                else:
                    # If this event is close to the previous one, extend the segment
                    if event.timestamp - current_segment_end <= 1.0:  # 1 second gap tolerance
                        current_segment_end = event.timestamp
                    else:
                        # Gap too large, save current segment and start new one
                        if current_segment_end > current_segment_start:
                            speech_segments.append((current_segment_start, current_segment_end))
                        current_segment_start = event.timestamp
                        current_segment_end = event.timestamp
            
            # Save final segment
            if current_segment_start is not None and current_segment_end > current_segment_start:
                speech_segments.append((current_segment_start, current_segment_end))
            
            logger.debug(f"Extracted {len(speech_segments)} speech segments from Whisper events")
            
        except Exception as e:
            logger.warning(f"Failed to extract speech segments: {e}")
        
        return speech_segments
    
    def _is_speech_artifact(self, event, speech_segments: List[tuple], combined_timeline=None) -> bool:
        """
        Determine if a LibROSA event is likely a speech artifact using multiple criteria.
        
        Args:
            event: LibROSA timeline event
            speech_segments: List of (start, end) speech timing tuples
            combined_timeline: Combined timeline for boundary and PyAudio analysis
            
        Returns:
            True if event should be filtered as speech artifact
        """
        event_time = event.timestamp
        merger_config = self.config.get('timeline_merger', {})
        
        # Check if filtering is enabled
        if not merger_config.get('enable_speech_artifact_filtering', True):
            return False
        
        # 1. Boundary Exception Rule - preserve events near structural boundaries
        if merger_config.get('preserve_boundary_events', True):
            if self._is_near_structural_boundary(event_time, combined_timeline, merger_config):
                logger.debug(f"LibROSA event at {event_time:.2f}s preserved due to boundary proximity")
                return False
        
        # 2. PyAudio-based filtering - check distance from PyAudio events
        pyaudio_threshold = merger_config.get('pyaudio_distance_threshold', 0.7)
        if combined_timeline and self._has_distant_pyaudio_events(event_time, combined_timeline, pyaudio_threshold):
            logger.debug(f"LibROSA event at {event_time:.2f}s preserved due to PyAudio distance > {pyaudio_threshold}s")
            return False
        
        # 3. Traditional speech overlap analysis as fallback
        for start, end in speech_segments:
            if start <= event_time <= end:
                # Event is during speech - apply classification logic
                return self._classify_speech_overlap_event(event)
        
        # Event is during silence/pure music - likely legitimate
        return False
    
    def _classify_speech_overlap_event(self, event) -> bool:
        """
        Classify whether an event during speech is likely a speech artifact.
        
        High probability speech artifacts:
        - Musical onsets with generic descriptions (speech syllables)
        - Frequent harmonic shifts (speech pitch variations)
        - High-frequency spectral events
        
        Lower probability (may be legitimate):
        - Tempo changes (could be background music tempo shift)
        - Structural boundaries (could be music section transitions)
        """
        event_description = event.description.lower()
        event_details = getattr(event, 'details', {})
        
        # High confidence speech artifacts - these are almost always speech during speech segments
        if 'musical onset' in event_description:
            # Musical onsets during speech are typically speech syllables/consonants
            return True
        
        if 'harmonic shift' in event_description:
            # Harmonic shifts during speech are likely speech pitch variations
            return True
        
        # Medium confidence speech artifacts
        if any(term in event_description for term in ['spectral', 'frequency', 'timbre']):
            # Spectral events during speech often reflect speech characteristics
            return True
        
        # Lower confidence - may be legitimate background music events
        if any(term in event_description for term in ['tempo', 'structural', 'rhythm']):
            # These could be legitimate background music changes during speech
            # Keep them unless we have strong evidence otherwise
            return False
        
        # Default: if during speech and no clear classification, lean toward artifact
        return True
    
    def _filter_librosa_speech_artifacts_from_timelines(
        self, 
        librosa_timelines: List, 
        whisper_timelines: List, 
        pyaudio_timelines: List, 
        temp_combined_timeline
    ) -> List:
        """
        Filter LibROSA speech artifacts from raw timelines before unified span creation.
        
        Args:
            librosa_timelines: Raw LibROSA timeline data
            whisper_timelines: Whisper timeline data for speech segments
            pyaudio_timelines: PyAudio timeline data for emotion events
            temp_combined_timeline: Temporary timeline with global data
            
        Returns:
            Filtered LibROSA timelines with speech artifacts removed
        """
        merger_config = self.config.get('timeline_merger', {})
        
        if not merger_config.get('enable_speech_artifact_filtering', True):
            logger.info("LibROSA speech artifact filtering disabled")
            return librosa_timelines
        
        filtered_timelines = []
        
        for librosa_timeline in librosa_timelines:
            # Extract speech segments from Whisper timelines
            speech_segments = []
            for whisper_timeline in whisper_timelines:
                for span in whisper_timeline.spans:
                    speech_segments.append((span.start, span.end))
            
            # Extract PyAudio events for distance analysis
            pyaudio_events = []
            for pyaudio_timeline in pyaudio_timelines:
                for event in pyaudio_timeline.events:
                    pyaudio_events.append(event.timestamp)
            
            # Filter events
            legitimate_events = []
            filtered_artifacts = []
            
            for event in librosa_timeline.events:
                if self._is_speech_artifact_from_raw_data(
                    event, speech_segments, pyaudio_events, temp_combined_timeline, merger_config
                ):
                    filtered_artifacts.append(event)
                    logger.debug(f"Filtered LibROSA speech artifact: {event.timestamp:.2f}s - {event.description}")
                else:
                    legitimate_events.append(event)
            
            # Create new timeline with filtered events
            filtered_timeline = librosa_timeline  # Keep original structure
            filtered_timeline.events = legitimate_events
            filtered_timelines.append(filtered_timeline)
            
            logger.info(f"LibROSA filtering: {len(filtered_artifacts)} artifacts removed, {len(legitimate_events)} events preserved")
        
        return filtered_timelines
    
    def _is_speech_artifact_from_raw_data(
        self, 
        event, 
        speech_segments: List[tuple], 
        pyaudio_events: List[float], 
        temp_combined_timeline, 
        config: Dict[str, Any]
    ) -> bool:
        """
        Determine if a LibROSA event is a speech artifact using raw timeline data.
        
        Args:
            event: LibROSA event to analyze
            speech_segments: List of (start, end) speech timing tuples
            pyaudio_events: List of PyAudio event timestamps
            temp_combined_timeline: Timeline with global data
            config: Timeline merger configuration
            
        Returns:
            True if event should be filtered as speech artifact
        """
        event_time = event.timestamp
        
        # 1. Boundary Exception Rule - preserve events near structural boundaries
        if config.get('preserve_boundary_events', True):
            boundary_distance = config.get('boundary_exception_distance', 0.5)
            video_duration = temp_combined_timeline.total_duration
            
            # Check video start/end boundaries
            if event_time <= boundary_distance or (video_duration - event_time) <= boundary_distance:
                logger.debug(f"LibROSA event at {event_time:.2f}s preserved due to video boundary proximity")
                return False
            
            # Check speech segment boundaries
            for start, end in speech_segments:
                if (abs(event_time - start) <= boundary_distance or 
                    abs(event_time - end) <= boundary_distance):
                    logger.debug(f"LibROSA event at {event_time:.2f}s preserved due to speech boundary proximity")
                    return False
        
        # 2. PyAudio-based filtering - check distance from PyAudio events
        pyaudio_threshold = config.get('pyaudio_distance_threshold', 0.7)
        if pyaudio_events:
            min_distance = min(abs(event_time - pyaudio_time) for pyaudio_time in pyaudio_events)
            if min_distance > pyaudio_threshold:
                logger.debug(f"LibROSA event at {event_time:.2f}s preserved due to PyAudio distance > {pyaudio_threshold}s")
                return False
        
        # 3. Traditional speech overlap analysis as fallback
        for start, end in speech_segments:
            if start <= event_time <= end:
                # Event is during speech - apply classification logic
                return self._classify_speech_overlap_event(event)
        
        # Event is during silence/pure music - likely legitimate
        return False
    
    def _is_near_structural_boundary(self, event_time: float, combined_timeline, config: Dict[str, Any]) -> bool:
        """
        Check if event is near structural boundaries (video start/end, segment transitions, span boundaries).
        
        Args:
            event_time: Timestamp of the LibROSA event
            combined_timeline: Master timeline containing all spans and events
            config: Timeline merger configuration
            
        Returns:
            True if event is within boundary_exception_distance of any structural boundary
        """
        if not combined_timeline:
            return False
        
        boundary_distance = config.get('boundary_exception_distance', 0.5)
        
        # Check video start/end boundaries
        video_duration = combined_timeline.get('global_data', {}).get('duration', 0)
        if event_time <= boundary_distance or (video_duration - event_time) <= boundary_distance:
            return True
        
        # Check Whisper segment boundaries (speech start/end points)
        for span in combined_timeline.get('timeline_spans', []):
            if span.get('source') == 'whisper':
                span_start = span.get('start', 0)
                span_end = span.get('end', 0)
                
                # Check proximity to segment start or end
                if (abs(event_time - span_start) <= boundary_distance or 
                    abs(event_time - span_end) <= boundary_distance):
                    return True
        
        return False
    
    def _has_distant_pyaudio_events(self, event_time: float, combined_timeline, threshold: float) -> bool:
        """
        Check if LibROSA event is sufficiently far from PyAudio emotion detection events.
        
        Args:
            event_time: Timestamp of the LibROSA event
            combined_timeline: Master timeline containing PyAudio events
            threshold: Minimum distance required from PyAudio events
            
        Returns:
            True if event is >threshold seconds from all PyAudio events
        """
        if not combined_timeline:
            return False
        
        # Find PyAudio events in the master timeline
        pyaudio_events = []
        
        # Check both standalone events and span events
        for event in combined_timeline.get('events', []):
            if event.get('source') == 'pyaudio':
                pyaudio_events.append(event['timestamp'])
        
        for span in combined_timeline.get('timeline_spans', []):
            # Check span events for PyAudio events (they're nested in unified spans)
            if 'events' in span:
                for event in span['events']:
                    if event.get('source') == 'pyaudio':
                        pyaudio_events.append(event['timestamp'])
        
        # Check distance from all PyAudio events
        if pyaudio_events:
            min_distance = min(abs(event_time - pyaudio_time) for pyaudio_time in pyaudio_events)
            return min_distance > threshold
        
        # No PyAudio events found - fall back to other filtering methods
        return False
    
    def _merge_timeline_metadata_simple(self, combined_timeline: EnhancedTimeline, timelines: List[EnhancedTimeline]):
        """Simple metadata merge for Demucs breakthrough"""
        # Combine speakers (deduplicated)
        unique_speakers = set()
        for timeline in timelines:
            if hasattr(timeline, 'speakers') and timeline.speakers:
                unique_speakers.update(timeline.speakers)
        combined_timeline.speakers = list(unique_speakers)
        
        # Combine transcripts
        combined_transcripts = []
        for timeline in timelines:
            if hasattr(timeline, 'full_transcript') and timeline.full_transcript:
                combined_transcripts.append(timeline.full_transcript)
        combined_timeline.full_transcript = " ".join(combined_transcripts)
        
        # Combine processing notes
        for timeline in timelines:
            if hasattr(timeline, 'processing_notes') and timeline.processing_notes:
                for note in timeline.processing_notes:
                    combined_timeline.processing_notes.append(f"[{timeline.sources_used[0] if timeline.sources_used else 'unknown'}] {note}")

    def cleanup(self):
        """Cleanup resources"""
        logger.debug("Audio timeline merger service cleanup complete")