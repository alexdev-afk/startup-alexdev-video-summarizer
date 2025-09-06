"""
Enhanced Timeline Merger Service

Combines EnhancedTimeline objects from multiple services into a single
hierarchical timeline with proper chronological organization.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from utils.logger import get_logger
from utils.enhanced_timeline_schema import EnhancedTimeline, TimelineEvent, TimelineSpan

logger = get_logger(__name__)


class EnhancedTimelineMergerError(Exception):
    """Enhanced timeline merger processing error"""
    pass


class EnhancedTimelineMergerService:
    """Service to merge multiple EnhancedTimeline objects into a single master timeline"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize enhanced timeline merger service"""
        self.config = config
        self.merger_config = config.get('timeline_merger', {})
        
        # Conflict resolution settings
        self.priority_order = self.merger_config.get('priority_order', ['whisper', 'librosa', 'pyaudio'])
        self.confidence_threshold = self.merger_config.get('confidence_threshold', 0.5)
        self.time_precision = self.merger_config.get('time_precision', 3)  # decimal places
        self.overlap_tolerance = self.merger_config.get('overlap_tolerance', 0.1)  # seconds
        
        logger.info(f"Enhanced timeline merger initialized - priority: {self.priority_order}")
    
    def merge_enhanced_timelines(
        self,
        timelines: List[EnhancedTimeline],
        output_path: Optional[str] = None
    ) -> EnhancedTimeline:
        """
        Merge multiple EnhancedTimeline objects into a single master timeline
        
        Args:
            timelines: List of EnhancedTimeline objects to merge
            output_path: Optional path to save the merged timeline
            
        Returns:
            EnhancedTimeline object with combined content from all services
        """
        start_time = time.time()
        
        try:
            if not timelines:
                raise EnhancedTimelineMergerError("No timelines provided for merging")
            
            # Use first timeline as base and merge others into it
            base_timeline = timelines[0]
            
            # Create master timeline with combined data
            master_timeline = EnhancedTimeline(
                audio_file=base_timeline.audio_file,
                total_duration=base_timeline.total_duration
            )
            
            # Add master as source
            master_timeline.sources_used.append("master")
            
            # Merge sources from all timelines (deduplicated)
            unique_sources = set()
            for timeline in timelines:
                unique_sources.update(timeline.sources_used)
            master_timeline.sources_used.extend(list(unique_sources))
            
            # Merge timeline data (speakers, transcripts, etc.)
            self._merge_timelines_data([master_timeline] + timelines)
            
            # Separate timelines by service type
            whisper_timelines = [t for t in timelines if "whisper" in t.sources_used]
            librosa_timelines = [t for t in timelines if "librosa" in t.sources_used]
            pyaudio_timelines = [t for t in timelines if "pyaudio" in t.sources_used]
            
            # Create unified LibROSA/PyAudio spans (same boundaries, merged events)
            unified_spans = self._create_unified_librosa_pyaudio_spans(librosa_timelines, pyaudio_timelines)
            
            # Get Whisper spans (keep separate)
            whisper_spans = []
            for timeline in whisper_timelines:
                for span in timeline.spans:
                    # Label as whisper span
                    span.details["span_group"] = "whisper"
                    whisper_spans.append(span)
            
            # Combine all spans and sort by start time
            all_spans = whisper_spans + unified_spans
            sorted_spans = sorted(all_spans, key=lambda s: s.start)
            
            # Add spans to master timeline
            for span in sorted_spans:
                master_timeline.add_span(span)
            
            # Collect standalone events (events not within any span)
            standalone_events = []
            for timeline in timelines:
                standalone_events.extend(timeline.events)
            
            # Sort standalone events chronologically
            sorted_standalone_events = sorted(standalone_events, key=lambda e: e.timestamp)
            
            # Add standalone events to master timeline
            for event in sorted_standalone_events:
                master_timeline.add_event(event)
            
            # Resolve double assignments (events in multiple spans)
            master_timeline.resolve_double_assignments()
            
            processing_time = time.time() - start_time
            logger.info(f"Enhanced timeline merge completed: {len(master_timeline.events)} events, {len(master_timeline.spans)} spans in {processing_time:.2f}s")
            
            # Save if output path provided
            if output_path:
                master_timeline.save_to_file(output_path)
                logger.info(f"Master enhanced timeline saved to: {output_path}")
            
            return master_timeline
            
        except Exception as e:
            logger.error(f"Enhanced timeline merge failed: {e}")
            raise EnhancedTimelineMergerError(f"Timeline merge failed: {e}")
    
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
            
            # Find matching LibROSA events for this span
            for timeline in librosa_timelines:
                for span in timeline.spans:
                    if abs(span.start - base_span.start) < 0.01 and abs(span.end - base_span.end) < 0.01:
                        # Add LibROSA events to unified span
                        for event in span.events:
                            try:
                                unified_span.add_event(event, is_first_span=is_first_span, is_last_span=is_last_span)
                            except ValueError:
                                # Event outside span boundaries - will be standalone
                                logger.debug(f"LibROSA event {event.timestamp} outside unified span {unified_span.start}-{unified_span.end}")
            
            # Find matching PyAudio events for this span
            for timeline in pyaudio_timelines:
                for span in timeline.spans:
                    if abs(span.start - base_span.start) < 0.01 and abs(span.end - base_span.end) < 0.01:
                        # Add PyAudio events to unified span
                        for event in span.events:
                            try:
                                unified_span.add_event(event, is_first_span=is_first_span, is_last_span=is_last_span)
                            except ValueError:
                                # Event outside span boundaries - will be standalone
                                logger.debug(f"PyAudio event {event.timestamp} outside unified span {unified_span.start}-{unified_span.end}")
            
            unified_spans.append(unified_span)
        
        logger.debug(f"Created {len(unified_spans)} unified LibROSA/PyAudio spans")
        return unified_spans
    
    def cleanup(self):
        """Cleanup resources"""
        logger.debug("Enhanced timeline merger service cleanup complete")