"""
Multi-Pass Audio Timeline Service

Leverages strengths of both LibROSA and pyAudioAnalysis:
- Pass 1: Collect all events from both libraries  
- Pass 2: Use pyAudio events as primary boundaries + LibROSA density analysis
- Pass 3: Generate intelligent spans within detected boundaries
"""

import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from services.librosa_timeline_service import LibROSATimelineService
from services.pyaudio_timeline_service import PyAudioTimelineService
from utils.logger import get_logger
from utils.timeline_schema import ServiceTimeline, TimelineEvent, TimelineSpan

logger = get_logger(__name__)


class MultiPassAudioTimelineError(Exception):
    """Multi-pass audio timeline processing error"""
    pass


class MultiPassAudioTimelineService:
    """Multi-pass audio timeline service combining LibROSA and pyAudioAnalysis strengths"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize multi-pass service"""
        self.config = config
        
        # Initialize individual services
        self.librosa_service = LibROSATimelineService(config)
        self.pyaudio_service = PyAudioTimelineService(config)
        
        # Multi-pass configuration
        self.multipass_config = config.get('multipass_timeline', {})
        self.boundary_merge_threshold = self.multipass_config.get('boundary_merge_threshold', 2.0)  # seconds
        self.min_span_duration = self.multipass_config.get('min_span_duration', 3.0)  # seconds
        
        logger.info("Multi-pass audio timeline service initialized")
    
    def generate_timeline(self, audio_path: str) -> ServiceTimeline:
        """
        Generate timeline using multi-pass approach
        
        Pass 1: Event detection from both libraries
        Pass 2: Boundary analysis using pyAudio events + LibROSA density
        Pass 3: Span characterization within detected boundaries
        """
        logger.info(f"Starting multi-pass timeline generation: {Path(audio_path).name}")
        start_time = time.time()
        
        try:
            # PASS 1: Event Detection
            logger.info("=== PASS 1: Event Detection ===")
            librosa_timeline, pyaudio_timeline = self._pass1_event_detection(audio_path)
            
            # PASS 2: Boundary Analysis  
            logger.info("=== PASS 2: Boundary Analysis ===")
            boundaries = self._pass2_boundary_analysis(librosa_timeline, pyaudio_timeline)
            
            # PASS 3: Span Characterization
            logger.info("=== PASS 3: Span Characterization ===")
            final_timeline = self._pass3_span_characterization(
                audio_path, boundaries, librosa_timeline, pyaudio_timeline
            )
            
            processing_time = time.time() - start_time
            logger.info(f"Multi-pass timeline complete: {len(final_timeline.events)} events, "
                       f"{len(final_timeline.spans)} spans in {processing_time:.2f}s")
            
            return final_timeline
            
        except Exception as e:
            logger.error(f"Multi-pass timeline generation failed: {e}")
            return self._create_fallback_timeline(audio_path, error=str(e))
    
    def _pass1_event_detection(self, audio_path: str) -> Tuple[ServiceTimeline, ServiceTimeline]:
        """Pass 1: Run both services to collect all events"""
        logger.info("Running LibROSA and pyAudioAnalysis event detection...")
        
        # Get timelines from both services  
        librosa_timeline = self.librosa_service.generate_timeline(audio_path)
        pyaudio_timeline = self.pyaudio_service.generate_timeline(audio_path)
        
        logger.info(f"LibROSA: {len(librosa_timeline.events)} events, {len(librosa_timeline.spans)} spans")
        logger.info(f"pyAudio: {len(pyaudio_timeline.events)} events, {len(pyaudio_timeline.spans)} spans")
        
        return librosa_timeline, pyaudio_timeline
    
    def _pass2_boundary_analysis(self, librosa_timeline: ServiceTimeline, 
                                pyaudio_timeline: ServiceTimeline) -> List[float]:
        """Pass 2: Analyze events to determine intelligent boundaries"""
        logger.info("Analyzing events for boundary detection...")
        
        # Primary boundaries from pyAudio content changes (these are the most important)
        primary_boundaries = [0.0]  # Always start at 0
        
        for event in pyaudio_timeline.events:
            # pyAudio events mark significant content changes
            if event.category in ["speech", "environment", "sfx"]:
                primary_boundaries.append(event.timestamp)
        
        # Add end boundary
        primary_boundaries.append(max(librosa_timeline.total_duration, pyaudio_timeline.total_duration))
        
        # Secondary boundaries from LibROSA event density analysis
        secondary_boundaries = self._analyze_librosa_density(librosa_timeline, primary_boundaries)
        
        # Merge and clean boundaries
        all_boundaries = sorted(set(primary_boundaries + secondary_boundaries))
        cleaned_boundaries = self._clean_boundaries(all_boundaries)
        
        logger.info(f"Boundary detection: {len(primary_boundaries)} primary + "
                   f"{len(secondary_boundaries)} secondary → {len(cleaned_boundaries)} final boundaries")
        logger.info(f"Final boundaries: {[f'{b:.1f}s' for b in cleaned_boundaries]}")
        
        return cleaned_boundaries
    
    def _analyze_librosa_density(self, librosa_timeline: ServiceTimeline, 
                                primary_boundaries: List[float]) -> List[float]:
        """Analyze LibROSA event density to suggest additional boundaries"""
        if len(librosa_timeline.events) < 4:
            return []  # Not enough events for density analysis
        
        # Analyze event density between primary boundaries
        secondary_boundaries = []
        
        for i in range(len(primary_boundaries) - 1):
            start_time = primary_boundaries[i]
            end_time = primary_boundaries[i + 1]
            segment_duration = end_time - start_time
            
            # Skip short segments
            if segment_duration < 10.0:
                continue
            
            # Count events in this segment
            events_in_segment = [
                e for e in librosa_timeline.events 
                if start_time <= e.timestamp <= end_time
            ]
            
            # If segment has high event density and is long, consider splitting
            if len(events_in_segment) > 6 and segment_duration > 15.0:
                # Find midpoint with lowest event density for clean split
                midpoint = start_time + segment_duration / 2
                secondary_boundaries.append(midpoint)
                logger.debug(f"Added density boundary at {midpoint:.1f}s "
                           f"(segment {start_time:.1f}-{end_time:.1f}s, {len(events_in_segment)} events)")
        
        return secondary_boundaries
    
    def _clean_boundaries(self, boundaries: List[float]) -> List[float]:
        """Clean and merge boundaries that are too close together"""
        if not boundaries:
            return []
        
        cleaned = [boundaries[0]]  # Always keep first boundary
        
        for boundary in boundaries[1:]:
            # Merge boundaries that are too close
            if boundary - cleaned[-1] >= self.boundary_merge_threshold:
                cleaned.append(boundary)
            else:
                logger.debug(f"Merged boundary {boundary:.1f}s (too close to {cleaned[-1]:.1f}s)")
        
        return cleaned
    
    def _pass3_span_characterization(self, audio_path: str, boundaries: List[float],
                                   librosa_timeline: ServiceTimeline, 
                                   pyaudio_timeline: ServiceTimeline) -> ServiceTimeline:
        """Pass 3: Generate intelligent spans within detected boundaries"""
        logger.info(f"Characterizing {len(boundaries)-1} spans from intelligent boundaries...")
        
        # Create final timeline
        total_duration = max(librosa_timeline.total_duration, pyaudio_timeline.total_duration)
        final_timeline = ServiceTimeline(
            source="multipass",
            audio_file=audio_path,
            total_duration=total_duration
        )
        
        # Add all events from both services (they're already good)
        for event in librosa_timeline.events:
            final_timeline.add_event(TimelineEvent(
                timestamp=event.timestamp,
                description=f"[LibROSA] {event.description}",
                category=event.category,
                source="multipass",
                confidence=event.confidence,
                details={**event.details, "original_source": "librosa"}
            ))
        
        for event in pyaudio_timeline.events:
            final_timeline.add_event(TimelineEvent(
                timestamp=event.timestamp,
                description=f"[pyAudio] {event.description}",
                category=event.category,
                source="multipass", 
                confidence=event.confidence,
                details={**event.details, "original_source": "pyaudio"}
            ))
        
        # Generate intelligent spans between boundaries
        for i in range(len(boundaries) - 1):
            start_time = boundaries[i]
            end_time = boundaries[i + 1]
            duration = end_time - start_time
            
            # Skip very short spans
            if duration < self.min_span_duration:
                continue
            
            # Analyze this span using both libraries' insights
            span_description = self._characterize_span(
                start_time, end_time, librosa_timeline, pyaudio_timeline
            )
            
            # Determine primary category for this span
            span_category = self._determine_span_category(
                start_time, end_time, librosa_timeline, pyaudio_timeline
            )
            
            # Create intelligent span
            final_timeline.add_span(TimelineSpan(
                start=start_time,
                end=end_time,
                description=span_description,
                category=span_category,
                source="multipass",
                confidence=0.8,  # High confidence from multi-source analysis
                details={
                    "boundary_driven": True,
                    "librosa_events_count": len([e for e in librosa_timeline.events 
                                               if start_time <= e.timestamp <= end_time]),
                    "pyaudio_events_count": len([e for e in pyaudio_timeline.events
                                               if start_time <= e.timestamp <= end_time])
                }
            ))
        
        logger.info(f"Generated {len(final_timeline.spans)} intelligent spans")
        return final_timeline
    
    def _characterize_span(self, start_time: float, end_time: float,
                          librosa_timeline: ServiceTimeline, 
                          pyaudio_timeline: ServiceTimeline) -> str:
        """Generate rich description for span using both libraries' insights"""
        duration = end_time - start_time
        
        # Count events in this span from each library
        librosa_events = [e for e in librosa_timeline.events 
                         if start_time <= e.timestamp <= end_time]
        pyaudio_events = [e for e in pyaudio_timeline.events
                         if start_time <= e.timestamp <= end_time]
        
        # Base description
        description_parts = []
        
        # LibROSA musical analysis
        if librosa_events:
            if len(librosa_events) >= 5:
                music_intensity = "dense musical activity"
            elif len(librosa_events) >= 2:
                music_intensity = "moderate musical activity" 
            else:
                music_intensity = "sparse musical activity"
            
            description_parts.append(f"{music_intensity} ({len(librosa_events)} accents)")
        else:
            description_parts.append("minimal musical activity")
        
        # pyAudioAnalysis content analysis
        if pyaudio_events:
            content_changes = []
            for event in pyaudio_events:
                if "speaker change" in event.description.lower():
                    content_changes.append("speaker transition")
                elif "emotion" in event.description.lower():
                    content_changes.append("emotional shift")
                elif "sound effect" in event.description.lower():
                    content_changes.append("sound effect")
                elif "transition" in event.description.lower():
                    content_changes.append("audio transition")
            
            if content_changes:
                description_parts.append(f"with {', '.join(content_changes)}")
        
        # Combine description
        base_description = f"Audio segment ({duration:.1f}s) - {' '.join(description_parts)}"
        
        # Add context based on position and events
        if start_time == 0:
            base_description = f"Opening: {base_description}"
        elif end_time >= max(librosa_timeline.total_duration, pyaudio_timeline.total_duration) - 1:
            base_description = f"Closing: {base_description}"
        elif len(pyaudio_events) > 0:
            base_description = f"Transition: {base_description}"
        
        return base_description
    
    def _determine_span_category(self, start_time: float, end_time: float,
                                librosa_timeline: ServiceTimeline,
                                pyaudio_timeline: ServiceTimeline) -> str:
        """Determine primary category for span"""
        pyaudio_events = [e for e in pyaudio_timeline.events
                         if start_time <= e.timestamp <= end_time]
        
        # Check pyAudio events for category hints
        for event in pyaudio_events:
            if event.category == "speech":
                return "speech"
            elif event.category == "sfx":
                return "sfx"
            elif event.category == "environment":
                return "environment"
        
        # Default to music if LibROSA has events
        librosa_events = [e for e in librosa_timeline.events
                         if start_time <= e.timestamp <= end_time]
        if librosa_events:
            return "music"
        
        return "environment"  # Safe default
    
    def _create_fallback_timeline(self, audio_path: str, error: Optional[str] = None) -> ServiceTimeline:
        """Create fallback timeline when processing fails"""
        try:
            from pathlib import Path
            file_size = Path(audio_path).stat().st_size
            estimated_duration = file_size / 88000
        except:
            estimated_duration = 30.0
        
        timeline = ServiceTimeline(
            source="multipass",
            audio_file=str(audio_path),
            total_duration=estimated_duration
        )
        
        # Add basic fallback span
        timeline.add_span(TimelineSpan(
            start=0.0,
            end=estimated_duration,
            description=f"Audio content ({estimated_duration:.1f}s) - analysis failed",
            category="environment",
            source="multipass",
            confidence=0.1,
            details={"fallback_mode": True, "error": error}
        ))
        
        logger.warning(f"Using fallback multi-pass timeline: {error}")
        return timeline