"""
Enhanced Timeline Schema - Clean hierarchical structure

Redesigned timeline format that avoids duplicate data and provides clean nesting:
- Global data first (transcript, speakers, etc.)
- Spans contain events that happen within them
- Services contribute their expertise only (no invalid cross-domain data)
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import json
from pathlib import Path


class TimelineEvent:
    """Single timestamp event within a span (no start/end times)"""
    
    def __init__(
        self,
        timestamp: float,
        description: str,
        source: str,
        confidence: float = 1.0,
        details: Optional[Dict[str, Any]] = None
    ):
        self.timestamp = round(timestamp, 2)  # 2 decimal precision
        self.description = description
        self.source = source
        self.confidence = round(confidence, 2) if confidence != 1.0 else None  # Only show if not perfect
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "timestamp": self.timestamp,
            "description": self.description,
            "source": self.source
        }
        
        # Only include confidence if meaningful (not perfect)
        if self.confidence is not None:
            result["confidence"] = self.confidence
            
        # Only include details if they exist and are meaningful
        if self.details:
            # Filter out None values and empty strings
            meaningful_details = {k: v for k, v in self.details.items() 
                                if v is not None and v != "" and v != 0}
            if meaningful_details:
                result["details"] = meaningful_details
        
        return result


class TimelineSpan:
    """Duration-based span containing events and features"""
    
    def __init__(
        self,
        start: float,
        end: float,
        description: str,
        source: str,
        confidence: float = 1.0,
        details: Optional[Dict[str, Any]] = None
    ):
        self.start = round(start, 2)
        self.end = round(end, 2)
        self.description = description
        self.source = source
        self.confidence = round(confidence, 2) if confidence != 1.0 else None
        self.details = details or {}
        self.events: List[TimelineEvent] = []
        self.audio_features: Dict[str, Any] = {}
    
    def add_event(self, event: TimelineEvent, is_first_span: bool = False, is_last_span: bool = False):
        """
        Add an event that occurs within this span
        
        Boundary handling rule: 
        - All spans accept both START and END boundaries (inclusive)
        - This prevents events from falling through gaps between non-contiguous spans
        - Double assignment resolution happens at the timeline level (later span wins)
        """
        # All spans: include both start AND end boundaries (fully inclusive)
        if self.start <= event.timestamp <= self.end:
            self.events.append(event)
        else:
            raise ValueError(f"Event timestamp {event.timestamp} outside span range [{self.start}, {self.end}]")
    
    def add_audio_feature(self, source: str, feature_name: str, value: Any):
        """Add audio feature from analysis services"""
        if source not in self.audio_features:
            self.audio_features[source] = {}
        self.audio_features[source][feature_name] = value
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "start": self.start,
            "end": self.end,
            "description": self.description,
            "source": self.source
        }
        
        # Only include confidence if meaningful
        if self.confidence is not None:
            result["confidence"] = self.confidence
        
        # Add events if they exist
        if self.events:
            result["events"] = [event.to_dict() for event in sorted(self.events, key=lambda e: e.timestamp)]
        
        # Add audio features if they exist
        if self.audio_features:
            result["audio_features"] = self.audio_features
        
        # Only include meaningful details
        if self.details:
            meaningful_details = {k: v for k, v in self.details.items() 
                                if v is not None and v != "" and v != 0}
            if meaningful_details:
                result["details"] = meaningful_details
        
        return result


class EnhancedTimeline:
    """Clean hierarchical timeline with global data and nested spans/events"""
    
    def __init__(self, audio_file: str, total_duration: float):
        self.audio_file = audio_file
        self.total_duration = round(total_duration, 2)
        self.created_at = datetime.now().isoformat()
        
        # Global data
        self.full_transcript = ""
        self.speakers: List[str] = []
        self.language = ""
        self.language_confidence = None
        
        # Timeline spans
        self.spans: List[TimelineSpan] = []
        
        # Standalone events (for services that detect point events)
        self.events: List[TimelineEvent] = []
        
        # Processing metadata
        self.sources_used: List[str] = []
        self.processing_notes: List[str] = []
    
    def set_global_data(
        self,
        transcript: str,
        speakers: List[str],
        language: str = "",
        language_confidence: float = None
    ):
        """Set global transcript data"""
        self.full_transcript = transcript.strip()
        self.speakers = speakers
        self.language = language
        self.language_confidence = round(language_confidence, 2) if language_confidence and language_confidence != 1.0 else None
    
    def add_span(self, span: TimelineSpan):
        """Add a timeline span"""
        self.spans.append(span)
        if span.source not in self.sources_used:
            self.sources_used.append(span.source)
    
    def add_event(self, event: TimelineEvent):
        """Add a standalone timeline event"""
        self.events.append(event)
        if event.source not in self.sources_used:
            self.sources_used.append(event.source)
    
    def add_processing_note(self, note: str):
        """Add processing note for auditing"""
        self.processing_notes.append(note)
    
    def resolve_double_assignments(self):
        """
        Resolve double assignments where events appear in multiple spans
        Rule: Later span (higher start time) wins the event
        """
        # Track events by timestamp
        event_assignments = {}  # timestamp -> list of (span_index, event_index)
        
        # Find all event assignments
        for span_idx, span in enumerate(self.spans):
            for event_idx, event in enumerate(span.events):
                timestamp = event.timestamp
                if timestamp not in event_assignments:
                    event_assignments[timestamp] = []
                event_assignments[timestamp].append((span_idx, event_idx, event))
        
        # Resolve double assignments
        events_to_remove = []  # (span_idx, event_idx) tuples to remove
        
        for timestamp, assignments in event_assignments.items():
            if len(assignments) > 1:
                # Multiple spans have this event - keep the later span (higher start time)
                assignments.sort(key=lambda x: self.spans[x[0]].start)  # Sort by span start time
                
                # Keep the last assignment (later span), remove others
                for span_idx, event_idx, event in assignments[:-1]:
                    events_to_remove.append((span_idx, event_idx))
        
        # Remove events in reverse order to maintain indices
        events_to_remove.sort(reverse=True)
        for span_idx, event_idx in events_to_remove:
            del self.spans[span_idx].events[event_idx]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to clean dictionary format"""
        result = {
            "global_data": {
                "audio_file": self.audio_file,
                "duration": self.total_duration,
                "full_transcript": self.full_transcript,
                "speakers": self.speakers,
            }
        }
        
        # Only add language data if meaningful
        if self.language:
            result["global_data"]["language"] = self.language
            if self.language_confidence is not None:
                result["global_data"]["language_confidence"] = self.language_confidence
        
        # Add timeline spans
        if self.spans:
            result["timeline_spans"] = [span.to_dict() for span in sorted(self.spans, key=lambda s: s.start)]
        
        # Add standalone events
        if self.events:
            result["events"] = [event.to_dict() for event in sorted(self.events, key=lambda e: e.timestamp)]
        
        # Add metadata
        total_span_events = sum(len(span.events) for span in self.spans)
        total_standalone_events = len(self.events)
        
        result["metadata"] = {
            "created_at": self.created_at,
            "sources_used": self.sources_used,
            "total_spans": len(self.spans),
            "total_events": total_span_events + total_standalone_events,
            "span_events": total_span_events,
            "standalone_events": total_standalone_events
        }
        
        # Only add processing notes if they exist (for debugging)
        if self.processing_notes:
            result["metadata"]["processing_notes"] = self.processing_notes
        
        return result
    
    def save_to_file(self, output_path: str):
        """Save to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    


# Utility functions for service integration
def create_speech_span(start: float, end: float, text: str, speaker: str, confidence: float, source: str) -> TimelineSpan:
    """Create a speech span with specified source"""
    assert source, "source parameter is required"
    span = TimelineSpan(
        start=start,
        end=end,
        description="Speech",
        source=source,
        confidence=confidence,
        details={
            "text": text, 
            "speaker": speaker, 
            "word_count": len(text.split())
        }
    )
    
    return span


def create_word_event(timestamp: float, word: str, confidence: float, speaker: str, source: str) -> TimelineEvent:
    """Create a word event with specified source"""
    assert source, "source parameter is required"
    return TimelineEvent(
        timestamp=timestamp,
        description="Word",
        source=source, 
        confidence=confidence,
        details={"word": word, "speaker": speaker}
    )


def create_speaker_change_event(timestamp: float, prev_speaker: str, new_speaker: str, source: str) -> TimelineEvent:
    """Create a speaker change event with specified source"""
    assert source, "source parameter is required"
    return TimelineEvent(
        timestamp=timestamp,
        description="Speaker Change",
        source=source,
        confidence=0.85,
        details={"previous_speaker": prev_speaker, "new_speaker": new_speaker}
    )


def create_librosa_event(timestamp: float, event_type: str, confidence: float, source: str, details: Optional[Dict[str, Any]] = None) -> TimelineEvent:
    """Create LibROSA music analysis timeline event"""
    assert source and isinstance(source, str) and len(source.strip()) > 0, f"Source tag must be a non-empty string, got: {source}"
    # Map event types to descriptions
    descriptions = {
        "tempo_change": "Tempo Change",
        "musical_onset": "Musical Onset", 
        "harmonic_shift": "Harmonic Shift",
        "energy_increase": "Energy Increase",
        "energy_decrease": "Energy Decrease"
    }
    
    return TimelineEvent(
        timestamp=timestamp,
        description=descriptions.get(event_type, event_type.replace('_', ' ').title()),
        source=source,
        confidence=confidence,
        details=details or {}
    )


def create_librosa_span(start: float, end: float, span_type: str, confidence: float, source: str, details: Optional[Dict[str, Any]] = None) -> TimelineSpan:
    """Create LibROSA music analysis timeline span"""
    assert source and isinstance(source, str) and len(source.strip()) > 0, f"Source tag must be a non-empty string, got: {source}"
    # Map span types to descriptions
    descriptions = {
        "structural_segment": "Structural Segment",
        "opening": "Opening Section",
        "middle": "Middle Section", 
        "closing": "Closing Section"
    }
    
    return TimelineSpan(
        start=start,
        end=end,
        description=descriptions.get(span_type, span_type.replace('_', ' ').title()),
        source=source,
        confidence=confidence,
        details=details or {}
    )


def create_pyaudio_event(timestamp: float, event_type: str, confidence: float, source: str, details: Optional[Dict[str, Any]] = None) -> TimelineEvent:
    """Create pyAudioAnalysis timeline event"""
    assert source and isinstance(source, str) and len(source.strip()) > 0, f"Source tag must be a non-empty string, got: {source}"
    # Map event types to descriptions
    descriptions = {
        "speaker_change": "Speaker Change",
        "emotion_change": "Emotion Change",
        "sound_effect": "Sound Effect",
        "audio_transition": "Audio Transition",
        "car_horn": "Car Horn",
        "applause": "Applause",
        "door_slam": "Door Slam",
        "speech": "Speech",
        "music": "Music"
    }
    
    return TimelineEvent(
        timestamp=timestamp,
        description=descriptions.get(event_type, event_type.replace('_', ' ').title()),
        source=source,
        confidence=confidence,
        details=details or {}
    )


def create_pyaudio_span(start: float, end: float, span_type: str, confidence: float, source: str, details: Optional[Dict[str, Any]] = None) -> TimelineSpan:
    """Create pyAudioAnalysis timeline span"""
    assert source and isinstance(source, str) and len(source.strip()) > 0, f"Source tag must be a non-empty string, got: {source}"
    # Map span types to descriptions
    descriptions = {
        "environment": "Environment Span"
    }
    
    return TimelineSpan(
        start=start,
        end=end,
        description=descriptions.get(span_type, span_type.replace('_', ' ').title()),
        source=source,
        confidence=confidence,
        details=details or {}
    )


# Example of clean output structure
EXAMPLE_ENHANCED_TIMELINE = {
    "global_data": {
        "audio_file": "bonita.wav",
        "duration": 7.16,
        "full_transcript": "Like many of you, we experienced pressure tactics...",
        "speakers": ["SPEAKER_02", "Speaker_2"],
        "language": "en"
    },
    "timeline_spans": [
        {
            "start": 0.0,
            "end": 2.58,
            "description": "Speech",
            "source": "whisper",
            "confidence": 0.99,
            "events": [
                {
                    "timestamp": 0.0,
                    "description": "Word",
                    "source": "whisper",
                    "confidence": 0.71,
                    "details": {
                        "word": "Like",
                        "speaker": "SPEAKER_02"
                    }
                },
                {
                    "timestamp": 1.48,
                    "description": "Speaker Change",
                    "source": "whisper",
                    "details": {
                        "previous_speaker": "SPEAKER_02",
                        "new_speaker": "Speaker_2"
                    }
                }
            ],
            "audio_features": {
                "librosa": {
                    "tempo": 120,
                    "key": "C major"
                },
                "pyaudio": {
                    "volume": "moderate",
                    "emotion": "neutral"
                }
            },
            "details": {
                "text": "Like many of you, we experienced pressure tactics",
                "speaker": "SPEAKER_02",
                "word_count": 8
            }
        }
    ],
    "metadata": {
        "created_at": "2025-01-15T10:30:00",
        "sources_used": ["whisper", "librosa", "pyaudio"],
        "total_spans": 1,
        "total_events": 2
    }
}