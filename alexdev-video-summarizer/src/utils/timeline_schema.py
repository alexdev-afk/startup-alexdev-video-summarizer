"""
Timeline JSON Schema Definitions for Advertisement Audio Analysis

Defines standard schemas for timeline events and spans across all audio analysis services.
"""

from typing import Dict, List, Any, Optional, Union, Literal
from datetime import datetime
import json


class TimelineEvent:
    """Single timestamp event (e.g., sound effect, tempo change)"""
    
    def __init__(
        self,
        timestamp: float,
        description: str,
        category: Literal["music", "speech", "sfx", "transition", "environment"],
        source: Literal["whisper", "librosa", "pyaudio"],
        confidence: float = 1.0,
        details: Optional[Dict[str, Any]] = None
    ):
        self.type = "event"
        self.timestamp = round(timestamp, 3)  # 3 decimal precision for milliseconds
        self.description = description
        self.category = category
        self.source = source
        self.confidence = round(confidence, 3)
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "timestamp": self.timestamp,
            "description": self.description,
            "category": self.category,
            "source": self.source,
            "confidence": self.confidence,
            "details": self.details
        }


class TimelineSpan:
    """Duration-based span (e.g., music segment, speech segment)"""
    
    def __init__(
        self,
        start: float,
        end: float,
        description: str,
        category: Literal["music", "speech", "sfx", "transition", "environment"],
        source: Literal["whisper", "librosa", "pyaudio"],
        confidence: float = 1.0,
        details: Optional[Dict[str, Any]] = None
    ):
        self.type = "span"
        self.start = round(start, 3)
        self.end = round(end, 3)
        self.duration = round(end - start, 3)
        self.description = description
        self.category = category
        self.source = source
        self.confidence = round(confidence, 3)
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "start": self.start,
            "end": self.end,
            "duration": self.duration,
            "description": self.description,
            "category": self.category,
            "source": self.source,
            "confidence": self.confidence,
            "details": self.details
        }


class ServiceTimeline:
    """Timeline output from a single service (Whisper, LibROSA, or pyAudioAnalysis)"""
    
    def __init__(
        self,
        source: Literal["whisper", "librosa", "pyaudio"],
        audio_file: str,
        total_duration: float
    ):
        self.source = source
        self.audio_file = audio_file
        self.total_duration = round(total_duration, 3)
        self.events: List[TimelineEvent] = []
        self.spans: List[TimelineSpan] = []
        self.created_at = datetime.now().isoformat()
        self.service_version = f"{source}_timeline_v1.0.0"
    
    def add_event(self, event: TimelineEvent):
        """Add a timeline event"""
        if event.source != self.source:
            raise ValueError(f"Event source {event.source} doesn't match timeline source {self.source}")
        self.events.append(event)
    
    def add_span(self, span: TimelineSpan):
        """Add a timeline span"""
        if span.source != self.source:
            raise ValueError(f"Span source {span.source} doesn't match timeline source {self.source}")
        self.spans.append(span)
    
    def get_all_timeline_objects(self) -> List[Dict[str, Any]]:
        """Get all events and spans as dictionaries"""
        timeline_objects = []
        timeline_objects.extend([event.to_dict() for event in self.events])
        timeline_objects.extend([span.to_dict() for span in self.spans])
        return timeline_objects
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "audio_file": self.audio_file,
            "total_duration": self.total_duration,
            "created_at": self.created_at,
            "service_version": self.service_version,
            "timeline": self.get_all_timeline_objects(),
            "summary": {
                "total_events": len(self.events),
                "total_spans": len(self.spans),
                "total_objects": len(self.events) + len(self.spans)
            }
        }
    
    def save_to_file(self, output_path: str):
        """Save timeline to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


class MergedTimeline:
    """Final merged timeline from all services, chronologically sorted"""
    
    def __init__(self, audio_file: str, total_duration: float):
        self.audio_file = audio_file
        self.total_duration = round(total_duration, 3)
        self.timeline_segments: List[Dict[str, Any]] = []
        self.created_at = datetime.now().isoformat()
        self.merger_version = "timeline_merger_v1.0.0"
        self.source_files = {
            "whisper": None,
            "librosa": None,  
            "pyaudio": None
        }
    
    def add_timeline_segment(
        self,
        time_range: str,
        content: List[Dict[str, Any]]
    ):
        """Add a time segment with its content from various sources"""
        self.timeline_segments.append({
            "time_range": time_range,
            "content": content
        })
    
    def set_source_file(self, source: str, file_path: str):
        """Record which source files were used"""
        if source in self.source_files:
            self.source_files[source] = file_path
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "audio_file": self.audio_file,
            "total_duration": self.total_duration,
            "created_at": self.created_at,
            "merger_version": self.merger_version,
            "source_files": self.source_files,
            "merged_timeline": self.timeline_segments,
            "summary": {
                "total_segments": len(self.timeline_segments),
                "sources_used": [k for k, v in self.source_files.items() if v is not None]
            }
        }
    
    def save_to_file(self, output_path: str):
        """Save merged timeline to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


# Schema validation helpers
def validate_timeline_object(obj: Dict[str, Any]) -> bool:
    """Validate a timeline object (event or span) structure"""
    required_fields = ["type", "description", "category", "source", "confidence"]
    
    # Check required fields
    for field in required_fields:
        if field not in obj:
            return False
    
    # Type-specific validation
    if obj["type"] == "event":
        return "timestamp" in obj
    elif obj["type"] == "span":
        return "start" in obj and "end" in obj and "duration" in obj
    
    return False


# Example timeline objects for reference
EXAMPLE_WHISPER_TIMELINE = {
    "source": "whisper",
    "audio_file": "advertisement.wav",
    "total_duration": 30.5,
    "timeline": [
        {
            "type": "span",
            "start": 3.2,
            "end": 15.7,
            "duration": 12.5,
            "description": "Male narrator: 'Introducing the amazing new product that will change your life'",
            "category": "speech",
            "source": "whisper",
            "confidence": 0.94,
            "details": {
                "speaker": "SPEAKER_00",
                "language": "en",
                "words": 12
            }
        },
        {
            "type": "event",
            "timestamp": 15.7,
            "description": "Speaker change detected",
            "category": "speech",
            "source": "whisper",
            "confidence": 0.87,
            "details": {
                "previous_speaker": "SPEAKER_00",
                "new_speaker": "SPEAKER_01"
            }
        }
    ]
}

EXAMPLE_LIBROSA_TIMELINE = {
    "source": "librosa",
    "audio_file": "advertisement.wav", 
    "total_duration": 30.5,
    "timeline": [
        {
            "type": "span",
            "start": 0.0,
            "end": 18.3,
            "duration": 18.3,
            "description": "Upbeat electronic music in C major, steady 128 BPM",
            "category": "music",
            "source": "librosa",
            "confidence": 0.89,
            "details": {
                "tempo": 128.4,
                "key": "C major",
                "harmonic_content": 0.73,
                "percussive_content": 0.27
            }
        },
        {
            "type": "event",
            "timestamp": 8.23,
            "description": "Tempo increase to 140 BPM",
            "category": "transition",
            "source": "librosa", 
            "confidence": 0.82,
            "details": {
                "previous_tempo": 128.4,
                "new_tempo": 140.1,
                "tempo_change": 11.7
            }
        }
    ]
}

EXAMPLE_PYAUDIO_TIMELINE = {
    "source": "pyaudio",
    "audio_file": "advertisement.wav",
    "total_duration": 30.5,
    "timeline": [
        {
            "type": "span", 
            "start": 3.0,
            "end": 16.0,
            "duration": 13.0,
            "description": "Male voice with excited emotional tone, high audio quality",
            "category": "speech",
            "source": "pyaudio",
            "confidence": 0.76,
            "details": {
                "emotion": "excited",
                "gender": "male",
                "audio_quality": "high",
                "speaking_rate": "moderate"
            }
        },
        {
            "type": "event",
            "timestamp": 8.45,
            "description": "Car horn sound effect detected",
            "category": "sfx",
            "source": "pyaudio",
            "confidence": 0.93,
            "details": {
                "sound_type": "vehicle_horn",
                "intensity": "medium",
                "duration_estimate": 1.2
            }
        }
    ]
}

EXAMPLE_MERGED_TIMELINE = {
    "audio_file": "advertisement.wav",
    "total_duration": 30.5,
    "merged_timeline": [
        {
            "time_range": "0.0-3.0s",
            "content": [
                {
                    "source": "librosa",
                    "description": "Upbeat electronic music in C major, steady 128 BPM",
                    "category": "music"
                }
            ]
        },
        {
            "time_range": "3.0-8.23s", 
            "content": [
                {
                    "source": "whisper",
                    "description": "Male narrator: 'Introducing the amazing new product'",
                    "category": "speech"
                },
                {
                    "source": "librosa",
                    "description": "Background music continues at 128 BPM",
                    "category": "music"
                },
                {
                    "source": "pyaudio",
                    "description": "Male voice with excited emotional tone",
                    "category": "speech"
                }
            ]
        },
        {
            "time_range": "8.23s",
            "content": [
                {
                    "source": "librosa",
                    "description": "Tempo increase to 140 BPM",
                    "category": "transition"
                }
            ]
        },
        {
            "time_range": "8.45s",
            "content": [
                {
                    "source": "pyaudio", 
                    "description": "Car horn sound effect detected",
                    "category": "sfx"
                }
            ]
        }
    ]
}