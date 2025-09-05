"""
Timeline Merger Service

Combines Whisper, LibROSA, and pyAudioAnalysis timeline outputs into a single
chronologically-sorted timeline for LLM consumption.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from utils.logger import get_logger
from utils.timeline_schema import MergedTimeline, validate_timeline_object

logger = get_logger(__name__)


class TimelineMergerError(Exception):
    """Timeline merger processing error"""
    pass


class TimelineMergerService:
    """Service to merge multiple timeline transcripts into chronologically sorted output"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize timeline merger service"""
        self.config = config
        self.merger_config = config.get('timeline_merger', {})
        
        # Conflict resolution settings
        self.priority_order = self.merger_config.get('priority_order', ['whisper', 'librosa', 'pyaudio'])
        self.confidence_threshold = self.merger_config.get('confidence_threshold', 0.5)
        self.time_precision = self.merger_config.get('time_precision', 3)  # decimal places
        self.overlap_tolerance = self.merger_config.get('overlap_tolerance', 0.1)  # seconds
        
        logger.info(f"Timeline merger initialized - priority: {self.priority_order}, confidence_threshold: {self.confidence_threshold}")
    
    def merge_timelines(
        self,
        whisper_timeline_path: Optional[str] = None,
        librosa_timeline_path: Optional[str] = None, 
        pyaudio_timeline_path: Optional[str] = None,
        audio_file: str = "",
        total_duration: float = 0.0
    ) -> MergedTimeline:
        """
        Merge timeline files from all three services
        
        Args:
            whisper_timeline_path: Path to Whisper timeline JSON
            librosa_timeline_path: Path to LibROSA timeline JSON  
            pyaudio_timeline_path: Path to pyAudioAnalysis timeline JSON
            audio_file: Original audio file name
            total_duration: Total audio duration in seconds
            
        Returns:
            MergedTimeline object with chronologically sorted content
        """
        start_time = time.time()
        
        try:
            # Load all available timelines
            timelines = self._load_timeline_files(
                whisper_timeline_path,
                librosa_timeline_path,
                pyaudio_timeline_path
            )
            
            if not timelines:
                raise TimelineMergerError("No valid timeline files found")
            
            # Extract and validate all timeline objects
            all_objects = self._extract_all_timeline_objects(timelines)
            
            # Sort chronologically
            sorted_objects = self._sort_chronologically(all_objects)
            
            # Create time segments
            time_segments = self._create_time_segments(sorted_objects, total_duration)
            
            # Build merged timeline
            merged_timeline = MergedTimeline(audio_file, total_duration)
            
            # Record source files
            source_map = {
                'whisper': whisper_timeline_path,
                'librosa': librosa_timeline_path,
                'pyaudio': pyaudio_timeline_path
            }
            for source, path in source_map.items():
                if path:
                    merged_timeline.set_source_file(source, path)
            
            # Add time segments
            for segment in time_segments:
                merged_timeline.add_timeline_segment(
                    segment['time_range'],
                    segment['content']
                )
            
            processing_time = time.time() - start_time
            logger.info(f"Timeline merge complete: {len(time_segments)} segments, {len(all_objects)} total objects, {processing_time:.2f}s")
            
            return merged_timeline
            
        except Exception as e:
            logger.error(f"Timeline merge failed: {e}")
            # Return empty timeline instead of failing
            empty_timeline = MergedTimeline(audio_file, total_duration)
            empty_timeline.add_timeline_segment(
                "0.0s-end",
                [{"source": "merger", "description": f"Timeline merge failed: {e}", "category": "error"}]
            )
            return empty_timeline
    
    def _load_timeline_files(
        self,
        whisper_path: Optional[str],
        librosa_path: Optional[str],
        pyaudio_path: Optional[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Load timeline JSON files"""
        timelines = {}
        
        file_map = {
            'whisper': whisper_path,
            'librosa': librosa_path, 
            'pyaudio': pyaudio_path
        }
        
        for source, path in file_map.items():
            if path and Path(path).exists():
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        timeline_data = json.load(f)
                        
                    # Validate basic structure
                    if 'timeline' in timeline_data and isinstance(timeline_data['timeline'], list):
                        timelines[source] = timeline_data
                        logger.info(f"Loaded {source} timeline: {len(timeline_data['timeline'])} objects")
                    else:
                        logger.warning(f"Invalid timeline structure in {path}")
                        
                except Exception as e:
                    logger.warning(f"Failed to load {source} timeline from {path}: {e}")
            else:
                logger.info(f"No {source} timeline provided or file not found")
        
        return timelines
    
    def _extract_all_timeline_objects(self, timelines: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract and validate all timeline objects from loaded timelines"""
        all_objects = []
        
        for source, timeline_data in timelines.items():
            timeline_objects = timeline_data.get('timeline', [])
            
            for obj in timeline_objects:
                if validate_timeline_object(obj):
                    all_objects.append(obj)
                else:
                    logger.warning(f"Invalid timeline object from {source}: {obj}")
        
        logger.info(f"Extracted {len(all_objects)} valid timeline objects")
        return all_objects
    
    def _sort_chronologically(self, objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort timeline objects chronologically"""
        def get_sort_key(obj: Dict[str, Any]) -> float:
            """Get timestamp for sorting (events use timestamp, spans use start time)"""
            if obj['type'] == 'event':
                return obj['timestamp']
            elif obj['type'] == 'span':
                return obj['start']
            else:
                return 0.0
        
        sorted_objects = sorted(objects, key=get_sort_key)
        logger.debug(f"Objects sorted chronologically: {len(sorted_objects)} objects")
        
        return sorted_objects
    
    def _create_time_segments(self, sorted_objects: List[Dict[str, Any]], total_duration: float) -> List[Dict[str, Any]]:
        """Create time segments by grouping overlapping or adjacent timeline objects"""
        if not sorted_objects:
            return []
        
        segments = []
        current_segment_content = []
        current_start_time = None
        current_end_time = None
        
        for obj in sorted_objects:
            obj_start, obj_end = self._get_object_time_range(obj)
            
            # First object
            if current_start_time is None:
                current_start_time = obj_start
                current_end_time = obj_end
                current_segment_content = [self._simplify_object(obj)]
                continue
            
            # Check if this object overlaps or is close to current segment
            if obj_start <= current_end_time + self.overlap_tolerance:
                # Add to current segment
                current_segment_content.append(self._simplify_object(obj))
                current_end_time = max(current_end_time, obj_end)
            else:
                # Start new segment - save current segment first
                segments.append({
                    'time_range': self._format_time_range(current_start_time, current_end_time),
                    'content': current_segment_content
                })
                
                # Start new segment
                current_start_time = obj_start
                current_end_time = obj_end
                current_segment_content = [self._simplify_object(obj)]
        
        # Add final segment
        if current_segment_content:
            segments.append({
                'time_range': self._format_time_range(current_start_time, current_end_time),
                'content': current_segment_content
            })
        
        logger.info(f"Created {len(segments)} time segments")
        return segments
    
    def _get_object_time_range(self, obj: Dict[str, Any]) -> Tuple[float, float]:
        """Get start and end time for any timeline object"""
        if obj['type'] == 'event':
            timestamp = obj['timestamp']
            return timestamp, timestamp
        elif obj['type'] == 'span':
            return obj['start'], obj['end']
        else:
            return 0.0, 0.0
    
    def _simplify_object(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        """Simplify timeline object for merged output"""
        return {
            'source': obj['source'],
            'description': obj['description'],
            'category': obj['category'],
            'confidence': obj.get('confidence', 1.0),
            'details': obj.get('details', {})
        }
    
    def _format_time_range(self, start_time: float, end_time: float) -> str:
        """Format time range as string"""
        if start_time == end_time:
            return f"{start_time:.{self.time_precision}f}s"
        else:
            return f"{start_time:.{self.time_precision}f}-{end_time:.{self.time_precision}f}s"
    
    def save_merged_timeline(self, merged_timeline: MergedTimeline, output_path: str):
        """Save merged timeline to file"""
        try:
            merged_timeline.save_to_file(output_path)
            logger.info(f"Merged timeline saved to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save merged timeline to {output_path}: {e}")
            raise TimelineMergerError(f"Save failed: {e}")
    
    def merge_and_save(
        self,
        output_path: str,
        whisper_timeline_path: Optional[str] = None,
        librosa_timeline_path: Optional[str] = None,
        pyaudio_timeline_path: Optional[str] = None,
        audio_file: str = "",
        total_duration: float = 0.0
    ) -> MergedTimeline:
        """Convenience method to merge timelines and save to file"""
        merged_timeline = self.merge_timelines(
            whisper_timeline_path,
            librosa_timeline_path, 
            pyaudio_timeline_path,
            audio_file,
            total_duration
        )
        
        self.save_merged_timeline(merged_timeline, output_path)
        return merged_timeline
    
    def cleanup(self):
        """Cleanup resources"""
        logger.debug("Timeline merger cleanup complete")


# Utility functions for working with merged timelines
def load_merged_timeline(file_path: str) -> Dict[str, Any]:
    """Load merged timeline from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load merged timeline from {file_path}: {e}")
        raise


def extract_llm_ready_text(merged_timeline: Dict[str, Any]) -> str:
    """Extract LLM-ready text description from merged timeline"""
    timeline_text = []
    
    for segment in merged_timeline.get('merged_timeline', []):
        time_range = segment.get('time_range', 'Unknown time')
        content_descriptions = []
        
        for content_item in segment.get('content', []):
            description = content_item.get('description', 'No description')
            source = content_item.get('source', 'unknown')
            content_descriptions.append(f"{description} [{source}]")
        
        segment_text = f"{time_range}: " + "; ".join(content_descriptions)
        timeline_text.append(segment_text)
    
    return "\n".join(timeline_text)


def get_timeline_summary(merged_timeline: Dict[str, Any]) -> Dict[str, Any]:
    """Get summary statistics from merged timeline"""
    timeline_segments = merged_timeline.get('merged_timeline', [])
    
    categories = {}
    sources = {}
    total_content_items = 0
    
    for segment in timeline_segments:
        for content_item in segment.get('content', []):
            category = content_item.get('category', 'unknown')
            source = content_item.get('source', 'unknown')
            
            categories[category] = categories.get(category, 0) + 1
            sources[source] = sources.get(source, 0) + 1
            total_content_items += 1
    
    return {
        'total_segments': len(timeline_segments),
        'total_content_items': total_content_items,
        'categories': categories,
        'sources': sources,
        'duration': merged_timeline.get('total_duration', 0.0)
    }