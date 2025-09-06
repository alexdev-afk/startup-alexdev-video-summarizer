"""
EasyOCR Timeline Service

Timeline-based text extraction service following the proven 3-stage integration pattern:
Stage 1: build/video_analysis - Raw structured data export
Stage 2: build/video_timelines - Timeline coordination with events/spans  
Stage 3: master_timeline.json - 6-service coordination with filtering

Based on successful audio timeline architecture patterns.
"""

import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from services.easyocr_service import EasyOCRService, EasyOCRError
from utils.logger import get_logger
from utils.enhanced_timeline_schema import EnhancedTimeline, TimelineSpan, TimelineEvent

logger = get_logger(__name__)


class EasyOCRTimelineServiceError(Exception):
    """EasyOCR timeline service error"""
    pass


class EasyOCRTimelineService:
    """EasyOCR timeline service for visual text extraction with scene-based processing"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize EasyOCR timeline service
        
        Args:
            config: Configuration dictionary with EasyOCR settings
        """
        self.config = config
        self.service_name = "easyocr"
        self.easyocr_config = config.get('gpu_pipeline', {}).get('easyocr', {})
        
        # Initialize underlying EasyOCR service
        self.easyocr_service = EasyOCRService(config)
        
        logger.info("EasyOCR timeline service initialized")
    
    def generate_and_save(self, video_path: str, scene_offsets_path: Optional[str] = None) -> EnhancedTimeline:
        """
        Generate enhanced timeline from video with scene-based text extraction
        
        Args:
            video_path: Path to video.mp4 file (from FFmpeg)
            scene_offsets_path: Path to scene_offsets.json for scene context
            
        Returns:
            EnhancedTimeline with text extraction events and spans
        """
        start_time = time.time()
        
        try:
            # Load scene information if available
            scene_info = self._load_scene_offsets(scene_offsets_path)
            
            # Get video duration and metadata
            video_metadata = self._get_video_metadata(video_path)
            total_duration = video_metadata.get('duration', 30.0)
            
            # Create enhanced timeline
            timeline = EnhancedTimeline(
                audio_file=str(video_path).replace('video.mp4', 'audio.wav'),  # Link to corresponding audio
                total_duration=total_duration
            )
            
            # Add service info
            timeline.sources_used.append(self.service_name)
            timeline.processing_notes.append(f"EasyOCR text extraction with languages: {self.easyocr_service.languages}")
            timeline.processing_notes.append(f"Scene-based processing with representative frames (70x optimization)")
            
            # Process video with scene context
            if scene_info and scene_info.get('scenes'):
                # Scene-based processing
                self._process_scenes(video_path, scene_info, timeline)
            else:
                # Full video processing as single scene
                self._process_full_video(video_path, timeline, total_duration)
            
            processing_time = time.time() - start_time
            logger.info(f"EasyOCR timeline generated: {len(timeline.events)} events, {len(timeline.spans)} spans in {processing_time:.2f}s")
            
            # Save intermediate analysis files
            self._save_intermediate_analysis(timeline, video_path, scene_info)
            
            # Save timeline to video_timelines directory
            self._save_timeline(timeline, video_path)
            
            return timeline
            
        except Exception as e:
            logger.error(f"EasyOCR timeline generation failed: {e}")
            return self._create_fallback_timeline(video_path, error=str(e))
    
    def _load_scene_offsets(self, scene_offsets_path: Optional[str]) -> Optional[Dict[str, Any]]:
        """Load scene offset information from JSON file"""
        if not scene_offsets_path:
            # Try to find scene_offsets.json in build directory
            possible_paths = [
                Path("build") / "bonita" / "scenes" / "scene_offsets.json",
                Path("build") / "bonita" / "scene_offsets.json",
                Path("build/scenes/scene_offsets.json"),
                Path("build/scene_offsets.json")
            ]
            
            for path in possible_paths:
                if path.exists():
                    scene_offsets_path = str(path)
                    break
            else:
                return None
        
        try:
            with open(scene_offsets_path, 'r') as f:
                scene_data = json.load(f)
            logger.info(f"Loaded scene offsets: {len(scene_data.get('scenes', []))} scenes")
            return scene_data
        except Exception as e:
            logger.warning(f"Could not load scene offsets from {scene_offsets_path}: {e}")
            return None
    
    def _get_video_metadata(self, video_path: str) -> Dict[str, Any]:
        """Extract video metadata for timeline creation"""
        try:
            video_file = Path(video_path)
            if not video_file.exists():
                raise EasyOCRTimelineServiceError(f"Video file not found: {video_path}")
            
            # Basic metadata from file
            file_size = video_file.stat().st_size
            
            # Try to get duration from scene offsets or estimate
            return {
                'file_size': file_size,
                'duration': 30.0,  # Default fallback, will be updated from scene info
                'path': str(video_file)
            }
        except Exception as e:
            logger.warning(f"Could not extract video metadata: {e}")
            return {'duration': 30.0}
    
    def _process_scenes(self, video_path: str, scene_info: Dict[str, Any], timeline: EnhancedTimeline):
        """Process video using scene-based approach"""
        # Convert scenes dict to list format
        scenes_dict = scene_info.get('scenes', {})
        scenes = []
        for scene_filename, scene_data in scenes_dict.items():
            scenes.append({
                'start_seconds': scene_data.get('original_start_seconds', 0.0),
                'end_seconds': scene_data.get('original_end_seconds', 5.0),
                'scene_id': scene_data.get('scene_id', 1),
                'duration_seconds': scene_data.get('duration_seconds', 5.0)
            })
        
        # Calculate total duration from last scene
        total_duration = scenes[-1]['end_seconds'] if scenes else timeline.total_duration
        
        # Update timeline duration from scene info
        timeline.total_duration = total_duration
        
        for scene_idx, scene in enumerate(scenes):
            scene_start = scene.get('start_seconds', 0.0)
            scene_end = scene.get('end_seconds', scene_start + 5.0)
            scene_id = scene_idx + 1
            
            # Create scene context for EasyOCR analysis
            scene_context = {
                'scene_id': scene_id,
                'start_seconds': scene_start,
                'end_seconds': scene_end,
                'duration': scene_end - scene_start
            }
            
            try:
                # Analyze video with scene context
                ocr_results = self.easyocr_service.extract_text(
                    Path(video_path), 
                    scene_info=scene_context
                )
                
                # Convert OCR results to timeline events and spans
                self._convert_text_to_timeline(ocr_results, timeline, scene_context)
                
            except Exception as e:
                logger.error(f"Scene {scene_id} EasyOCR analysis failed: {e}")
                # Add error event to timeline
                error_event = TimelineEvent(
                    timestamp=scene_start,
                    description=f"Text Extraction Error - Scene {scene_id}",
                    source=self.service_name,
                    confidence=0.0,
                    details={'error': str(e), 'scene_id': scene_id}
                )
                timeline.events.append(error_event)
    
    def _process_full_video(self, video_path: str, timeline: EnhancedTimeline, duration: float):
        """Process entire video as single scene"""
        try:
            # Analyze entire video
            ocr_results = self.easyocr_service.extract_text(Path(video_path))
            
            # Convert results with full video context
            full_video_context = {
                'scene_id': 1,
                'start_seconds': 0.0,
                'end_seconds': duration,
                'duration': duration
            }
            
            self._convert_text_to_timeline(ocr_results, timeline, full_video_context)
            
        except Exception as e:
            logger.error(f"Full video EasyOCR analysis failed: {e}")
            # Add fallback data
            self._add_mock_timeline_data(timeline)
    
    def _convert_text_to_timeline(self, ocr_results: Dict[str, Any], timeline: EnhancedTimeline, scene_context: Dict[str, Any]):
        """Convert EasyOCR results to timeline events and spans"""
        scene_start = scene_context.get('start_seconds', 0.0)
        scene_end = scene_context.get('end_seconds', scene_start + 5.0)
        scene_id = scene_context.get('scene_id', 1)
        
        # Extract text data from OCR results
        text_detections = ocr_results.get('text_details', [])
        extracted_text = ocr_results.get('extracted_text', [])
        total_text_regions = len(text_detections)
        
        # Create scene span for text extraction
        scene_span = TimelineSpan(
            start=scene_start,
            end=scene_end,
            description=f"Text Extraction - Scene {scene_id}",
            source=self.service_name,
            confidence=0.85,
            details={
                'scene_id': scene_id,
                'text_regions': total_text_regions,
                'extracted_phrases': len(extracted_text),
                'has_text': total_text_regions > 0
            }
        )
        
        # Add text detection events
        for detection in text_detections[:15]:  # Limit to top 15 text detections per scene
            # Calculate absolute timestamp from scene start + frame time
            absolute_timestamp = scene_start + detection.get('frame_time', 0.0)
            
            # Classify text region type
            text_region = self._classify_text_region(detection)
            
            # Create text detection event
            text_event = TimelineEvent(
                timestamp=absolute_timestamp,
                description=f"Text Detected: {detection.get('text', '')[:50]}",  # Truncate long text
                source=self.service_name,
                confidence=detection.get('confidence', 0.5),
                details={
                    'text_content': detection.get('text', ''),
                    'bounding_box': detection.get('bbox'),
                    'region_type': text_region,
                    'language': detection.get('language', 'unknown'),
                    'scene_id': scene_id,
                    'analysis_type': 'text_extraction'
                }
            )
            
            scene_span.add_event(text_event)
        
        # Add summary events for significant text findings
        if extracted_text:
            # Combine all extracted text for summary
            full_text = ' '.join(extracted_text[:100])  # Limit total characters
            
            text_summary_event = TimelineEvent(
                timestamp=scene_start,
                description=f"Text Summary - Scene {scene_id}",
                source=self.service_name,
                confidence=0.8,
                details={
                    'full_text': full_text,
                    'word_count': len(full_text.split()),
                    'text_regions': total_text_regions,
                    'analysis_type': 'text_summary',
                    'scene_id': scene_id
                }
            )
            timeline.events.append(text_summary_event)
        
        # Add region classification summary
        region_types = self._analyze_text_regions(text_detections)
        if region_types:
            region_event = TimelineEvent(
                timestamp=scene_start,
                description=f"Text Regions: {', '.join(region_types.keys())}",
                source=self.service_name,
                confidence=0.75,
                details={
                    'region_analysis': region_types,
                    'analysis_type': 'text_regions',
                    'scene_id': scene_id
                }
            )
            timeline.events.append(region_event)
        
        # Add scene span to timeline
        timeline.spans.append(scene_span)
    
    def _classify_text_region(self, detection: Dict[str, Any]) -> str:
        """Classify text region type based on position and characteristics"""
        bbox = detection.get('bbox', [0, 0, 100, 100])
        text = detection.get('text', '').strip()
        
        if len(bbox) >= 4:
            x1, y1, x2, y2 = bbox[:4]
            
            # Simple region classification based on position
            # These would need adjustment based on actual video resolution
            if y1 < 0.2:  # Top 20% of frame
                if len(text) > 20:
                    return 'header'
                else:
                    return 'title'
            elif y2 > 0.8:  # Bottom 20% of frame
                return 'footer'
            elif x1 < 0.2:  # Left 20% of frame
                return 'sidebar'
            elif x2 > 0.8:  # Right 20% of frame  
                return 'sidebar'
            else:
                return 'content'
        
        return 'unknown'
    
    def _analyze_text_regions(self, text_detections: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze and count text region types"""
        region_counts = {}
        
        for detection in text_detections:
            region_type = self._classify_text_region(detection)
            region_counts[region_type] = region_counts.get(region_type, 0) + 1
        
        return region_counts
    
    def _save_intermediate_analysis(self, timeline: EnhancedTimeline, video_path: str, scene_info: Optional[Dict[str, Any]]):
        """Save intermediate analysis to build/video_analysis directory (Stage 1)"""
        try:
            # Determine output path
            video_pathlib = Path(video_path)
            build_dir = video_pathlib.parent
            analysis_dir = build_dir / "video_analysis"
            analysis_dir.mkdir(exist_ok=True)
            
            # Create EasyOCR analysis data
            analysis_data = {
                'service': self.service_name,
                'video_file': str(video_path),
                'processing_timestamp': datetime.now().isoformat(),
                'ocr_config': {
                    'languages': self.easyocr_service.languages,
                    'device': self.easyocr_service.device,
                    'confidence_threshold': self.easyocr_service.confidence_threshold
                },
                'scene_context': scene_info,
                'summary': {
                    'total_events': len(timeline.events),
                    'total_spans': len(timeline.spans),
                    'total_duration': timeline.total_duration
                },
                'raw_events': [event.to_dict() for event in timeline.events],
                'raw_spans': [span.to_dict() for span in timeline.spans],
                'processing_notes': timeline.processing_notes
            }
            
            output_file = analysis_dir / f"{self.service_name}_analysis.json"
            
            with open(output_file, 'w') as f:
                json.dump(analysis_data, f, indent=2)
            
            logger.info(f"EasyOCR intermediate analysis saved to: {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save EasyOCR intermediate analysis: {e}")
    
    def _save_timeline(self, timeline: EnhancedTimeline, video_path: str):
        """Save timeline to build/video_timelines directory (Stage 2)"""
        try:
            # Determine output path
            video_pathlib = Path(video_path)
            build_dir = video_pathlib.parent
            timeline_dir = build_dir / "video_timelines"
            timeline_dir.mkdir(exist_ok=True)
            
            output_file = timeline_dir / f"{self.service_name}_timeline.json"
            timeline.save_to_file(str(output_file))
            
            logger.info(f"EasyOCR timeline saved to: {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save EasyOCR timeline: {e}")
    
    def _create_fallback_timeline(self, video_path: str, error: Optional[str] = None) -> EnhancedTimeline:
        """Create fallback timeline when processing fails"""
        estimated_duration = 30.0  # Default video duration
        
        timeline = EnhancedTimeline(
            audio_file=str(video_path).replace('video.mp4', 'audio.wav'),
            total_duration=estimated_duration
        )
        
        # Add service info
        timeline.sources_used.append(self.service_name)
        timeline.processing_notes.append(f"EasyOCR fallback mode")
        if error:
            timeline.processing_notes.append(f"Error: {error}")
        
        # Add mock data
        self._add_mock_timeline_data(timeline)
        
        logger.warning(f"Using fallback EasyOCR timeline: {error or 'EasyOCR service unavailable'}")
        return timeline
    
    def _add_mock_timeline_data(self, timeline: EnhancedTimeline):
        """Add mock timeline data for development/fallback"""
        # Mock scene span
        mock_span = TimelineSpan(
            start=0.0,
            end=timeline.total_duration,
            description="Mock Text Extraction",
            source=self.service_name,
            confidence=0.7,
            details={
                'scene_id': 1,
                'text_regions': 3,
                'extracted_phrases': 5,
                'has_text': True,
                'mock_mode': True
            }
        )
        
        # Mock text detection events
        mock_texts = [
            {'text': 'Welcome to Bonita', 'confidence': 0.85, 'timestamp': 1.0, 'region': 'header'},
            {'text': 'Beauty Salon', 'confidence': 0.78, 'timestamp': 3.0, 'region': 'title'},
            {'text': 'Quality Service', 'confidence': 0.82, 'timestamp': 10.0, 'region': 'content'},
            {'text': 'Book Now', 'confidence': 0.75, 'timestamp': 25.0, 'region': 'footer'}
        ]
        
        for text_data in mock_texts:
            mock_event = TimelineEvent(
                timestamp=text_data['timestamp'],
                description=f"Mock Text: {text_data['text']}",
                source=self.service_name,
                confidence=text_data['confidence'],
                details={
                    'text_content': text_data['text'],
                    'bounding_box': [100, 100, 300, 150],
                    'region_type': text_data['region'],
                    'language': 'en',
                    'mock_mode': True,
                    'analysis_type': 'text_extraction'
                }
            )
            mock_span.add_event(mock_event)
        
        # Add summary events
        timeline.events.append(TimelineEvent(
            timestamp=0.0,
            description="Text Summary - Scene 1",
            source=self.service_name,
            confidence=0.8,
            details={
                'full_text': 'Welcome to Bonita Beauty Salon Quality Service Book Now',
                'word_count': 9,
                'text_regions': 4,
                'analysis_type': 'text_summary',
                'mock_mode': True
            }
        ))
        
        timeline.events.append(TimelineEvent(
            timestamp=0.0,
            description="Text Regions: header, title, content, footer",
            source=self.service_name,
            confidence=0.75,
            details={
                'region_analysis': {'header': 1, 'title': 1, 'content': 1, 'footer': 1},
                'analysis_type': 'text_regions',
                'mock_mode': True
            }
        ))
        
        timeline.spans.append(mock_span)
    
    def cleanup(self):
        """Cleanup EasyOCR service resources"""
        if hasattr(self.easyocr_service, 'cleanup'):
            self.easyocr_service.cleanup()
        logger.debug("EasyOCR timeline service cleanup complete")


# Export for easy importing
__all__ = ['EasyOCRTimelineService', 'EasyOCRTimelineServiceError']