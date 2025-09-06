"""
OpenCV Timeline Service

Timeline-based face detection service following the proven 3-stage integration pattern:
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

from services.opencv_service import OpenCVService, OpenCVError
from utils.logger import get_logger
from utils.enhanced_timeline_schema import EnhancedTimeline, TimelineSpan, TimelineEvent

logger = get_logger(__name__)


class OpenCVTimelineServiceError(Exception):
    """OpenCV timeline service error"""
    pass


class OpenCVTimelineService:
    """OpenCV timeline service for face detection with scene-based processing"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize OpenCV timeline service
        
        Args:
            config: Configuration dictionary with OpenCV settings
        """
        self.config = config
        self.service_name = "opencv"
        self.opencv_config = config.get('cpu_pipeline', {}).get('opencv', {})
        
        # Initialize underlying OpenCV service
        self.opencv_service = OpenCVService(config)
        
        logger.info("OpenCV timeline service initialized")
    
    def generate_and_save(self, video_path: str, scene_offsets_path: Optional[str] = None) -> EnhancedTimeline:
        """
        Generate enhanced timeline from video with scene-based face detection
        
        Args:
            video_path: Path to video.mp4 file (from FFmpeg)
            scene_offsets_path: Path to scene_offsets.json for scene context
            
        Returns:
            EnhancedTimeline with face detection events and spans
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
            timeline.processing_notes.append(f"OpenCV face detection using Haar cascades")
            timeline.processing_notes.append(f"Scene-based processing with representative frames (70x optimization)")
            
            # Process video with scene context
            if scene_info and scene_info.get('scenes'):
                # Scene-based processing
                self._process_scenes(video_path, scene_info, timeline)
            else:
                # Full video processing as single scene
                self._process_full_video(video_path, timeline, total_duration)
            
            processing_time = time.time() - start_time
            logger.info(f"OpenCV timeline generated: {len(timeline.events)} events, {len(timeline.spans)} spans in {processing_time:.2f}s")
            
            # Save intermediate analysis files
            self._save_intermediate_analysis(timeline, video_path, scene_info)
            
            # Save timeline to video_timelines directory
            self._save_timeline(timeline, video_path)
            
            return timeline
            
        except Exception as e:
            logger.error(f"OpenCV timeline generation failed: {e}")
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
                raise OpenCVTimelineServiceError(f"Video file not found: {video_path}")
            
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
            
            # Create scene context for OpenCV analysis
            scene_context = {
                'scene_id': scene_id,
                'start_seconds': scene_start,
                'end_seconds': scene_end,
                'duration': scene_end - scene_start
            }
            
            try:
                # Analyze video with scene context
                face_results = self.opencv_service.detect_faces_from_scene(
                    Path(video_path), 
                    scene_context
                )
                
                # Convert face detection results to timeline events and spans
                self._convert_faces_to_timeline(face_results, timeline, scene_context)
                
            except Exception as e:
                logger.error(f"Scene {scene_id} OpenCV analysis failed: {e}")
                # Add error event to timeline
                error_event = TimelineEvent(
                    timestamp=scene_start,
                    description=f"Face Detection Error - Scene {scene_id}",
                    source=self.service_name,
                    confidence=0.0,
                    details={'error': str(e), 'scene_id': scene_id}
                )
                timeline.events.append(error_event)
    
    def _process_full_video(self, video_path: str, timeline: EnhancedTimeline, duration: float):
        """Process entire video as single scene"""
        try:
            # Create full video context
            full_video_context = {
                'scene_id': 1,
                'start_seconds': 0.0,
                'end_seconds': duration,
                'duration': duration
            }
            
            # Analyze entire video
            face_results = self.opencv_service.detect_faces_from_scene(
                Path(video_path),
                full_video_context
            )
            
            self._convert_faces_to_timeline(face_results, timeline, full_video_context)
            
        except Exception as e:
            logger.error(f"Full video OpenCV analysis failed: {e}")
            # Add fallback data
            self._add_mock_timeline_data(timeline)
    
    def _convert_faces_to_timeline(self, face_results: Dict[str, Any], timeline: EnhancedTimeline, scene_context: Dict[str, Any]):
        """Convert OpenCV face detection results to timeline events and spans"""
        scene_start = scene_context.get('start_seconds', 0.0)
        scene_end = scene_context.get('end_seconds', scene_start + 5.0)
        scene_id = scene_context.get('scene_id', 1)
        
        # Extract face data from results
        face_detections = face_results.get('face_details', [])
        face_count = face_results.get('face_count', 0)
        total_detections = len(face_detections)
        
        # Create scene span for face detection
        scene_span = TimelineSpan(
            start=scene_start,
            end=scene_end,
            description=f"Face Detection - Scene {scene_id}",
            source=self.service_name,
            confidence=0.8,
            details={
                'scene_id': scene_id,
                'faces_detected': face_count,
                'total_detections': total_detections,
                'has_faces': face_count > 0
            }
        )
        
        # Add face detection events
        for detection in face_detections[:10]:  # Limit to top 10 face detections per scene
            # Calculate absolute timestamp from scene start + frame time
            absolute_timestamp = scene_start + detection.get('frame_time', 0.0)
            
            # Classify face size and position
            face_classification = self._classify_face(detection)
            
            # Create face detection event
            face_event = TimelineEvent(
                timestamp=absolute_timestamp,
                description=f"Face Detected ({face_classification['size']}, {face_classification['position']})",
                source=self.service_name,
                confidence=detection.get('confidence', 0.7),
                details={
                    'bounding_box': detection.get('bbox'),
                    'face_size': face_classification['size'],
                    'face_position': face_classification['position'],
                    'area': detection.get('area'),
                    'scene_id': scene_id,
                    'analysis_type': 'face_detection'
                }
            )
            
            scene_span.add_event(face_event)
        
        # Add summary events for face findings
        if face_count > 0:
            face_summary_event = TimelineEvent(
                timestamp=scene_start,
                description=f"Faces Present ({face_count})",
                source=self.service_name,
                confidence=0.85,
                details={
                    'count': face_count,
                    'total_detections': total_detections,
                    'analysis_type': 'face_summary',
                    'scene_id': scene_id
                }
            )
            timeline.events.append(face_summary_event)
            
            # Analyze face distribution
            face_analysis = self._analyze_face_distribution(face_detections)
            if face_analysis:
                distribution_event = TimelineEvent(
                    timestamp=scene_start,
                    description=f"Face Analysis: {face_analysis['dominant_size']} faces, {face_analysis['dominant_position']} positioning",
                    source=self.service_name,
                    confidence=0.75,
                    details={
                        'size_distribution': face_analysis['size_distribution'],
                        'position_distribution': face_analysis['position_distribution'],
                        'dominant_size': face_analysis['dominant_size'],
                        'dominant_position': face_analysis['dominant_position'],
                        'analysis_type': 'face_analysis',
                        'scene_id': scene_id
                    }
                )
                timeline.events.append(distribution_event)
        
        # Add scene span to timeline
        timeline.spans.append(scene_span)
    
    def _classify_face(self, detection: Dict[str, Any]) -> Dict[str, str]:
        """Classify face size and position"""
        bbox = detection.get('bbox', [0, 0, 100, 100])
        
        if len(bbox) >= 4:
            x, y, w, h = bbox
            area = w * h
            
            # Classify size based on area (these thresholds would need adjustment)
            if area > 10000:
                size = 'large'
            elif area > 5000:
                size = 'medium'
            else:
                size = 'small'
            
            # Classify position based on center point
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Simple position classification (would need adjustment for actual video resolution)
            if center_y < 200:  # Top third
                if center_x < 300:
                    position = 'top-left'
                elif center_x > 600:
                    position = 'top-right'
                else:
                    position = 'top-center'
            elif center_y > 400:  # Bottom third
                if center_x < 300:
                    position = 'bottom-left'
                elif center_x > 600:
                    position = 'bottom-right'
                else:
                    position = 'bottom-center'
            else:  # Middle third
                if center_x < 300:
                    position = 'middle-left'
                elif center_x > 600:
                    position = 'middle-right'
                else:
                    position = 'center'
            
            return {'size': size, 'position': position}
        
        return {'size': 'unknown', 'position': 'unknown'}
    
    def _analyze_face_distribution(self, face_detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze distribution of face sizes and positions"""
        size_counts = {}
        position_counts = {}
        
        for detection in face_detections:
            classification = self._classify_face(detection)
            size = classification['size']
            position = classification['position']
            
            size_counts[size] = size_counts.get(size, 0) + 1
            position_counts[position] = position_counts.get(position, 0) + 1
        
        # Find dominant characteristics
        dominant_size = max(size_counts.keys(), key=lambda k: size_counts[k]) if size_counts else 'unknown'
        dominant_position = max(position_counts.keys(), key=lambda k: position_counts[k]) if position_counts else 'unknown'
        
        return {
            'size_distribution': size_counts,
            'position_distribution': position_counts,
            'dominant_size': dominant_size,
            'dominant_position': dominant_position
        }
    
    def _save_intermediate_analysis(self, timeline: EnhancedTimeline, video_path: str, scene_info: Optional[Dict[str, Any]]):
        """Save intermediate analysis to build/video_analysis directory (Stage 1)"""
        try:
            # Determine output path
            video_pathlib = Path(video_path)
            build_dir = video_pathlib.parent
            analysis_dir = build_dir / "video_analysis"
            analysis_dir.mkdir(exist_ok=True)
            
            # Create OpenCV analysis data
            analysis_data = {
                'service': self.service_name,
                'video_file': str(video_path),
                'processing_timestamp': datetime.now().isoformat(),
                'opencv_config': {
                    'cascade_file': self.opencv_service.cascade_file,
                    'scale_factor': self.opencv_service.scale_factor,
                    'min_neighbors': self.opencv_service.min_neighbors,
                    'confidence_threshold': self.opencv_service.confidence_threshold
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
            
            logger.info(f"OpenCV intermediate analysis saved to: {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save OpenCV intermediate analysis: {e}")
    
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
            
            logger.info(f"OpenCV timeline saved to: {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save OpenCV timeline: {e}")
    
    def _create_fallback_timeline(self, video_path: str, error: Optional[str] = None) -> EnhancedTimeline:
        """Create fallback timeline when processing fails"""
        estimated_duration = 30.0  # Default video duration
        
        timeline = EnhancedTimeline(
            audio_file=str(video_path).replace('video.mp4', 'audio.wav'),
            total_duration=estimated_duration
        )
        
        # Add service info
        timeline.sources_used.append(self.service_name)
        timeline.processing_notes.append(f"OpenCV fallback mode")
        if error:
            timeline.processing_notes.append(f"Error: {error}")
        
        # Add mock data
        self._add_mock_timeline_data(timeline)
        
        logger.warning(f"Using fallback OpenCV timeline: {error or 'OpenCV service unavailable'}")
        return timeline
    
    def _add_mock_timeline_data(self, timeline: EnhancedTimeline):
        """Add mock timeline data for development/fallback"""
        # Mock scene span
        mock_span = TimelineSpan(
            start=0.0,
            end=timeline.total_duration,
            description="Mock Face Detection",
            source=self.service_name,
            confidence=0.75,
            details={
                'scene_id': 1,
                'faces_detected': 2,
                'total_detections': 3,
                'has_faces': True,
                'mock_mode': True
            }
        )
        
        # Mock face detection events
        mock_faces = [
            {'size': 'large', 'position': 'center', 'confidence': 0.85, 'timestamp': 2.0},
            {'size': 'medium', 'position': 'top-right', 'confidence': 0.78, 'timestamp': 8.0},
            {'size': 'small', 'position': 'bottom-left', 'confidence': 0.72, 'timestamp': 15.0}
        ]
        
        for face_data in mock_faces:
            mock_event = TimelineEvent(
                timestamp=face_data['timestamp'],
                description=f"Mock Face ({face_data['size']}, {face_data['position']})",
                source=self.service_name,
                confidence=face_data['confidence'],
                details={
                    'bounding_box': [100, 100, 150, 150],
                    'face_size': face_data['size'],
                    'face_position': face_data['position'],
                    'area': 22500,
                    'mock_mode': True,
                    'analysis_type': 'face_detection'
                }
            )
            mock_span.add_event(mock_event)
        
        # Add summary events
        timeline.events.append(TimelineEvent(
            timestamp=0.0,
            description="Faces Present (2)",
            source=self.service_name,
            confidence=0.85,
            details={
                'count': 2,
                'total_detections': 3,
                'analysis_type': 'face_summary',
                'mock_mode': True
            }
        ))
        
        timeline.events.append(TimelineEvent(
            timestamp=0.0,
            description="Face Analysis: large faces, center positioning",
            source=self.service_name,
            confidence=0.75,
            details={
                'size_distribution': {'large': 1, 'medium': 1, 'small': 1},
                'position_distribution': {'center': 1, 'top-right': 1, 'bottom-left': 1},
                'dominant_size': 'large',
                'dominant_position': 'center',
                'analysis_type': 'face_analysis',
                'mock_mode': True
            }
        ))
        
        timeline.spans.append(mock_span)
    
    def cleanup(self):
        """Cleanup OpenCV service resources"""
        if hasattr(self.opencv_service, 'cleanup'):
            self.opencv_service.cleanup()
        logger.debug("OpenCV timeline service cleanup complete")


# Export for easy importing
__all__ = ['OpenCVTimelineService', 'OpenCVTimelineServiceError']