"""
YOLO Timeline Service

Semantic object detection timeline service following the proven 3-stage integration pattern:
Stage 1: build/video_analysis - Processing metadata and summary statistics only
Stage 2: build/video_timelines - Semantic timeline events with LLM-ready descriptions  
Stage 3: master_timeline.json - 6-service coordination with filtering

Transforms raw YOLO detections into semantic events like:
- "Person enters scene from left" 
- "Two people begin interaction"
- "Object composition changes significantly"
"""

import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime
import numpy as np

from services.yolo_service import YOLOService, YOLOError
from services.motion_aware_sampler import MotionAwareSampler
from utils.logger import get_logger
from utils.enhanced_timeline_schema import EnhancedTimeline, TimelineSpan, TimelineEvent

logger = get_logger(__name__)


class YOLOTimelineServiceError(Exception):
    """YOLO timeline service error"""
    pass


class SceneState:
    """Tracks scene composition for semantic change detection"""
    
    def __init__(self):
        self.active_objects = {}  # class_name -> {count: int, positions: List[bbox], last_seen: timestamp}
        self.scene_center_objects = set()  # Objects in center of scene
        self.interaction_pairs = set()  # Tuples of interacting object IDs
        self.last_composition_hash = ""
        
    def update(self, detections: List[Dict], timestamp: float):
        """Update scene state with new detections"""
        current_objects = {}
        
        for detection in detections:
            class_name = detection['class']
            bbox = detection['bounding_box']
            
            if class_name not in current_objects:
                current_objects[class_name] = {'count': 0, 'positions': [], 'last_seen': timestamp}
            
            current_objects[class_name]['count'] += 1
            current_objects[class_name]['positions'].append(bbox)
            current_objects[class_name]['last_seen'] = timestamp
        
        self.active_objects = current_objects
    
    def detect_changes(self, new_detections: List[Dict], timestamp: float, prev_state: 'SceneState') -> List[Dict]:
        """Detect semantic changes between scene states"""
        changes = []
        
        # Object appearance/disappearance
        current_classes = set(d['class'] for d in new_detections)
        previous_classes = set(prev_state.active_objects.keys()) if prev_state else set()
        
        # New objects entering scene
        for new_class in current_classes - previous_classes:
            entry_direction = self._detect_entry_direction(new_detections, new_class)
            changes.append({
                'type': 'object_appearance',
                'timestamp': timestamp,
                'description': f"{new_class} enters scene from {entry_direction}",
                'confidence': 0.85,
                'details': {
                    'object_class': new_class,
                    'entry_direction': entry_direction,
                    'analysis_type': 'object_appearance'
                }
            })
        
        # Objects exiting scene
        for exited_class in previous_classes - current_classes:
            changes.append({
                'type': 'object_disappearance', 
                'timestamp': timestamp,
                'description': f"{exited_class} exits scene",
                'confidence': 0.80,
                'details': {
                    'object_class': exited_class,
                    'analysis_type': 'object_disappearance'
                }
            })
        
        # Significant composition changes
        if self._detect_composition_change(new_detections, prev_state):
            changes.append({
                'type': 'composition_change',
                'timestamp': timestamp, 
                'description': "Scene composition changes significantly",
                'confidence': 0.75,
                'details': {
                    'analysis_type': 'scene_change',
                    'change_magnitude': self._calculate_change_magnitude(new_detections, prev_state)
                }
            })
        
        return changes
    
    def _detect_entry_direction(self, detections: List[Dict], class_name: str) -> str:
        """Detect which direction an object entered from"""
        for detection in detections:
            if detection['class'] == class_name:
                bbox = detection['bounding_box']
                center_x = (bbox[0] + bbox[2]) / 2
                
                # Assume 1920x1080 resolution for direction detection
                if center_x < 640:
                    return "left"
                elif center_x > 1280:
                    return "right"
                else:
                    return "center"
        return "unknown"
    
    def _detect_composition_change(self, new_detections: List[Dict], prev_state: 'SceneState') -> bool:
        """Detect if scene composition changed significantly"""
        if not prev_state or not prev_state.active_objects:
            return len(new_detections) > 0
        
        # Simple heuristic: significant change if object count changed by >30%
        current_total = len(new_detections)
        previous_total = sum(obj['count'] for obj in prev_state.active_objects.values())
        
        if previous_total == 0:
            return current_total > 0
        
        change_ratio = abs(current_total - previous_total) / previous_total
        return change_ratio > 0.3
    
    def _calculate_change_magnitude(self, new_detections: List[Dict], prev_state: 'SceneState') -> float:
        """Calculate magnitude of scene change (0.0 to 1.0)"""
        if not prev_state:
            return 0.5
        
        current_total = len(new_detections)
        previous_total = sum(obj['count'] for obj in prev_state.active_objects.values())
        
        if previous_total == 0:
            return 0.8 if current_total > 0 else 0.0
        
        return min(1.0, abs(current_total - previous_total) / previous_total)


class YOLOTimelineService:
    """Semantic YOLO timeline service for visual object detection with motion-aware processing"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize YOLO timeline service with semantic transformation capabilities
        
        Args:
            config: Configuration dictionary with YOLO settings
        """
        self.config = config
        self.service_name = "yolo"
        self.yolo_config = config.get('gpu_pipeline', {}).get('yolo', {})
        
        # Semantic event thresholds
        self.scene_change_threshold = self.yolo_config.get('scene_change_threshold', 0.3)
        self.new_object_confidence = self.yolo_config.get('new_object_confidence', 0.7)
        self.motion_significance = self.yolo_config.get('motion_significance', 25)  # pixels
        
        # Initialize underlying YOLO service and motion sampler
        self.yolo_service = YOLOService(config)
        self.motion_sampler = MotionAwareSampler(config)
        
        # Scene state tracking
        self.scene_states = {}  # scene_id -> SceneState
        
        logger.info("YOLO timeline service initialized with semantic transformation")
    
    def generate_and_save(self, video_path: str, scene_offsets_path: Optional[str] = None) -> EnhancedTimeline:
        """
        Generate enhanced timeline from video with scene-based processing
        
        Args:
            video_path: Path to video.mp4 file (from FFmpeg)
            scene_offsets_path: Path to scene_offsets.json for scene context
            
        Returns:
            EnhancedTimeline with object detection events and spans
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
            timeline.processing_notes.append(f"Motion-aware YOLO object detection using {self.yolo_service.model_name}")
            timeline.processing_notes.append(f"Semantic event detection with {self.scene_change_threshold} change threshold")
            timeline.processing_notes.append(f"Scene-based processing with motion-aware keyframes (70x optimization)")
            
            # Process video with semantic transformation
            raw_analysis_data = {}
            if scene_info and scene_info.get('scenes'):
                # Scene-based processing with motion-aware keyframes
                raw_analysis_data = self._process_scenes_with_semantic_events(video_path, scene_info, timeline)
            else:
                # Full video processing as single scene
                raw_analysis_data = self._process_full_video_semantic(video_path, timeline, total_duration)
            
            processing_time = time.time() - start_time
            logger.info(f"YOLO timeline generated: {len(timeline.events)} events, {len(timeline.spans)} spans in {processing_time:.2f}s")
            
            # Save intermediate analysis files (metadata only)
            self._save_intermediate_analysis(raw_analysis_data, video_path, scene_info)
            
            # Save semantic timeline to video_timelines directory
            self._save_timeline(timeline, video_path)
            
            return timeline
            
        except Exception as e:
            logger.error(f"YOLO timeline generation failed: {e}")
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
                raise YOLOTimelineServiceError(f"Video file not found: {video_path}")
            
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
    
    def _process_scenes_with_semantic_events(self, video_path: str, scene_info: Dict, timeline: EnhancedTimeline) -> Dict:
        """Process scenes with motion-aware keyframes and semantic event detection"""
        
        raw_analysis_data = {
            'service': self.service_name,
            'video_file': video_path,
            'processing_timestamp': datetime.now().isoformat(),
            'model_info': {
                'model_name': self.yolo_service.model_name,
                'device': self.yolo_service.device,
                'confidence_threshold': self.yolo_service.confidence_threshold,
                'development_mode': self.yolo_service.development_config.get('mock_ai_services', False)
            },
            'semantic_thresholds': {
                'scene_change_threshold': self.scene_change_threshold,
                'new_object_confidence': self.new_object_confidence,
                'motion_significance': self.motion_significance
            },
            'scene_context': scene_info,
            'processing_summary': {
                'total_scenes': len(scene_info.get('scenes', {})),
                'total_keyframes_analyzed': 0,
                'semantic_events_created': 0,
                'timeline_spans_created': 0
            },
            # CRITICAL: Include all raw detection data for validation
            'raw_detections_by_scene': {},
            'raw_keyframes_analyzed': {},
            'raw_semantic_transformations': {}
        }
        
        scenes_data = scene_info.get('scenes', {})
        prev_scene_state = None
        total_keyframes = 0
        
        for scene_file, scene_data in scenes_data.items():
            scene_id = scene_data['scene_id']
            start_time = scene_data['original_start_seconds'] 
            end_time = scene_data['original_end_seconds']
            duration = scene_data['duration_seconds']
            
            # Get motion-aware keyframes for this scene
            keyframes = self.motion_sampler.extract_motion_keyframes(
                Path(video_path), {
                    'scene_id': scene_id,
                    'start_seconds': start_time,
                    'end_seconds': end_time
                }
            )
            
            total_keyframes += len(keyframes)
            
            # CAPTURE RAW DATA: Store keyframe information
            raw_analysis_data['raw_keyframes_analyzed'][f'scene_{scene_id:03d}'] = [
                {
                    'timestamp': kf['timestamp'],
                    'keyframe_type': kf.get('keyframe_type', 'motion_based'),
                    'motion_score': kf.get('motion_score', 0.5)
                }
                for kf in keyframes
            ]
            
            # Process keyframes and detect objects
            scene_detections = []
            scene_raw_detections = []
            for keyframe in keyframes:
                detections = self._analyze_keyframe_at_timestamp(
                    video_path, keyframe['timestamp']
                )
                if detections:
                    scene_detections.extend(detections)
                    # CAPTURE RAW DATA: Store all raw YOLO detections
                    scene_raw_detections.extend([
                        {
                            'timestamp': det['timestamp'],
                            'class': det['class'],
                            'class_id': det['class_id'],
                            'confidence': det['confidence'],
                            'bounding_box': det['bounding_box'],  # [x1, y1, x2, y2]
                            'area': det['area'],
                            'keyframe_source': True
                        }
                        for det in detections
                    ])
            
            # CAPTURE RAW DATA: Store all raw detections for this scene
            raw_analysis_data['raw_detections_by_scene'][f'scene_{scene_id:03d}'] = scene_raw_detections
            
            # Create semantic events from scene analysis
            semantic_events = self._create_semantic_events_from_detections(
                scene_detections, scene_id, start_time, end_time, prev_scene_state
            )
            
            # CAPTURE RAW DATA: Store semantic transformation mapping
            raw_analysis_data['raw_semantic_transformations'][f'scene_{scene_id:03d}'] = {
                'input_raw_detections': len(scene_raw_detections),
                'output_semantic_events': len(semantic_events),
                'semantic_events': semantic_events  # Include the actual semantic events for validation
            }
            
            # Add events to timeline
            for event in semantic_events:
                timeline.events.append(TimelineEvent(
                    timestamp=event['timestamp'],
                    description=event['description'],
                    source=self.service_name,
                    confidence=event['confidence'],
                    details=event['details']
                ))
            
            # Create timeline span for this scene
            scene_span = self._create_scene_timeline_span(
                scene_detections, scene_id, start_time, end_time
            )
            timeline.spans.append(scene_span)
            
            # Update scene state for next iteration
            if scene_id not in self.scene_states:
                self.scene_states[scene_id] = SceneState()
            
            self.scene_states[scene_id].update(scene_detections, end_time)
            prev_scene_state = self.scene_states[scene_id]
        
        # Update processing summary
        raw_analysis_data['processing_summary'].update({
            'total_keyframes_analyzed': total_keyframes,
            'semantic_events_created': len(timeline.events),
            'timeline_spans_created': len(timeline.spans)
        })
        
        return raw_analysis_data
    
    def _analyze_keyframe_at_timestamp(self, video_path: str, timestamp: float) -> List[Dict]:
        """Analyze single keyframe and extract real YOLO object detections"""
        try:
            import cv2
            
            # Extract frame at specific timestamp
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"Could not open video: {video_path}")
                return []
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30.0
            
            # Seek to timestamp
            frame_number = int(timestamp * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            # Read frame
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                logger.warning(f"Could not read frame at {timestamp}s")
                return []
            
            # Run YOLO detection on frame
            if not hasattr(self.yolo_service, 'model') or self.yolo_service.model is None:
                self.yolo_service._load_model()
            
            if self.yolo_service.model is None:
                logger.warning("YOLO model not available")
                return []
            
            # Run YOLO inference
            results = self.yolo_service.model(frame, verbose=False)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i in range(len(boxes)):
                        # Extract detection info
                        box = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
                        confidence = float(boxes.conf[i].cpu().numpy())
                        class_id = int(boxes.cls[i].cpu().numpy())
                        class_name = self.yolo_service.model.names[class_id]
                        
                        # Filter by confidence threshold and basic validation
                        if confidence >= self.yolo_service.confidence_threshold:
                            # Basic size validation to reduce noise
                            if self._is_valid_detection_size(box, (frame.shape[1], frame.shape[0])):
                                detections.append({
                                    'class': class_name,
                                    'confidence': confidence,
                                    'bounding_box': box.tolist(),  # [x1, y1, x2, y2]
                                    'timestamp': timestamp,
                                    'class_id': class_id,
                                    'area': (box[2] - box[0]) * (box[3] - box[1])
                                })
            
            logger.debug(f"YOLO detected {len(detections)} objects at {timestamp:.2f}s")
            return detections
            
        except Exception as e:
            logger.error(f"Real YOLO keyframe analysis failed at {timestamp}s: {e}")
            return []
    
    def _is_valid_detection_size(self, box, frame_size) -> bool:
        """Basic size validation to filter out noise detections"""
        frame_width, frame_height = frame_size
        x1, y1, x2, y2 = box
        
        # Calculate detection area
        det_width = x2 - x1
        det_height = y2 - y1
        det_area = det_width * det_height
        frame_area = frame_width * frame_height
        
        if det_area <= 0:
            return False
            
        # Filter detections that are too small (likely noise) or too large (likely false)
        area_ratio = det_area / frame_area
        if area_ratio < 0.0001 or area_ratio > 0.9:  # Less than 0.01% or more than 90% of frame
            return False
        
        # Filter detections with unrealistic aspect ratios (likely artifacts)
        aspect_ratio = det_width / det_height if det_height > 0 else float('inf')
        if aspect_ratio > 10 or aspect_ratio < 0.1:  # Very wide or very tall boxes
            return False
            
        return True
    
    def _create_mock_detections(self, timestamp: float) -> List[Dict]:
        """Create mock detections for development/testing"""
        # Vary detections based on timestamp to simulate scene changes
        base_detections = [
            {
                'class': 'person',
                'confidence': 0.85,
                'bounding_box': [100 + int(timestamp * 10), 200, 300 + int(timestamp * 5), 500],
                'timestamp': timestamp
            }
        ]
        
        # Add more objects for later timestamps
        if timestamp > 10:
            base_detections.append({
                'class': 'chair',
                'confidence': 0.78,
                'bounding_box': [400, 300, 600, 500],
                'timestamp': timestamp
            })
        
        if timestamp > 20:
            base_detections.append({
                'class': 'person',
                'confidence': 0.82,
                'bounding_box': [700, 150, 900, 450],
                'timestamp': timestamp
            })
        
        return base_detections
    
    def _create_semantic_events_from_detections(self, detections: List[Dict], scene_id: int, 
                                              start_time: float, end_time: float, 
                                              prev_scene_state: Optional[SceneState]) -> List[Dict]:
        """Transform raw detections into semantic events"""
        semantic_events = []
        
        if not detections:
            return semantic_events
        
        # Group detections by timestamp for change detection
        detections_by_time = {}
        for detection in detections:
            ts = detection['timestamp']
            if ts not in detections_by_time:
                detections_by_time[ts] = []
            detections_by_time[ts].append(detection)
        
        # Create scene state and detect changes
        current_scene_state = SceneState()
        
        # Process each timestamp
        sorted_timestamps = sorted(detections_by_time.keys())
        for i, timestamp in enumerate(sorted_timestamps):
            frame_detections = detections_by_time[timestamp]
            
            # Update scene state
            current_scene_state.update(frame_detections, timestamp)
            
            # Detect changes (use previous scene state for first frame of new scene)
            reference_state = prev_scene_state if i == 0 and prev_scene_state else None
            changes = current_scene_state.detect_changes(frame_detections, timestamp, reference_state)
            
            semantic_events.extend(changes)
        
        # Add scene summary events
        if detections:
            object_classes = list(set(d['class'] for d in detections))
            people_count = len([d for d in detections if d['class'] == 'person'])
            
            # Create semantic description
            if people_count > 1:
                description = f"Scene contains {people_count} people"
                if len(object_classes) > 1:
                    other_objects = [obj for obj in object_classes if obj != 'person']
                    description += f" and {', '.join(other_objects)}"
            elif people_count == 1:
                description = "Single person in scene"
                if len(object_classes) > 1:
                    other_objects = [obj for obj in object_classes if obj != 'person']
                    description += f" with {', '.join(other_objects)}"
            else:
                description = f"Scene contains: {', '.join(object_classes)}"
            
            semantic_events.append({
                'type': 'scene_summary',
                'timestamp': start_time + 0.1,  # Slight offset from scene start
                'description': description,
                'confidence': 0.8,
                'details': {
                    'analysis_type': 'scene_summary',
                    'scene_id': scene_id,
                    'object_classes': object_classes,
                    'people_count': people_count,
                    'total_objects': len(detections)
                }
            })
        
        return semantic_events
    
    def _create_scene_timeline_span(self, detections: List[Dict], scene_id: int, 
                                  start_time: float, end_time: float) -> TimelineSpan:
        """Create timeline span for scene with semantic description"""
        
        if not detections:
            return TimelineSpan(
                start=start_time,
                end=end_time,
                description=f"Empty scene - Scene {scene_id}",
                source=self.service_name,
                confidence=0.6,
                details={
                    'scene_id': scene_id,
                    'object_count': 0,
                    'analysis_type': 'scene_span'
                }
            )
        
        # Analyze scene composition
        object_classes = list(set(d['class'] for d in detections))
        people_count = len([d for d in detections if d['class'] == 'person'])
        avg_confidence = sum(d['confidence'] for d in detections) / len(detections)
        
        # Create semantic span description
        if people_count > 1:
            span_description = f"Multi-person scene with {len(object_classes)} object types"
        elif people_count == 1:
            span_description = f"Single person scene with {len(object_classes)} object types"
        else:
            span_description = f"Object scene: {', '.join(object_classes)}"
        
        return TimelineSpan(
            start=start_time,
            end=end_time,
            description=span_description,
            source=self.service_name,
            confidence=min(0.9, avg_confidence + 0.1),
            details={
                'scene_id': scene_id,
                'object_classes': object_classes,
                'people_count': people_count,
                'total_objects': len(detections),
                'avg_detection_confidence': avg_confidence,
                'analysis_type': 'scene_span'
            }
        )
    
    def _save_intermediate_analysis(self, raw_analysis_data: Dict, video_path: str, scene_info: Optional[Dict]):
        """Save metadata-only analysis file following timeline service pattern"""
        
        # Create build directory structure
        # Extract video name from path like 'build/bonita/video.mp4' -> 'bonita'
        video_path_parts = Path(video_path).parts
        if 'build' in video_path_parts:
            video_name = video_path_parts[video_path_parts.index('build') + 1]
        else:
            video_name = Path(video_path).stem.replace('video', '') or 'unknown'
        
        analysis_dir = Path('build') / video_name / 'video_analysis'
        analysis_dir.mkdir(parents=True, exist_ok=True)
        
        analysis_file = analysis_dir / 'yolo_analysis.json'
        
        try:
            with open(analysis_file, 'w') as f:
                json.dump(raw_analysis_data, f, indent=2, default=str)
            
            logger.info(f"YOLO intermediate analysis saved to: {analysis_file}")
            
        except Exception as e:
            logger.error(f"Failed to save YOLO intermediate analysis: {e}")
    
    def _save_timeline(self, timeline: EnhancedTimeline, video_path: str):
        """Save full semantic timeline file following timeline service pattern"""
        
        # Create build directory structure  
        # Extract video name from path like 'build/bonita/video.mp4' -> 'bonita'
        video_path_parts = Path(video_path).parts
        if 'build' in video_path_parts:
            video_name = video_path_parts[video_path_parts.index('build') + 1]
        else:
            video_name = Path(video_path).stem.replace('video', '') or 'unknown'
            
        timeline_dir = Path('build') / video_name / 'video_timelines'
        timeline_dir.mkdir(parents=True, exist_ok=True)
        
        timeline_file = timeline_dir / 'yolo_timeline.json'
        
        try:
            # Convert timeline to dictionary format
            timeline_dict = timeline.to_dict()
            
            with open(timeline_file, 'w') as f:
                json.dump(timeline_dict, f, indent=2, default=str)
            
            logger.info(f"YOLO timeline saved to: {timeline_file}")
            
        except Exception as e:
            logger.error(f"Failed to save YOLO timeline: {e}")
    
    def _create_fallback_timeline(self, video_path: str, error: str = "") -> EnhancedTimeline:
        """Create fallback timeline when processing fails"""
        
        timeline = EnhancedTimeline(
            audio_file=str(video_path).replace('video.mp4', 'audio.wav'),
            total_duration=30.0
        )
        
        timeline.sources_used.append(self.service_name)
        timeline.processing_notes.append(f"YOLO processing failed: {error}")
        timeline.processing_notes.append("Fallback timeline created with minimal data")
        
        # Add a basic failure event
        timeline.events.append(TimelineEvent(
            timestamp=0.0,
            description="YOLO analysis failed - no object detection available",
            source=self.service_name,
            confidence=0.1,
            details={
                'analysis_type': 'processing_failure',
                'error': error,
                'fallback_mode': True
            }
        ))
        
        return timeline
