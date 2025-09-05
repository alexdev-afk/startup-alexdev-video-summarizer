"""
OpenCV Face Detection Service

Handles CPU-based face detection from video scenes using OpenCV.
Implements parallel CPU processing integrated with object detection.
"""

import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import gc

# Optional imports for development mode
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from utils.logger import get_logger

logger = get_logger(__name__)


class OpenCVError(Exception):
    """OpenCV processing error"""
    pass


class OpenCVService:
    """OpenCV face detection service for visual analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize OpenCV service
        
        Args:
            config: Configuration dictionary with OpenCV settings
        """
        self.config = config
        self.opencv_config = config.get('cpu_pipeline', {}).get('opencv', {})
        self.development_config = config.get('development', {})
        
        # Face detection configuration
        self.cascade_file = self.opencv_config.get('cascade_file', 'haarcascade_frontalface_default.xml')
        self.scale_factor = self.opencv_config.get('scale_factor', 1.1)
        self.min_neighbors = self.opencv_config.get('min_neighbors', 5)
        self.min_face_size = self.opencv_config.get('min_face_size', (30, 30))
        self.confidence_threshold = self.opencv_config.get('confidence_threshold', 0.7)
        
        # Runtime state
        self.face_cascade = None
        self.cascade_loaded = False
        
        logger.info(f"OpenCV service initialized - cascade: {self.cascade_file}")
    
    def _load_cascade(self):
        """Load Haar cascade classifier (lazy loading)"""
        if self.cascade_loaded:
            return
        
        # Skip cascade loading in development mode or if dependencies missing
        if self.development_config.get('mock_mode', False) or not CV2_AVAILABLE:
            logger.info("OpenCV cascade loading skipped - development/mock mode or missing dependencies")
            self.cascade_loaded = True
            return
        
        try:
            logger.info("Loading OpenCV Haar cascade...")
            start_time = time.time()
            
            # Load pre-trained Haar cascade for face detection
            cascade_path = cv2.data.haarcascades + self.cascade_file
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            if self.face_cascade.empty():
                logger.error(f"Failed to load cascade from {cascade_path}")
                self.face_cascade = None
            else:
                load_time = time.time() - start_time
                logger.info(f"OpenCV cascade loaded in {load_time:.2f}s")
            
            self.cascade_loaded = True
            
        except Exception as e:
            logger.error(f"Failed to load OpenCV cascade: {e}")
            logger.info("Falling back to mock mode for development")
            self.face_cascade = None
            self.cascade_loaded = True
    
    def detect_faces_from_scene(self, video_path: Path, scene_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect faces from video scene using representative frame analysis
        
        Args:
            video_path: Path to video file
            scene_info: Scene boundary information
            
        Returns:
            Face detection results with bounding boxes and confidence
        """
        scene_id = scene_info['scene_id']
        start_time = time.time()
        
        try:
            # Load cascade if needed
            self._load_cascade()
            
            # Extract representative frame
            representative_frame = self._extract_representative_frame(video_path, scene_info)
            
            if representative_frame is None:
                logger.warning(f"Could not extract representative frame for scene {scene_id}")
                return self._fallback_face_result(scene_info, "No representative frame")
            
            # Perform face detection
            if self.face_cascade is None:
                logger.warning(f"OpenCV cascade not available for scene {scene_id}")
                return self._fallback_face_result(scene_info, "Cascade not available")
            
            # Convert to grayscale for face detection
            gray_frame = cv2.cvtColor(representative_frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray_frame,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=self.min_face_size
            )
            
            # Process results
            face_detections = []
            
            for (x, y, w, h) in faces:
                # Calculate confidence based on size and position
                confidence = self._calculate_face_confidence(x, y, w, h, representative_frame.shape)
                
                if confidence >= self.confidence_threshold:
                    face_detections.append({
                        'bbox': self._normalize_bbox((x, y, w, h), representative_frame.shape),
                        'confidence': confidence,
                        'size_category': self._classify_face_size(w, h, representative_frame.shape),
                        'position': self._classify_face_position(x, y, w, h, representative_frame.shape),
                        'raw_bbox': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)}
                    })
            
            processing_time = time.time() - start_time
            
            return {
                'scene_id': scene_id,
                'face_detections': face_detections,
                'face_count': len(face_detections),
                'total_faces_found': len(faces),  # Before confidence filtering
                'confidence_threshold': self.confidence_threshold,
                'processing_time': processing_time,
                'representative_frame_info': {
                    'timestamp': scene_info.get('start_time', 0) + (scene_info.get('end_time', 0) - scene_info.get('start_time', 0)) / 2,
                    'shape': representative_frame.shape if representative_frame is not None else None
                },
                'detection_parameters': {
                    'scale_factor': self.scale_factor,
                    'min_neighbors': self.min_neighbors,
                    'min_face_size': self.min_face_size
                }
            }
            
        except Exception as e:
            logger.error(f"OpenCV face detection failed for scene {scene_id}: {e}")
            raise OpenCVError(f"Face detection failed: {str(e)}")
    
    def _extract_representative_frame(self, video_path: Path, scene_info: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract representative frame from scene for face detection"""
        if not CV2_AVAILABLE:
            logger.warning("OpenCV not available for frame extraction")
            return None
        
        try:
            # Calculate middle timestamp of scene
            start_time = scene_info.get('start_time', 0)
            end_time = scene_info.get('end_time', start_time + 1)
            middle_time = start_time + (end_time - start_time) / 2
            
            # Open video
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                logger.warning(f"Could not open video: {video_path}")
                return None
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0:
                fps = 30  # Fallback FPS
            
            # Calculate frame number
            frame_number = int(middle_time * fps)
            
            # Seek to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            # Read frame
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                return frame
            else:
                logger.warning(f"Could not read frame {frame_number} from {video_path}")
                return None
                
        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")
            return None
    
    def _normalize_bbox(self, bbox: Tuple[int, int, int, int], frame_shape: Tuple[int, int, int]) -> Dict[str, float]:
        """Normalize bounding box coordinates to 0-1 range"""
        x, y, w, h = bbox
        height, width = frame_shape[:2]
        
        return {
            'x_min': x / width,
            'y_min': y / height,
            'x_max': (x + w) / width,
            'y_max': (y + h) / height,
            'center_x': (x + w/2) / width,
            'center_y': (y + h/2) / height,
            'width': w / width,
            'height': h / height
        }
    
    def _calculate_face_confidence(self, x: int, y: int, w: int, h: int, frame_shape: Tuple[int, int, int]) -> float:
        """Calculate confidence score based on face characteristics"""
        height, width = frame_shape[:2]
        
        # Base confidence from size (larger faces are more confident)
        face_area = w * h
        frame_area = width * height
        size_ratio = face_area / frame_area
        
        base_confidence = min(0.9, 0.6 + size_ratio * 10)  # 0.6-0.9 based on size
        
        # Position bonus (centered faces get higher confidence)
        center_x = x + w/2
        center_y = y + h/2
        distance_from_center = abs(center_x - width/2) + abs(center_y - height/2)
        max_distance = width/2 + height/2
        position_bonus = 0.1 * (1 - distance_from_center / max_distance)
        
        # Aspect ratio bonus (more square faces get higher confidence)
        aspect_ratio = w / h if h > 0 else 1
        ideal_ratio = 0.8  # Slightly wider than square
        aspect_bonus = 0.05 * (1 - abs(aspect_ratio - ideal_ratio))
        
        return min(0.95, base_confidence + position_bonus + aspect_bonus)
    
    def _classify_face_size(self, width: int, height: int, frame_shape: Tuple[int, int, int]) -> str:
        """Classify face size category"""
        frame_height, frame_width = frame_shape[:2]
        face_area = width * height
        frame_area = frame_width * frame_height
        size_ratio = face_area / frame_area
        
        if size_ratio > 0.15:
            return 'large'
        elif size_ratio > 0.05:
            return 'medium'
        else:
            return 'small'
    
    def _classify_face_position(self, x: int, y: int, width: int, height: int, frame_shape: Tuple[int, int, int]) -> str:
        """Classify face position in frame"""
        frame_height, frame_width = frame_shape[:2]
        center_x = x + width/2
        center_y = y + height/2
        
        # Determine horizontal position
        if center_x < frame_width * 0.33:
            h_pos = 'left'
        elif center_x > frame_width * 0.67:
            h_pos = 'right'
        else:
            h_pos = 'center'
        
        # Determine vertical position
        if center_y < frame_height * 0.33:
            v_pos = 'top'
        elif center_y > frame_height * 0.67:
            v_pos = 'bottom'
        else:
            v_pos = 'middle'
        
        return f"{v_pos}_{h_pos}"
    
    def _fallback_face_result(self, scene_info: Dict[str, Any], error: str) -> Dict[str, Any]:
        """Generate fallback result when face detection fails"""
        return {
            'scene_id': scene_info['scene_id'],
            'face_detections': [],
            'face_count': 0,
            'total_faces_found': 0,
            'confidence_threshold': self.confidence_threshold,
            'processing_time': 0.0,
            'representative_frame_info': {
                'timestamp': scene_info.get('start_time', 0),
                'shape': None
            },
            'detection_parameters': {
                'scale_factor': self.scale_factor,
                'min_neighbors': self.min_neighbors,
                'min_face_size': self.min_face_size
            },
            'error': error,
            'fallback_mode': True
        }
    
    def cleanup_resources(self):
        """Clean up OpenCV resources"""
        if self.face_cascade is not None:
            self.face_cascade = None
        
        # Force garbage collection
        gc.collect()
        logger.debug("OpenCV resources cleaned up")
    
    def __del__(self):
        """Cleanup on service destruction"""
        self.cleanup_resources()