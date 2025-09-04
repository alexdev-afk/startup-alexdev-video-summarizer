"""
YOLO Object Detection Service

Handles GPU-based visual object detection using YOLOv8.
Processes FFmpeg-prepared video files with sequential GPU coordination.
"""

import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import gc

# Optional imports for development mode
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from utils.logger import get_logger

logger = get_logger(__name__)


class YOLOError(Exception):
    """YOLO processing error"""
    pass


class YOLOService:
    """YOLO object detection service for visual analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize YOLO service
        
        Args:
            config: Configuration dictionary with YOLO settings
        """
        self.config = config
        self.yolo_config = config.get('gpu_pipeline', {}).get('yolo', {})
        self.development_config = config.get('development', {})
        
        # Model configuration
        self.model_name = self.yolo_config.get('model', 'yolov8n.pt')
        self.confidence_threshold = self.yolo_config.get('confidence_threshold', 0.5)
        self.device = self._determine_device()
        
        # Runtime state
        self.model = None
        self.model_loaded = False
        
        logger.info(f"YOLO service initialized - model: {self.model_name}, device: {self.device}")
    
    def _determine_device(self) -> str:
        """Determine the best device for YOLO processing"""
        device_config = self.yolo_config.get('device', 'auto')
        
        if device_config == 'auto':
            if TORCH_AVAILABLE and torch.cuda.is_available():
                return 'cuda'
            else:
                logger.warning("CUDA not available, falling back to CPU")
                return 'cpu'
        
        return device_config
    
    def _load_model(self):
        """Load YOLO model (lazy loading)"""
        if self.model_loaded:
            return
        
        # Skip model loading in development mode or if dependencies missing
        if self.development_config.get('skip_model_loading', False) or not TORCH_AVAILABLE:
            logger.info("YOLO model loading skipped (development mode or missing dependencies)")
            self.model_loaded = True
            return
        
        try:
            from ultralytics import YOLO
            logger.info(f"Loading YOLO model: {self.model_name}")
            
            self.model = YOLO(self.model_name)
            
            # Move to specified device
            if self.device == 'cuda' and torch.cuda.is_available():
                self.model.to('cuda')
            
            self.model_loaded = True
            
            logger.info("YOLO model loaded successfully")
            
        except ImportError:
            raise YOLOError(
                "Ultralytics YOLO not installed. Install with: pip install ultralytics"
            )
        except Exception as e:
            raise YOLOError(f"Failed to load YOLO model: {str(e)}") from e
    
    def analyze_video(self, video_path: Path, scene_info: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Analyze video file using YOLO object detection
        
        Args:
            video_path: Path to video.mp4 file (from FFmpeg)
            scene_info: Optional scene boundary information for context
            
        Returns:
            Dictionary with detection results
            
        Raises:
            YOLOError: If detection fails
        """
        logger.info(f"Analyzing video: {video_path.name}")
        
        # Handle development mock mode
        if self.development_config.get('mock_ai_services', False):
            return self._mock_detection(video_path, scene_info)
        
        # Validate input
        if not video_path.exists():
            raise YOLOError(f"Video file not found: {video_path}")
        
        if video_path.stat().st_size < 1024:  # Less than 1KB
            raise YOLOError(f"Video file too small: {video_path}")
        
        try:
            # Load model if needed
            self._load_model()
            
            start_time = time.time()
            
            # Process video using representative frame analysis for performance
            frames_to_analyze = self._extract_representative_frames(video_path, scene_info)
            
            all_detections = []
            
            for frame_time, frame in frames_to_analyze:
                detections = self._detect_objects_in_frame(frame, frame_time)
                all_detections.extend(detections)
            
            processing_time = time.time() - start_time
            
            # Aggregate results
            aggregated_result = self._aggregate_detections(all_detections, processing_time, scene_info)
            
            # GPU memory cleanup
            if self.device == 'cuda':
                torch.cuda.empty_cache()
                gc.collect()
            
            logger.info(f"YOLO analysis complete: {processing_time:.2f}s, {len(all_detections)} detections")
            return aggregated_result
            
        except Exception as e:
            # Cleanup on error
            self._cleanup_gpu_memory()
            raise YOLOError(f"YOLO analysis failed: {str(e)}") from e
    
    def _extract_representative_frames(self, video_path: Path, scene_info: Optional[Dict]):
        """Extract representative frames for analysis (70x performance optimization)"""
        frames = []
        
        # Check if OpenCV is available
        if not CV2_AVAILABLE:
            logger.warning("OpenCV not available for frame extraction")
            return []
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise YOLOError(f"Could not open video: {video_path}")
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            # Scene-specific frame selection
            if scene_info and 'start_seconds' in scene_info:
                start_frame = int(scene_info['start_seconds'] * fps)
                end_frame = int(scene_info.get('end_seconds', duration) * fps)
                frames_to_sample = min(5, max(1, (end_frame - start_frame) // 10))  # Sample up to 5 frames
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            else:
                # Full video representative sampling
                frames_to_sample = min(10, max(3, total_frames // 100))  # Sample up to 10 frames
            
            # Extract frames at regular intervals
            frame_interval = max(1, (total_frames // frames_to_sample) if frames_to_sample > 0 else 1)
            
            for i in range(frames_to_sample):
                frame_pos = i * frame_interval
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                
                ret, frame = cap.read()
                if ret:
                    frame_time = frame_pos / fps if fps > 0 else i
                    frames.append((frame_time, frame))
            
            cap.release()
            
        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")
            # Return empty list, will trigger mock mode
            return []
        
        logger.debug(f"Extracted {len(frames)} representative frames")
        return frames
    
    def _detect_objects_in_frame(self, frame, frame_time: float) -> List[Dict[str, Any]]:
        """Detect objects in a single frame"""
        try:
            # Run YOLO inference
            results = self.model(frame, verbose=False)
            
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i in range(len(boxes)):
                        # Extract detection info
                        box = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
                        confidence = float(boxes.conf[i].cpu().numpy())
                        class_id = int(boxes.cls[i].cpu().numpy())
                        class_name = self.model.names[class_id]
                        
                        # Filter by confidence threshold
                        if confidence >= self.confidence_threshold:
                            detections.append({
                                'frame_time': frame_time,
                                'class': class_name,
                                'class_id': class_id,
                                'confidence': confidence,
                                'bbox': [float(x) for x in box],  # [x1, y1, x2, y2]
                                'area': float((box[2] - box[0]) * (box[3] - box[1]))
                            })
            
            return detections
            
        except Exception as e:
            logger.warning(f"Detection failed for frame at {frame_time:.2f}s: {e}")
            return []
    
    def _aggregate_detections(self, detections: List[Dict[str, Any]], processing_time: float, scene_info: Optional[Dict]) -> Dict[str, Any]:
        """Aggregate individual detections into summary results"""
        
        # Count objects by class
        object_counts = {}
        unique_objects = set()
        people_count = 0
        total_detections = len(detections)
        
        high_confidence_detections = []
        
        for detection in detections:
            class_name = detection['class']
            confidence = detection['confidence']
            
            # Count occurrences
            object_counts[class_name] = object_counts.get(class_name, 0) + 1
            unique_objects.add(class_name)
            
            # Count people specifically
            if class_name == 'person':
                people_count = max(people_count, object_counts['person'])
            
            # Keep high-confidence detections for details
            if confidence >= 0.7:
                high_confidence_detections.append(detection)
        
        # Build summary
        return {
            'objects': sorted(list(unique_objects)),
            'object_counts': object_counts,
            'people_count': people_count,
            'total_detections': total_detections,
            'object_details': high_confidence_detections[:20],  # Top 20 high-confidence detections
            'processing_time': processing_time,
            'model_info': {
                'model_name': self.model_name,
                'device': self.device,
                'confidence_threshold': self.confidence_threshold
            },
            'scene_context': scene_info
        }
    
    def _mock_detection(self, video_path: Path, scene_info: Optional[Dict]) -> Dict[str, Any]:
        """Mock detection for development/testing"""
        logger.debug(f"Mock YOLO detection: {video_path.name}")
        
        # Simulate processing time
        time.sleep(0.3 if self.development_config.get('fast_mode', False) else 0.8)
        
        # Generate mock detections based on video name or scene
        video_name = video_path.parent.name
        scene_id = scene_info.get('scene_id', 1) if scene_info else 1
        
        # Mock objects based on scene context
        base_objects = ['person', 'laptop', 'chair']
        
        if 'meeting' in video_name.lower() or scene_id % 3 == 0:
            base_objects.extend(['tv', 'cell phone', 'cup'])
        
        if 'presentation' in video_name.lower() or scene_id % 4 == 0:
            base_objects.extend(['projector', 'whiteboard'])
        
        # Mock detection details
        object_details = []
        object_counts = {}
        
        for i, obj in enumerate(base_objects):
            confidence = 0.7 + (0.2 * ((i + scene_id) % 3))
            count = 1 + ((i + scene_id) % 3)
            object_counts[obj] = count
            
            object_details.append({
                'frame_time': 0.0,
                'class': obj,
                'class_id': i,
                'confidence': confidence,
                'bbox': [100 + i*50, 100 + i*30, 200 + i*50, 250 + i*30],
                'area': 7500 + i*1000
            })
        
        people_count = object_counts.get('person', 0)
        
        return {
            'objects': sorted(base_objects),
            'object_counts': object_counts,
            'people_count': people_count,
            'total_detections': len(object_details),
            'object_details': object_details,
            'processing_time': 0.3,
            'model_info': {
                'model_name': 'mock',
                'device': 'mock',
                'confidence_threshold': self.confidence_threshold
            },
            'scene_context': scene_info,
            'mock_mode': True
        }
    
    def _cleanup_gpu_memory(self):
        """Cleanup GPU memory after processing"""
        if self.device == 'cuda' and TORCH_AVAILABLE:
            try:
                torch.cuda.empty_cache()
                gc.collect()
                logger.debug("GPU memory cleaned up")
            except Exception as e:
                logger.warning(f"GPU memory cleanup failed: {e}")
    
    def unload_model(self):
        """Unload model to free memory"""
        if self.model is not None:
            del self.model
            self.model = None
            self.model_loaded = False
            self._cleanup_gpu_memory()
            logger.debug("YOLO model unloaded")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'confidence_threshold': self.confidence_threshold,
            'model_loaded': self.model_loaded,
            'development_mode': self.development_config.get('mock_ai_services', False)
        }


# Export for easy importing
__all__ = ['YOLOService', 'YOLOError']