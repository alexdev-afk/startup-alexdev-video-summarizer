"""
EasyOCR Text Extraction Service

Handles GPU-based text extraction from video scenes using EasyOCR.
Implements sequential GPU coordination after YOLO processing.
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

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

from utils.logger import get_logger

logger = get_logger(__name__)


class EasyOCRError(Exception):
    """EasyOCR processing error"""
    pass


class EasyOCRService:
    """EasyOCR text extraction service for visual analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize EasyOCR service
        
        Args:
            config: Configuration dictionary with EasyOCR settings
        """
        self.config = config
        self.easyocr_config = config.get('gpu_pipeline', {}).get('easyocr', {})
        self.development_config = config.get('development', {})
        
        # OCR configuration
        self.languages = self.easyocr_config.get('languages', ['en'])
        self.gpu_enabled = self.easyocr_config.get('gpu', True)
        self.confidence_threshold = self.easyocr_config.get('confidence_threshold', 0.5)
        self.device = self._determine_device()
        
        # Runtime state
        self.reader = None
        self.reader_loaded = False
        
        logger.info(f"EasyOCR service initialized - languages: {self.languages}, device: {self.device}")
    
    def _determine_device(self) -> str:
        """Determine the best device for EasyOCR processing"""
        device_config = self.easyocr_config.get('device', 'auto')
        
        if device_config == 'auto':
            if TORCH_AVAILABLE and torch.cuda.is_available() and self.gpu_enabled:
                return 'cuda'
            else:
                logger.warning("CUDA not available or disabled, falling back to CPU")
                return 'cpu'
        
        return device_config
    
    def _load_reader(self):
        """Load EasyOCR reader (lazy loading)"""
        if self.reader_loaded:
            return
        
        # Skip reader loading in development mode or if dependencies missing
        if self.development_config.get('mock_mode', False) or not EASYOCR_AVAILABLE:
            logger.info("EasyOCR reader loading skipped - development/mock mode or missing dependencies")
            self.reader_loaded = True
            return
        
        try:
            logger.info("Loading EasyOCR reader...")
            start_time = time.time()
            
            # Initialize EasyOCR reader with GPU support
            use_gpu = self.device == 'cuda'
            self.reader = easyocr.Reader(
                self.languages, 
                gpu=use_gpu,
                verbose=False
            )
            
            load_time = time.time() - start_time
            logger.info(f"EasyOCR reader loaded in {load_time:.2f}s - GPU: {use_gpu}")
            self.reader_loaded = True
            
        except Exception as e:
            logger.error(f"Failed to load EasyOCR reader: {e}")
            logger.info("Falling back to mock mode for development")
            self.reader = None
            self.reader_loaded = True
    
    def extract_text_from_scene(self, video_path: Path, scene_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract text from video scene using representative frame analysis
        
        Args:
            video_path: Path to video file
            scene_info: Scene boundary information
            
        Returns:
            Text extraction results with bounding boxes and confidence
        """
        scene_id = scene_info['scene_id']
        start_time = time.time()
        
        try:
            # Load reader if needed
            self._load_reader()
            
            # Extract representative frame
            representative_frame = self._extract_representative_frame(video_path, scene_info)
            
            if representative_frame is None:
                logger.warning(f"Could not extract representative frame for scene {scene_id}")
                return self._fallback_text_result(scene_info, "No representative frame")
            
            # Perform OCR analysis
            if self.reader is None:
                logger.warning(f"EasyOCR reader not available for scene {scene_id}")
                return self._fallback_text_result(scene_info, "Reader not available")
            
            # Extract text with bounding boxes
            ocr_results = self.reader.readtext(representative_frame)
            
            # Process results
            text_extractions = []
            full_text = []
            
            for detection in ocr_results:
                bbox, text, confidence = detection
                
                if confidence >= self.confidence_threshold:
                    text_extractions.append({
                        'text': text.strip(),
                        'confidence': float(confidence),
                        'bbox': self._normalize_bbox(bbox, representative_frame.shape),
                        'region_type': self._classify_text_region(bbox, representative_frame.shape)
                    })
                    full_text.append(text.strip())
            
            processing_time = time.time() - start_time
            
            return {
                'scene_id': scene_id,
                'text_extractions': text_extractions,
                'full_text': ' '.join(full_text),
                'text_count': len(text_extractions),
                'confidence_threshold': self.confidence_threshold,
                'processing_time': processing_time,
                'representative_frame_info': {
                    'timestamp': scene_info.get('start_time', 0) + (scene_info.get('end_time', 0) - scene_info.get('start_time', 0)) / 2,
                    'shape': representative_frame.shape if representative_frame is not None else None
                },
                'languages': self.languages,
                'device': self.device
            }
            
        except Exception as e:
            logger.error(f"EasyOCR text extraction failed for scene {scene_id}: {e}")
            raise EasyOCRError(f"Text extraction failed: {str(e)}")
    
    def _extract_representative_frame(self, video_path: Path, scene_info: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract representative frame from scene for OCR analysis"""
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
    
    def _normalize_bbox(self, bbox: List[List[int]], frame_shape: Tuple[int, int, int]) -> Dict[str, float]:
        """Normalize bounding box coordinates to 0-1 range"""
        height, width = frame_shape[:2]
        
        # Get min/max coordinates
        x_coords = [point[0] for point in bbox]
        y_coords = [point[1] for point in bbox]
        
        return {
            'x_min': min(x_coords) / width,
            'y_min': min(y_coords) / height,
            'x_max': max(x_coords) / width,
            'y_max': max(y_coords) / height,
            'center_x': sum(x_coords) / (4 * width),
            'center_y': sum(y_coords) / (4 * height)
        }
    
    def _classify_text_region(self, bbox: List[List[int]], frame_shape: Tuple[int, int, int]) -> str:
        """Classify text region type based on position and size"""
        height, width = frame_shape[:2]
        
        # Calculate region properties
        x_coords = [point[0] for point in bbox]
        y_coords = [point[1] for point in bbox]
        
        center_y = sum(y_coords) / 4
        center_x = sum(x_coords) / 4
        bbox_height = max(y_coords) - min(y_coords)
        bbox_width = max(x_coords) - min(x_coords)
        
        # Region classification
        if center_y < height * 0.2:
            return 'header'
        elif center_y > height * 0.8:
            return 'footer'
        elif center_x < width * 0.3:
            return 'sidebar_left'
        elif center_x > width * 0.7:
            return 'sidebar_right'
        elif bbox_width > width * 0.6:
            return 'banner'
        else:
            return 'content'
    
    def _fallback_text_result(self, scene_info: Dict[str, Any], error: str) -> Dict[str, Any]:
        """Generate fallback result when OCR processing fails"""
        return {
            'scene_id': scene_info['scene_id'],
            'text_extractions': [],
            'full_text': '',
            'text_count': 0,
            'confidence_threshold': self.confidence_threshold,
            'processing_time': 0.0,
            'representative_frame_info': {
                'timestamp': scene_info.get('start_time', 0),
                'shape': None
            },
            'languages': self.languages,
            'device': self.device,
            'error': error,
            'fallback_mode': True
        }
    
    def cleanup_gpu_memory(self):
        """Clean up GPU memory after processing"""
        if self.reader and hasattr(self.reader, 'clear'):
            try:
                self.reader.clear()
            except:
                pass
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("GPU memory cache cleared")
    
    def __del__(self):
        """Cleanup on service destruction"""
        self.cleanup_gpu_memory()