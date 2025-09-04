"""
Video GPU Pipeline Controller for visual analysis.

Handles YOLO and EasyOCR coordination with sequential GPU processing.
"""

import time
from pathlib import Path
from typing import Dict, Any

from services.yolo_service import YOLOService, YOLOError
from utils.logger import get_logger

logger = get_logger(__name__)


class VideoGPUPipelineController:
    """Video GPU pipeline controller for visual analysis (YOLO + EasyOCR)"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize GPU pipeline controller
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.gpu_config = config.get('gpu_pipeline', {})
        
        # Initialize services
        self.yolo_service = YOLOService(config)
        
        logger.info("Video GPU pipeline controller initialized")
    
    def process_scene(self, scene: Dict[str, Any], context) -> Dict[str, Any]:
        """
        Process scene through Video GPU pipeline
        
        Sequential GPU processing: YOLO → EasyOCR
        
        Args:
            scene: Scene boundary data
            context: Video processing context
            
        Returns:
            Visual analysis results from YOLO + EasyOCR
        """
        scene_id = scene['scene_id']
        logger.info(f"Processing scene {scene_id} through Video GPU pipeline")
        
        results = {}
        
        try:
            # Step 1: YOLO object detection (using FFmpeg-extracted video)
            if hasattr(context, 'video_path') and context.video_path:
                results['yolo'] = self.yolo_service.analyze_video(
                    context.video_path, scene_info=scene
                )
            else:
                logger.warning(f"No video file available for scene {scene_id}")
                results['yolo'] = self._fallback_yolo_result(scene)
            
            # Step 2: EasyOCR text extraction (mock for now - Phase 4)
            results['easyocr'] = self._mock_easyocr_processing(scene, context)
            
            logger.info(f"Video GPU pipeline complete for scene {scene_id}")
            return results
            
        except Exception as e:
            logger.error(f"Video GPU pipeline failed for scene {scene_id}: {e}")
            # Return fallback results to prevent complete failure
            return {
                'yolo': self._fallback_yolo_result(scene),
                'easyocr': self._fallback_easyocr_result(scene),
                'error': str(e)
            }
    
    def _fallback_yolo_result(self, scene: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback YOLO result when video processing fails"""
        return {
            'objects': [],
            'object_counts': {},
            'people_count': 0,
            'total_detections': 0,
            'object_details': [],
            'processing_time': 0.0,
            'error': 'YOLO processing unavailable',
            'fallback_mode': True
        }
    
    def _fallback_easyocr_result(self, scene: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback EasyOCR result when processing fails"""
        return {
            'text': [],
            'text_details': [],
            'processing_time': 0.0,
            'error': 'EasyOCR processing unavailable',
            'fallback_mode': True
        }
    
    def _mock_easyocr_processing(self, scene: Dict[str, Any], context) -> Dict[str, Any]:
        """Mock EasyOCR text extraction processing"""
        logger.debug(f"Mock EasyOCR processing for scene {scene['scene_id']}")
        
        # Simulate processing time
        time.sleep(0.3)
        
        # Mock text content based on scene
        mock_texts = []
        if scene['scene_id'] == 1:
            mock_texts = ['Welcome to the Presentation', 'Q4 Budget Review']
        elif scene['scene_id'] % 2 == 0:
            mock_texts = ['Key Metrics', 'Performance Data', '2024 Goals']
        
        return {
            'text': mock_texts,
            'text_details': [
                {
                    'text': text,
                    'confidence': 0.88 + (0.05 * i),
                    'bbox': [200 + i*100, 50 + i*40, 300 + i*100, 80 + i*40]
                }
                for i, text in enumerate(mock_texts)
            ],
            'processing_time': 0.3,
            'mock_mode': True
        }