"""
Scene Detection Service using PySceneDetect.

Implements content-aware scene boundary detection for 70x performance improvement
through representative frame analysis instead of frame-by-frame processing.
Based on feature specification: scene-detection.md
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import time

try:
    from scenedetect import detect, ContentDetector
    SCENEDETECT_AVAILABLE = True
except ImportError:
    SCENEDETECT_AVAILABLE = False

from utils.logger import get_logger

logger = get_logger(__name__)


class SceneDetectionError(Exception):
    """Scene detection processing error"""
    pass


class SceneDetectionService:
    """PySceneDetect scene boundary detection service"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize scene detection service
        
        Args:
            config: Configuration dictionary with scene detection settings
        """
        self.config = config
        self.scene_config = config.get('scene_detection', {})
        
        # Scene detection parameters
        self.threshold = self.scene_config.get('threshold', 27.0)
        self.min_scene_length = self.scene_config.get('min_scene_length', 2.0)
        self.downscale_factor = self.scene_config.get('downscale_factor', 1)
        self.frame_skip = self.scene_config.get('frame_skip', 0)
        
        # Fallback settings
        self.fallback_to_time_based = self.scene_config.get('fallback_to_time_based', True)
        self.time_based_scene_length = self.scene_config.get('time_based_scene_length', 120)
        
        if not SCENEDETECT_AVAILABLE:
            raise SceneDetectionError("PySceneDetect not available. Install with: pip install scenedetect>=0.6.2")
        
        logger.info(f"Scene detection service initialized - threshold: {self.threshold}")
    
    def analyze_video_scenes(self, video_path: Path) -> Dict[str, Any]:
        """
        Analyze video for scene boundaries using PySceneDetect
        
        Args:
            video_path: Path to video file
            
        Returns:
            Scene analysis data structure with boundaries and representative frames
        """
        # Scene detection requires real PySceneDetect - no mock modes
        
        try:
            logger.info(f"Analyzing scenes for: {video_path.name}")
            
            # 1. Detect scene boundaries using PySceneDetect
            scene_list, fps = self._detect_scenes(video_path)
            
            if not scene_list:
                logger.warning("No scenes detected, falling back to time-based splitting")
                return self._fallback_time_based_scenes(video_path)
            
            # 2. Process boundaries into usable format
            boundaries = self._process_scene_boundaries(scene_list, fps)
            
            # 3. Extract representative frames for each scene
            representative_frames = self._extract_representative_frames(boundaries, fps)
            
            # 4. Calculate total duration
            total_duration = boundaries[-1]['end_seconds'] if boundaries else 0.0
            
            scene_data = {
                'scene_count': len(boundaries),
                'scenes': boundaries,
                'boundaries': boundaries,
                'representative_frames': representative_frames,
                'scene_files': [],  # Will be populated by FFmpeg coordination
                'fps': fps,
                'total_duration': total_duration,
                'production_mode': True
            }
            
            logger.info(f"Scene analysis complete: {len(boundaries)} scenes detected")
            return scene_data
            
        except Exception as e:
            logger.error(f"Scene detection failed for {video_path.name}: {str(e)}")
            if self.fallback_to_time_based:
                logger.info("Falling back to time-based scene splitting")
                return self._fallback_time_based_scenes(video_path)
            else:
                raise SceneDetectionError(f"Scene detection failed: {str(e)}")
    
    def _detect_scenes(self, video_path: Path) -> Tuple[List, float]:
        """
        Detect scene boundaries using PySceneDetect modern API
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tuple of (scene_list, fps)
        """
        # Use modern detect() function - much simpler than VideoManager
        scene_list = detect(
            str(video_path), 
            ContentDetector(threshold=self.threshold)
        )
        
        # Get video info with OpenCV  
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        
        if fps <= 0:
            fps = 30.0  # Fallback FPS
        
        return scene_list, fps
    
    def _process_scene_boundaries(self, scene_list: List, fps: float) -> List[Dict[str, Any]]:
        """
        Convert PySceneDetect scene list to processing boundaries
        
        Args:
            scene_list: PySceneDetect scene list
            fps: Video frame rate
            
        Returns:
            List of scene boundary dictionaries
        """
        boundaries = []
        
        for i, (start_time, end_time) in enumerate(scene_list):
            # Handle last scene (may not have end time)
            end_seconds = end_time.get_seconds() if end_time else None
            duration = (end_time - start_time).get_seconds() if end_time else None
            
            boundary = {
                'scene_id': i + 1,
                'start_seconds': start_time.get_seconds(),
                'end_seconds': end_seconds,
                'duration': duration,
                'start_frame': start_time.get_frames(),
                'end_frame': end_time.get_frames() if end_time else None
            }
            
            boundaries.append(boundary)
        
        return boundaries
    
    def _extract_representative_frames(self, boundaries: List[Dict], fps: float) -> List[Dict[str, Any]]:
        """
        Extract representative frame information for each scene
        
        Args:
            boundaries: Scene boundary data
            fps: Video frame rate
            
        Returns:
            List of representative frame data
        """
        representative_frames = []
        
        for boundary in boundaries:
            start_frame = boundary['start_frame']
            end_frame = boundary['end_frame']
            
            if end_frame is not None:
                # Use middle frame for best scene representation
                middle_frame = int(start_frame + (end_frame - start_frame) / 2)
            else:
                # For last scene without end, use frame 3 seconds after start
                middle_frame = int(start_frame + (3 * fps))
            
            frame_info = {
                'scene_id': boundary['scene_id'],
                'representative_frame': middle_frame,
                'frame_timestamp': middle_frame / fps,
                'scene_context': boundary
            }
            
            representative_frames.append(frame_info)
        
        return representative_frames
    
    def _fallback_time_based_scenes(self, video_path: Path) -> Dict[str, Any]:
        """
        Fallback to time-based scene splitting when content detection fails
        
        Args:
            video_path: Path to video file
            
        Returns:
            Scene data with time-based boundaries
        """
        logger.info(f"Using time-based scene splitting: {self.time_based_scene_length}s intervals")
        
        # Get actual video duration with OpenCV
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        
        if fps <= 0:
            fps = 30.0  # Fallback FPS
        
        total_duration = frame_count / fps if frame_count > 0 else 360.0
        scene_duration = self.time_based_scene_length
        
        boundaries = []
        scene_count = int(total_duration // scene_duration) + (1 if total_duration % scene_duration > 0 else 0)
        
        for i in range(scene_count):
            start_seconds = i * scene_duration
            end_seconds = min((i + 1) * scene_duration, total_duration)
            
            boundary = {
                'scene_id': i + 1,
                'start_seconds': start_seconds,
                'end_seconds': end_seconds,
                'duration': end_seconds - start_seconds,
                'start_frame': int(start_seconds * fps),
                'end_frame': int(end_seconds * fps)
            }
            boundaries.append(boundary)
        
        representative_frames = self._extract_representative_frames(boundaries, fps)
        
        return {
            'scene_count': len(boundaries),
            'scenes': boundaries,
            'boundaries': boundaries,
            'representative_frames': representative_frames,
            'scene_files': [],
            'fps': fps,
            'total_duration': total_duration,
            'production_mode': True,
            'fallback_mode': True
        }
    
    
    def coordinate_frame_extraction(self, video_path: Path, boundaries: List[Dict], ffmpeg_service=None) -> Dict[str, Any]:
        """
        Coordinate with FFmpeg service to extract 3 frames per scene instead of full scene videos
        
        Args:
            video_path: Path to video file
            boundaries: Scene boundary data
            ffmpeg_service: FFmpeg service instance
            
        Returns:
            Dictionary containing frame extraction metadata and paths
        """
        # Real frame extraction only - no mock modes
        
        if not ffmpeg_service:
            logger.warning("No FFmpeg service provided, cannot extract frames")
            return {}
        
        try:
            logger.info(f"Coordinating frame extraction: 3 frames × {len(boundaries)} scenes = {len(boundaries) * 3} total frames")
            frame_data = ffmpeg_service.extract_scene_frames(video_path, boundaries)
            
            total_frames = sum(len(scene_data.get('frames', {})) for scene_data in frame_data.get('scenes', {}).values())
            logger.info(f"Frame extraction complete: {total_frames} frames created")
            return frame_data
            
        except Exception as e:
            logger.error(f"Frame extraction failed: {str(e)}")
            return {}
    
    
