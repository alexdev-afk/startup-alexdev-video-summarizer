"""
Motion-Aware Frame Sampling Service

Provides high-quality keyframe selection using optical flow analysis for 
optimal temporal coverage in video AI processing. Designed to maximize
information density for downstream LLM semantic integration.
"""

import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import gc

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from utils.logger import get_logger

logger = get_logger(__name__)


class MotionAwareError(Exception):
    """Motion analysis error"""
    pass


class MotionAwareSampler:
    """Motion-aware keyframe sampling for high-quality video analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize motion-aware sampler
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.development_config = config.get('development', {})
        
        # Motion analysis parameters
        self.motion_sample_interval = 10  # Analyze every 10th frame for performance
        self.motion_threshold = 2.0       # Minimum motion magnitude to consider
        self.max_keyframes = 12           # Maximum keyframes per scene
        self.min_keyframes = 5            # Minimum keyframes per scene
        
        # Optical flow parameters
        self.flow_params = dict(
            winSize=(15, 15),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        if not CV2_AVAILABLE:
            logger.warning("OpenCV not available - motion analysis will use fallback sampling")
        else:
            logger.info(f"Motion-aware sampler initialized - max keyframes: {self.max_keyframes}")
    
    def extract_motion_keyframes(self, video_path: Path, scene_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract keyframes based on motion analysis and scene context
        
        Args:
            video_path: Path to video file
            scene_info: Scene boundary information
            
        Returns:
            List of keyframe dictionaries with timestamps and motion scores
        """
        if self.development_config.get('mock_ai_services', False):
            return self._mock_motion_keyframes(scene_info)
        
        if not CV2_AVAILABLE:
            return self._fallback_temporal_keyframes(scene_info)
        
        try:
            scene_id = scene_info.get('scene_id', 1)
            start_time = scene_info.get('start_seconds', 0.0)
            end_time = scene_info.get('end_seconds', start_time + 5.0)
            
            logger.debug(f"Motion analysis for scene {scene_id}: {start_time:.2f}s - {end_time:.2f}s")
            
            # Analyze motion throughout scene
            motion_events = self._analyze_scene_motion(video_path, start_time, end_time)
            
            # Select optimal keyframes
            keyframes = self._select_optimal_keyframes(motion_events, start_time, end_time)
            
            # Ensure scene boundaries are included
            keyframes = self._ensure_boundary_coverage(keyframes, start_time, end_time)
            
            logger.debug(f"Selected {len(keyframes)} motion-aware keyframes for scene {scene_id}")
            return keyframes
            
        except Exception as e:
            logger.error(f"Motion analysis failed for scene {scene_info.get('scene_id')}: {e}")
            return self._fallback_temporal_keyframes(scene_info)
    
    def _analyze_scene_motion(self, video_path: Path, start_time: float, end_time: float) -> List[Dict[str, Any]]:
        """Analyze motion throughout the scene using optical flow"""
        motion_events = []
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise MotionAwareError(f"Could not open video: {video_path}")
        
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30.0
            
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            
            # Initialize for optical flow
            prev_gray = None
            prev_corners = None
            
            # Sample frames for motion analysis
            for frame_num in range(start_frame, end_frame, self.motion_sample_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                timestamp = frame_num / fps
                
                if prev_gray is not None:
                    # Detect corners for tracking
                    if prev_corners is None or len(prev_corners) < 100:
                        prev_corners = cv2.goodFeaturesToTrack(
                            prev_gray, maxCorners=200, qualityLevel=0.01, 
                            minDistance=10, blockSize=3
                        )
                    
                    if prev_corners is not None and len(prev_corners) > 0:
                        # Calculate optical flow
                        corners, status, error = cv2.calcOpticalFlowPyrLK(
                            prev_gray, gray, prev_corners, None, **self.flow_params
                        )
                        
                        # Filter good points
                        if corners is not None and status is not None:
                            good_corners = corners[status == 1]
                            good_prev = prev_corners[status == 1]
                            
                            if len(good_corners) > 0:
                                # Calculate motion magnitude
                                motion_vectors = good_corners - good_prev
                                motion_magnitudes = np.sqrt(
                                    motion_vectors[:, 0]**2 + motion_vectors[:, 1]**2
                                )
                                
                                # Motion statistics
                                mean_motion = np.mean(motion_magnitudes)
                                max_motion = np.max(motion_magnitudes)
                                motion_density = len(good_corners) / len(prev_corners)
                                
                                # Combined motion score
                                motion_score = mean_motion * motion_density + (max_motion * 0.3)
                                
                                motion_events.append({
                                    'timestamp': timestamp,
                                    'motion_score': float(motion_score),
                                    'mean_motion': float(mean_motion),
                                    'max_motion': float(max_motion),
                                    'motion_density': float(motion_density),
                                    'tracked_points': len(good_corners)
                                })
                
                prev_gray = gray.copy()
                prev_corners = cv2.goodFeaturesToTrack(
                    gray, maxCorners=200, qualityLevel=0.01, 
                    minDistance=10, blockSize=3
                )
                
        finally:
            cap.release()
        
        return motion_events
    
    def _select_optimal_keyframes(self, motion_events: List[Dict], start_time: float, end_time: float) -> List[Dict[str, Any]]:
        """Select optimal keyframes based on motion analysis"""
        if not motion_events:
            return []
        
        # Sort by motion score
        motion_events.sort(key=lambda x: x['motion_score'], reverse=True)
        
        # Select high-motion moments
        high_motion_count = min(self.max_keyframes - 2, len(motion_events) // 2)  # Reserve 2 for boundaries
        high_motion_frames = motion_events[:high_motion_count]
        
        # Add temporal distribution for coverage
        scene_duration = end_time - start_time
        temporal_frames = []
        
        if scene_duration > 10:  # For longer scenes, add temporal samples
            temporal_count = min(3, self.max_keyframes - len(high_motion_frames) - 2)
            for i in range(temporal_count):
                t = start_time + (scene_duration * (i + 1) / (temporal_count + 1))
                temporal_frames.append({
                    'timestamp': t,
                    'motion_score': 0.5,  # Neutral score
                    'keyframe_type': 'temporal_coverage'
                })
        
        # Combine and deduplicate
        all_keyframes = high_motion_frames + temporal_frames
        
        # Remove frames too close together (min 1 second apart)
        filtered_keyframes = []
        all_keyframes.sort(key=lambda x: x['timestamp'])
        
        for keyframe in all_keyframes:
            if not filtered_keyframes or keyframe['timestamp'] - filtered_keyframes[-1]['timestamp'] > 1.0:
                keyframe['keyframe_type'] = keyframe.get('keyframe_type', 'motion_based')
                filtered_keyframes.append(keyframe)
        
        return filtered_keyframes[:self.max_keyframes - 2]  # Reserve space for boundaries
    
    def _ensure_boundary_coverage(self, keyframes: List[Dict], start_time: float, end_time: float) -> List[Dict[str, Any]]:
        """Ensure scene start and end are represented"""
        # Add scene start (unless too close to existing keyframe)
        start_needed = True
        end_needed = True
        
        for kf in keyframes:
            if abs(kf['timestamp'] - start_time) < 0.5:
                start_needed = False
            if abs(kf['timestamp'] - end_time) < 0.5:
                end_needed = False
        
        if start_needed:
            keyframes.append({
                'timestamp': start_time + 0.2,  # Slight offset for scene establishment
                'motion_score': 1.0,
                'keyframe_type': 'scene_start'
            })
        
        if end_needed and end_time - start_time > 2.0:  # Only for longer scenes
            keyframes.append({
                'timestamp': end_time - 0.2,  # Slight offset before scene end
                'motion_score': 1.0,
                'keyframe_type': 'scene_end'
            })
        
        # Sort by timestamp and ensure we don't exceed max keyframes
        keyframes.sort(key=lambda x: x['timestamp'])
        return keyframes[:self.max_keyframes]
    
    def _fallback_temporal_keyframes(self, scene_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fallback to temporal sampling when motion analysis unavailable"""
        start_time = scene_info.get('start_seconds', 0.0)
        end_time = scene_info.get('end_seconds', start_time + 5.0)
        duration = end_time - start_time
        
        # Create temporal keyframes
        if duration <= 3.0:
            # Short scenes - just start and middle
            timestamps = [start_time + 0.2, start_time + duration * 0.6]
        elif duration <= 8.0:
            # Medium scenes - start, early, middle, late
            timestamps = [
                start_time + 0.2,
                start_time + duration * 0.3,
                start_time + duration * 0.6,
                start_time + duration * 0.9
            ]
        else:
            # Long scenes - more coverage
            timestamps = [
                start_time + 0.2,
                start_time + duration * 0.2,
                start_time + duration * 0.4,
                start_time + duration * 0.6,
                start_time + duration * 0.8,
                end_time - 0.2
            ]
        
        return [
            {
                'timestamp': ts,
                'motion_score': 0.5,
                'keyframe_type': 'temporal_fallback'
            }
            for ts in timestamps
        ]
    
    def _mock_motion_keyframes(self, scene_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Mock motion keyframes for development"""
        logger.debug("Using mock motion keyframes for development")
        return self._fallback_temporal_keyframes(scene_info)


# Export for easy importing
__all__ = ['MotionAwareSampler', 'MotionAwareError']