"""
Semantic Scene Detection Service

Enhances PySceneDetect's pixel-based scene detection with VLM semantic understanding.
Uses dual-image comparison to optimize scene boundaries based on content meaning rather than just visual changes.

Key capabilities:
- Two-phase boundary stabilization (cross-scene → intra-scene)
- Semantic scene merging and splitting using VLM analysis
- Rapid cut detection with minimum scene time constraints
- Content-aware boundary optimization
"""

import time
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass

from PIL import Image
import cv2
import numpy as np

from utils.logger import get_logger
from services.internvl3_timeline_service import InternVL3SceneAnalyzer

logger = get_logger(__name__)


@dataclass
class SceneBoundary:
    """Represents a scene boundary with semantic metadata"""
    start_time: float
    end_time: float
    start_frame: int
    end_frame: int
    confidence: float = 1.0
    semantic_type: str = "normal"  # normal, rapid_cut, merged, split
    merge_count: int = 0
    split_count: int = 0
    stability_iterations: int = 0


class SemanticSceneDetectionError(Exception):
    """Semantic scene detection error"""
    pass


class SemanticSceneDetectionService:
    """Semantic scene detection service using VLM differential analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize semantic scene detection service"""
        self.config = config
        self.service_name = "semantic_scene_detection"
        
        # Semantic detection configuration
        self.semantic_config = config.get('semantic_scene_detection', {})
        self.min_scene_duration = self.semantic_config.get('min_scene_duration', 2.0)
        self.rapid_cut_threshold = self.semantic_config.get('rapid_cut_threshold', 1.0)
        self.max_stabilization_iterations = self.semantic_config.get('max_stabilization_iterations', 5)
        self.merge_threshold = self.semantic_config.get('merge_threshold', 0.75)
        self.split_threshold = self.semantic_config.get('split_threshold', 0.25)
        self.enable_cross_scene = self.semantic_config.get('enable_cross_scene_analysis', True)
        self.enable_intra_scene = self.semantic_config.get('enable_intra_scene_analysis', True)
        self.boundary_confidence_threshold = self.semantic_config.get('boundary_confidence_threshold', 0.6)
        
        # Initialize VLM analyzer for semantic comparison
        self.vlm_analyzer = None
        self._vlm_loaded = False
        
        logger.info(f"Semantic scene detection service initialized")
        logger.info(f"Min scene duration: {self.min_scene_duration}s")
        logger.info(f"Rapid cut threshold: {self.rapid_cut_threshold}s")
        logger.info(f"Max stabilization iterations: {self.max_stabilization_iterations}")
        logger.info(f"Cross-scene analysis: {self.enable_cross_scene}")
        logger.info(f"Intra-scene analysis: {self.enable_intra_scene}")
    
    def _ensure_vlm_loaded(self):
        """Ensure VLM analyzer is loaded for semantic comparison"""
        if self._vlm_loaded:
            return
            
        logger.info("Loading VLM analyzer for semantic scene detection...")
        try:
            # Create a full InternVL3 service to get the properly initialized analyzer
            from services.internvl3_timeline_service import InternVL3TimelineService
            
            # Initialize the full service
            vlm_service = InternVL3TimelineService(self.config)
            vlm_service._ensure_model_loaded()
            
            # Extract the scene analyzer
            self.vlm_analyzer = vlm_service.scene_analyzer
            self._vlm_loaded = True
            logger.info("VLM analyzer loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load VLM analyzer: {e}")
            raise SemanticSceneDetectionError(f"Failed to load VLM analyzer: {str(e)}") from e
    
    def optimize_scene_boundaries(self, video_path: str, pyscenedetect_scenes: List[Tuple[float, float]]) -> List[SceneBoundary]:
        """
        Optimize PySceneDetect boundaries using semantic analysis
        
        Args:
            video_path: Path to video file
            pyscenedetect_scenes: List of (start_time, end_time) tuples from PySceneDetect
            
        Returns:
            List of optimized SceneBoundary objects
        """
        start_time = time.time()
        self._ensure_vlm_loaded()
        
        try:
            # Convert PySceneDetect output to SceneBoundary objects
            initial_boundaries = self._convert_to_scene_boundaries(pyscenedetect_scenes)
            logger.info(f"Initial scene count: {len(initial_boundaries)}")
            
            # Phase 1: Cross-scene stabilization (merge similar adjacent scenes)
            if self.enable_cross_scene:
                stabilized_boundaries = self._phase1_cross_scene_stabilization(video_path, initial_boundaries)
                logger.info(f"After cross-scene stabilization: {len(stabilized_boundaries)} scenes")
            else:
                stabilized_boundaries = initial_boundaries
                logger.info("Cross-scene analysis disabled")
            
            # Phase 2: Intra-scene analysis (split semantically different content within scenes)  
            if self.enable_intra_scene:
                final_boundaries = self._phase2_intra_scene_analysis(video_path, stabilized_boundaries)
                logger.info(f"Final semantic scene count: {len(final_boundaries)}")
            else:
                final_boundaries = stabilized_boundaries
                logger.info("Intra-scene analysis disabled")
            
            # Final pass: Mark rapid cuts
            final_boundaries = self._mark_rapid_cuts(final_boundaries)
            
            processing_time = time.time() - start_time
            logger.info(f"Semantic scene optimization completed in {processing_time:.2f}s")
            
            return final_boundaries
            
        except Exception as e:
            logger.error(f"Semantic scene optimization failed: {e}")
            # Fallback to original boundaries
            return self._convert_to_scene_boundaries(pyscenedetect_scenes)
    
    def _convert_to_scene_boundaries(self, scenes: List[Tuple[float, float]]) -> List[SceneBoundary]:
        """Convert PySceneDetect output to SceneBoundary objects"""
        boundaries = []
        for i, (start_time, end_time) in enumerate(scenes):
            boundary = SceneBoundary(
                start_time=start_time,
                end_time=end_time,
                start_frame=int(start_time * 30),  # Assume 30 FPS
                end_frame=int(end_time * 30),
                confidence=1.0,
                semantic_type="normal"
            )
            boundaries.append(boundary)
        return boundaries
    
    def _phase1_cross_scene_stabilization(self, video_path: str, boundaries: List[SceneBoundary]) -> List[SceneBoundary]:
        """
        Phase 1: Stabilize cross-scene boundaries by merging semantically similar adjacent scenes
        """
        logger.info("Phase 1: Cross-scene boundary stabilization")
        
        current_boundaries = boundaries.copy()
        iteration = 0
        
        while iteration < self.max_stabilization_iterations:
            iteration += 1
            logger.info(f"Stabilization iteration {iteration}")
            
            changes_made = False
            new_boundaries = []
            i = 0
            
            while i < len(current_boundaries):
                current_scene = current_boundaries[i]
                
                # Check if we can merge with next scene
                if i + 1 < len(current_boundaries):
                    next_scene = current_boundaries[i + 1]
                    
                    # Compare representative frames semantically
                    similarity = self._compare_scenes_semantically(video_path, current_scene, next_scene)
                    
                    if similarity > self.merge_threshold:
                        # Merge scenes
                        merged_scene = self._merge_scenes(current_scene, next_scene)
                        new_boundaries.append(merged_scene)
                        logger.info(f"Merged scenes at {current_scene.start_time:.1f}s-{next_scene.end_time:.1f}s (similarity: {similarity:.2f})")
                        changes_made = True
                        i += 2  # Skip next scene as it's merged
                    else:
                        new_boundaries.append(current_scene)
                        i += 1
                else:
                    new_boundaries.append(current_scene)
                    i += 1
            
            current_boundaries = new_boundaries
            
            if not changes_made:
                logger.info(f"Stabilization converged after {iteration} iterations")
                break
        
        # Mark stability iterations
        for boundary in current_boundaries:
            boundary.stability_iterations = iteration
        
        return current_boundaries
    
    def _phase2_intra_scene_analysis(self, video_path: str, boundaries: List[SceneBoundary]) -> List[SceneBoundary]:
        """
        Phase 2: Analyze within scenes for semantic changes and split if needed
        """
        logger.info("Phase 2: Intra-scene semantic analysis")
        
        final_boundaries = []
        
        for boundary in boundaries:
            # Check if scene is long enough to analyze for splits
            duration = boundary.end_time - boundary.start_time
            
            if duration < self.min_scene_duration * 2:  # Too short to split
                final_boundaries.append(boundary)
                continue
            
            # Analyze for intra-scene semantic changes
            split_points = self._find_intra_scene_splits(video_path, boundary)
            
            if not split_points:
                final_boundaries.append(boundary)
            else:
                # Split the scene
                split_scenes = self._split_scene(boundary, split_points)
                final_boundaries.extend(split_scenes)
                logger.info(f"Split scene at {boundary.start_time:.1f}s-{boundary.end_time:.1f}s into {len(split_scenes)} parts")
        
        return final_boundaries
    
    def _compare_scenes_semantically(self, video_path: str, scene1: SceneBoundary, scene2: SceneBoundary) -> float:
        """
        Compare two scenes semantically using VLM dual-image analysis
        
        Returns:
            Similarity score (0.0 to 1.0)
        """
        try:
            # Extract representative frames from both scenes
            frame1 = self._extract_representative_frame(video_path, scene1)
            frame2 = self._extract_representative_frame(video_path, scene2)
            
            if frame1 is None or frame2 is None:
                return 0.0
            
            # Use VLM dual-image comparison
            prompt = "Compare these two images. Are they from the same scene or activity? Answer with SAME_SCENE or DIFFERENT_SCENE, followed by a confidence score from 0.0 to 1.0."
            
            response = self.vlm_analyzer._query_vlm_dual(frame1, frame2, prompt)
            
            # Parse response for similarity
            if "SAME_SCENE" in response.upper():
                # Try to extract confidence score
                try:
                    import re
                    confidence_match = re.search(r'(\d+\.?\d*)', response)
                    if confidence_match:
                        confidence = float(confidence_match.group(1))
                        return confidence if confidence <= 1.0 else confidence / 100.0
                    return 0.8  # Default high confidence for SAME_SCENE
                except:
                    return 0.8
            else:
                return 0.2  # Low similarity for DIFFERENT_SCENE
                
        except Exception as e:
            logger.warning(f"Failed to compare scenes semantically: {e}")
            return 0.5  # Neutral similarity on error
    
    def _extract_representative_frame(self, video_path: str, scene: SceneBoundary) -> Optional[Image.Image]:
        """Extract representative frame from scene middle"""
        try:
            mid_time = (scene.start_time + scene.end_time) / 2
            
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_number = int(mid_time * fps)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                # Convert BGR to RGB and create PIL Image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return Image.fromarray(frame_rgb)
            return None
            
        except Exception as e:
            logger.warning(f"Failed to extract frame at {scene.start_time:.1f}s: {e}")
            return None
    
    def _merge_scenes(self, scene1: SceneBoundary, scene2: SceneBoundary) -> SceneBoundary:
        """Merge two adjacent scenes"""
        merged = SceneBoundary(
            start_time=scene1.start_time,
            end_time=scene2.end_time,
            start_frame=scene1.start_frame,
            end_frame=scene2.end_frame,
            confidence=min(scene1.confidence, scene2.confidence),
            semantic_type="merged",
            merge_count=scene1.merge_count + scene2.merge_count + 1,
            split_count=max(scene1.split_count, scene2.split_count)
        )
        return merged
    
    def _find_intra_scene_splits(self, video_path: str, boundary: SceneBoundary) -> List[float]:
        """
        Find points within a scene where semantic content changes significantly
        
        Returns:
            List of timestamps where splits should occur
        """
        split_points = []
        duration = boundary.end_time - boundary.start_time
        
        # Sample frames at regular intervals within the scene
        sample_interval = max(1.0, duration / 5)  # Sample every 1s or duration/5, whichever is larger
        sample_times = []
        
        current_time = boundary.start_time + sample_interval
        while current_time < boundary.end_time - sample_interval:
            sample_times.append(current_time)
            current_time += sample_interval
        
        if len(sample_times) < 2:
            return split_points
        
        # Compare consecutive sample frames
        for i in range(len(sample_times) - 1):
            frame1 = self._extract_frame_at_time(video_path, sample_times[i])
            frame2 = self._extract_frame_at_time(video_path, sample_times[i + 1])
            
            if frame1 and frame2:
                # Check for semantic difference
                similarity = self._compare_frames_semantically(frame1, frame2)
                
                if similarity < self.split_threshold:
                    # Potential split point - check if it meets minimum duration
                    split_time = sample_times[i + 1]
                    
                    # Ensure minimum duration constraints
                    time_from_start = split_time - boundary.start_time
                    time_to_end = boundary.end_time - split_time
                    
                    if time_from_start >= self.min_scene_duration and time_to_end >= self.min_scene_duration:
                        split_points.append(split_time)
                        logger.info(f"Found intra-scene split at {split_time:.1f}s (similarity: {similarity:.2f})")
        
        return split_points
    
    def _extract_frame_at_time(self, video_path: str, timestamp: float) -> Optional[Image.Image]:
        """Extract frame at specific timestamp"""
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_number = int(timestamp * fps)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return Image.fromarray(frame_rgb)
            return None
            
        except Exception as e:
            logger.warning(f"Failed to extract frame at {timestamp:.1f}s: {e}")
            return None
    
    def _compare_frames_semantically(self, frame1: Image.Image, frame2: Image.Image) -> float:
        """Compare two frames semantically"""
        try:
            prompt = "Compare these images. How similar are they in content and scene? Answer with a similarity score from 0.0 (completely different) to 1.0 (identical)."
            response = self.vlm_analyzer._query_vlm_dual(frame1, frame2, prompt)
            
            # Extract similarity score
            import re
            score_match = re.search(r'(\d+\.?\d*)', response)
            if score_match:
                score = float(score_match.group(1))
                return score if score <= 1.0 else score / 100.0
            
            # Fallback parsing
            if "identical" in response.lower() or "very similar" in response.lower():
                return 0.9
            elif "similar" in response.lower():
                return 0.7
            elif "different" in response.lower():
                return 0.3
            else:
                return 0.5  # Neutral
                
        except Exception as e:
            logger.warning(f"Failed to compare frames semantically: {e}")
            return 0.5
    
    def _split_scene(self, boundary: SceneBoundary, split_points: List[float]) -> List[SceneBoundary]:
        """Split a scene at specified points"""
        if not split_points:
            return [boundary]
        
        # Sort split points
        sorted_splits = sorted(split_points)
        
        # Create new scene boundaries
        scenes = []
        current_start = boundary.start_time
        
        for split_time in sorted_splits:
            # Create scene from current_start to split_time
            scene = SceneBoundary(
                start_time=current_start,
                end_time=split_time,
                start_frame=int(current_start * 30),
                end_frame=int(split_time * 30),
                confidence=boundary.confidence,
                semantic_type="split",
                merge_count=boundary.merge_count,
                split_count=boundary.split_count + 1
            )
            scenes.append(scene)
            current_start = split_time
        
        # Add final scene from last split to end
        final_scene = SceneBoundary(
            start_time=current_start,
            end_time=boundary.end_time,
            start_frame=int(current_start * 30),
            end_frame=boundary.end_frame,
            confidence=boundary.confidence,
            semantic_type="split",
            merge_count=boundary.merge_count,
            split_count=boundary.split_count + 1
        )
        scenes.append(final_scene)
        
        # Check for rapid cuts (very short scenes)
        for scene in scenes:
            duration = scene.end_time - scene.start_time
            if duration < self.rapid_cut_threshold:
                scene.semantic_type = "rapid_cut"
        
        return scenes
    
    def _mark_rapid_cuts(self, boundaries: List[SceneBoundary]) -> List[SceneBoundary]:
        """Mark scenes shorter than rapid_cut_threshold as rapid cuts"""
        rapid_cut_count = 0
        
        for boundary in boundaries:
            duration = boundary.end_time - boundary.start_time
            if duration < self.rapid_cut_threshold:
                boundary.semantic_type = "rapid_cut"
                rapid_cut_count += 1
                logger.info(f"Marked scene [{boundary.start_time:.1f}s-{boundary.end_time:.1f}s] as rapid cut ({duration:.1f}s)")
        
        if rapid_cut_count > 0:
            logger.info(f"Total rapid cuts detected: {rapid_cut_count}")
        
        return boundaries
    
    def save_semantic_boundaries(self, boundaries: List[SceneBoundary], video_path: str):
        """Save semantic scene boundaries to JSON file"""
        try:
            # Create output directory
            video_path_parts = Path(video_path).parts
            if 'build' in video_path_parts:
                video_name = video_path_parts[video_path_parts.index('build') + 1]
            else:
                video_name = Path(video_path).stem.replace('video', '') or 'unknown'
            
            output_dir = Path('build') / video_name / 'semantic_scenes'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Convert boundaries to JSON-serializable format
            boundaries_data = []
            for boundary in boundaries:
                boundaries_data.append({
                    'start_time': boundary.start_time,
                    'end_time': boundary.end_time,
                    'start_frame': boundary.start_frame,
                    'end_frame': boundary.end_frame,
                    'duration': boundary.end_time - boundary.start_time,
                    'confidence': boundary.confidence,
                    'semantic_type': boundary.semantic_type,
                    'merge_count': boundary.merge_count,
                    'split_count': boundary.split_count,
                    'stability_iterations': boundary.stability_iterations
                })
            
            # Create output data
            output_data = {
                'video_path': str(video_path),
                'processing_timestamp': datetime.now().isoformat(),
                'service': self.service_name,
                'total_scenes': len(boundaries),
                'config': {
                    'min_scene_duration': self.min_scene_duration,
                    'max_stabilization_iterations': self.max_stabilization_iterations,
                    'merge_threshold': self.merge_threshold,
                    'split_threshold': self.split_threshold
                },
                'boundaries': boundaries_data
            }
            
            # Save to file
            output_file = output_dir / 'semantic_boundaries.json'
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            logger.info(f"Semantic boundaries saved to: {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save semantic boundaries: {e}")
    
    def cleanup(self):
        """Cleanup VLM analyzer to free memory"""
        if self._vlm_loaded and self.vlm_analyzer:
            try:
                logger.info("Cleaning up semantic scene detection service...")
                if hasattr(self.vlm_analyzer, 'cleanup'):
                    self.vlm_analyzer.cleanup()
                self.vlm_analyzer = None
                self._vlm_loaded = False
                logger.info("Semantic scene detection cleanup complete")
            except Exception as e:
                logger.warning(f"Error during cleanup: {e}")


# Export for easy importing
__all__ = ['SemanticSceneDetectionService', 'SceneBoundary']