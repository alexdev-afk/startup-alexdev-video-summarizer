"""
Video processing context management.

Manages processing state and data for individual videos throughout the pipeline.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class VideoProcessingContext:
    """Processing context for a single video"""
    
    video_path: Path
    audio_path: Optional[Path] = None
    processed_video_path: Optional[Path] = None
    vocals_path: Optional[Path] = None
    no_vocals_path: Optional[Path] = None
    scene_data: Optional[Dict[str, Any]] = None
    build_directory: Optional[Path] = None
    scene_analysis: Dict[str, Any] = field(default_factory=dict)
    video_analysis_results: Optional[Dict[str, Any]] = None
    processing_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize processing context"""
        self.video_name = self.video_path.stem
        self.build_directory = Path('build') / self.video_name
        self.build_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize processing metadata
        self.processing_metadata = {
            'video_name': self.video_name,
            'original_path': str(self.video_path),
            'build_directory': str(self.build_directory),
            'processing_stages': {},
            'errors': []
        }
        
        logger.debug(f"Processing context initialized for: {self.video_name}")
    
    def validate_ffmpeg_output(self) -> bool:
        """Validate that FFmpeg extraction was successful"""
        if not self.audio_path or not self.audio_path.exists():
            logger.error(f"Audio file missing: {self.audio_path}")
            return False
            
        if not self.processed_video_path or not self.processed_video_path.exists():
            logger.error(f"Video file missing: {self.processed_video_path}")
            return False
            
        # Check file sizes
        if self.audio_path.stat().st_size < 1024:
            logger.error(f"Audio file too small: {self.audio_path}")
            return False
            
        if self.processed_video_path.stat().st_size < 1024:
            logger.error(f"Video file too small: {self.processed_video_path}")
            return False
            
        logger.debug("FFmpeg outputs validated successfully")
        return True
    
    def store_scene_analysis(self, scene_id: int, audio_results: Dict[str, Any], video_gpu_results: Dict[str, Any], video_cpu_results: Dict[str, Any]):
        """
        Store analysis results for a scene from all 3 pipelines
        
        Args:
            scene_id: Scene identifier
            audio_results: Results from Audio pipeline (Whisper → LibROSA → pyAudioAnalysis)
            video_gpu_results: Results from Video GPU pipeline (InternVL3 VLM)
            video_cpu_results: Results from Video CPU pipeline (Visual analysis)
        """
        self.scene_analysis[scene_id] = {
            'scene_id': scene_id,
            'audio_analysis': audio_results,
            'video_gpu_analysis': video_gpu_results,
            'video_cpu_analysis': video_cpu_results,
            'combined_analysis': self._combine_analysis_results(audio_results, video_gpu_results, video_cpu_results)
        }
        
        # Save to disk for persistence
        self._save_scene_analysis(scene_id)
        
        logger.debug(f"Scene {scene_id} analysis stored")
    
    def _combine_analysis_results(self, audio_results: Dict[str, Any], video_gpu_results: Dict[str, Any], video_cpu_results: Dict[str, Any]) -> Dict[str, Any]:
        """Combine results from all 3 pipelines"""
        combined = {
            'audio_analysis': {
                'transcript': audio_results.get('whisper', {}).get('transcript', ''),
                'speakers': audio_results.get('whisper', {}).get('speakers', []),
                'music_features': audio_results.get('librosa', {}).get('features', {}),
                'audio_classification': audio_results.get('pyaudioanalysis', {}).get('classification', {})
            },
            'visual_analysis': {
                'objects': video_gpu_results.get('internvl3', {}).get('objects', []),
                'people_count': len([obj for obj in video_gpu_results.get('internvl3', {}).get('objects', []) if obj.get('class') == 'person']),
                'text_content': video_gpu_results.get('internvl3', {}).get('text_content', []),
                'faces': video_gpu_results.get('internvl3', {}).get('faces', [])
            }
        }
        
        return combined
    
    def _save_scene_analysis(self, scene_id: int):
        """Save scene analysis to disk"""
        analysis_dir = self.build_directory / 'analysis'
        analysis_dir.mkdir(exist_ok=True)
        
        analysis_file = analysis_dir / f'scene_{scene_id:03d}_analysis.json'
        
        try:
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(self.scene_analysis[scene_id], f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save scene {scene_id} analysis: {e}")
    
    def get_all_analysis(self) -> Dict[str, Any]:
        """
        Get complete analysis data for knowledge base generation
        
        Returns:
            Complete analysis data structure
        """
        return {
            'video_metadata': {
                'name': self.video_name,
                'original_path': str(self.video_path),
                'audio_path': str(self.audio_path) if self.audio_path else None,
                'processed_video_path': str(self.video_path) if self.video_path else None,
                'build_directory': str(self.build_directory)
            },
            'scene_data': self.scene_data,
            'scene_analysis': self.scene_analysis,
            'video_analysis_results': self.video_analysis_results,
            'processing_metadata': self.processing_metadata
        }
    
    def update_processing_stage(self, stage: str, status: str, details: Dict[str, Any] = None):
        """
        Update processing stage status
        
        Args:
            stage: Processing stage name
            status: Status (starting, completed, failed)
            details: Optional stage-specific details
        """
        self.processing_metadata['processing_stages'][stage] = {
            'status': status,
            'details': details or {},
            'timestamp': None  # Could add timestamp here
        }
        
        # Save updated metadata
        self._save_processing_metadata()
    
    def add_error(self, stage: str, error: str):
        """Add error to processing metadata"""
        error_entry = {
            'stage': stage,
            'error': error,
            'timestamp': None  # Could add timestamp here
        }
        
        self.processing_metadata['errors'].append(error_entry)
        self._save_processing_metadata()
        
        logger.error(f"Processing error in {stage}: {error}")
    
    def _save_processing_metadata(self):
        """Save processing metadata to disk"""
        metadata_file = self.build_directory / 'processing_metadata.json'
        
        try:
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.processing_metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save processing metadata: {e}")
    
    def cleanup_artifacts(self):
        """Clean up processing artifacts for failed video"""
        try:
            if self.build_directory and self.build_directory.exists():
                shutil.rmtree(self.build_directory)
                logger.debug(f"Cleaned up artifacts for: {self.video_name}")
        except Exception as e:
            logger.warning(f"Failed to cleanup artifacts for {self.video_name}: {e}")
    
    def get_scene_count(self) -> int:
        """Get number of detected scenes"""
        if self.scene_data:
            return self.scene_data.get('scene_count', 0)
        return 0
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get processing summary for reporting"""
        return {
            'video_name': self.video_name,
            'scene_count': self.get_scene_count(),
            'analysis_complete': len(self.scene_analysis),
            'processing_stages': list(self.processing_metadata['processing_stages'].keys()),
            'error_count': len(self.processing_metadata['errors']),
            'build_directory': str(self.build_directory)
        }
    
    def get_audio_file_path(self) -> Path:
        """Get path to the main audio file"""
        return self.build_directory / "audio.wav"
    
    def get_audio_analysis_path(self, analysis_type: str) -> Path:
        """Get path to audio analysis directory or specific analysis file"""
        return self.build_directory / "audio_analysis" / analysis_type
    
    def get_extraction_directory(self) -> Path:
        """Get path to extraction directory for audio/video files"""
        return self.build_directory / "extraction"