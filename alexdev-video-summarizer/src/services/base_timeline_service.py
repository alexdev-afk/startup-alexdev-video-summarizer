"""
Base Timeline Service

Abstract base class providing common functionality for timeline services.
Eliminates code duplication between LibROSA and PyAudio timeline services.
"""

import time
import gc
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import json

from utils.logger import get_logger
from utils.timeline_schema import ServiceTimeline
from utils.enhanced_timeline_schema import EnhancedTimeline

logger = get_logger(__name__)


class BaseTimelineServiceError(Exception):
    """Base timeline service error"""
    pass


class BaseTimelineService(ABC):
    """Abstract base class for timeline services"""
    
    def __init__(self, config: Dict[str, Any], service_name: str):
        """
        Initialize base timeline service
        
        Args:
            config: Configuration dictionary
            service_name: Name of the service (e.g., 'librosa', 'pyaudio')
        """
        self.config = config
        self.service_name = service_name
        self.service_config = self._get_service_config(config, service_name)
        
        logger.info(f"{service_name.title()} timeline service base initialized")
    
    @abstractmethod
    def _get_service_config(self, config: Dict[str, Any], service_name: str) -> Dict[str, Any]:
        """Get service-specific configuration"""
        pass
    
    @abstractmethod
    def _detect_enhanced_events_and_spans(self, audio_data: Any, timeline: EnhancedTimeline, **kwargs):
        """Detect events and spans for enhanced timeline - implemented by subclasses"""
        pass
    
    @abstractmethod
    def _detect_events_and_spans(self, audio_data: Any, timeline: ServiceTimeline, **kwargs):
        """Detect events and spans for service timeline - implemented by subclasses"""
        pass
    
    @abstractmethod
    def _load_audio(self, audio_path: str) -> Tuple[Any, Any]:
        """Load audio file - implemented by subclasses"""
        pass
    
    @abstractmethod
    def _add_enhanced_mock_data(self, timeline: EnhancedTimeline):
        """Add mock data when service is unavailable - enhanced timeline"""
        pass
    
    @abstractmethod
    def _add_mock_data(self, timeline: ServiceTimeline):
        """Add mock data when service is unavailable - service timeline"""
        pass
    
    def generate_and_save(self, audio_path: str) -> EnhancedTimeline:
        """
        Generate enhanced timeline and save intermediate files
        
        Args:
            audio_path: Path to audio file (FFmpeg-prepared WAV)
            
        Returns:
            EnhancedTimeline with detected events and spans
        """
        start_time = time.time()
        
        try:
            # Load audio data
            audio_data, audio_metadata = self._load_audio(audio_path)
            if audio_data is None:
                return self._create_fallback_enhanced_timeline(audio_path)
            
            # Calculate total duration
            total_duration = self._calculate_duration(audio_data, audio_metadata)
            
            # Create enhanced timeline object
            timeline = EnhancedTimeline(
                audio_file=str(audio_path),
                total_duration=total_duration
            )
            
            # Add service to sources used
            timeline.sources_used.append(self.service_name)
            
            # Add processing notes
            self._add_processing_notes(timeline, audio_metadata)
            
            # Generate events and spans using service-specific detection
            self._detect_enhanced_events_and_spans(audio_data, timeline, metadata=audio_metadata)
            
            processing_time = time.time() - start_time
            logger.info(f"{self.service_name.title()} enhanced timeline generated: {len(timeline.events)} events, {len(timeline.spans)} spans in {processing_time:.2f}s")
            
            # Save intermediate analysis files
            self._save_intermediate_analysis(timeline, audio_path, audio_data, audio_metadata)
            
            # Save timeline to audio_timelines directory
            self._save_enhanced_timeline(timeline, audio_path)
            
            return timeline
            
        except Exception as e:
            logger.error(f"{self.service_name.title()} enhanced timeline generation failed: {e}")
            return self._create_fallback_enhanced_timeline(audio_path, error=str(e))
    
    def generate_timeline(self, audio_path: str) -> ServiceTimeline:
        """
        Generate service timeline from audio
        
        Args:
            audio_path: Path to audio file (FFmpeg-prepared WAV)
            
        Returns:
            ServiceTimeline with detected events and spans
        """
        start_time = time.time()
        
        try:
            # Load audio data
            audio_data, audio_metadata = self._load_audio(audio_path)
            if audio_data is None:
                return self._create_fallback_timeline(audio_path)
            
            # Calculate total duration
            total_duration = self._calculate_duration(audio_data, audio_metadata)
            
            # Create timeline object
            timeline = ServiceTimeline(
                source=self.service_name,
                audio_file=str(audio_path),
                total_duration=total_duration
            )
            
            # Generate events and spans using service-specific detection
            self._detect_events_and_spans(audio_data, timeline, metadata=audio_metadata)
            
            processing_time = time.time() - start_time
            logger.info(f"{self.service_name.title()} timeline generated: {len(timeline.events)} events, {len(timeline.spans)} spans in {processing_time:.2f}s")
            
            # Save timeline to file
            self._save_timeline(timeline, audio_path)
            
            return timeline
            
        except Exception as e:
            logger.error(f"{self.service_name.title()} timeline generation failed: {e}")
            return self._create_fallback_timeline(audio_path, error=str(e))
    
    @abstractmethod
    def _calculate_duration(self, audio_data: Any, audio_metadata: Any) -> float:
        """Calculate audio duration - implemented by subclasses"""
        pass
    
    @abstractmethod
    def _add_processing_notes(self, timeline: EnhancedTimeline, audio_metadata: Any):
        """Add processing notes to timeline - implemented by subclasses"""
        pass
    
    def _create_fallback_enhanced_timeline(self, audio_path: str, error: Optional[str] = None) -> EnhancedTimeline:
        """Create fallback enhanced timeline when processing fails"""
        estimated_duration = self._estimate_duration_from_file(audio_path)
        
        timeline = EnhancedTimeline(
            audio_file=str(audio_path),
            total_duration=estimated_duration
        )
        
        # Add service to sources used
        timeline.sources_used.append(self.service_name)
        
        # Add fallback processing notes
        timeline.processing_notes.append(f"{self.service_name.title()} fallback mode")
        if error:
            timeline.processing_notes.append(f"Error: {error}")
        
        # Add mock data
        self._add_enhanced_mock_data(timeline)
        
        logger.warning(f"Using fallback {self.service_name} enhanced timeline: {error or f'{self.service_name} unavailable'}")
        return timeline
    
    def _create_fallback_timeline(self, audio_path: str, error: Optional[str] = None) -> ServiceTimeline:
        """Create fallback timeline when processing fails"""
        estimated_duration = self._estimate_duration_from_file(audio_path)
        
        timeline = ServiceTimeline(
            source=self.service_name,
            audio_file=str(audio_path),
            total_duration=estimated_duration
        )
        
        # Add mock data
        self._add_mock_data(timeline)
        
        logger.warning(f"Using fallback {self.service_name} timeline: {error or f'{self.service_name} unavailable'}")
        return timeline
    
    def _estimate_duration_from_file(self, audio_path: str) -> float:
        """Estimate duration from file info"""
        try:
            file_size = Path(audio_path).stat().st_size
            # Rough estimate: 44.1 kHz * 16-bit * mono = ~88KB per second
            estimated_duration = file_size / 88000
            return max(30.0, estimated_duration)  # Minimum 30 seconds
        except:
            return 30.0  # Default fallback
    
    def _save_intermediate_analysis(self, timeline: EnhancedTimeline, audio_path: str, audio_data: Any, audio_metadata: Any):
        """Save intermediate analysis files to audio_analysis directory"""
        try:
            # Determine output path
            audio_pathlib = Path(audio_path)
            build_dir = audio_pathlib.parent
            analysis_dir = build_dir / "audio_analysis"
            analysis_dir.mkdir(exist_ok=True)
            
            # Create service-specific analysis data
            analysis_data = self._create_analysis_data(timeline, audio_path, audio_data, audio_metadata)
            
            output_file = analysis_dir / f"{self.service_name}_analysis.json"
            
            with open(output_file, 'w') as f:
                json.dump(analysis_data, f, indent=2)
            
            logger.info(f"{self.service_name.title()} intermediate analysis saved to: {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save {self.service_name} intermediate analysis: {e}")
    
    @abstractmethod
    def _create_analysis_data(self, timeline: EnhancedTimeline, audio_path: str, audio_data: Any, audio_metadata: Any) -> Dict[str, Any]:
        """Create analysis data dictionary - implemented by subclasses"""
        pass
    
    def _save_enhanced_timeline(self, timeline: EnhancedTimeline, audio_path: str):
        """Save enhanced timeline to audio_timelines directory"""
        try:
            # Determine output path
            audio_pathlib = Path(audio_path)
            build_dir = audio_pathlib.parent
            timeline_dir = build_dir / "audio_timelines"
            timeline_dir.mkdir(exist_ok=True)
            
            output_file = timeline_dir / f"{self.service_name}_timeline.json"
            timeline.save_to_file(str(output_file))
            
            logger.info(f"{self.service_name.title()} enhanced timeline saved to: {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save {self.service_name} enhanced timeline: {e}")
    
    def _save_timeline(self, timeline: ServiceTimeline, audio_path: str):
        """Save timeline to file"""
        try:
            # Determine output path
            audio_pathlib = Path(audio_path)
            build_dir = audio_pathlib.parent
            timeline_dir = build_dir / "audio_timelines"
            timeline_dir.mkdir(exist_ok=True)
            
            output_file = timeline_dir / f"{self.service_name}_timeline.json"
            timeline.save_to_file(str(output_file))
            
            logger.info(f"{self.service_name.title()} timeline saved to: {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save {self.service_name} timeline: {e}")
    
    def cleanup(self):
        """Cleanup resources"""
        gc.collect()
        logger.debug(f"{self.service_name.title()} timeline service cleanup complete")