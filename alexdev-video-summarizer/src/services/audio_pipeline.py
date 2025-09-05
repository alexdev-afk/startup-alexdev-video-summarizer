"""
Audio Pipeline Controller for sequential audio processing.

Handles Whisper → LibROSA → pyAudioAnalysis pipeline.
Sequential processing of audio content per scene.
"""

import time
from pathlib import Path
from typing import Dict, Any

from services.whisper_service import WhisperService, WhisperError
from services.librosa_service import LibROSAService, LibROSAError
from utils.logger import get_logger

logger = get_logger(__name__)


class AudioPipelineController:
    """Audio pipeline controller for sequential audio processing"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize audio pipeline controller
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.audio_config = config.get('audio_pipeline', {})
        
        # Initialize services
        self.whisper_service = WhisperService(config)
        self.librosa_service = LibROSAService(config)
        
        logger.info("Audio pipeline controller initialized")
    
    def process_scene(self, scene: Dict[str, Any], context) -> Dict[str, Any]:
        """
        Process scene through Audio pipeline
        
        Sequential processing: Whisper → LibROSA → pyAudioAnalysis
        
        Args:
            scene: Scene boundary data
            context: Video processing context
            
        Returns:
            Combined audio analysis results
        """
        scene_id = scene['scene_id']
        logger.info(f"Processing scene {scene_id} through Audio pipeline")
        
        results = {}
        
        try:
            # Step 1: Whisper transcription (using FFmpeg-extracted audio)
            if hasattr(context, 'audio_path') and context.audio_path:
                results['whisper'] = self.whisper_service.transcribe_audio(
                    context.audio_path, scene_info=scene
                )
            else:
                logger.warning(f"No audio file available for scene {scene_id}")
                results['whisper'] = self._fallback_whisper_result(scene)
            
            # Step 2: LibROSA music analysis
            if hasattr(context, 'audio_path') and context.audio_path:
                results['librosa'] = self.librosa_service.analyze_audio_segment(
                    context.audio_path, scene_info=scene
                )
            else:
                logger.warning(f"No audio file available for LibROSA analysis in scene {scene_id}")
                results['librosa'] = self._fallback_librosa_result(scene)
            
            # Step 3: pyAudioAnalysis feature extraction (mock for now - Phase 3)
            results['pyaudioanalysis'] = self._mock_pyaudioanalysis_processing(scene, context)
            
            logger.info(f"Audio pipeline complete for scene {scene_id}")
            return results
            
        except (WhisperError, LibROSAError) as e:
            logger.error(f"Audio pipeline failed for scene {scene_id}: {e}")
            # Return fallback results to prevent complete failure
            return {
                'whisper': self._fallback_whisper_result(scene),
                'librosa': self.librosa_service._fallback_analysis_result(scene, error=str(e)),
                'pyaudioanalysis': self._fallback_pyaudioanalysis_result(scene),
                'error': str(e)
            }
        except Exception as e:
            logger.error(f"Audio pipeline unexpected failure for scene {scene_id}: {e}")
            # Return fallback results to prevent complete failure
            return {
                'whisper': self._fallback_whisper_result(scene),
                'librosa': self.librosa_service._fallback_analysis_result(scene, error=str(e)),
                'pyaudioanalysis': self._fallback_pyaudioanalysis_result(scene),
                'error': str(e)
            }
    
    def _fallback_whisper_result(self, scene: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback Whisper result when audio processing fails"""
        return {
            'transcript': f"[Audio processing failed for scene {scene['scene_id']}]",
            'speakers': [],
            'language': 'unknown',
            'language_probability': 0.0,
            'segments': [],
            'processing_time': 0.0,
            'error': 'Audio processing unavailable',
            'fallback_mode': True
        }
    
    def _fallback_librosa_result(self, scene: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback LibROSA result when processing fails"""
        return self.librosa_service._fallback_analysis_result(scene, error='LibROSA processing unavailable')
    
    def _fallback_pyaudioanalysis_result(self, scene: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback pyAudioAnalysis result when processing fails"""
        return {
            'features': {},
            'processing_time': 0.0,
            'error': 'pyAudioAnalysis processing unavailable',
            'fallback_mode': True
        }
    
    def _mock_pyaudioanalysis_processing(self, scene: Dict[str, Any], context) -> Dict[str, Any]:
        """Mock pyAudioAnalysis 68-feature extraction processing"""
        logger.debug(f"Mock pyAudioAnalysis processing for scene {scene['scene_id']}")
        
        # Simulate processing time
        time.sleep(0.8)
        
        # Mock 68 audio features (simplified representation)
        mock_features = {
            f'feature_{i:02d}': 0.5 + (scene['scene_id'] * 0.01) + (i * 0.005)
            for i in range(1, 69)
        }
        
        return {
            'features_68': mock_features,
            'classification': {
                'type': 'speech',
                'subtype': 'presentation' if scene['scene_id'] == 1 else 'discussion',
                'confidence': 0.87,
                'speaker_emotion': 'neutral',
                'audio_quality': 'high'
            },
            'summary_statistics': {
                'mean_energy': 0.45,
                'std_energy': 0.12,
                'spectral_rolloff': 3200.0,
                'spectral_flux': 0.023
            },
            'processing_time': 0.8,
            'mock_mode': True
        }