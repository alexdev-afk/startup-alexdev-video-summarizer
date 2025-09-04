"""
Whisper Transcription Service

Handles GPU-based audio transcription using OpenAI Whisper.
Optimized for FFmpeg-prepared audio.wav files with speaker identification.
"""

import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import gc

# Optional imports for development mode
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from utils.logger import get_logger

logger = get_logger(__name__)


class WhisperError(Exception):
    """Whisper processing error"""
    pass


class WhisperService:
    """Whisper transcription service for audio processing"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Whisper service
        
        Args:
            config: Configuration dictionary with Whisper settings
        """
        self.config = config
        self.whisper_config = config.get('gpu_pipeline', {}).get('whisper', {})
        self.development_config = config.get('development', {})
        
        # Model configuration
        self.model_size = self.whisper_config.get('model_size', 'large-v2')
        self.device = self._determine_device()
        self.language = self.whisper_config.get('language', 'auto')
        
        # Runtime state
        self.model = None
        self.model_loaded = False
        
        logger.info(f"Whisper service initialized - model: {self.model_size}, device: {self.device}")
    
    def _determine_device(self) -> str:
        """Determine the best device for Whisper processing"""
        device_config = self.whisper_config.get('device', 'auto')
        
        if device_config == 'auto':
            if TORCH_AVAILABLE and torch.cuda.is_available():
                return 'cuda'
            else:
                logger.warning("CUDA not available, falling back to CPU")
                return 'cpu'
        
        return device_config
    
    def _load_model(self):
        """Load Whisper model (lazy loading)"""
        if self.model_loaded:
            return
        
        # Skip model loading in development mode or if dependencies missing
        if self.development_config.get('skip_model_loading', False) or not TORCH_AVAILABLE:
            logger.info("Whisper model loading skipped (development mode or missing dependencies)")
            self.model_loaded = True
            return
        
        try:
            import whisper
            logger.info(f"Loading Whisper model: {self.model_size}")
            
            self.model = whisper.load_model(self.model_size, device=self.device)
            self.model_loaded = True
            
            logger.info("Whisper model loaded successfully")
            
        except ImportError:
            raise WhisperError(
                "OpenAI Whisper not installed. Install with: pip install openai-whisper"
            )
        except Exception as e:
            raise WhisperError(f"Failed to load Whisper model: {str(e)}") from e
    
    def transcribe_audio(self, audio_path: Path, scene_info: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Transcribe audio file using Whisper
        
        Args:
            audio_path: Path to audio.wav file (from FFmpeg)
            scene_info: Optional scene boundary information for context
            
        Returns:
            Dictionary with transcription results
            
        Raises:
            WhisperError: If transcription fails
        """
        logger.info(f"Transcribing audio: {audio_path.name}")
        
        # Handle development mock mode
        if self.development_config.get('mock_ai_services', False):
            return self._mock_transcription(audio_path, scene_info)
        
        # Validate input
        if not audio_path.exists():
            raise WhisperError(f"Audio file not found: {audio_path}")
        
        if audio_path.stat().st_size < 1024:  # Less than 1KB
            raise WhisperError(f"Audio file too small: {audio_path}")
        
        try:
            # Load model if needed
            self._load_model()
            
            start_time = time.time()
            
            # Transcribe with Whisper
            transcription_options = {
                'task': 'transcribe',
                'verbose': False,
            }
            
            # Set language if not auto-detect
            if self.language != 'auto':
                transcription_options['language'] = self.language
            
            logger.debug(f"Starting Whisper transcription with options: {transcription_options}")
            result = self.model.transcribe(str(audio_path), **transcription_options)
            
            processing_time = time.time() - start_time
            
            # Process results
            processed_result = self._process_whisper_result(result, processing_time, scene_info)
            
            # GPU memory cleanup
            if self.device == 'cuda' and TORCH_AVAILABLE:
                torch.cuda.empty_cache()
                gc.collect()
            
            logger.info(f"Transcription complete: {processing_time:.2f}s, {len(result['segments'])} segments")
            return processed_result
            
        except Exception as e:
            # Cleanup on error
            self._cleanup_gpu_memory()
            raise WhisperError(f"Whisper transcription failed: {str(e)}") from e
    
    def _process_whisper_result(self, result: Dict, processing_time: float, scene_info: Optional[Dict]) -> Dict[str, Any]:
        """Process raw Whisper result into standardized format"""
        
        # Extract segments with speaker identification attempt
        segments = []
        speakers = set()
        
        for segment in result.get('segments', []):
            # Basic speaker identification (Whisper doesn't do this natively)
            # This would need additional diarization for true speaker separation
            speaker = 'Speaker_1'  # Placeholder - would use speaker diarization
            speakers.add(speaker)
            
            segments.append({
                'start': segment.get('start', 0),
                'end': segment.get('end', 0),
                'text': segment.get('text', '').strip(),
                'speaker': speaker,
                'confidence': segment.get('no_speech_prob', 0.0),
                'words': segment.get('words', []) if 'words' in segment else []
            })
        
        # Build final result
        processed_result = {
            'transcript': result.get('text', '').strip(),
            'language': result.get('language', 'unknown'),
            'language_probability': result.get('language_probability', 0.0),
            'segments': segments,
            'speakers': sorted(list(speakers)),
            'processing_time': processing_time,
            'model_info': {
                'model_size': self.model_size,
                'device': self.device
            },
            'scene_context': scene_info
        }
        
        return processed_result
    
    def _mock_transcription(self, audio_path: Path, scene_info: Optional[Dict]) -> Dict[str, Any]:
        """Mock transcription for development/testing"""
        logger.debug(f"Mock Whisper transcription: {audio_path.name}")
        
        # Simulate processing time
        time.sleep(0.5 if self.development_config.get('fast_mode', False) else 1.0)
        
        # Generate mock content based on filename or scene
        video_name = audio_path.parent.name
        scene_id = scene_info.get('scene_id', 1) if scene_info else 1
        
        mock_transcript = (
            f"This is a mock transcription for {video_name}, scene {scene_id}. "
            f"The speaker discusses various topics related to the video content. "
            f"Key points include project updates, technical details, and strategic planning. "
            f"Multiple speakers may be present in this {scene_info.get('duration', 60):.1f} second segment."
        )
        
        return {
            'transcript': mock_transcript,
            'language': 'en',
            'language_probability': 0.95,
            'segments': [
                {
                    'start': scene_info.get('start_seconds', 0) if scene_info else 0,
                    'end': scene_info.get('end_seconds', 60) if scene_info else 60,
                    'text': mock_transcript,
                    'speaker': 'Speaker_1',
                    'confidence': 0.92,
                    'words': []
                }
            ],
            'speakers': ['Speaker_1', 'Speaker_2'] if scene_id % 2 == 0 else ['Speaker_1'],
            'processing_time': 0.5,
            'model_info': {
                'model_size': 'mock',
                'device': 'mock'
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
            logger.debug("Whisper model unloaded")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            'model_size': self.model_size,
            'device': self.device,
            'language': self.language,
            'model_loaded': self.model_loaded,
            'development_mode': self.development_config.get('mock_ai_services', False)
        }


# Export for easy importing
__all__ = ['WhisperService', 'WhisperError']