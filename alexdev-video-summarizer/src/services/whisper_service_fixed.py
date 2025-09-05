"""
WhisperX + Silero VAD Transcription Service with Advanced Chunking

Handles GPU-based audio transcription using WhisperX with Silero VAD.
Optimized for FFmpeg-prepared audio.wav files with enhanced speaker identification.
Includes advanced chunking strategies for advertisement content.
"""

import time
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import gc
from datetime import datetime
import numpy as np

# Optional imports for development mode
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import whisperx
    WHISPERX_AVAILABLE = True
except ImportError:
    WHISPERX_AVAILABLE = False

try:
    # Silero VAD model and utilities
    silero_vad_model = None
    silero_utils = None
    SILERO_AVAILABLE = False
    
    def load_silero_vad():
        global silero_vad_model, silero_utils, SILERO_AVAILABLE
        if not SILERO_AVAILABLE:
            try:
                silero_vad_model, silero_utils = torch.hub.load(
                    repo_or_dir="snakers4/silero-vad", 
                    model="silero_vad", 
                    onnx=False
                )
                SILERO_AVAILABLE = True
            except Exception as e:
                logger.warning(f"Failed to load Silero VAD: {e}")
        return silero_vad_model, silero_utils
except ImportError:
    SILERO_AVAILABLE = False
    def load_silero_vad():
        return None, None

from utils.logger import get_logger
from .whisper_vad_chunking import AdvancedVADChunking

logger = get_logger(__name__)


class WhisperError(Exception):
    """Whisper processing error"""
    pass


class WhisperService:
    """WhisperX + Silero VAD transcription service for audio processing"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize WhisperX + Silero VAD service
        
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
        
        # VAD configuration
        self.vad_threshold = self.whisper_config.get('vad_threshold', 0.4)
        self.chunk_threshold = self.whisper_config.get('chunk_threshold', 3.0)
        
        # Advertisement-optimized chunking settings
        self.chunking_strategy = self.whisper_config.get('chunking_strategy', 'multi_strategy')  # 'gap_based', 'duration_based', 'multi_strategy'
        self.max_chunk_duration = self.whisper_config.get('max_chunk_duration', 20.0)  # 20 seconds for ads
        self.min_chunk_duration = self.whisper_config.get('min_chunk_duration', 5.0)   # 5 seconds minimum
        self.energy_based_splitting = self.whisper_config.get('energy_based_splitting', True)
        
        # Hybrid mode configuration
        self.use_original_whisper = self.whisper_config.get('use_original_whisper', False)
        self.enable_silero_vad = self.whisper_config.get('enable_silero_vad', False)
        self.enable_word_timestamps = self.whisper_config.get('enable_word_timestamps', True)
        self.enable_word_alignment = self.whisper_config.get('enable_word_alignment', True)
        
        # Sequential model loading for GPU memory management (WhisperX only)
        self.sequential_model_loading = self.whisper_config.get('sequential_model_loading', False)
        
        # Diarization configuration
        self.huggingface_token = self.whisper_config.get('huggingface_token')
        self.enable_diarization = self.whisper_config.get('enable_diarization', True)
        self.max_speakers = self.whisper_config.get('max_speakers', 10)
        self.min_speakers = self.whisper_config.get('min_speakers', 1)
        
        # Runtime state
        self.whisperx_model = None
        self.original_whisper_model = None  # Original Whisper model for hybrid mode
        self.diarize_model = None
        self.silero_model = None
        self.silero_utils = None
        self.model_loaded = False
        
        # Initialize advanced VAD chunking
        self.advanced_chunking = AdvancedVADChunking(config)
        
        # Setup FFmpeg path for Whisper compatibility
        self._setup_ffmpeg_path()
        
        logger.info(f"WhisperX + Silero VAD service initialized - model: {self.model_size}, device: {self.device}, chunking: {self.chunking_strategy}")
    
    # ... [rest of the methods remain the same until _perform_vad_segmentation] ...
    
    def _perform_vad_segmentation(self, audio_path: Path) -> List[Dict[str, Any]]:
        """
        Perform Voice Activity Detection using advanced chunking strategies for advertisements
        
        Returns:
            List of VAD segments with start/end timestamps and audio data
        """
        if not self.silero_model or not self.silero_utils:
            logger.warning("Silero VAD not available, using fallback segmentation")
            return self._fallback_vad_segmentation(audio_path)
        
        try:
            # Extract VAD utilities
            get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks = self.silero_utils
            
            # Read audio at 16kHz for VAD (Silero VAD requirement)
            VAD_SR = 16000
            wav = read_audio(str(audio_path), sampling_rate=VAD_SR)
            
            # Get speech timestamps using Silero VAD
            speech_timestamps = get_speech_timestamps(
                wav, 
                self.silero_model, 
                sampling_rate=VAD_SR, 
                threshold=self.vad_threshold
            )
            
            logger.debug(f"Raw VAD detected {len(speech_timestamps)} speech regions")
            
            # Apply advanced chunking strategy
            if self.chunking_strategy == 'gap_based':
                processed_chunks = self.advanced_chunking.chunk_by_gaps(speech_timestamps, wav, VAD_SR, collect_chunks)
            elif self.chunking_strategy == 'duration_based':
                processed_chunks = self.advanced_chunking.chunk_by_duration(speech_timestamps, wav, VAD_SR, collect_chunks)
            elif self.chunking_strategy == 'multi_strategy':
                processed_chunks = self.advanced_chunking.chunk_multi_strategy(speech_timestamps, wav, VAD_SR, collect_chunks)
            else:
                # Default to multi-strategy for advertisements
                processed_chunks = self.advanced_chunking.chunk_multi_strategy(speech_timestamps, wav, VAD_SR, collect_chunks)
            
            logger.info(f"VAD segmentation complete ({self.chunking_strategy}): {len(speech_timestamps)} regions -> {len(processed_chunks)} chunks")
            return processed_chunks
            
        except Exception as e:
            logger.error(f"Silero VAD failed: {e}, falling back to time-based segmentation")
            return self._fallback_vad_segmentation(audio_path)

# ... [rest of the file would continue with all other methods] ...