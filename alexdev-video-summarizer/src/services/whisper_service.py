"""
WhisperX + Silero VAD Transcription Service

Handles GPU-based audio transcription using WhisperX with Silero VAD.
Optimized for FFmpeg-prepared audio.wav files with enhanced speaker identification.
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
        
        # Setup FFmpeg path for Whisper compatibility
        self._setup_ffmpeg_path()
        
        logger.info(f"WhisperX + Silero VAD service initialized - model: {self.model_size}, device: {self.device}, vad_threshold: {self.vad_threshold}")
    
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
    
    def _setup_ffmpeg_path(self):
        """Setup FFmpeg path in environment for Whisper compatibility"""
        import platform
        import subprocess
        
        # First, test if FFmpeg is already available
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            logger.info("FFmpeg already available in PATH")
            return
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass
        
        # Try to find FFmpeg using the same logic as FFmpeg service
        if platform.system() == 'Windows':
            # WinGet installation path
            winget_path = Path.home() / "AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-8.0-full_build/bin/ffmpeg.exe"
            if winget_path.exists():
                # Add the directory containing ffmpeg.exe to PATH at the beginning
                ffmpeg_dir = str(winget_path.parent)
                current_path = os.environ.get('PATH', '')
                if ffmpeg_dir not in current_path:
                    os.environ['PATH'] = ffmpeg_dir + os.pathsep + current_path
                    logger.info(f"Added FFmpeg directory to PATH for WhisperX: {ffmpeg_dir}")
                    
                # Verify FFmpeg is now accessible
                try:
                    subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
                    logger.info("FFmpeg PATH setup successful")
                    return
                except (FileNotFoundError, subprocess.CalledProcessError):
                    logger.warning("FFmpeg found but not accessible after PATH setup")
            
            # WinGet Links path
            links_path = Path.home() / "AppData/Local/Microsoft/WinGet/Links/ffmpeg.exe"
            if links_path.exists():
                ffmpeg_dir = str(links_path.parent)
                current_path = os.environ.get('PATH', '')
                if ffmpeg_dir not in current_path:
                    os.environ['PATH'] = ffmpeg_dir + os.pathsep + current_path
                    logger.info(f"Added FFmpeg directory to PATH for WhisperX: {ffmpeg_dir}")
                    
                # Verify FFmpeg is now accessible
                try:
                    subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
                    logger.info("FFmpeg PATH setup successful")
                    return
                except (FileNotFoundError, subprocess.CalledProcessError):
                    logger.warning("FFmpeg found but not accessible after PATH setup")
        
        # If ffmpeg still not found, log error
        logger.error("FFmpeg not found in common locations and not in PATH. WhisperX will fail without FFmpeg.")
        raise WhisperError("FFmpeg not found. Install FFmpeg and ensure it's in PATH for WhisperX compatibility.")
    
    def _load_model(self):
        """Load WhisperX and Silero VAD models (lazy loading)"""
        if self.model_loaded:
            return
        
        # Skip model loading in development mode or if dependencies missing
        if self.development_config.get('skip_model_loading', False) or not TORCH_AVAILABLE:
            logger.info("WhisperX + Silero VAD model loading skipped (development mode or missing dependencies)")
            self.model_loaded = True
            return
        
        try:
            # Choose between original Whisper and WhisperX
            if self.use_original_whisper:
                # Load original Whisper (GPU compatible)
                logger.info(f"Loading original Whisper model: {self.model_size}")
                import whisper
                self.original_whisper_model = whisper.load_model(self.model_size, device=self.device)
                logger.info("Original Whisper model loaded successfully on GPU")
            
            elif WHISPERX_AVAILABLE:
                # Load WhisperX model (fallback mode)
                logger.info(f"Loading WhisperX model: {self.model_size}")
                
                # Set compute type based on device
                if self.device == 'cuda':
                    compute_type = 'float16'  # GPU supports float16
                else:
                    compute_type = 'int8'     # CPU fallback to int8
                
                self.whisperx_model = whisperx.load_model(
                    self.model_size, 
                    device=self.device,
                    compute_type=compute_type
                )
                
                # Sequential loading: Only load diarization if not using sequential mode
                if not self.sequential_model_loading:
                    # Load diarization model immediately (original behavior)
                    if self.enable_diarization and self.huggingface_token:
                        try:
                            self.diarize_model = whisperx.diarize.DiarizationPipeline(
                                use_auth_token=self.huggingface_token,
                                device=self.device
                            )
                            logger.info("WhisperX model and diarization pipeline loaded successfully")
                            logger.info(f"Diarization enabled: speakers {self.min_speakers}-{self.max_speakers}")
                        except Exception as e:
                            logger.warning(f"Diarization pipeline failed to load: {e}")
                            logger.info("WhisperX model loaded successfully (without diarization)")
                            self.diarize_model = None
                    elif not self.huggingface_token and self.enable_diarization:
                        logger.warning("Diarization enabled but no Hugging Face token provided")
                        logger.info("Set 'huggingface_token' in config for speaker diarization")
                        self.diarize_model = None
                    else:
                        logger.info("WhisperX model loaded (diarization disabled in config)")
                        self.diarize_model = None
                else:
                    # Sequential loading: Defer diarization model loading
                    logger.info("WhisperX model loaded successfully (sequential mode - diarization deferred)")
                    self.diarize_model = None
            else:
                # Fallback to original Whisper
                import whisper
                logger.warning("WhisperX not available, falling back to OpenAI Whisper")
                self.whisperx_model = whisper.load_model(self.model_size, device=self.device)
                logger.info("OpenAI Whisper model loaded successfully")
            
            # Load Silero VAD model
            self.silero_model, self.silero_utils = load_silero_vad()
            if self.silero_model:
                logger.info("Silero VAD model loaded successfully")
            else:
                logger.warning("Silero VAD not available, using basic VAD")
            
            self.model_loaded = True
            
        except ImportError as e:
            raise WhisperError(
                f"Required dependencies not installed: {str(e)}. "
                "Install with: pip install whisperx torch"
            )
        except Exception as e:
            raise WhisperError(f"Failed to load models: {str(e)}") from e
    
    def _load_diarization_model(self):
        """Load diarization model for sequential processing"""
        if self.diarize_model or not self.huggingface_token:
            return
            
        try:
            logger.info("Loading diarization model for sequential processing...")
            self.diarize_model = whisperx.diarize.DiarizationPipeline(
                use_auth_token=self.huggingface_token,
                device=self.device
            )
            logger.info(f"Diarization model loaded: speakers {self.min_speakers}-{self.max_speakers}")
        except Exception as e:
            logger.warning(f"Failed to load diarization model: {e}")
            self.diarize_model = None
    
    def _unload_diarization_model(self):
        """Unload diarization model to free VRAM"""
        if self.diarize_model:
            logger.info("Unloading diarization model to free VRAM...")
            del self.diarize_model
            self.diarize_model = None
            
            # GPU memory cleanup
            if self.device == 'cuda' and TORCH_AVAILABLE:
                import torch
                torch.cuda.empty_cache()
                gc.collect()
                logger.debug("GPU memory cleared after diarization model unload")
    
    def transcribe_audio(self, audio_path: Path, scene_info: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Transcribe audio file using WhisperX + Silero VAD
        
        Args:
            audio_path: Path to audio.wav file (from FFmpeg)
            scene_info: Optional scene boundary information for context
            
        Returns:
            Dictionary with transcription results with whole-file timestamps
            
        Raises:
            WhisperError: If transcription fails
        """
        logger.info(f"Transcribing audio with VAD: {audio_path.name}")
        
        # Handle development mock mode
        if self.development_config.get('mock_ai_services', False):
            return self._mock_transcription(audio_path, scene_info)
        
        # Validate input
        if not audio_path.exists():
            raise WhisperError(f"Audio file not found: {audio_path}")
        
        if audio_path.stat().st_size < 1024:  # Less than 1KB
            raise WhisperError(f"Audio file too small: {audio_path}")
        
        try:
            # Load models if needed
            self._load_model()
            
            start_time = time.time()
            
            # STAGE 1: VAD-based audio segmentation
            vad_segments = self._perform_vad_segmentation(audio_path)
            logger.info(f"VAD detected {len(vad_segments)} speech segments")
            
            # STAGE 2: Process each VAD segment with WhisperX
            all_transcription_segments = []
            for i, vad_chunk in enumerate(vad_segments):
                logger.debug(f"Processing VAD chunk {i+1}/{len(vad_segments)}")
                chunk_result = self._transcribe_vad_chunk(audio_path, vad_chunk, i)
                if chunk_result:
                    all_transcription_segments.extend(chunk_result)
            
            # STAGE 3: Reconstruct whole-file timestamps and merge segments
            reconstructed_segments = self._reconstruct_whole_file_timestamps(
                all_transcription_segments, vad_segments
            )
            
            # STAGE 4: Apply speaker diarization if available
            if self.enable_diarization and WHISPERX_AVAILABLE:
                # Load diarization model if using sequential loading
                if self.sequential_model_loading and not self.diarize_model:
                    self._load_diarization_model()
                
                if self.diarize_model:
                    reconstructed_segments = self._apply_speaker_diarization(
                        audio_path, reconstructed_segments
                    )
                    
                    # Unload diarization model if using sequential loading
                    if self.sequential_model_loading:
                        self._unload_diarization_model()
            
            processing_time = time.time() - start_time
            
            # Build final result
            processed_result = self._build_final_result(
                reconstructed_segments, processing_time, scene_info, vad_segments
            )
            
            # GPU memory cleanup
            self._cleanup_gpu_memory()
            
            logger.info(f"VAD + WhisperX transcription complete: {processing_time:.2f}s, "
                       f"{len(vad_segments)} VAD chunks -> {len(reconstructed_segments)} final segments")
            
            # Save analysis to intermediate file
            self._save_analysis_to_file(audio_path, processed_result)
            
            return processed_result
            
        except Exception as e:
            # Cleanup on error
            self._cleanup_gpu_memory()
            raise WhisperError(f"VAD + WhisperX transcription failed: {str(e)}") from e
    
    def _perform_vad_segmentation(self, audio_path: Path) -> List[Dict[str, Any]]:
        """
        Perform Voice Activity Detection using Silero VAD
        
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
            
            # Add padding and remove small gaps (following WhisperWithVAD approach)
            for i, timestamp in enumerate(speech_timestamps):
                # Add padding: 0.2s head, 1.3s tail
                timestamp["start"] = max(0, timestamp["start"] - int(0.2 * VAD_SR))
                timestamp["end"] = min(wav.shape[0] - 16, timestamp["end"] + int(1.3 * VAD_SR))
                
                # Remove overlaps
                if i > 0 and timestamp["start"] < speech_timestamps[i - 1]["end"]:
                    timestamp["start"] = speech_timestamps[i - 1]["end"]
            
            # Process each VAD region as its own chunk (simplified approach)
            # No grouping - each VAD region becomes one chunk for individual Whisper processing
            processed_chunks = []
            for region_idx, timestamp in enumerate(speech_timestamps):
                # Each region becomes one chunk
                chunk_timestamps = [timestamp]
                
                # Collect audio data for this region
                chunk_audio = collect_chunks(chunk_timestamps, wav)
                
                # Calculate region timing in whole-file context
                chunk_start_seconds = timestamp["start"] / VAD_SR
                chunk_end_seconds = timestamp["end"] / VAD_SR
                
                # Offset is just the start time (no complex calculation needed)
                offset = chunk_start_seconds
                
                processed_chunks.append({
                    'chunk_id': region_idx,
                    'start_seconds': chunk_start_seconds,
                    'end_seconds': chunk_end_seconds,
                    'duration': chunk_end_seconds - chunk_start_seconds,
                    'audio_data': chunk_audio,
                    'offset': offset,
                    'vad_segments': [
                        {
                            'start': timestamp["start"] / VAD_SR,
                            'end': timestamp["end"] / VAD_SR
                        }
                    ],
                    'sampling_rate': VAD_SR
                })
            
            logger.info(f"VAD segmentation complete: {len(speech_timestamps)} regions -> {len(processed_chunks)} chunks")
            return processed_chunks
            
        except Exception as e:
            logger.error(f"Silero VAD failed: {e}, falling back to time-based segmentation")
            return self._fallback_vad_segmentation(audio_path)
    
    def _fallback_vad_segmentation(self, audio_path: Path) -> List[Dict[str, Any]]:
        """Fallback time-based segmentation when VAD is unavailable"""
        try:
            # Read audio file to get duration
            import librosa
            y, sr = librosa.load(str(audio_path), sr=None)
            duration = len(y) / sr
            
            # Create 30-second chunks as fallback
            chunk_duration = 30.0
            chunks = []
            
            for i in range(0, int(duration), int(chunk_duration)):
                start_seconds = i
                end_seconds = min(i + chunk_duration, duration)
                
                # Extract audio chunk
                start_sample = int(start_seconds * sr)
                end_sample = int(end_seconds * sr)
                chunk_audio = y[start_sample:end_sample]
                
                chunks.append({
                    'chunk_id': i // int(chunk_duration),
                    'start_seconds': start_seconds,
                    'end_seconds': end_seconds,
                    'duration': end_seconds - start_seconds,
                    'audio_data': chunk_audio,
                    'offset': start_seconds,
                    'vad_segments': [{'start': start_seconds, 'end': end_seconds}],
                    'sampling_rate': sr,
                    'fallback': True
                })
            
            logger.info(f"Fallback segmentation: {len(chunks)} time-based chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Fallback segmentation failed: {e}")
            # Final fallback - single segment
            return [{
                'chunk_id': 0,
                'start_seconds': 0,
                'end_seconds': 60,  # Assume 60s
                'duration': 60,
                'audio_data': None,
                'offset': 0,
                'vad_segments': [{'start': 0, 'end': 60}],
                'sampling_rate': 16000,
                'fallback': True,
                'single_segment': True
            }]
    
    def _transcribe_vad_chunk(self, audio_path: Path, vad_chunk: Dict[str, Any], chunk_idx: int) -> List[Dict[str, Any]]:
        """
        Transcribe a single VAD chunk using WhisperX
        
        Returns:
            List of transcription segments with chunk-relative timestamps
        """
        try:
            # Save chunk audio to temporary file for WhisperX
            temp_chunk_path = audio_path.parent / f"temp_chunk_{chunk_idx}.wav"
            
            if vad_chunk.get('audio_data') is not None:
                # Save audio data using soundfile
                import soundfile as sf
                sf.write(temp_chunk_path, vad_chunk['audio_data'], vad_chunk['sampling_rate'])
            else:
                # Fallback - extract chunk from original file
                self._extract_audio_chunk(audio_path, temp_chunk_path, vad_chunk)
            
            # Choose transcription method based on hybrid mode
            if self.use_original_whisper and self.original_whisper_model:
                # Original Whisper transcription (hybrid mode)
                logger.debug("Using original Whisper for transcription")
                transcription_options = {
                    'task': 'transcribe',
                    'word_timestamps': self.enable_word_timestamps,
                    'condition_on_previous_text': False,
                    'temperature': 0,
                    'no_speech_threshold': 0.6,
                }
                
                # Set language if not auto-detect
                if self.language != 'auto':
                    transcription_options['language'] = self.language
                    
                result = self.original_whisper_model.transcribe(str(temp_chunk_path), **transcription_options)
                
            elif WHISPERX_AVAILABLE and hasattr(self.whisperx_model, 'transcribe'):
                # WhisperX transcription
                result = self.whisperx_model.transcribe(str(temp_chunk_path))
                
                # WhisperX alignment for better word timestamps (optional)
                if self.enable_word_alignment and hasattr(whisperx, 'load_align_model'):
                    try:
                        logger.debug("Loading alignment model for word-level timestamps")
                        align_model, metadata = whisperx.load_align_model(
                            language_code=result.get("language", "en"), 
                            device=self.device
                        )
                        result = whisperx.align(result["segments"], align_model, metadata, str(temp_chunk_path), self.device)
                        logger.debug("Word alignment completed successfully")
                    except Exception as e:
                        logger.warning(f"Word alignment failed, using segment-level timestamps: {e}")
                else:
                    logger.debug("Word alignment disabled - using segment-level timestamps only")
                    
            else:
                # Fallback to original Whisper
                transcription_options = {
                    'task': 'transcribe',
                    'word_timestamps': True,
                    'condition_on_previous_text': False,
                    'temperature': 0,
                    'no_speech_threshold': 0.6,
                    'logprob_threshold': -1.0
                }
                
                if self.language != 'auto':
                    transcription_options['language'] = self.language
                
                result = self.whisperx_model.transcribe(str(temp_chunk_path), **transcription_options)
            
            # Clean up temporary file
            try:
                temp_chunk_path.unlink()
            except:
                pass
            
            # Process segments - add chunk context for timestamp reconstruction
            chunk_segments = []
            for segment in result.get('segments', []):
                # Apply hallucination filtering (from WhisperWithVAD)
                if self._is_hallucination(segment):
                    continue
                
                chunk_segments.append({
                    'chunk_id': chunk_idx,
                    'chunk_start': segment.get('start', 0),
                    'chunk_end': segment.get('end', 0),
                    'text': segment.get('text', '').strip(),
                    'confidence': 1.0 - segment.get('no_speech_prob', 0.0),
                    'words': segment.get('words', []),
                    'vad_chunk_info': vad_chunk  # Store chunk info for timestamp reconstruction
                })
            
            return chunk_segments
            
        except Exception as e:
            logger.error(f"Failed to transcribe VAD chunk {chunk_idx}: {e}")
            return []
    
    def _extract_audio_chunk(self, audio_path: Path, output_path: Path, vad_chunk: Dict[str, Any]):
        """Extract audio chunk using ffmpeg when audio_data is not available"""
        try:
            import subprocess
            
            start_time = vad_chunk['start_seconds']
            duration = vad_chunk['duration']
            
            cmd = [
                'ffmpeg', '-y',
                '-i', str(audio_path),
                '-ss', str(start_time),
                '-t', str(duration),
                '-ar', '16000',
                '-ac', '1',
                str(output_path)
            ]
            
            subprocess.run(cmd, capture_output=True, check=True)
            
        except Exception as e:
            logger.error(f"Failed to extract audio chunk: {e}")
            raise
    
    def _is_hallucination(self, segment: Dict[str, Any]) -> bool:
        """
        Filter common Whisper hallucinations (adapted from WhisperWithVAD)
        """
        text = segment.get('text', '').strip().lower()
        
        # Common hallucination patterns
        suppress_patterns = [
            "thank you", "thanks for", "subscribe", "like and",
            "bye", "bye bye", "please sub", "the end",
            "my channel", "the channel", "for watching",
            "thank you for watching", "see you next", "full video"
        ]
        
        # Low confidence or high no-speech probability
        confidence = 1.0 - segment.get('no_speech_prob', 0.0)
        avg_logprob = segment.get('avg_logprob', 0)
        
        if confidence < 0.3 or avg_logprob < -1.0:
            return True
        
        # Check for hallucination patterns
        for pattern in suppress_patterns:
            if pattern in text:
                # Apply additional confidence penalty
                confidence -= 0.35 if len(pattern) > 10 else 0.15
                if confidence < 0.5:
                    return True
        
        return False
    
    def _reconstruct_whole_file_timestamps(self, all_segments: List[Dict[str, Any]], 
                                          vad_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Reconstruct whole-file timestamps from chunk-relative timestamps
        
        This is the critical step that preserves timing context for institutional knowledge
        """
        reconstructed = []
        
        for segment in all_segments:
            chunk_id = segment['chunk_id']
            vad_chunk = vad_chunks[chunk_id]
            
            # Reconstruct whole-file timestamps
            whole_file_start = vad_chunk['start_seconds'] + segment['chunk_start']
            whole_file_end = vad_chunk['start_seconds'] + segment['chunk_end']
            
            reconstructed_segment = {
                'start': whole_file_start,
                'end': whole_file_end,
                'duration': whole_file_end - whole_file_start,
                'text': segment['text'],
                'confidence': segment['confidence'],
                'words': self._reconstruct_word_timestamps(segment['words'], vad_chunk['start_seconds']),
                'speaker': 'Unknown',  # Will be filled by diarization
                'vad_chunk_id': chunk_id,
                'original_chunk_timing': {
                    'chunk_start': segment['chunk_start'],
                    'chunk_end': segment['chunk_end']
                }
            }
            
            reconstructed.append(reconstructed_segment)
        
        # Sort by start time to ensure chronological order
        reconstructed.sort(key=lambda x: x['start'])
        
        logger.debug(f"Timestamp reconstruction: {len(all_segments)} chunk segments -> {len(reconstructed)} whole-file segments")
        return reconstructed
    
    def _reconstruct_word_timestamps(self, words: List[Dict[str, Any]], chunk_offset: float) -> List[Dict[str, Any]]:
        """Reconstruct word-level timestamps to whole-file context"""
        if not words:
            return []
        
        reconstructed_words = []
        for word in words:
            reconstructed_words.append({
                'word': word.get('word', ''),
                'start': chunk_offset + word.get('start', 0),
                'end': chunk_offset + word.get('end', 0),
                'confidence': word.get('confidence', word.get('probability', 1.0))
            })
        
        return reconstructed_words
    
    def _apply_speaker_diarization(self, audio_path: Path, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply speaker diarization using WhisperX diarization pipeline
        """
        if not self.diarize_model:
            logger.warning("Diarization model not available, using fallback speaker assignment")
            return self._fallback_speaker_assignment(segments)
        
        try:
            logger.info("Applying WhisperX speaker diarization...")
            
            # Load audio using WhisperX
            audio = whisperx.load_audio(str(audio_path))
            
            # Run diarization on the audio with speaker constraints
            diarize_segments = self.diarize_model(
                audio,
                min_speakers=self.min_speakers,
                max_speakers=self.max_speakers
            )
            
            # Convert our segments to WhisperX format for speaker assignment
            whisperx_result = {
                'segments': [
                    {
                        'start': seg['start'],
                        'end': seg['end'], 
                        'text': seg['text'],
                        'words': seg.get('words', [])
                    } for seg in segments
                ]
            }
            
            # Apply speaker assignment using WhisperX
            try:
                result_with_speakers = whisperx.assign_word_speakers(diarize_segments, whisperx_result)
                logger.debug("Speaker assignment completed successfully")
            except Exception as e:
                logger.warning(f"Speaker assignment failed, using diarization segments only: {e}")
                # Fallback: manually assign speakers based on diarization segments
                result_with_speakers = self._manual_speaker_assignment(diarize_segments, whisperx_result)
            
            # Update our segments with speaker information
            for i, whisperx_segment in enumerate(result_with_speakers['segments']):
                if i < len(segments):
                    # Extract speaker from WhisperX result
                    speaker = whisperx_segment.get('speaker', 'Unknown')
                    if speaker == 'Unknown' or speaker is None:
                        # Try to get speaker from words if segment speaker is unknown
                        words = whisperx_segment.get('words', [])
                        if words:
                            word_speakers = [w.get('speaker') for w in words if w.get('speaker')]
                            if word_speakers:
                                # Use most common speaker in the segment
                                from collections import Counter
                                speaker = Counter(word_speakers).most_common(1)[0][0]
                    
                    segments[i]['speaker'] = speaker if speaker != 'Unknown' else f'Speaker_{i % 2 + 1}'
                    
                    # Update word-level speaker information if available
                    if 'words' in whisperx_segment:
                        segments[i]['words'] = whisperx_segment['words']
            
            unique_speakers = set(seg['speaker'] for seg in segments if seg['speaker'] != 'Unknown')
            logger.info(f"WhisperX speaker diarization complete: {len(unique_speakers)} speakers detected")
            
        except Exception as e:
            logger.error(f"WhisperX speaker diarization failed: {e}")
            logger.info("Falling back to basic speaker assignment")
            segments = self._fallback_speaker_assignment(segments)
        
        return segments
    
    def _manual_speaker_assignment(self, diarize_segments: Dict, whisperx_result: Dict) -> Dict:
        """
        Manual speaker assignment when WhisperX assign_word_speakers fails
        """
        try:
            # Extract speaker segments from diarization result
            speaker_timeline = []
            for segment in diarize_segments:
                speaker_timeline.append({
                    'start': segment['start'],
                    'end': segment['end'],
                    'speaker': segment['speaker']
                })
            
            # Sort by start time
            speaker_timeline.sort(key=lambda x: x['start'])
            
            # Assign speakers to whisperx segments based on temporal overlap
            result_segments = []
            for whisperx_segment in whisperx_result['segments']:
                seg_start = whisperx_segment['start']
                seg_end = whisperx_segment['end']
                seg_midpoint = (seg_start + seg_end) / 2
                
                # Find best matching speaker segment
                assigned_speaker = 'Unknown'
                best_overlap = 0.0
                
                for speaker_seg in speaker_timeline:
                    # Calculate overlap
                    overlap_start = max(seg_start, speaker_seg['start'])
                    overlap_end = min(seg_end, speaker_seg['end'])
                    overlap = max(0, overlap_end - overlap_start)
                    
                    if overlap > best_overlap:
                        best_overlap = overlap
                        assigned_speaker = speaker_seg['speaker']
                
                # Create segment with speaker assignment
                result_segment = whisperx_segment.copy()
                result_segment['speaker'] = assigned_speaker
                result_segments.append(result_segment)
            
            return {'segments': result_segments}
            
        except Exception as e:
            logger.error(f"Manual speaker assignment failed: {e}")
            # Ultimate fallback - return original without speaker info
            return whisperx_result
    
    def _fallback_speaker_assignment(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Fallback speaker assignment when diarization is not available
        """
        logger.debug("Using fallback speaker assignment based on timing patterns")
        
        for i, segment in enumerate(segments):
            if len(segments) > 1:
                # Simple heuristic: long pauses suggest speaker changes
                if i == 0:
                    speaker = 'Speaker_1'
                elif segment['start'] - segments[i-1]['end'] > 2.0:  # 2 second pause threshold
                    prev_speaker = segments[i-1]['speaker']
                    speaker = 'Speaker_2' if prev_speaker == 'Speaker_1' else 'Speaker_1'
                else:
                    speaker = segments[i-1]['speaker']  # Continue with same speaker
            else:
                speaker = 'Speaker_1'
                
            segments[i]['speaker'] = speaker
        
        unique_speakers = set(seg['speaker'] for seg in segments)
        logger.debug(f"Fallback speaker assignment: {len(unique_speakers)} speakers assigned")
        
        return segments
    
    def _build_final_result(self, segments: List[Dict[str, Any]], processing_time: float, 
                           scene_info: Optional[Dict], vad_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build the final transcription result with comprehensive metadata"""
        
        # Extract unique speakers
        speakers = sorted(list(set(seg['speaker'] for seg in segments if seg['speaker'] != 'Unknown')))
        if not speakers:
            speakers = ['Speaker_1']
        
        # Build full transcript
        full_transcript = ' '.join(seg['text'] for seg in segments if seg['text'].strip())
        
        # Calculate statistics
        total_speech_duration = sum(chunk['duration'] for chunk in vad_chunks)
        total_segments = len(segments)
        
        final_result = {
            'transcript': full_transcript,
            'language': 'auto-detected',  # Would be detected by WhisperX
            'language_probability': 0.95,  # Placeholder
            'segments': segments,
            'speakers': speakers,
            'processing_time': processing_time,
            'model_info': {
                'model_size': self.model_size,
                'device': self.device,
                'whisperx_enabled': WHISPERX_AVAILABLE,
                'silero_vad_enabled': self.silero_model is not None,
                'diarization_enabled': self.diarize_model is not None
            },
            'scene_context': scene_info,
            'vad_analysis': {
                'total_chunks': len(vad_chunks),
                'total_speech_duration': total_speech_duration,
                'vad_threshold': self.vad_threshold,
                'chunk_threshold': self.chunk_threshold
            },
            'quality_metrics': {
                'segments_count': total_segments,
                'average_segment_duration': sum(seg['duration'] for seg in segments) / max(1, total_segments),
                'average_confidence': sum(seg['confidence'] for seg in segments) / max(1, total_segments),
                'speaker_count': len(speakers)
            }
        }
        
        return final_result
    
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
        
        # STAGE 2: Intelligent post-processing segment merging
        original_count = len(segments)
        if len(segments) > 1:
            segments = self._merge_segments_intelligently(segments)
            logger.debug(f"Segment merging: {original_count} -> {len(segments)} segments")
        
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
            'scene_context': scene_info,
            'segment_processing': {
                'original_segments': original_count,
                'merged_segments': len(segments),
                'merge_ratio': f"{original_count}/{len(segments)}" if len(segments) > 0 else "0/0"
            }
        }
        
        return processed_result

    def _merge_segments_intelligently(self, segments: List[Dict]) -> List[Dict]:
        """
        Intelligent segment merging using multiple criteria
        
        Best practice approach:
        - Merge short segments (< 2.5 seconds)
        - Merge segments with small gaps (< 0.8 seconds)
        - Keep incomplete sentences together
        - Preserve semantic coherence
        """
        if len(segments) <= 1:
            return segments
        
        merged = []
        current = segments[0].copy()
        
        for next_seg in segments[1:]:
            # Calculate metrics
            current_duration = current['end'] - current['start']
            gap_duration = next_seg['start'] - current['end']
            current_text = current['text'].strip()
            current_word_count = len(current_text.split())
            
            # Merge criteria (industry best practices)
            should_merge = (
                # Duration too short (less than 2.5 seconds)
                current_duration < 2.5 or
                
                # Gap too small (less than 0.8 seconds) - likely same phrase
                gap_duration < 0.8 or
                
                # Incomplete sentence (no ending punctuation)
                (current_text and not current_text.endswith(('.', '!', '?', ':')) and 
                 not next_seg['text'].strip().startswith(('And', 'But', 'So', 'Now', 'Then', 'However'))) or
                
                # Very short text (less than 4 words) - likely incomplete
                current_word_count < 4 or
                
                # Next segment is very short continuation
                len(next_seg['text'].strip().split()) < 3
            )
            
            if should_merge:
                # Merge the segments
                gap_text = " " if gap_duration < 0.3 else "... "  # Add pause indicator for longer gaps
                current['text'] = current_text + gap_text + next_seg['text'].strip()
                current['end'] = next_seg['end']
                
                # Average confidence (weighted by duration)
                current_weight = current_duration
                next_weight = next_seg['end'] - next_seg['start']
                total_weight = current_weight + next_weight
                
                if total_weight > 0:
                    current['confidence'] = (
                        (current['confidence'] * current_weight + 
                         next_seg['confidence'] * next_weight) / total_weight
                    )
                
                # Merge word timestamps if available
                if current.get('words') and next_seg.get('words'):
                    current['words'].extend(next_seg['words'])
                    
            else:
                # Don't merge - add current segment and move to next
                merged.append(current)
                current = next_seg.copy()
        
        # Add the last segment
        merged.append(current)
        
        # Post-processing: Clean up merged segments
        for segment in merged:
            # Clean up text
            segment['text'] = ' '.join(segment['text'].split())  # Remove extra whitespace
            
            # Recalculate duration
            segment['duration'] = segment['end'] - segment['start']
        
        return merged
    
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
            f"Multiple speakers may be present in this {scene_info.get('duration', 60) if scene_info else 60:.1f} second segment."
        )
        
        mock_result = {
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
        
        # Save mock analysis to intermediate file
        self._save_analysis_to_file(audio_path, mock_result)
        
        return mock_result
    
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
        """Unload models to free memory"""
        if self.whisperx_model is not None:
            del self.whisperx_model
            self.whisperx_model = None
        
        if self.diarize_model is not None:
            del self.diarize_model
            self.diarize_model = None
            
        if self.silero_model is not None:
            del self.silero_model
            self.silero_model = None
            
        self.silero_utils = None
        self.model_loaded = False
        self._cleanup_gpu_memory()
        logger.debug("WhisperX + Silero VAD models unloaded")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded models"""
        return {
            'model_size': self.model_size,
            'device': self.device,
            'language': self.language,
            'model_loaded': self.model_loaded,
            'whisperx_available': WHISPERX_AVAILABLE,
            'silero_vad_available': self.silero_model is not None,
            'diarization_available': self.diarize_model is not None,
            'vad_threshold': self.vad_threshold,
            'chunk_threshold': self.chunk_threshold,
            'development_mode': self.development_config.get('mock_ai_services', False),
            'enhancement_features': {
                'vad_segmentation': True,
                'hallucination_filtering': True,
                'speaker_diarization': self.diarize_model is not None,
                'word_level_timestamps': True,
                'whole_file_timestamp_reconstruction': True
            }
        }
    
    def _save_analysis_to_file(self, audio_path: Path, analysis_result: Dict[str, Any]):
        """Save Whisper analysis results to intermediate JSON file"""
        try:
            # Determine build directory from audio path
            # audio_path format: build/[video_name]/audio.wav
            build_dir = audio_path.parent
            analysis_dir = build_dir / "audio_analysis"
            analysis_dir.mkdir(exist_ok=True)
            
            # Create timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = analysis_dir / "whisper_transcription.json"
            
            # Add metadata to analysis result
            analysis_with_metadata = {
                **analysis_result,
                'analysis_timestamp': timestamp,
                'input_file': str(audio_path),
                'service_version': 'whisperx_silero_vad_v1.0.0',
                'processing_pipeline': {
                    'vad_stage': 'silero_vad' if self.silero_model else 'fallback_segmentation',
                    'transcription_stage': 'whisperx' if WHISPERX_AVAILABLE else 'openai_whisper',
                    'diarization_stage': 'whisperx_diarization' if self.diarize_model else 'basic_speaker_assignment',
                    'timestamp_reconstruction': 'whole_file_context_preserved'
                }
            }
            
            # Save to JSON file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(analysis_with_metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Whisper analysis saved to: {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save Whisper analysis to file: {e}")
            # Don't raise - file saving is supplementary to main processing


# Export for easy importing
__all__ = ['WhisperService', 'WhisperError']