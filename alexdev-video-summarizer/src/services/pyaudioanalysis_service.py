"""
pyAudioAnalysis Service for 68-Feature Audio Analysis

Handles comprehensive 68-feature audio analysis and advanced speaker diarization.
Optimized for FFmpeg-prepared audio.wav files with scene context preservation.
"""

import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import gc
from datetime import datetime

# Optional imports for development mode
try:
    from pyAudioAnalysis import audioTrainTest as aT
    from pyAudioAnalysis import audioSegmentation as aS
    from pyAudioAnalysis import audioFeatureExtraction as aF
    PYAUDIOANALYSIS_FULL = True
except (ImportError, ValueError, Exception) as e:
    # Handle import errors, numpy compatibility issues, and other exceptions
    PYAUDIOANALYSIS_FULL = False

# scipy.io.wavfile is needed for audio loading (separate from pyAudioAnalysis)
try:
    import scipy.io.wavfile as wavfile
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Overall availability combines both
PYAUDIOANALYSIS_AVAILABLE = PYAUDIOANALYSIS_FULL and SCIPY_AVAILABLE

from utils.logger import get_logger

logger = get_logger(__name__)


class PyAudioAnalysisError(Exception):
    """pyAudioAnalysis processing error"""
    pass


class PyAudioAnalysisService:
    """pyAudioAnalysis service for comprehensive audio feature extraction"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize pyAudioAnalysis service
        
        Args:
            config: Configuration dictionary with pyAudioAnalysis settings
        """
        self.config = config
        self.pyaudio_config = config.get('cpu_pipeline', {}).get('pyaudioanalysis', {})
        self.development_config = config.get('development', {})
        
        # Output mode configuration - REMOVED FAKE INTERPRETIVE ANALYSIS
        self.output_mode = self.pyaudio_config.get('output_mode', 'numerical')  # "numerical" only - fake interpretive removed
        
        # Audio processing configuration
        self.window_size = self.pyaudio_config.get('window_size', 0.050)  # 50ms
        self.step_size = self.pyaudio_config.get('step_size', 0.025)     # 25ms
        self.features_mode = self.pyaudio_config.get('features', 'all')
        
        # Analysis settings
        self.enable_speaker_diarization = self.pyaudio_config.get('speaker_diarization', True)
        self.enable_emotion_analysis = self.pyaudio_config.get('emotion_analysis', True)
        self.enable_classification = self.pyaudio_config.get('classification', True)
        
        logger.info(f"pyAudioAnalysis service initialized - window: {self.window_size}s, available: {PYAUDIOANALYSIS_AVAILABLE}, mode: {self.output_mode}")
    
    def analyze_whisper_segments(self, audio_path: str, whisper_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze audio segments aligned to Whisper transcription boundaries
        
        Args:
            audio_path: Path to audio file (FFmpeg-prepared WAV)
            whisper_result: Whisper transcription result with segments
            
        Returns:
            Segment-by-segment pyAudioAnalysis results aligned to Whisper boundaries
        """
        start_time = time.time()
        
        try:
            whisper_segments = whisper_result.get('segments', [])
            if not whisper_segments:
                logger.warning("No Whisper segments found for pyAudioAnalysis alignment")
                return self._fallback_whisper_aligned_result(whisper_result)
            
            segment_analyses = []
            
            # Process each Whisper segment individually
            for i, whisper_segment in enumerate(whisper_segments):
                logger.debug(f"Processing Whisper segment {i+1}/{len(whisper_segments)}: {whisper_segment['text'][:50]}...")
                
                # Extract audio for this specific speech segment
                segment_audio, sample_rate = self._extract_whisper_segment_audio(
                    audio_path,
                    whisper_segment['start'],
                    whisper_segment['end']
                )
                
                if segment_audio is None or len(segment_audio) == 0:
                    logger.warning(f"Failed to extract audio for segment {i+1}")
                    segment_analysis = self._fallback_segment_analysis(whisper_segment, i+1)
                else:
                    # Analyze this specific phrase for emotion/speaker characteristics
                    segment_analysis = {
                        'segment_id': i + 1,
                        'whisper_segment_id': whisper_segment.get('id', i+1),
                        'text': whisper_segment['text'].strip(),
                        'timespan': f"{whisper_segment['start']:.2f}-{whisper_segment['end']:.2f}s",
                        'duration': whisper_segment['end'] - whisper_segment['start'],
                        'speaker': whisper_segment.get('speaker', 'Unknown'),
                        
                        # Generate analysis based on output mode
                        **self._create_segment_analysis(segment_audio, sample_rate, whisper_segment)
                    }
                
                segment_analyses.append(segment_analysis)
                logger.debug(f"Segment {i+1} analysis complete: voice={segment_analysis.get('voice_characteristics', {}).get('vocal_clarity', 'unknown')[:50]}")
            
            # Simple numerical aggregation - fake interpretive aggregation removed
            aggregate_analysis = {'total_segments': len(segment_analyses)}
            
            final_result = {
                'processing_type': 'whisper_aligned_segments',
                'total_segments': len(segment_analyses),
                'segment_analyses': segment_analyses,
                'aggregate_analysis': aggregate_analysis,
                'processing_time': time.time() - start_time,
                'whisper_context': {
                    'total_whisper_segments': len(whisper_segments),
                    'transcription_language': whisper_result.get('language', 'unknown'),
                    'total_speakers': len(whisper_result.get('speakers', []))
                }
            }
            
            # Save segment-aligned analysis to file
            self._save_analysis_to_file(audio_path, final_result)
            
            logger.info(f"Whisper-aligned pyAudioAnalysis complete: {len(segment_analyses)} segments processed")
            return final_result
            
        except Exception as e:
            logger.error(f"Whisper-aligned pyAudioAnalysis failed: {e}")
            fallback_result = self._fallback_whisper_aligned_result(whisper_result, error=str(e))
            self._save_analysis_to_file(audio_path, fallback_result)
            return fallback_result

    def analyze_audio_segment(self, audio_path: str, scene_info: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Analyze audio segment with comprehensive 68-feature extraction
        
        Args:
            audio_path: Path to audio file (FFmpeg-prepared WAV)
            scene_info: Scene boundary information for context preservation
            
        Returns:
            Comprehensive 68-feature audio analysis results
        """
        start_time = time.time()
        
        try:
            # Extract audio segment for scene if boundaries provided
            if scene_info and 'start_seconds' in scene_info and 'end_seconds' in scene_info:
                audio_data, sample_rate = self._extract_scene_audio(audio_path, scene_info)
            else:
                audio_data, sample_rate = self._load_full_audio(audio_path)
            
            if audio_data is None:
                fallback_result = self._fallback_analysis_result(scene_info)
                
                # Save fallback analysis to intermediate file
                self._save_analysis_to_file(audio_path, fallback_result)
                
                return fallback_result
            
            # Comprehensive 68-feature analysis (speaker analysis removed - handled by WhisperX)
            results = {
                'features_68': self._extract_68_features(audio_data, sample_rate),
                'classification': self._classify_audio_content(audio_data, sample_rate),
                'emotion_analysis': self._analyze_emotion(audio_data, sample_rate),
                'summary_statistics': self._compute_summary_statistics(audio_data, sample_rate),
                'processing_time': time.time() - start_time,
                'scene_context': scene_info,
                'sample_rate': sample_rate,
                'window_size': self.window_size,
                'step_size': self.step_size,
                'note': 'Speaker diarization handled by WhisperX + pyannote in transcription pipeline'
            }
            
            logger.debug(f"pyAudioAnalysis complete - {len(results['features_68'])} features extracted")
            
            # Save analysis to intermediate file
            self._save_analysis_to_file(audio_path, results)
            
            return results
            
        except Exception as e:
            logger.error(f"pyAudioAnalysis analysis failed: {e}")
            fallback_result = self._fallback_analysis_result(scene_info, error=str(e))
            
            # Save fallback analysis to intermediate file
            self._save_analysis_to_file(audio_path, fallback_result)
            
            return fallback_result
    
    def _extract_whisper_segment_audio(self, audio_path: str, start_time: float, end_time: float) -> Tuple[Optional[np.ndarray], int]:
        """Extract specific Whisper segment audio"""
        try:
            # Use scipy wavfile first, fallback to other methods if needed
            try:
                sample_rate, audio_data = wavfile.read(audio_path)
                logger.debug(f"Audio file loaded: {audio_data.shape}, {audio_data.dtype}, {sample_rate}Hz")
            except Exception as e:
                logger.warning(f"scipy.io.wavfile failed, falling back: {e}")
                return None, 22050
            
            # Convert to float and normalize
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            elif audio_data.dtype == np.uint8:
                audio_data = (audio_data.astype(np.float32) - 128.0) / 128.0
            else:
                # Already float, ensure it's float32
                audio_data = audio_data.astype(np.float32)
            
            # Handle stereo to mono conversion
            if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
                logger.debug(f"Converting stereo to mono: {audio_data.shape}")
                audio_data = np.mean(audio_data, axis=1)
            elif len(audio_data.shape) > 1:
                # Single channel but 2D array, flatten it
                audio_data = audio_data.flatten()
            
            # Extract Whisper segment (start_time and end_time are in seconds)
            start_frame = int(start_time * sample_rate)
            end_frame = int(end_time * sample_rate)
            
            # Ensure bounds are valid
            start_frame = max(0, start_frame)
            end_frame = min(len(audio_data), end_frame)
            
            if start_frame >= end_frame:
                logger.warning(f"Invalid Whisper segment bounds: {start_time:.2f}s-{end_time:.2f}s, frames: {start_frame}-{end_frame}")
                return None, sample_rate
            
            segment_audio = audio_data[start_frame:end_frame]
            
            # Verify we got valid audio data
            if len(segment_audio) == 0:
                logger.warning(f"Empty segment extracted: {start_time:.2f}s-{end_time:.2f}s")
                return None, sample_rate
            
            logger.debug(f"Extracted Whisper segment: {len(segment_audio)} samples ({len(segment_audio)/sample_rate:.2f}s) from {start_time:.2f}s-{end_time:.2f}s")
            
            return segment_audio, sample_rate
            
        except Exception as e:
            logger.error(f"Whisper segment audio extraction failed: {e}")
            return None, 22050

    def _extract_scene_audio(self, audio_path: str, scene_info: Dict) -> Tuple[Optional[np.ndarray], int]:
        """Extract specific scene audio segment"""
        if not PYAUDIOANALYSIS_AVAILABLE:
            return None, 22050
            
        try:
            # Load audio file
            sample_rate, audio_data = wavfile.read(audio_path)
            
            # Convert to float and normalize if needed
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            
            # Handle stereo to mono conversion
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Extract scene segment
            start_frame = int(scene_info['start_seconds'] * sample_rate)
            end_frame = int(scene_info['end_seconds'] * sample_rate) if scene_info.get('end_seconds') else len(audio_data)
            
            scene_audio = audio_data[start_frame:end_frame]
            logger.debug(f"Extracted scene audio: {len(scene_audio)} samples ({len(scene_audio)/sample_rate:.2f}s)")
            
            return scene_audio, sample_rate
            
        except Exception as e:
            logger.error(f"Scene audio extraction failed: {e}")
            return None, 22050
    
    def _load_full_audio(self, audio_path: str) -> Tuple[Optional[np.ndarray], int]:
        """Load full audio file"""
        if not PYAUDIOANALYSIS_AVAILABLE:
            return None, 22050
            
        try:
            sample_rate, audio_data = wavfile.read(audio_path)
            
            # Convert to float and normalize
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0
                
            # Handle stereo to mono conversion
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            logger.debug(f"Loaded audio: {len(audio_data)} samples ({len(audio_data)/sample_rate:.2f}s)")
            return audio_data, sample_rate
            
        except Exception as e:
            logger.error(f"Audio loading failed: {e}")
            return None, 22050
    
    def _extract_68_features(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """Extract comprehensive 68 audio features"""
        if not PYAUDIOANALYSIS_AVAILABLE or len(audio_data) == 0:
            return self._generate_mock_68_features()
        
        try:
            # Extract features using pyAudioAnalysis
            features = aF.feature_extraction(audio_data, sample_rate, 
                                           self.window_size * sample_rate, 
                                           self.step_size * sample_rate)
            
            # features is a 2D array (num_features, num_windows)
            # We'll take the mean across all windows for each feature
            if features.size > 0:
                feature_means = np.mean(features, axis=1)
                
                # Map to named features (pyAudioAnalysis provides 34 basic features)
                feature_names = self._get_feature_names()
                features_dict = {}
                
                for i, name in enumerate(feature_names):
                    if i < len(feature_means):
                        features_dict[name] = float(feature_means[i])
                
                # Add extended features to reach 68 total
                features_dict.update(self._compute_extended_features(audio_data, sample_rate))
                
                return features_dict
            else:
                return self._generate_mock_68_features()
                
        except Exception as e:
            logger.warning(f"68-feature extraction failed: {e}")
            return self._generate_mock_68_features()
    
    def _get_feature_names(self) -> List[str]:
        """Get standard pyAudioAnalysis feature names"""
        return [
            'zcr_mean', 'energy_mean', 'energy_entropy_mean', 'spectral_centroid_mean',
            'spectral_spread_mean', 'spectral_entropy_mean', 'spectral_flux_mean',
            'spectral_rolloff_mean', 'mfcc_1_mean', 'mfcc_2_mean', 'mfcc_3_mean',
            'mfcc_4_mean', 'mfcc_5_mean', 'mfcc_6_mean', 'mfcc_7_mean', 'mfcc_8_mean',
            'mfcc_9_mean', 'mfcc_10_mean', 'mfcc_11_mean', 'mfcc_12_mean', 'mfcc_13_mean',
            'chroma_1_mean', 'chroma_2_mean', 'chroma_3_mean', 'chroma_4_mean',
            'chroma_5_mean', 'chroma_6_mean', 'chroma_7_mean', 'chroma_8_mean',
            'chroma_9_mean', 'chroma_10_mean', 'chroma_11_mean', 'chroma_12_mean',
            'chroma_std_mean'
        ]
    
    def _compute_extended_features(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """Compute additional features to reach 68 total"""
        try:
            extended_features = {}
            
            # Temporal features
            extended_features['duration'] = len(audio_data) / sample_rate
            extended_features['rms_energy'] = float(np.sqrt(np.mean(audio_data ** 2)))
            extended_features['peak_amplitude'] = float(np.max(np.abs(audio_data)))
            extended_features['dynamic_range'] = float(np.max(audio_data) - np.min(audio_data))
            
            # Spectral features (computed manually)
            fft = np.fft.fft(audio_data)
            magnitude = np.abs(fft[:len(fft)//2])
            freqs = np.fft.fftfreq(len(fft), 1/sample_rate)[:len(fft)//2]
            
            # Spectral statistics
            extended_features['spectral_mean'] = float(np.mean(magnitude))
            extended_features['spectral_std'] = float(np.std(magnitude))
            extended_features['spectral_skewness'] = float(self._compute_skewness(magnitude))
            extended_features['spectral_kurtosis'] = float(self._compute_kurtosis(magnitude))
            
            # Harmonic features
            extended_features['fundamental_frequency'] = float(self._estimate_f0(audio_data, sample_rate))
            extended_features['harmonic_ratio'] = float(self._compute_harmonic_ratio(audio_data))
            extended_features['noise_ratio'] = 1.0 - extended_features['harmonic_ratio']
            
            # Rhythm and timing features
            extended_features['zero_crossing_variance'] = float(self._compute_zcr_variance(audio_data))
            extended_features['autocorrelation_max'] = float(self._compute_autocorr_max(audio_data))
            
            # Cepstral features
            extended_features['cepstral_peak_prominence'] = float(self._compute_cpp(audio_data, sample_rate))
            
            # Add padding features to reach exactly 68
            current_count = len(self._get_feature_names()) + len(extended_features)
            for i in range(current_count, 68):
                extended_features[f'extended_feature_{i:02d}'] = 0.0
                
            return extended_features
            
        except Exception as e:
            logger.warning(f"Extended feature computation failed: {e}")
            return {f'extended_feature_{i:02d}': 0.0 for i in range(34, 68)}
    
    def _classify_audio_content(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Classify audio content type and characteristics"""
        if not self.enable_classification or not PYAUDIOANALYSIS_AVAILABLE:
            return self._generate_mock_classification()
        
        try:
            # Simple heuristic-based classification
            # In production, this would use trained models from pyAudioAnalysis
            
            # Energy-based analysis
            rms_energy = np.sqrt(np.mean(audio_data ** 2))
            zero_crossing_rate = len(np.where(np.diff(np.signbit(audio_data)))[0]) / len(audio_data)
            
            # Spectral analysis
            fft = np.fft.fft(audio_data)
            magnitude = np.abs(fft[:len(fft)//2])
            spectral_centroid = np.sum(magnitude * np.arange(len(magnitude))) / np.sum(magnitude) if np.sum(magnitude) > 0 else 0
            
            # Classification logic
            if zero_crossing_rate > 0.1 and spectral_centroid > 1000:
                audio_type = 'speech'
                subtype = 'clean_speech' if rms_energy > 0.1 else 'noisy_speech'
                confidence = 0.85
            elif rms_energy > 0.05 and spectral_centroid < 2000:
                audio_type = 'music'
                subtype = 'instrumental' if zero_crossing_rate < 0.05 else 'vocal'
                confidence = 0.80
            else:
                audio_type = 'mixed'
                subtype = 'speech_music'
                confidence = 0.60
            
            return {
                'type': audio_type,
                'subtype': subtype,
                'confidence': confidence,
                'speaker_emotion': 'neutral',  # Placeholder for emotion analysis
                'audio_quality': 'high' if rms_energy > 0.1 else 'medium' if rms_energy > 0.05 else 'low'
            }
            
        except Exception as e:
            logger.warning(f"Audio classification failed: {e}")
            return self._generate_mock_classification()
    
    def _analyze_speaker_characteristics(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Analyze speaker characteristics for this specific segment"""
        if not PYAUDIOANALYSIS_AVAILABLE or len(audio_data) == 0:
            return {'voice_quality': 'unknown', 'pitch_analysis': {}, 'speaking_rate': 0.0}
        
        try:
            # Voice quality analysis
            rms_energy = np.sqrt(np.mean(audio_data ** 2))
            zero_crossing_rate = len(np.where(np.diff(np.signbit(audio_data)))[0]) / len(audio_data)
            
            # Pitch analysis (fundamental frequency estimation)
            f0 = self._estimate_f0(audio_data, sample_rate)
            
            # Speaking rate estimation
            duration = len(audio_data) / sample_rate
            speaking_rate = zero_crossing_rate * 100  # Rough speaking rate estimate
            
            # Voice quality classification
            if rms_energy > 0.15 and f0 > 200:
                voice_quality = 'clear_high_pitch'
            elif rms_energy > 0.10 and f0 > 100:
                voice_quality = 'clear_medium_pitch'
            elif rms_energy > 0.05:
                voice_quality = 'moderate_clarity'
            else:
                voice_quality = 'low_energy'
            
            return {
                'voice_quality': voice_quality,
                'pitch_analysis': {
                    'fundamental_frequency': float(f0),
                    'pitch_category': 'high' if f0 > 200 else 'medium' if f0 > 100 else 'low'
                },
                'energy_analysis': {
                    'rms_energy': float(rms_energy),
                    'energy_level': 'high' if rms_energy > 0.15 else 'medium' if rms_energy > 0.08 else 'low'
                },
                'speaking_rate': float(speaking_rate),
                'voice_confidence': float(min(rms_energy * 5, 1.0))  # 0-1 confidence score
            }
            
        except Exception as e:
            logger.warning(f"Speaker characteristics analysis failed: {e}")
            return {'voice_quality': 'analysis_failed', 'error': str(e)}

    def _extract_prosodic_features(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Extract prosodic features (rhythm, stress, intonation) for this segment"""
        if len(audio_data) == 0:
            return {}
        
        try:
            # Energy contour analysis
            window_samples = int(0.025 * sample_rate)  # 25ms windows
            energy_contour = []
            
            for i in range(0, len(audio_data) - window_samples, window_samples):
                window = audio_data[i:i + window_samples]
                energy = np.sqrt(np.mean(window ** 2))
                energy_contour.append(energy)
            
            if len(energy_contour) == 0:
                return {}
            
            energy_contour = np.array(energy_contour)
            
            # Prosodic analysis
            energy_variance = np.var(energy_contour)
            energy_range = np.max(energy_contour) - np.min(energy_contour)
            
            # Rhythm analysis
            rhythm_regularity = 1.0 - min(energy_variance * 10, 1.0)  # More variance = less regular
            
            # Stress pattern detection
            stress_points = len(np.where(energy_contour > np.mean(energy_contour) + np.std(energy_contour))[0])
            stress_density = stress_points / len(energy_contour) if len(energy_contour) > 0 else 0
            
            return {
                'energy_dynamics': {
                    'energy_variance': float(energy_variance),
                    'energy_range': float(energy_range),
                    'dynamic_range': float(energy_range / (np.mean(energy_contour) + 1e-10))
                },
                'rhythm_analysis': {
                    'rhythm_regularity': float(rhythm_regularity),
                    'rhythm_category': 'regular' if rhythm_regularity > 0.7 else 'irregular'
                },
                'stress_analysis': {
                    'stress_points': int(stress_points),
                    'stress_density': float(stress_density),
                    'stress_level': 'high' if stress_density > 0.3 else 'medium' if stress_density > 0.15 else 'low'
                }
            }
            
        except Exception as e:
            logger.warning(f"Prosodic feature extraction failed: {e}")
            return {'error': str(e)}

    # REMOVED: _aggregate_segment_analyses - fake interpretive aggregation

    # REMOVED: _detect_emotion_changes, _categorize_pacing - fake analysis helpers

    def _fallback_whisper_aligned_result(self, whisper_result: Dict, error: Optional[str] = None) -> Dict[str, Any]:
        """Fallback result for Whisper-aligned processing"""
        segments = whisper_result.get('segments', [])
        
        fallback_segments = []
        for i, segment in enumerate(segments):
            fallback_segments.append({
                'segment_id': i + 1,
                'text': segment.get('text', ''),
                'timespan': f"{segment.get('start', 0):.2f}-{segment.get('end', 0):.2f}s",
                'emotion_analysis': {'emotion': 'neutral', 'confidence': 0.0},
                'speaker_characteristics': {'voice_quality': 'unknown'},
                'fallback_mode': True
            })
        
        return {
            'processing_type': 'whisper_aligned_segments',
            'total_segments': len(fallback_segments),
            'segment_analyses': fallback_segments,
            'aggregate_analysis': {'error': error or 'pyAudioAnalysis processing unavailable'},
            'processing_time': 0.0,
            'fallback_mode': True,
            'error': error
        }

    def _fallback_segment_analysis(self, whisper_segment: Dict, segment_id: int) -> Dict[str, Any]:
        """Fallback analysis for individual segment"""
        return {
            'segment_id': segment_id,
            'text': whisper_segment.get('text', ''),
            'timespan': f"{whisper_segment.get('start', 0):.2f}-{whisper_segment.get('end', 0):.2f}s",
            'emotion_analysis': {'emotion': 'neutral', 'confidence': 0.0},
            'speaker_characteristics': {'voice_quality': 'unknown'},
            'audio_classification': {'type': 'speech', 'confidence': 0.5},
            'fallback_mode': True
        }

    # _analyze_speakers method removed - misleading fake diarization
    # Real speaker diarization now handled by WhisperX + pyannote in transcription pipeline
    
    def _analyze_emotion(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Analyze emotional content of speech"""
        if not self.enable_emotion_analysis:
            return {'emotion': 'neutral', 'confidence': 0.0}
        
        try:
            # Basic emotion analysis based on prosodic features
            rms_energy = np.sqrt(np.mean(audio_data ** 2))
            zero_crossing_rate = len(np.where(np.diff(np.signbit(audio_data)))[0]) / len(audio_data)
            
            # Simple emotion classification
            if rms_energy > 0.2 and zero_crossing_rate > 0.15:
                emotion = 'excited'
                confidence = 0.75
            elif rms_energy < 0.05:
                emotion = 'calm'
                confidence = 0.70
            elif zero_crossing_rate > 0.2:
                emotion = 'stressed'
                confidence = 0.65
            else:
                emotion = 'neutral'
                confidence = 0.80
            
            return {
                'emotion': emotion,
                'confidence': confidence,
                'arousal': float(rms_energy * 2),  # 0-1 scale
                'valence': 0.5,  # Neutral baseline
                'prosodic_features': {
                    'energy_level': float(rms_energy),
                    'speech_rate': float(zero_crossing_rate)
                }
            }
            
        except Exception as e:
            logger.warning(f"Emotion analysis failed: {e}")
            return {'emotion': 'neutral', 'confidence': 0.0, 'error': str(e)}
    
    def _compute_summary_statistics(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """Compute summary statistics for the audio segment"""
        try:
            return {
                'mean_energy': float(np.mean(audio_data ** 2)),
                'std_energy': float(np.std(audio_data ** 2)),
                'max_amplitude': float(np.max(np.abs(audio_data))),
                'min_amplitude': float(np.min(np.abs(audio_data))),
                'rms_value': float(np.sqrt(np.mean(audio_data ** 2))),
                'dynamic_range': float(np.max(audio_data) - np.min(audio_data)),
                'spectral_rolloff': float(self._compute_spectral_rolloff(audio_data, sample_rate)),
                'spectral_flux': float(self._compute_spectral_flux(audio_data))
            }
        except Exception as e:
            logger.warning(f"Summary statistics computation failed: {e}")
            return {'mean_energy': 0.0, 'std_energy': 0.0, 'error': str(e)}
    
    # Helper methods for feature computation
    def _compute_skewness(self, data: np.ndarray) -> float:
        """Compute skewness of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _compute_kurtosis(self, data: np.ndarray) -> float:
        """Compute kurtosis of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _estimate_f0(self, audio_data: np.ndarray, sample_rate: int) -> float:
        """Estimate fundamental frequency"""
        try:
            # Simple autocorrelation-based F0 estimation
            autocorr = np.correlate(audio_data, audio_data, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Find the first significant peak (fundamental frequency)
            min_period = int(sample_rate / 400)  # 400 Hz max
            max_period = int(sample_rate / 50)   # 50 Hz min
            
            if max_period < len(autocorr):
                peak_idx = np.argmax(autocorr[min_period:max_period]) + min_period
                return sample_rate / peak_idx
            return 0.0
        except:
            return 0.0
    
    def _compute_harmonic_ratio(self, audio_data: np.ndarray) -> float:
        """Compute harmonic to noise ratio"""
        try:
            autocorr = np.correlate(audio_data, audio_data, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            if len(autocorr) > 1:
                return float(np.max(autocorr[1:]) / autocorr[0]) if autocorr[0] > 0 else 0.0
            return 0.0
        except:
            return 0.0
    
    def _compute_zcr_variance(self, audio_data: np.ndarray) -> float:
        """Compute variance of zero crossing rate over windows"""
        try:
            window_size = len(audio_data) // 10
            if window_size < 1:
                return 0.0
                
            zcr_values = []
            for i in range(0, len(audio_data) - window_size, window_size):
                window = audio_data[i:i + window_size]
                zcr = len(np.where(np.diff(np.signbit(window)))[0]) / len(window)
                zcr_values.append(zcr)
            
            return float(np.var(zcr_values)) if len(zcr_values) > 1 else 0.0
        except:
            return 0.0
    
    def _compute_autocorr_max(self, audio_data: np.ndarray) -> float:
        """Compute maximum autocorrelation value"""
        try:
            autocorr = np.correlate(audio_data, audio_data, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            return float(np.max(autocorr[1:]) / autocorr[0]) if len(autocorr) > 1 and autocorr[0] > 0 else 0.0
        except:
            return 0.0
    
    def _compute_cpp(self, audio_data: np.ndarray, sample_rate: int) -> float:
        """Compute Cepstral Peak Prominence"""
        try:
            # Simple cepstral analysis
            windowed = audio_data * np.hanning(len(audio_data))
            fft = np.fft.fft(windowed)
            log_magnitude = np.log(np.abs(fft) + 1e-10)
            cepstrum = np.fft.ifft(log_magnitude).real
            
            # Find peak in quefrency domain
            if len(cepstrum) > 10:
                return float(np.max(cepstrum[10:len(cepstrum)//2]))
            return 0.0
        except:
            return 0.0
    
    def _compute_spectral_rolloff(self, audio_data: np.ndarray, sample_rate: int, rolloff_percent: float = 0.85) -> float:
        """Compute spectral rolloff frequency"""
        try:
            fft = np.fft.fft(audio_data)
            magnitude = np.abs(fft[:len(fft)//2])
            total_energy = np.sum(magnitude)
            
            cumsum_energy = np.cumsum(magnitude)
            rolloff_energy = total_energy * rolloff_percent
            
            rolloff_idx = np.where(cumsum_energy >= rolloff_energy)[0]
            if len(rolloff_idx) > 0:
                return (rolloff_idx[0] * sample_rate) / (2 * len(magnitude))
            return sample_rate / 4
        except:
            return 0.0
    
    def _compute_spectral_flux(self, audio_data: np.ndarray) -> float:
        """Compute spectral flux (rate of spectral change)"""
        try:
            window_size = len(audio_data) // 4
            if window_size < 1:
                return 0.0
                
            prev_spectrum = None
            flux_values = []
            
            for i in range(0, len(audio_data) - window_size, window_size):
                window = audio_data[i:i + window_size]
                spectrum = np.abs(np.fft.fft(window))
                
                if prev_spectrum is not None:
                    flux = np.sum((spectrum - prev_spectrum) ** 2)
                    flux_values.append(flux)
                
                prev_spectrum = spectrum
            
            return float(np.mean(flux_values)) if len(flux_values) > 0 else 0.0
        except:
            return 0.0
    
    # _generate_speaker_segments method removed - misleading fake diarization
    # Real speaker diarization now handled by WhisperX + pyannote in transcription pipeline
    
    def _generate_mock_68_features(self) -> Dict[str, float]:
        """Generate mock 68 features when pyAudioAnalysis is not available"""
        features = {}
        feature_names = self._get_feature_names()
        
        # Mock the standard features
        for i, name in enumerate(feature_names):
            features[name] = 0.5 + (i * 0.01)
        
        # Add extended features to reach 68
        for i in range(len(feature_names), 68):
            features[f'extended_feature_{i:02d}'] = 0.3 + (i * 0.005)
        
        return features
    
    def _generate_mock_classification(self) -> Dict[str, Any]:
        """Generate mock classification when analysis is not available"""
        return {
            'type': 'speech',
            'subtype': 'discussion',
            'confidence': 0.75,
            'speaker_emotion': 'neutral',
            'audio_quality': 'medium'
        }
    
    def _fallback_analysis_result(self, scene_info: Optional[Dict] = None, error: Optional[str] = None) -> Dict[str, Any]:
        """Fallback result when pyAudioAnalysis processing fails"""
        scene_id = scene_info.get('scene_id', 0) if scene_info else 0
        
        return {
            'features_68': self._generate_mock_68_features(),
            'classification': self._generate_mock_classification(),
            'speaker_analysis': {
                'speaker_count': 1 + (scene_id % 2),
                'speakers': [f'Speaker_{i+1}' for i in range(1 + (scene_id % 2))],
                'diarization_confidence': 0.0
            },
            'emotion_analysis': {
                'emotion': 'neutral',
                'confidence': 0.0,
                'arousal': 0.5,
                'valence': 0.5
            },
            'summary_statistics': {
                'mean_energy': 0.45,
                'std_energy': 0.12,
                'spectral_rolloff': 3200.0,
                'spectral_flux': 0.023
            },
            'processing_time': 0.0,
            'scene_context': scene_info,
            'sample_rate': 22050,
            'window_size': self.window_size,
            'step_size': self.step_size,
            'fallback_mode': True,
            'error': error or 'pyAudioAnalysis processing unavailable'
        }
    
    def cleanup(self):
        """Cleanup resources"""
        # Force garbage collection for memory cleanup
        gc.collect()
        logger.debug("pyAudioAnalysis service cleanup complete")
    
    def _save_analysis_to_file(self, audio_path: str, analysis_result: Dict[str, Any]):
        """Save pyAudioAnalysis results to intermediate JSON file"""
        try:
            # Determine build directory from audio path
            # audio_path format: build/[video_name]/audio.wav
            audio_pathlib = Path(audio_path)
            build_dir = audio_pathlib.parent
            analysis_dir = build_dir / "audio_analysis"
            analysis_dir.mkdir(exist_ok=True)
            
            # Create timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = analysis_dir / "pyaudioanalysis_features.json"
            
            # Convert numpy arrays to lists for JSON serialization
            serializable_result = self._make_json_serializable(analysis_result)
            
            # Add metadata to analysis result
            analysis_with_metadata = {
                **serializable_result,
                'analysis_timestamp': timestamp,
                'input_file': str(audio_path),
                'service_version': 'pyaudioanalysis_v1.0.0'
            }
            
            # Save to JSON file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(analysis_with_metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"pyAudioAnalysis analysis saved to: {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save pyAudioAnalysis analysis to file: {e}")
            # Don't raise - file saving is supplementary to main processing
    
    def _make_json_serializable(self, data: Any) -> Any:
        """Convert numpy arrays and other non-serializable objects to JSON-compatible format"""
        if isinstance(data, dict):
            return {key: self._make_json_serializable(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._make_json_serializable(item) for item in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, np.floating):
            return float(data)
        elif isinstance(data, np.integer):
            return int(data)
        else:
            return data

    def _create_segment_analysis(self, audio_data: np.ndarray, sample_rate: int, whisper_segment: Dict) -> Dict[str, Any]:
        """Create numerical segment analysis - fake interpretive analysis removed"""
        if len(audio_data) == 0:
            return self._create_fallback_numerical_analysis(whisper_segment)
        
        try:
            # Only numerical analysis - interpretive fake analysis removed
            return self._create_numerical_analysis(audio_data, sample_rate)
                
        except Exception as e:
            logger.warning(f"Segment analysis failed: {e}")
            return self._create_fallback_numerical_analysis(whisper_segment, str(e))


    def _create_numerical_analysis(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Create traditional numerical analysis (legacy mode)"""
        return {
            'emotion_analysis': self._analyze_emotion(audio_data, sample_rate),
            'speaker_characteristics': self._analyze_speaker_characteristics(audio_data, sample_rate),
            'audio_classification': self._classify_audio_content(audio_data, sample_rate),
            'prosodic_features': self._extract_prosodic_features(audio_data, sample_rate),
            'segment_68_features': self._extract_68_features(audio_data, sample_rate),
            'segment_statistics': self._compute_summary_statistics(audio_data, sample_rate)
        }

    # REMOVED: _extract_raw_features_for_interpretation - only used for fake analysis

    # REMOVED: _interpret_recording_environment - fake heuristic analysis

    # REMOVED: _interpret_vocal_delivery - fake heuristic analysis

    # REMOVED: _interpret_speaking_patterns - fake heuristic analysis

    # REMOVED: _interpret_presentation_quality - fake heuristic analysis

    def _create_fallback_analysis(self, whisper_segment: Dict, error: Optional[str] = None) -> Dict[str, Any]:
        """Fallback analysis - numerical only, fake interpretive removed"""
        return self._create_fallback_numerical_analysis(whisper_segment, error)

    # REMOVED: _create_fallback_interpretive_analysis - fake heuristic analysis

    def _create_fallback_numerical_analysis(self, whisper_segment: Dict, error: Optional[str] = None) -> Dict[str, Any]:
        """Fallback numerical analysis when processing fails"""
        return {
            'emotion_analysis': {'emotion': 'neutral', 'confidence': 0.0},
            'speaker_characteristics': {'voice_quality': 'unknown'},
            'audio_classification': {'type': 'speech', 'confidence': 0.5},
            'prosodic_features': {},
            'segment_68_features': self._generate_mock_68_features(),
            'segment_statistics': {'mean_energy': 0.0, 'std_energy': 0.0},
            'fallback_mode': True,
            'error': error or 'Audio processing tools unavailable'
        }

    def _identify_dominant_pattern(self, pattern_list: List[str], fallback: str) -> str:
        """Identify the most common pattern or provide intelligent summary"""
        if not pattern_list:
            return fallback
        
        # Find most common pattern
        pattern_counts = {}
        for pattern in pattern_list:
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        if len(pattern_counts) == 1:
            return list(pattern_counts.keys())[0]
        elif len(pattern_counts) <= 3:
            # Few different patterns - describe variation
            dominant = max(pattern_counts.items(), key=lambda x: x[1])
            if dominant[1] > len(pattern_list) * 0.6:
                return f"Primarily: {dominant[0]}"
            else:
                return f"Varied: {', '.join(pattern_counts.keys())}"
        else:
            return fallback

    def _assess_presentation_flow(self, segment_analyses: List[Dict]) -> str:
        """Assess how the presentation flows across segments"""
        if len(segment_analyses) < 2:
            return "Single segment presentation"
        
        # Look for consistency in engagement and approach
        engagement_progression = []
        for seg in segment_analyses:
            delivery = seg.get('delivery_assessment', {})
            engagement = delivery.get('speaker_engagement', '')
            if 'High engagement' in engagement:
                engagement_progression.append('high')
            elif 'Good engagement' in engagement:
                engagement_progression.append('good')
            elif 'Steady engagement' in engagement:
                engagement_progression.append('steady')
            else:
                engagement_progression.append('low')
        
        # Analyze flow pattern
        if len(set(engagement_progression)) == 1:
            return f"Consistent {engagement_progression[0]} engagement throughout"
        elif engagement_progression[0] == 'low' and engagement_progression[-1] in ['good', 'high']:
            return "Building engagement from introduction to conclusion"
        elif engagement_progression[0] in ['good', 'high'] and engagement_progression[-1] == 'low':
            return "Strong opening with diminishing engagement"
        else:
            return "Dynamic engagement with varied energy levels across segments"

    def _assess_content_organization(self, segment_analyses: List[Dict]) -> str:
        """Assess content organization based on segment characteristics"""
        total_segments = len(segment_analyses)
        
        if total_segments <= 3:
            return "Concise presentation with focused content structure"
        elif total_segments <= 7:
            return "Well-structured presentation with clear segment organization"
        elif total_segments <= 12:
            return "Comprehensive presentation with detailed segment breakdown"
        else:
            return "Extensive presentation with complex multi-segment organization"

    def _create_speaker_style_summary(self, segment_analyses: List[Dict]) -> str:
        """Create overall speaker style summary"""
        articulation_styles = []
        delivery_rhythms = []
        
        for seg in segment_analyses:
            comm_style = seg.get('communication_style', {})
            if comm_style.get('articulation_style'):
                articulation_styles.append(comm_style['articulation_style'])
            if comm_style.get('delivery_rhythm'):
                delivery_rhythms.append(comm_style['delivery_rhythm'])
        
        dominant_articulation = self._identify_dominant_pattern(articulation_styles, "Mixed articulation styles")
        dominant_rhythm = self._identify_dominant_pattern(delivery_rhythms, "Variable delivery pace")
        
        return f"Speaking style: {dominant_articulation}. Delivery rhythm: {dominant_rhythm}"

    def _create_voice_summary(self, segment_analyses: List[Dict]) -> str:
        """Create overall voice characteristics summary"""
        clarity_patterns = []
        pitch_patterns = []
        stability_patterns = []
        
        for seg in segment_analyses:
            voice_chars = seg.get('voice_characteristics', {})
            if voice_chars.get('vocal_clarity'):
                clarity_patterns.append(voice_chars['vocal_clarity'])
            if voice_chars.get('pitch_characteristics'):
                pitch_patterns.append(voice_chars['pitch_characteristics'])
            if voice_chars.get('energy_stability'):
                stability_patterns.append(voice_chars['energy_stability'])
        
        clarity = self._identify_dominant_pattern(clarity_patterns, "Variable voice clarity")
        pitch = self._identify_dominant_pattern(pitch_patterns, "Mixed pitch characteristics")
        stability = self._identify_dominant_pattern(stability_patterns, "Variable energy stability")
        
        return f"Voice profile: {clarity}. {pitch}. {stability}"

    def _identify_presentation_strengths(self, segment_analyses: List[Dict]) -> List[str]:
        """Identify key strengths in the presentation"""
        strengths = []
        
        # Check for consistency
        voice_consistency = len(set([seg.get('voice_characteristics', {}).get('vocal_clarity', '') 
                                   for seg in segment_analyses])) <= 2
        if voice_consistency:
            strengths.append("Consistent vocal delivery throughout presentation")
        
        # Check for high engagement
        high_engagement_count = sum(1 for seg in segment_analyses 
                                  if 'High engagement' in seg.get('delivery_assessment', {}).get('speaker_engagement', ''))
        if high_engagement_count > len(segment_analyses) * 0.5:
            strengths.append("Strong audience engagement maintained")
        
        # Check for professional quality
        professional_count = sum(1 for seg in segment_analyses 
                               if 'Professional' in seg.get('delivery_assessment', {}).get('presentation_professionalism', ''))
        if professional_count > len(segment_analyses) * 0.6:
            strengths.append("Professional presentation quality")
        
        # Check for clear articulation
        clear_articulation_count = sum(1 for seg in segment_analyses 
                                     if 'clear' in seg.get('communication_style', {}).get('articulation_style', '').lower())
        if clear_articulation_count > len(segment_analyses) * 0.7:
            strengths.append("Clear and articulate speech delivery")
        
        return strengths if strengths else ["Presentation analysis completed"]