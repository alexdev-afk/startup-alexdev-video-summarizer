"""
LibROSA Music Analysis Service

Handles comprehensive music feature extraction using LibROSA.
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
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

from utils.logger import get_logger

logger = get_logger(__name__)


class LibROSAError(Exception):
    """LibROSA processing error"""
    pass


class LibROSAService:
    """LibROSA music analysis service for audio processing"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LibROSA service
        
        Args:
            config: Configuration dictionary with LibROSA settings
        """
        self.config = config
        self.librosa_config = config.get('cpu_pipeline', {}).get('librosa', {})
        self.development_config = config.get('development', {})
        
        # Audio processing configuration
        self.sample_rate = self.librosa_config.get('sample_rate', 22050)
        self.hop_length = self.librosa_config.get('hop_length', 512)
        self.n_fft = self.librosa_config.get('n_fft', 2048)
        self.n_mels = self.librosa_config.get('n_mels', 128)
        self.n_mfcc = self.librosa_config.get('n_mfcc', 13)
        
        # Analysis settings
        self.tempo_detection = self.librosa_config.get('tempo_detection', True)
        self.harmonic_analysis = self.librosa_config.get('harmonic_analysis', True)
        self.spectral_features = self.librosa_config.get('spectral_features', True)
        
        logger.info(f"LibROSA service initialized - sample_rate: {self.sample_rate}, available: {LIBROSA_AVAILABLE}")
    
    def analyze_audio_segment(self, audio_path: str, scene_info: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Analyze audio segment with comprehensive feature extraction
        
        Args:
            audio_path: Path to audio file (FFmpeg-prepared WAV)
            scene_info: Scene boundary information for context preservation
            
        Returns:
            Comprehensive music analysis results
        """
        start_time = time.time()
        
        try:
            # Extract audio segment for scene if boundaries provided
            if scene_info and 'start_seconds' in scene_info and 'end_seconds' in scene_info:
                audio_data = self._extract_scene_audio(audio_path, scene_info)
            else:
                audio_data = self._load_full_audio(audio_path)
            
            if audio_data is None:
                return self._fallback_analysis_result(scene_info)
            
            # Comprehensive music analysis
            results = {
                'tempo_analysis': self._analyze_tempo(audio_data),
                'spectral_features': self._extract_spectral_features(audio_data),
                'harmonic_features': self._extract_harmonic_features(audio_data),
                'rhythmic_features': self._extract_rhythmic_features(audio_data),
                'audio_classification': self._classify_audio_content(audio_data),
                'processing_time': time.time() - start_time,
                'scene_context': scene_info,
                'sample_rate': self.sample_rate
            }
            
            logger.debug(f"LibROSA analysis complete - tempo: {results['tempo_analysis'].get('tempo', 'N/A')}")
            
            # Save analysis to intermediate file
            self._save_analysis_to_file(audio_path, results)
            
            return results
            
        except Exception as e:
            logger.error(f"LibROSA analysis failed: {e}")
            fallback_result = self._fallback_analysis_result(scene_info, error=str(e))
            
            # Save fallback analysis to intermediate file
            self._save_analysis_to_file(audio_path, fallback_result)
            
            return fallback_result
    
    def _extract_scene_audio(self, audio_path: str, scene_info: Dict) -> Optional[np.ndarray]:
        """Extract specific scene audio segment"""
        if not LIBROSA_AVAILABLE:
            return None
            
        try:
            # Load full audio file
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Extract scene segment
            start_frame = int(scene_info['start_seconds'] * sr)
            end_frame = int(scene_info['end_seconds'] * sr) if scene_info.get('end_seconds') else len(y)
            
            scene_audio = y[start_frame:end_frame]
            logger.debug(f"Extracted scene audio: {len(scene_audio)} samples ({len(scene_audio)/sr:.2f}s)")
            
            return scene_audio
            
        except Exception as e:
            logger.error(f"Scene audio extraction failed: {e}")
            return None
    
    def _load_full_audio(self, audio_path: str) -> Optional[np.ndarray]:
        """Load full audio file"""
        if not LIBROSA_AVAILABLE:
            return None
            
        try:
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            logger.debug(f"Loaded audio: {len(y)} samples ({len(y)/sr:.2f}s)")
            return y
        except Exception as e:
            logger.error(f"Audio loading failed: {e}")
            return None
    
    def _analyze_tempo(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Extract tempo and rhythmic information"""
        if not self.tempo_detection or not LIBROSA_AVAILABLE:
            return {'tempo': 120.0, 'tempo_confidence': 0.0, 'beat_track': [], 'mock_mode': True}
        
        try:
            # Tempo estimation
            tempo, beats = librosa.beat.beat_track(y=audio_data, sr=self.sample_rate)
            
            # Onset detection for rhythm analysis
            onset_frames = librosa.onset.onset_detect(y=audio_data, sr=self.sample_rate)
            onset_times = librosa.frames_to_time(onset_frames, sr=self.sample_rate)
            
            return {
                'tempo': float(tempo),
                'beat_count': len(beats),
                'onset_count': len(onset_times),
                'rhythm_regularity': self._calculate_rhythm_regularity(onset_times),
                'tempo_confidence': 0.85  # LibROSA doesn't provide confidence, use default
            }
            
        except Exception as e:
            logger.warning(f"Tempo analysis failed: {e}")
            return {'tempo': 120.0, 'tempo_confidence': 0.0, 'error': str(e)}
    
    def _extract_spectral_features(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Extract spectral features for audio characterization"""
        if not self.spectral_features or not LIBROSA_AVAILABLE:
            return {'spectral_centroid': 2500.0, 'mock_mode': True}
        
        try:
            # Core spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=self.sample_rate)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)[0]
            
            # MFCC features
            mfccs = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=self.n_mfcc)
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=audio_data, sr=self.sample_rate)
            
            return {
                'spectral_centroid_mean': float(np.mean(spectral_centroid)),
                'spectral_centroid_std': float(np.std(spectral_centroid)),
                'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
                'zero_crossing_rate_mean': float(np.mean(zero_crossing_rate)),
                'mfcc_mean': [float(np.mean(mfcc)) for mfcc in mfccs],
                'mfcc_std': [float(np.std(mfcc)) for mfcc in mfccs],
                'chroma_mean': [float(np.mean(chroma[i])) for i in range(chroma.shape[0])],
                'spectral_bandwidth': float(np.mean(librosa.feature.spectral_bandwidth(y=audio_data, sr=self.sample_rate))),
                'spectral_flatness': float(np.mean(librosa.feature.spectral_flatness(y=audio_data)))
            }
            
        except Exception as e:
            logger.warning(f"Spectral feature extraction failed: {e}")
            return {'spectral_centroid_mean': 2500.0, 'error': str(e)}
    
    def _extract_harmonic_features(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Extract harmonic and timbral features"""
        if not self.harmonic_analysis or not LIBROSA_AVAILABLE:
            return {'harmonic_energy': 0.5, 'mock_mode': True}
        
        try:
            # Harmonic-percussive separation
            harmonic, percussive = librosa.effects.hpss(audio_data)
            
            # Tonal centroid features
            tonnetz = librosa.feature.tonnetz(y=harmonic, sr=self.sample_rate)
            
            # Energy analysis
            harmonic_energy = np.sum(harmonic ** 2)
            percussive_energy = np.sum(percussive ** 2)
            total_energy = harmonic_energy + percussive_energy
            
            return {
                'harmonic_energy_ratio': float(harmonic_energy / total_energy) if total_energy > 0 else 0.5,
                'percussive_energy_ratio': float(percussive_energy / total_energy) if total_energy > 0 else 0.5,
                'tonnetz_mean': [float(np.mean(tonnetz[i])) for i in range(tonnetz.shape[0])],
                'harmonic_complexity': float(np.std(tonnetz)),
                'timbral_richness': float(np.mean(np.abs(harmonic)))
            }
            
        except Exception as e:
            logger.warning(f"Harmonic feature extraction failed: {e}")
            return {'harmonic_energy_ratio': 0.5, 'error': str(e)}
    
    def _extract_rhythmic_features(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Extract rhythmic and temporal features"""
        if not LIBROSA_AVAILABLE:
            return {'rhythm_strength': 0.5, 'mock_mode': True}
        
        try:
            # RMS energy for dynamics
            rms_energy = librosa.feature.rms(y=audio_data)[0]
            
            # Tempo stability analysis
            tempogram = librosa.feature.tempogram(y=audio_data, sr=self.sample_rate)
            tempo_stability = float(np.std(tempogram))
            
            return {
                'rms_energy_mean': float(np.mean(rms_energy)),
                'rms_energy_std': float(np.std(rms_energy)),
                'dynamic_range': float(np.max(rms_energy) - np.min(rms_energy)),
                'tempo_stability': tempo_stability,
                'rhythm_strength': float(1.0 / (1.0 + tempo_stability))  # Inverse relationship
            }
            
        except Exception as e:
            logger.warning(f"Rhythmic feature extraction failed: {e}")
            return {'rhythm_strength': 0.5, 'error': str(e)}
    
    def _classify_audio_content(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Classify audio content type and characteristics"""
        try:
            # Simple heuristic-based classification
            # This could be enhanced with ML models in the future
            
            # Energy-based classification
            rms_energy = np.mean(librosa.feature.rms(y=audio_data)[0]) if LIBROSA_AVAILABLE else 0.3
            
            # Spectral characteristics
            if LIBROSA_AVAILABLE:
                spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate)[0])
                zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio_data)[0])
            else:
                spectral_centroid = 2500.0
                zero_crossing_rate = 0.15
            
            # Classification logic
            if spectral_centroid > 3000 and zero_crossing_rate > 0.2:
                audio_type = 'speech'
                mood = 'neutral'
            elif spectral_centroid < 2000 and rms_energy > 0.1:
                audio_type = 'music'
                mood = 'energetic' if rms_energy > 0.3 else 'calm'
            else:
                audio_type = 'mixed'
                mood = 'neutral'
            
            # Quality assessment
            quality_score = min(10.0, max(1.0, 10.0 * rms_energy))
            noise_level = 'low' if rms_energy > 0.05 else 'high'
            
            return {
                'type': audio_type,
                'mood': mood,
                'energy_level': 'high' if rms_energy > 0.3 else 'medium' if rms_energy > 0.1 else 'low',
                'quality_score': float(quality_score),
                'noise_level': noise_level,
                'speech_probability': 0.8 if audio_type == 'speech' else 0.3,
                'music_probability': 0.8 if audio_type == 'music' else 0.2
            }
            
        except Exception as e:
            logger.warning(f"Audio classification failed: {e}")
            return {'type': 'unknown', 'mood': 'neutral', 'error': str(e)}
    
    def _calculate_rhythm_regularity(self, onset_times: np.ndarray) -> float:
        """Calculate rhythm regularity score from onset times"""
        if len(onset_times) < 3:
            return 0.0
        
        # Calculate inter-onset intervals
        intervals = np.diff(onset_times)
        
        # Regularity is inverse of interval variance
        if len(intervals) > 1:
            regularity = 1.0 / (1.0 + np.std(intervals))
        else:
            regularity = 0.5
        
        return float(regularity)
    
    def _fallback_analysis_result(self, scene_info: Optional[Dict] = None, error: Optional[str] = None) -> Dict[str, Any]:
        """Fallback result when LibROSA processing fails"""
        scene_id = scene_info.get('scene_id', 0) if scene_info else 0
        
        return {
            'tempo_analysis': {
                'tempo': 120.0 + (scene_id * 5),
                'tempo_confidence': 0.0,
                'beat_count': 0
            },
            'spectral_features': {
                'spectral_centroid_mean': 2500.0,
                'zero_crossing_rate_mean': 0.15,
                'mfcc_mean': [12.5, -8.2, 4.1, -2.3, 1.8, 0.9, -1.1, 0.3, -0.8, 0.5, -0.2, 0.1, -0.3]
            },
            'harmonic_features': {
                'harmonic_energy_ratio': 0.6,
                'percussive_energy_ratio': 0.4
            },
            'rhythmic_features': {
                'rms_energy_mean': 0.3,
                'rhythm_strength': 0.5
            },
            'audio_classification': {
                'type': 'speech' if scene_id % 3 != 0 else 'background_music',
                'mood': 'neutral',
                'energy_level': 'medium',
                'quality_score': 7.5,
                'noise_level': 'low'
            },
            'processing_time': 0.0,
            'scene_context': scene_info,
            'sample_rate': self.sample_rate,
            'fallback_mode': True,
            'error': error or 'LibROSA processing unavailable'
        }
    
    def cleanup(self):
        """Cleanup resources"""
        # Force garbage collection for memory cleanup
        gc.collect()
        logger.debug("LibROSA service cleanup complete")
    
    def _save_analysis_to_file(self, audio_path: str, analysis_result: Dict[str, Any]):
        """Save LibROSA analysis results to intermediate JSON file"""
        try:
            # Determine build directory from audio path
            # audio_path format: build/[video_name]/audio.wav
            audio_pathlib = Path(audio_path)
            build_dir = audio_pathlib.parent
            analysis_dir = build_dir / "audio_analysis"
            analysis_dir.mkdir(exist_ok=True)
            
            # Create timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = analysis_dir / "librosa_music_analysis.json"
            
            # Convert numpy arrays to lists for JSON serialization
            serializable_result = self._make_json_serializable(analysis_result)
            
            # Add metadata to analysis result
            analysis_with_metadata = {
                **serializable_result,
                'analysis_timestamp': timestamp,
                'input_file': str(audio_path),
                'service_version': 'librosa_v1.0.0'
            }
            
            # Save to JSON file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(analysis_with_metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"LibROSA analysis saved to: {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save LibROSA analysis to file: {e}")
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