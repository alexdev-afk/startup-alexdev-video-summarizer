"""
LibROSA Timeline Service

Converts LibROSA music analysis into event-based timeline format for advertisement audio analysis.
Focuses on musical transitions, tempo changes, and structural events rather than duration spans.
"""

import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import gc
from datetime import datetime

# Optional imports for development mode
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

from utils.logger import get_logger
from utils.timeline_schema import ServiceTimeline, TimelineEvent, TimelineSpan

logger = get_logger(__name__)


class LibROSATimelineError(Exception):
    """LibROSA timeline processing error"""
    pass


class LibROSATimelineService:
    """LibROSA timeline service for event-based music analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LibROSA timeline service
        
        Args:
            config: Configuration dictionary with LibROSA settings
        """
        self.config = config
        self.librosa_config = config.get('cpu_pipeline', {}).get('librosa', {})
        
        # Audio processing configuration
        self.sample_rate = self.librosa_config.get('sample_rate', 22050)
        self.hop_length = self.librosa_config.get('hop_length', 512)
        self.n_fft = self.librosa_config.get('n_fft', 2048)
        
        # Event detection thresholds
        self.tempo_change_threshold = self.librosa_config.get('tempo_change_threshold', 5.0)  # BPM
        self.key_change_confidence = self.librosa_config.get('key_change_confidence', 0.15)
        self.structural_change_threshold = self.librosa_config.get('structural_change_threshold', 0.3)
        
        # Analysis window settings for event detection
        self.analysis_window = self.librosa_config.get('analysis_window', 5.0)  # seconds
        self.overlap_window = self.librosa_config.get('overlap_window', 2.5)  # seconds
        
        logger.info(f"LibROSA timeline service initialized - sample_rate: {self.sample_rate}, event_detection: enabled, available: {LIBROSA_AVAILABLE}")
    
    def generate_timeline(self, audio_path: str) -> ServiceTimeline:
        """
        Generate event-based timeline from audio using LibROSA's real capabilities
        
        Args:
            audio_path: Path to audio file (FFmpeg-prepared WAV)
            
        Returns:
            ServiceTimeline with music events and transitions
        """
        start_time = time.time()
        
        try:
            # Load audio data
            audio_data = self._load_audio(audio_path)
            if audio_data is None:
                return self._create_fallback_timeline(audio_path)
            
            # Create timeline object
            total_duration = len(audio_data) / self.sample_rate
            timeline = ServiceTimeline(
                source="librosa",
                audio_file=str(audio_path),
                total_duration=total_duration
            )
            
            # Generate music events and spans using real LibROSA analysis
            self._detect_music_events(audio_data, timeline)
            self._detect_structural_spans(audio_data, timeline)
            
            processing_time = time.time() - start_time
            logger.info(f"LibROSA timeline generated: {len(timeline.events)} events, {len(timeline.spans)} spans in {processing_time:.2f}s")
            
            # Save timeline to file
            self._save_timeline(timeline, audio_path)
            
            return timeline
            
        except Exception as e:
            logger.error(f"LibROSA timeline generation failed: {e}")
            return self._create_fallback_timeline(audio_path, error=str(e))
    
    def _detect_music_events(self, audio_data: np.ndarray, timeline: ServiceTimeline):
        """
        Detect musical events using real LibROSA capabilities
        Focus on transitions and moments of change
        """
        if not LIBROSA_AVAILABLE:
            self._add_mock_music_events(timeline)
            return
        
        try:
            # 1. Tempo transitions using beat tracking
            self._detect_tempo_events(audio_data, timeline)
            
            # 2. Musical onsets and rhythm changes
            self._detect_onset_events(audio_data, timeline)
            
            # 3. Harmonic/key transitions using chroma analysis
            self._detect_harmonic_events(audio_data, timeline)
            
            # 4. Energy/volume transitions
            self._detect_energy_events(audio_data, timeline)
            
        except Exception as e:
            logger.warning(f"Music event detection failed: {e}, using fallback events")
            self._add_mock_music_events(timeline)
    
    def _detect_tempo_events(self, audio_data: np.ndarray, timeline: ServiceTimeline):
        """Detect tempo changes as timeline events"""
        try:
            # Analyze tempo over sliding windows
            window_samples = int(self.analysis_window * self.sample_rate)
            step_samples = int(self.overlap_window * self.sample_rate)
            
            tempos = []
            timestamps = []
            
            for start_idx in range(0, len(audio_data) - window_samples, step_samples):
                end_idx = start_idx + window_samples
                window_audio = audio_data[start_idx:end_idx]
                
                # Real tempo estimation using LibROSA
                tempo, _ = librosa.beat.beat_track(y=window_audio, sr=self.sample_rate)
                
                tempos.append(tempo)
                timestamps.append(start_idx / self.sample_rate)
            
            # Detect significant tempo changes
            for i in range(1, len(tempos)):
                tempo_change = abs(tempos[i] - tempos[i-1])
                
                if tempo_change > self.tempo_change_threshold:
                    # Create tempo change event
                    event = TimelineEvent(
                        timestamp=timestamps[i],
                        description=f"Tempo change: {tempos[i-1]:.1f} -> {tempos[i]:.1f} BPM",
                        category="transition",
                        source="librosa",
                        confidence=min(0.95, 0.6 + (tempo_change / 20.0)),
                        details={
                            "previous_tempo": float(tempos[i-1]),
                            "new_tempo": float(tempos[i]),
                            "tempo_change": float(tempo_change),
                            "analysis_type": "tempo_transition"
                        }
                    )
                    timeline.add_event(event)
            
            logger.debug(f"Detected {len([e for e in timeline.events if 'tempo_change' in e.details.get('analysis_type', '')])} tempo change events")
            
        except Exception as e:
            logger.warning(f"Tempo event detection failed: {e}")
    
    def _detect_onset_events(self, audio_data: np.ndarray, timeline: ServiceTimeline):
        """Detect significant musical onsets as events"""
        try:
            # Real onset detection using LibROSA
            onset_frames = librosa.onset.onset_detect(
                y=audio_data, 
                sr=self.sample_rate,
                units='time'
            )
            
            # Filter for significant onsets (not every small beat)
            if len(onset_frames) > 50:  # Too many onsets - filter for strongest
                onset_strength = librosa.onset.onset_strength(y=audio_data, sr=self.sample_rate)
                onset_strength_times = librosa.frames_to_time(
                    np.arange(len(onset_strength)), 
                    sr=self.sample_rate
                )
                
                # Keep only strongest onsets (top 20%)
                strength_threshold = np.percentile(onset_strength, 80)
                strong_onsets = []
                
                for onset_time in onset_frames:
                    # Find closest strength measurement
                    closest_idx = np.argmin(np.abs(onset_strength_times - onset_time))
                    if onset_strength[closest_idx] > strength_threshold:
                        strong_onsets.append(onset_time)
                
                onset_frames = strong_onsets[:15]  # Limit to 15 strongest
            
            # Create onset events
            for i, onset_time in enumerate(onset_frames):
                if i < 15:  # Limit events to avoid timeline clutter
                    # Determine onset character based on surrounding context
                    onset_description = self._characterize_onset(audio_data, onset_time)
                    
                    event = TimelineEvent(
                        timestamp=float(onset_time),
                        description=onset_description,
                        category="music",
                        source="librosa", 
                        confidence=0.75,
                        details={
                            "analysis_type": "musical_onset",
                            "onset_index": i,
                            "total_onsets": len(onset_frames)
                        }
                    )
                    timeline.add_event(event)
            
            logger.debug(f"Detected {len(onset_frames)} significant musical onset events")
            
        except Exception as e:
            logger.warning(f"Onset event detection failed: {e}")
    
    def _detect_harmonic_events(self, audio_data: np.ndarray, timeline: ServiceTimeline):
        """Detect harmonic/key changes as events"""
        try:
            # Real chroma analysis for harmonic content
            chroma = librosa.feature.chroma_cqt(y=audio_data, sr=self.sample_rate)
            
            # Analyze chroma over time windows for key changes
            window_frames = int(self.analysis_window * self.sample_rate / self.hop_length)
            step_frames = int(self.overlap_window * self.sample_rate / self.hop_length)
            
            key_profiles = []
            timestamps = []
            
            for start_frame in range(0, chroma.shape[1] - window_frames, step_frames):
                end_frame = start_frame + window_frames
                window_chroma = chroma[:, start_frame:end_frame]
                
                # Get dominant pitch class (simplified key detection)
                chroma_mean = np.mean(window_chroma, axis=1)
                dominant_pitch = np.argmax(chroma_mean)
                
                key_profiles.append(dominant_pitch)
                timestamps.append(start_frame * self.hop_length / self.sample_rate)
            
            # Detect key changes
            for i in range(1, len(key_profiles)):
                if key_profiles[i] != key_profiles[i-1]:
                    # Map pitch class to key name (simplified)
                    pitch_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                    old_key = pitch_names[key_profiles[i-1]]
                    new_key = pitch_names[key_profiles[i]]
                    
                    event = TimelineEvent(
                        timestamp=timestamps[i],
                        description=f"Harmonic shift: {old_key} -> {new_key} tonality",
                        category="transition",
                        source="librosa",
                        confidence=0.65,  # Lower confidence for simplified key detection
                        details={
                            "previous_key": old_key,
                            "new_key": new_key,
                            "pitch_class_change": int(key_profiles[i] - key_profiles[i-1]),
                            "analysis_type": "harmonic_transition"
                        }
                    )
                    timeline.add_event(event)
            
            logger.debug(f"Detected {len([e for e in timeline.events if 'harmonic' in e.details.get('analysis_type', '')])} harmonic transition events")
            
        except Exception as e:
            logger.warning(f"Harmonic event detection failed: {e}")
    
    def _detect_energy_events(self, audio_data: np.ndarray, timeline: ServiceTimeline):
        """Detect energy/volume transitions as events"""
        try:
            # Real RMS energy analysis
            rms_energy = librosa.feature.rms(y=audio_data, frame_length=self.n_fft, hop_length=self.hop_length)[0]
            rms_times = librosa.frames_to_time(np.arange(len(rms_energy)), sr=self.sample_rate, hop_length=self.hop_length)
            
            # Smooth RMS for trend detection
            from scipy.signal import savgol_filter
            if len(rms_energy) > 51:
                rms_smooth = savgol_filter(rms_energy, 51, 3)
            else:
                rms_smooth = rms_energy
            
            # Detect significant energy changes
            energy_threshold = np.std(rms_smooth) * 1.5  # 1.5 standard deviations
            
            for i in range(1, len(rms_smooth)):
                energy_change = abs(rms_smooth[i] - rms_smooth[i-1])
                
                if energy_change > energy_threshold:
                    # Determine if increase or decrease
                    if rms_smooth[i] > rms_smooth[i-1]:
                        description = f"Energy increase: building intensity"
                        transition_type = "energy_increase"
                    else:
                        description = f"Energy decrease: calming dynamics"
                        transition_type = "energy_decrease"
                    
                    event = TimelineEvent(
                        timestamp=float(rms_times[i]),
                        description=description,
                        category="transition",
                        source="librosa",
                        confidence=0.7,
                        details={
                            "previous_energy": float(rms_smooth[i-1]),
                            "new_energy": float(rms_smooth[i]),
                            "energy_change": float(energy_change),
                            "transition_type": transition_type,
                            "analysis_type": "energy_transition"
                        }
                    )
                    timeline.add_event(event)
            
            logger.debug(f"Detected {len([e for e in timeline.events if 'energy' in e.details.get('analysis_type', '')])} energy transition events")
            
        except Exception as e:
            logger.warning(f"Energy event detection failed: {e}")
    
    def _detect_structural_spans(self, audio_data: np.ndarray, timeline: ServiceTimeline):
        """
        Detect structural music spans using real LibROSA segmentation
        Focus on coherent musical sections rather than arbitrary time spans
        """
        if not LIBROSA_AVAILABLE:
            self._add_mock_structural_spans(timeline)
            return
        
        try:
            # Real structural segmentation using LibROSA
            # This finds boundaries between different musical sections
            chroma = librosa.feature.chroma_cqt(y=audio_data, sr=self.sample_rate)
            
            # Use recurrence matrix for structure detection
            R = librosa.segment.recurrence_matrix(
                chroma, 
                mode='affinity',
                metric='cosine'
            )
            
            # Detect segment boundaries
            boundaries = librosa.segment.agglomerative(
                chroma, 
                k=None,  # Let algorithm determine number of segments
                clusterer=None
            )
            
            # Convert frame boundaries to time boundaries
            boundary_times = librosa.frames_to_time(boundaries, sr=self.sample_rate, hop_length=self.hop_length)
            
            # Create structural spans between boundaries
            for i in range(len(boundary_times) - 1):
                start_time = float(boundary_times[i])
                end_time = float(boundary_times[i + 1])
                
                # Analyze this segment's characteristics
                segment_description = self._characterize_segment(
                    audio_data, start_time, end_time
                )
                
                span = TimelineSpan(
                    start=start_time,
                    end=end_time,
                    description=segment_description,
                    category="music",
                    source="librosa",
                    confidence=0.8,
                    details={
                        "segment_index": i,
                        "total_segments": len(boundary_times) - 1,
                        "analysis_type": "structural_segment"
                    }
                )
                timeline.add_span(span)
            
            logger.debug(f"Detected {len(boundary_times) - 1} structural music segments")
            
        except Exception as e:
            logger.warning(f"Structural segmentation failed: {e}, using fallback spans")
            self._add_mock_structural_spans(timeline)
    
    def _characterize_onset(self, audio_data: np.ndarray, onset_time: float) -> str:
        """Characterize a musical onset based on surrounding context"""
        try:
            # Analyze a small window around the onset
            onset_sample = int(onset_time * self.sample_rate)
            window_size = int(0.5 * self.sample_rate)  # 0.5 second window
            
            start_idx = max(0, onset_sample - window_size // 2)
            end_idx = min(len(audio_data), onset_sample + window_size // 2)
            window_audio = audio_data[start_idx:end_idx]
            
            # Analyze spectral characteristics around onset
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=window_audio, sr=self.sample_rate))
            rms_energy = np.mean(librosa.feature.rms(y=window_audio))
            
            # Characterize based on features
            if spectral_centroid > 3000:
                if rms_energy > 0.3:
                    return "Sharp percussive hit with bright timbre"
                else:
                    return "Subtle high-frequency accent"
            elif spectral_centroid > 1500:
                if rms_energy > 0.2:
                    return "Strong musical accent with warm timbre"
                else:
                    return "Gentle melodic emphasis"
            else:
                if rms_energy > 0.25:
                    return "Deep bass accent or drum hit"
                else:
                    return "Soft low-frequency musical event"
        
        except Exception:
            # Fallback to generic description
            return f"Musical accent at {onset_time:.1f}s"
    
    def _characterize_segment(self, audio_data: np.ndarray, start_time: float, end_time: float) -> str:
        """Characterize a structural music segment"""
        try:
            # Extract segment audio
            start_sample = int(start_time * self.sample_rate)
            end_sample = int(end_time * self.sample_rate)
            segment_audio = audio_data[start_sample:end_sample]
            
            # Analyze segment characteristics
            tempo, _ = librosa.beat.beat_track(y=segment_audio, sr=self.sample_rate)
            
            # Energy analysis
            rms_energy = np.mean(librosa.feature.rms(y=segment_audio))
            
            # Spectral characteristics
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=segment_audio, sr=self.sample_rate))
            
            # Harmonic content
            harmonic, percussive = librosa.effects.hpss(segment_audio)
            harmonic_ratio = np.sum(harmonic ** 2) / (np.sum(harmonic ** 2) + np.sum(percussive ** 2))
            
            # Create description based on analysis
            duration = end_time - start_time
            
            if harmonic_ratio > 0.7:
                texture = "melodic"
            elif harmonic_ratio < 0.3:
                texture = "percussive"
            else:
                texture = "mixed"
            
            if rms_energy > 0.3:
                intensity = "high energy"
            elif rms_energy > 0.15:
                intensity = "moderate energy"
            else:
                intensity = "low energy"
            
            return f"{texture.capitalize()} section ({duration:.1f}s) - {intensity}, {tempo:.0f} BPM"
            
        except Exception:
            # Fallback description
            duration = end_time - start_time
            return f"Musical section ({duration:.1f}s)"
    
    def _add_mock_music_events(self, timeline: ServiceTimeline):
        """Add mock music events when LibROSA is not available"""
        duration = timeline.total_duration
        
        # Add some mock tempo events
        if duration > 10:
            timeline.add_event(TimelineEvent(
                timestamp=duration * 0.3,
                description="Tempo builds, increasing excitement",
                category="transition",
                source="librosa",
                confidence=0.5,
                details={"mock_mode": True, "analysis_type": "tempo_transition"}
            ))
        
        if duration > 20:
            timeline.add_event(TimelineEvent(
                timestamp=duration * 0.7,
                description="Energy decrease, calming dynamics",
                category="transition", 
                source="librosa",
                confidence=0.5,
                details={"mock_mode": True, "analysis_type": "energy_transition"}
            ))
    
    def _add_mock_structural_spans(self, timeline: ServiceTimeline):
        """Add mock structural spans when LibROSA is not available"""
        duration = timeline.total_duration
        
        # Create simple structural segments
        segment_length = duration / 3
        
        for i in range(3):
            start_time = i * segment_length
            end_time = min((i + 1) * segment_length, duration)
            
            if i == 0:
                description = f"Opening musical section ({end_time - start_time:.1f}s) - building energy"
            elif i == 1:
                description = f"Middle section ({end_time - start_time:.1f}s) - sustained intensity"
            else:
                description = f"Closing section ({end_time - start_time:.1f}s) - resolution"
            
            timeline.add_span(TimelineSpan(
                start=start_time,
                end=end_time,
                description=description,
                category="music",
                source="librosa",
                confidence=0.5,
                details={"mock_mode": True, "analysis_type": "structural_segment"}
            ))
    
    def _load_audio(self, audio_path: str) -> Optional[np.ndarray]:
        """Load audio file using LibROSA"""
        if not LIBROSA_AVAILABLE:
            logger.warning("LibROSA not available for audio loading")
            return None
        
        try:
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            logger.debug(f"Loaded audio: {len(y)} samples ({len(y)/sr:.2f}s)")
            return y
        except Exception as e:
            logger.error(f"Audio loading failed: {e}")
            return None
    
    def _create_fallback_timeline(self, audio_path: str, error: Optional[str] = None) -> ServiceTimeline:
        """Create fallback timeline when processing fails"""
        # Estimate duration from file info or use default
        try:
            from pathlib import Path
            file_size = Path(audio_path).stat().st_size
            # Rough estimate: 44.1 kHz * 16-bit * mono = ~88KB per second
            estimated_duration = file_size / 88000
        except:
            estimated_duration = 30.0  # Default fallback
        
        timeline = ServiceTimeline(
            source="librosa",
            audio_file=str(audio_path),
            total_duration=estimated_duration
        )
        
        # Add basic fallback events
        self._add_mock_music_events(timeline)
        self._add_mock_structural_spans(timeline)
        
        logger.warning(f"Using fallback LibROSA timeline: {error or 'LibROSA unavailable'}")
        return timeline
    
    def _save_timeline(self, timeline: ServiceTimeline, audio_path: str):
        """Save timeline to file"""
        try:
            # Determine output path
            audio_pathlib = Path(audio_path)
            build_dir = audio_pathlib.parent
            timeline_dir = build_dir / "audio_timelines"
            timeline_dir.mkdir(exist_ok=True)
            
            output_file = timeline_dir / "librosa_timeline.json"
            timeline.save_to_file(str(output_file))
            
            logger.info(f"LibROSA timeline saved to: {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save LibROSA timeline: {e}")
    
    def cleanup(self):
        """Cleanup resources"""
        gc.collect()
        logger.debug("LibROSA timeline service cleanup complete")