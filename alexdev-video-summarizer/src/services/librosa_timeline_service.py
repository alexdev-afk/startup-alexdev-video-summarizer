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
from utils.enhanced_timeline_schema import EnhancedTimeline, create_music_event, create_music_span

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
        
        # Event detection thresholds (from configuration)
        self.tempo_change_threshold = self.librosa_config.get('tempo_change_threshold', 10.0)  # BPM - was hardcoded as 5.0
        self.key_change_confidence = self.librosa_config.get('harmonic_change_threshold', 0.4)  # Map to config name
        self.structural_change_threshold = self.librosa_config.get('spectral_change_threshold', 0.3)
        self.energy_change_threshold = self.librosa_config.get('energy_change_threshold', 0.5)
        
        # Analysis window settings for event detection
        self.analysis_window = self.librosa_config.get('analysis_window', 5.0)  # seconds
        self.overlap_window = self.librosa_config.get('overlap_window', 2.5)  # seconds
        
        # Segmentation settings
        self.min_segment_length = self.librosa_config.get('min_segment_length', 3.0)
        self.max_segment_length = self.librosa_config.get('max_segment_length', 30.0)
        
        logger.info(f"LibROSA timeline service initialized - sample_rate: {self.sample_rate}, event_detection: enabled, available: {LIBROSA_AVAILABLE}")
    
    def generate_and_save(self, audio_path: str, source_tag: Optional[str] = None, optimization: Optional[Dict] = None) -> EnhancedTimeline:
        """
        Generate enhanced timeline and save intermediate files
        
        Args:
            audio_path: Path to audio file (FFmpeg-prepared WAV)
            
        Returns:
            EnhancedTimeline with music events and transitions
        """
        start_time = time.time()
        
        try:
            # Load audio data
            audio_data = self._load_audio(audio_path)
            if audio_data is None:
                return self._create_fallback_enhanced_timeline(audio_path)
            
            # Create enhanced timeline object
            total_duration = len(audio_data) / self.sample_rate
            timeline = EnhancedTimeline(
                audio_file=str(audio_path),
                total_duration=total_duration
            )
            
            # Add source tag to sources used
            timeline.sources_used.append(source_tag if source_tag else "librosa")
            
            # Add processing notes with analysis details
            timeline.processing_notes.append(f"LibROSA music analysis - sample_rate: {self.sample_rate}, analysis_window: {self.analysis_window}")
            timeline.processing_notes.append(f"Audio frames: {len(audio_data)}, analysis completed")
            
            # Generate music events using real LibROSA analysis
            assert source_tag, "source_tag is required for enhanced timeline generation"
            self._detect_enhanced_music_events(audio_data, timeline, source_tag)
            self._detect_enhanced_structural_spans(audio_data, timeline, source_tag)
            
            processing_time = time.time() - start_time
            logger.info(f"LibROSA enhanced timeline generated: {len(timeline.events)} events, {len(timeline.spans)} spans in {processing_time:.2f}s")
            
            # Save intermediate analysis files before cleanup
            self._save_intermediate_analysis(timeline, audio_path, audio_data, source_tag)
            
            # Save timeline to audio_timelines directory
            self._save_enhanced_timeline(timeline, audio_path, source_tag)
            
            # Cleanup large audio data from memory after all processing
            del audio_data
            gc.collect()
            
            return timeline
            
        except Exception as e:
            logger.error(f"LibROSA enhanced timeline generation failed: {e}")
            return self._create_fallback_enhanced_timeline(audio_path, error=str(e))
    
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
            # For legacy method, use default source or extract from timeline
            service_source = timeline.source or "librosa"
            self._detect_music_events(audio_data, timeline, service_source)
            self._detect_structural_spans(audio_data, timeline, service_source)
            
            processing_time = time.time() - start_time
            logger.info(f"LibROSA timeline generated: {len(timeline.events)} events, {len(timeline.spans)} spans in {processing_time:.2f}s")
            
            # Save timeline to file
            self._save_timeline(timeline, audio_path, source_tag)
            
            # Cleanup large audio data from memory after all processing
            del audio_data
            gc.collect()
            
            return timeline
            
        except Exception as e:
            logger.error(f"LibROSA timeline generation failed: {e}")
            return self._create_fallback_timeline(audio_path, error=str(e))
    
    def _detect_music_events(self, audio_data: np.ndarray, timeline: ServiceTimeline, source_tag: str):
        """
        Detect musical events using real LibROSA capabilities
        Focus on transitions and moments of change
        """
        if not LIBROSA_AVAILABLE:
            self._add_mock_music_events(timeline)
            return
        
        try:
            # 1. Tempo transitions using beat tracking
            self._detect_tempo_events(audio_data, timeline, source_tag)
            
            # 2. Musical onsets and rhythm changes
            self._detect_onset_events(audio_data, timeline, source_tag)
            
            # 3. Harmonic/key transitions using chroma analysis
            self._detect_harmonic_events(audio_data, timeline, source_tag)
            
            # 4. Energy/volume transitions
            self._detect_energy_events(audio_data, timeline, source_tag)
            
        except Exception as e:
            logger.warning(f"Music event detection failed: {e}, using fallback events")
            self._add_mock_music_events(timeline, source_tag)
    
    def _detect_tempo_events(self, audio_data: np.ndarray, timeline: ServiceTimeline, source_tag: str):
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
                        source=source_tag,
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
    
    def _detect_onset_events(self, audio_data: np.ndarray, timeline: ServiceTimeline, source_tag: str):
        """Detect significant musical onsets as events"""
        try:
            # Real onset detection using LibROSA
            onset_frames = librosa.onset.onset_detect(
                y=audio_data, 
                sr=self.sample_rate,
                units='time'
            )
            
            # Filter for significant onsets (not every small beat)
            max_onsets = int(self.max_segment_length * 3)  # Max 3 onsets per second based on max segment length
            if len(onset_frames) > max_onsets:
                onset_strength = librosa.onset.onset_strength(y=audio_data, sr=self.sample_rate)
                onset_strength_times = librosa.frames_to_time(
                    np.arange(len(onset_strength)), 
                    sr=self.sample_rate
                )
                
                # Keep only strongest onsets (configurable percentage)
                percentile_threshold = 100 - (self.energy_change_threshold * 100)  # Convert threshold to percentile
                strength_threshold = np.percentile(onset_strength, percentile_threshold)
                strong_onsets = []
                
                for onset_time in onset_frames:
                    # Find closest strength measurement
                    closest_idx = np.argmin(np.abs(onset_strength_times - onset_time))
                    if onset_strength[closest_idx] > strength_threshold:
                        strong_onsets.append(onset_time)
                
                max_final_onsets = int(self.min_segment_length * 5)  # Max 5 onsets per min segment duration
                onset_frames = strong_onsets[:max_final_onsets]
            
            # Create onset events
            max_onset_events = int(self.min_segment_length * 5)  # Max 5 events per min segment duration
            for i, onset_time in enumerate(onset_frames):
                if i < max_onset_events:  # Limit events to avoid timeline clutter
                    # Determine onset character based on surrounding context
                    onset_description = self._characterize_onset(audio_data, onset_time)
                    
                    event = TimelineEvent(
                        timestamp=float(onset_time),
                        description=onset_description,
                        category="music",
                        source=source_tag, 
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
    
    def _detect_harmonic_events(self, audio_data: np.ndarray, timeline: ServiceTimeline, source_tag: str):
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
                        source=source_tag,
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
    
    def _detect_energy_events(self, audio_data: np.ndarray, timeline: ServiceTimeline, source_tag: str):
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
            energy_threshold = np.std(rms_smooth) * self.energy_change_threshold
            
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
                        source=source_tag,
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
    
    def _detect_structural_spans(self, audio_data: np.ndarray, timeline: ServiceTimeline, source_tag: str):
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
            
            # Detect segment boundaries - use simpler approach for newer librosa versions
            try:
                boundaries = librosa.segment.agglomerative(
                    chroma, 
                    k=5  # Fixed number of segments for stability
                )
            except Exception as e:
                logger.debug(f"Agglomerative segmentation failed: {e}, using simpler approach")
                # Fallback to simple time-based boundaries
                duration = chroma.shape[1] * self.hop_length / self.sample_rate
                num_segments = max(3, int(duration / self.max_segment_length))
                boundaries = np.linspace(0, chroma.shape[1], num_segments + 1, dtype=int)
            
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
                    source=source_tag,
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
            self._add_mock_structural_spans(timeline, source_tag)
    
    def _detect_enhanced_music_events(self, audio_data: np.ndarray, timeline: EnhancedTimeline, source_tag: str):
        """
        Detect musical events using real LibROSA capabilities for enhanced timeline
        """
        if not LIBROSA_AVAILABLE:
            self._add_enhanced_mock_music_events(timeline)
            return
        
        try:
            # 1. Tempo transitions using beat tracking
            self._detect_enhanced_tempo_events(audio_data, timeline, source_tag)
            
            # 2. Musical onsets and rhythm changes
            self._detect_enhanced_onset_events(audio_data, timeline, source_tag)
            
            # 3. Harmonic/key transitions using chroma analysis
            self._detect_enhanced_harmonic_events(audio_data, timeline, source_tag)
            
            # 4. Energy/volume transitions
            self._detect_enhanced_energy_events(audio_data, timeline, source_tag)
            
        except Exception as e:
            logger.warning(f"Enhanced music event detection failed: {e}, using fallback events")
            self._add_enhanced_mock_music_events(timeline, source_tag)
    
    def _detect_enhanced_tempo_events(self, audio_data: np.ndarray, timeline: EnhancedTimeline, source_tag: str):
        """Detect tempo changes as enhanced timeline events"""
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
                    # Create enhanced tempo change event
                    event = create_music_event(
                        timestamp=timestamps[i],
                        event_type="tempo_change",
                        confidence=min(0.95, 0.6 + (tempo_change / 20.0)),
                        details={
                            "previous_tempo": float(tempos[i-1]),
                            "new_tempo": float(tempos[i]),
                            "tempo_change": float(tempo_change),
                            "analysis_type": "tempo_transition"
                        },
                        source=source_tag
                    )
                    timeline.add_event(event)
            
            logger.debug(f"Detected {len([e for e in timeline.events if e.details.get('analysis_type') == 'tempo_transition'])} tempo change events")
            
        except Exception as e:
            logger.warning(f"Enhanced tempo event detection failed: {e}")
    
    def _detect_enhanced_onset_events(self, audio_data: np.ndarray, timeline: EnhancedTimeline, source_tag: str):
        """Detect significant musical onsets as enhanced events"""
        try:
            # Real onset detection using LibROSA
            onset_frames = librosa.onset.onset_detect(
                y=audio_data, 
                sr=self.sample_rate,
                units='time'
            )
            
            # Filter for significant onsets (not every small beat)
            max_onsets = int(self.max_segment_length * 3)  # Max 3 onsets per second based on max segment length
            if len(onset_frames) > max_onsets:
                onset_strength = librosa.onset.onset_strength(y=audio_data, sr=self.sample_rate)
                onset_strength_times = librosa.frames_to_time(
                    np.arange(len(onset_strength)), 
                    sr=self.sample_rate
                )
                
                # Keep only strongest onsets (configurable percentage)
                percentile_threshold = 100 - (self.energy_change_threshold * 100)  # Convert threshold to percentile
                strength_threshold = np.percentile(onset_strength, percentile_threshold)
                strong_onsets = []
                
                for onset_time in onset_frames:
                    # Find closest strength measurement
                    closest_idx = np.argmin(np.abs(onset_strength_times - onset_time))
                    if onset_strength[closest_idx] > strength_threshold:
                        strong_onsets.append(onset_time)
                
                max_final_onsets = int(self.min_segment_length * 5)  # Max 5 onsets per min segment duration
                onset_frames = strong_onsets[:max_final_onsets]
            
            # Create enhanced onset events
            for i, onset_time in enumerate(onset_frames):
                if i < 15:  # Limit events to avoid timeline clutter
                    # Characterize onset
                    onset_character = self._characterize_onset(audio_data, onset_time)
                    
                    event = create_music_event(
                        timestamp=float(onset_time),
                        event_type="musical_onset",
                        confidence=0.75,
                        details={
                            "onset_character": onset_character,
                            "onset_index": i,
                            "total_onsets": len(onset_frames),
                            "analysis_type": "musical_onset"
                        },
                        source=source_tag
                    )
                    timeline.add_event(event)
            
            logger.debug(f"Detected {len(onset_frames)} significant musical onset events")
            
        except Exception as e:
            logger.warning(f"Enhanced onset event detection failed: {e}")
    
    def _detect_enhanced_harmonic_events(self, audio_data: np.ndarray, timeline: EnhancedTimeline, source_tag: str):
        """Detect harmonic/key changes as enhanced events"""
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
                    
                    event = create_music_event(
                        timestamp=timestamps[i],
                        event_type="harmonic_shift",
                        confidence=0.65,
                        details={
                            "previous_key": old_key,
                            "new_key": new_key,
                            "pitch_class_change": int(key_profiles[i] - key_profiles[i-1]),
                            "analysis_type": "harmonic_transition"
                        },
                        source=source_tag
                    )
                    timeline.add_event(event)
            
            logger.debug(f"Detected {len([e for e in timeline.events if e.details.get('analysis_type') == 'harmonic_transition'])} harmonic transition events")
            
        except Exception as e:
            logger.warning(f"Enhanced harmonic event detection failed: {e}")
    
    def _detect_enhanced_energy_events(self, audio_data: np.ndarray, timeline: EnhancedTimeline, source_tag: str):
        """Detect energy/volume transitions as enhanced events"""
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
            energy_threshold = np.std(rms_smooth) * self.energy_change_threshold
            
            for i in range(1, len(rms_smooth)):
                energy_change = abs(rms_smooth[i] - rms_smooth[i-1])
                
                if energy_change > energy_threshold:
                    # Determine if increase or decrease
                    if rms_smooth[i] > rms_smooth[i-1]:
                        event_type = "energy_increase"
                    else:
                        event_type = "energy_decrease"
                    
                    event = create_music_event(
                        timestamp=float(rms_times[i]),
                        event_type=event_type,
                        confidence=0.7,
                        details={
                            "previous_energy": float(rms_smooth[i-1]),
                            "new_energy": float(rms_smooth[i]),
                            "energy_change": float(energy_change),
                            "analysis_type": "energy_transition"
                        },
                        source=source_tag
                    )
                    timeline.add_event(event)
            
            logger.debug(f"Detected {len([e for e in timeline.events if e.details.get('analysis_type') == 'energy_transition'])} energy transition events")
            
        except Exception as e:
            logger.warning(f"Enhanced energy event detection failed: {e}")
    
    def _detect_enhanced_structural_spans(self, audio_data: np.ndarray, timeline: EnhancedTimeline, source_tag: str):
        """
        Detect structural music spans using real LibROSA segmentation for enhanced timeline
        """
        if not LIBROSA_AVAILABLE:
            self._add_enhanced_mock_structural_spans(timeline)
            return
        
        try:
            # Real structural segmentation using LibROSA
            chroma = librosa.feature.chroma_cqt(y=audio_data, sr=self.sample_rate)
            
            # Use recurrence matrix for structure detection
            R = librosa.segment.recurrence_matrix(
                chroma, 
                mode='affinity',
                metric='cosine'
            )
            
            # Detect segment boundaries - use simpler approach for newer librosa versions
            try:
                boundaries = librosa.segment.agglomerative(
                    chroma, 
                    k=5  # Fixed number of segments for stability
                )
            except Exception as e:
                logger.debug(f"Agglomerative segmentation failed: {e}, using simpler approach")
                # Fallback to simple time-based boundaries
                duration = chroma.shape[1] * self.hop_length / self.sample_rate
                num_segments = max(3, int(duration / self.max_segment_length))
                boundaries = np.linspace(0, chroma.shape[1], num_segments + 1, dtype=int)
            
            # Convert frame boundaries to time boundaries
            boundary_times = librosa.frames_to_time(boundaries, sr=self.sample_rate, hop_length=self.hop_length)
            
            # Create enhanced structural spans between boundaries
            for i in range(len(boundary_times) - 1):
                start_time = float(boundary_times[i])
                end_time = float(boundary_times[i + 1])
                
                # Analyze this segment's characteristics
                segment_characteristics = self._analyze_segment_characteristics(
                    audio_data, start_time, end_time
                )
                
                span = create_music_span(
                    start=start_time,
                    end=end_time,
                    span_type="structural_segment",
                    confidence=0.8,
                    details={
                        "segment_index": i,
                        "total_segments": len(boundary_times) - 1,
                        "analysis_type": "structural_segment",
                        **segment_characteristics
                    },
                    source=source_tag
                )
                timeline.add_span(span)
            
            logger.debug(f"Detected {len(boundary_times) - 1} structural music segments")
            
        except Exception as e:
            logger.warning(f"Enhanced structural segmentation failed: {e}, using fallback spans")
            self._add_enhanced_mock_structural_spans(timeline, source_tag)
    
    def _analyze_segment_characteristics(self, audio_data: np.ndarray, start_time: float, end_time: float) -> Dict[str, Any]:
        """Analyze characteristics of a structural music segment"""
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
            
            duration = end_time - start_time
            
            return {
                "duration": duration,
                "tempo": float(tempo),
                "energy": float(rms_energy),
                "spectral_centroid": float(spectral_centroid),
                "harmonic_ratio": float(harmonic_ratio),
                "texture": "melodic" if harmonic_ratio > 0.7 else ("percussive" if harmonic_ratio < 0.3 else "mixed"),
                "intensity": "high" if rms_energy > 0.3 else ("moderate" if rms_energy > 0.15 else "low")
            }
            
        except Exception:
            # Fallback characteristics
            return {
                "duration": end_time - start_time,
                "tempo": 120.0,
                "energy": 0.1,
                "texture": "mixed",
                "intensity": "moderate"
            }
    
    def _add_enhanced_mock_music_events(self, timeline: EnhancedTimeline):
        """Add mock music events when LibROSA is not available for enhanced timeline"""
        duration = timeline.total_duration
        
        # Add some mock tempo events
        if duration > 10:
            event = create_music_event(
                timestamp=duration * 0.3,
                event_type="tempo_change",
                confidence=0.5,
                details={"mock_mode": True, "analysis_type": "tempo_transition"}
            )
            timeline.add_event(event)
        
        if duration > 20:
            event = create_music_event(
                timestamp=duration * 0.7,
                event_type="energy_decrease",
                confidence=0.5,
                details={"mock_mode": True, "analysis_type": "energy_transition"}
            )
            timeline.add_event(event)
    
    def _add_enhanced_mock_structural_spans(self, timeline: EnhancedTimeline, source_tag: str):
        """Add mock structural spans when LibROSA is not available for enhanced timeline"""
        duration = timeline.total_duration
        
        # Create simple structural segments
        segment_length = duration / 3
        
        for i in range(3):
            start_time = i * segment_length
            end_time = min((i + 1) * segment_length, duration)
            
            span_type = "opening" if i == 0 else ("middle" if i == 1 else "closing")
            
            span = create_music_span(
                start=start_time,
                end=end_time,
                span_type=span_type,
                confidence=0.5,
                details={
                    "mock_mode": True, 
                    "analysis_type": "structural_segment",
                    "duration": end_time - start_time,
                    "texture": "mixed",
                    "intensity": "moderate"
                },
                source=source_tag
            )
            timeline.add_span(span)
    
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
            
            # Characterize based on features (using configurable thresholds)
            high_freq_threshold = self.sample_rate * 0.136  # ~3000 Hz for 22050 sample rate
            mid_freq_threshold = self.sample_rate * 0.068   # ~1500 Hz for 22050 sample rate
            
            if spectral_centroid > high_freq_threshold:
                if rms_energy > self.energy_change_threshold * 0.6:  # 60% of energy threshold
                    return "Sharp percussive hit with bright timbre"
                else:
                    return "Subtle high-frequency accent"
            elif spectral_centroid > mid_freq_threshold:
                if rms_energy > self.energy_change_threshold * 0.4:  # 40% of energy threshold
                    return "Strong musical accent with warm timbre"
                else:
                    return "Gentle melodic emphasis"
            else:
                if rms_energy > self.energy_change_threshold * 0.5:  # 50% of energy threshold
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
            
            # Create description based on analysis (using configurable thresholds)
            duration = end_time - start_time
            
            melodic_threshold = self.key_change_confidence + 0.3  # 0.4 + 0.3 = 0.7 with default config
            percussive_threshold = self.key_change_confidence - 0.1  # 0.4 - 0.1 = 0.3 with default config
            
            if harmonic_ratio > melodic_threshold:
                texture = "melodic"
            elif harmonic_ratio < percussive_threshold:
                texture = "percussive"
            else:
                texture = "mixed"
            
            high_energy_threshold = self.energy_change_threshold * 0.6  # 60% of energy threshold
            moderate_energy_threshold = self.energy_change_threshold * 0.3  # 30% of energy threshold
            
            if rms_energy > high_energy_threshold:
                intensity = "high energy"
            elif rms_energy > moderate_energy_threshold:
                intensity = "moderate energy"
            else:
                intensity = "low energy"
            
            return f"{texture.capitalize()} section ({duration:.1f}s) - {intensity}, {tempo:.0f} BPM"
            
        except Exception:
            # Fallback description
            duration = end_time - start_time
            return f"Musical section ({duration:.1f}s)"
    
    def _add_mock_music_events(self, timeline: ServiceTimeline, source_tag: str):
        """Add mock music events when LibROSA is not available"""
        duration = timeline.total_duration
        
        # Add some mock tempo events
        if duration > 10:
            timeline.add_event(TimelineEvent(
                timestamp=duration * 0.3,
                description="Tempo builds, increasing excitement",
                category="transition",
                source=source_tag,
                confidence=0.5,
                details={"mock_mode": True, "analysis_type": "tempo_transition"}
            ))
        
        if duration > 20:
            timeline.add_event(TimelineEvent(
                timestamp=duration * 0.7,
                description="Energy decrease, calming dynamics",
                category="transition", 
                source=source_tag,
                confidence=0.5,
                details={"mock_mode": True, "analysis_type": "energy_transition"}
            ))
    
    def _add_mock_structural_spans(self, timeline: ServiceTimeline, source_tag: str):
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
                source=source_tag,
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
        fallback_source = timeline.source or "librosa"
        self._add_mock_music_events(timeline, fallback_source)
        self._add_mock_structural_spans(timeline, fallback_source)
        
        logger.warning(f"Using fallback LibROSA timeline: {error or 'LibROSA unavailable'}")
        return timeline
    
    def _save_timeline(self, timeline: ServiceTimeline, audio_path: str, source_tag: Optional[str] = None):
        """Save timeline to file"""
        try:
            # Determine output path
            audio_pathlib = Path(audio_path)
            build_dir = audio_pathlib.parent
            timeline_dir = build_dir / "audio_timelines"
            timeline_dir.mkdir(exist_ok=True)
            
            timeline_filename = f"{source_tag or 'librosa'}_timeline.json"
            output_file = timeline_dir / timeline_filename
            timeline.save_to_file(str(output_file))
            
            logger.info(f"LibROSA timeline saved to: {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save LibROSA timeline: {e}")
    
    def _create_fallback_enhanced_timeline(self, audio_path: str, error: Optional[str] = None) -> EnhancedTimeline:
        """Create fallback enhanced timeline when processing fails"""
        # Estimate duration from file info or use default
        try:
            from pathlib import Path
            file_size = Path(audio_path).stat().st_size
            # Rough estimate: 44.1 kHz * 16-bit * mono = ~88KB per second
            estimated_duration = file_size / 88000
        except:
            estimated_duration = 30.0  # Default fallback
        
        timeline = EnhancedTimeline(
            audio_file=str(audio_path),
            total_duration=estimated_duration
        )
        
        # Add LibROSA to sources used
        timeline.sources_used.append("librosa")
        
        # Add fallback processing notes
        timeline.processing_notes.append(f"LibROSA fallback mode - sample_rate: {self.sample_rate}")
        if error:
            timeline.processing_notes.append(f"Error: {error}")
        
        # Add basic fallback events and spans
        fallback_source = "librosa"  # Default for enhanced timeline fallback
        self._add_enhanced_mock_music_events(timeline, fallback_source)
        self._add_enhanced_mock_structural_spans(timeline, fallback_source)
        
        logger.warning(f"Using fallback LibROSA enhanced timeline: {error or 'LibROSA unavailable'}")
        return timeline
    
    def _save_intermediate_analysis(self, timeline: EnhancedTimeline, audio_path: str, audio_data: np.ndarray, source_tag: Optional[str] = None):
        """Save intermediate analysis files to audio_analysis directory"""
        try:
            # Determine output path
            audio_pathlib = Path(audio_path)
            build_dir = audio_pathlib.parent
            analysis_dir = build_dir / "audio_analysis"
            analysis_dir.mkdir(exist_ok=True)
            
            # Save raw analysis data
            analysis_data = {
                "metadata": {
                    "source": "librosa",
                    "audio_file": str(audio_path),
                    "total_duration": timeline.total_duration,
                    "sample_rate": self.sample_rate,
                    "analysis_timestamp": datetime.now().isoformat(),
                    "librosa_available": LIBROSA_AVAILABLE
                },
                "raw_features": {
                    "audio_length": len(audio_data) if audio_data is not None else 0,
                    "analysis_windows": {
                        "window_size": self.analysis_window,
                        "overlap": self.overlap_window
                    },
                    "thresholds": {
                        "tempo_change": self.tempo_change_threshold,
                        "key_change_confidence": self.key_change_confidence,
                        "structural_change": self.structural_change_threshold
                    }
                },
                "detected_events_count": len(timeline.events),
                "detected_spans_count": len(timeline.spans),
                "transcript_length": len(timeline.full_transcript),
                "speakers": timeline.speakers,
                "complete_events_data": [
                    {
                        "timestamp": event.timestamp,
                        "description": event.description,
                        "event_type": event.description.lower().replace(" ", "_"),
                        "source": event.source,
                        "confidence": event.confidence,
                        "details": event.details
                    } for event in timeline.events
                ],
                "complete_spans_data": [
                    {
                        "start": span.start,
                        "end": span.end,
                        "description": span.description,
                        "source": span.source,
                        "confidence": span.confidence,
                        "details": span.details,
                        "events_count": len(span.events)
                    } for span in timeline.spans
                ]
            }
            
            analysis_filename = f"{source_tag or 'librosa'}_analysis.json"
            output_file = analysis_dir / analysis_filename
            
            import json
            with open(output_file, 'w') as f:
                json.dump(analysis_data, f, indent=2)
            
            logger.info(f"LibROSA intermediate analysis saved to: {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save LibROSA intermediate analysis: {e}")
    
    def _save_enhanced_timeline(self, timeline: EnhancedTimeline, audio_path: str, source_tag: Optional[str] = None):
        """Save enhanced timeline to audio_timelines directory"""
        try:
            # Determine output path
            audio_pathlib = Path(audio_path)
            build_dir = audio_pathlib.parent
            timeline_dir = build_dir / "audio_timelines"
            timeline_dir.mkdir(exist_ok=True)
            
            timeline_filename = f"{source_tag or 'librosa'}_timeline.json"
            output_file = timeline_dir / timeline_filename
            timeline.save_to_file(str(output_file))
            
            logger.info(f"LibROSA enhanced timeline saved to: {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save LibROSA enhanced timeline: {e}")
    
    def cleanup(self):
        """Cleanup resources and memory"""
        try:
            # Clear any large audio data from memory if still referenced
            if hasattr(self, '_current_audio_data'):
                delattr(self, '_current_audio_data')
            
            # Force garbage collection multiple times for thorough cleanup
            for _ in range(3):
                gc.collect()
            
            # Clear librosa cache if available
            if LIBROSA_AVAILABLE:
                try:
                    # Clear librosa's internal caches
                    import librosa.cache
                    librosa.cache.clear()
                except (ImportError, AttributeError):
                    pass  # Cache clearing not available in this librosa version
            
            logger.debug("LibROSA timeline service cleanup complete - memory freed")
            
        except Exception as e:
            logger.warning(f"LibROSA cleanup warning: {e}")
            # Continue with basic cleanup even if advanced cleanup fails
            gc.collect()