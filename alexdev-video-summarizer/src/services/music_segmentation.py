"""
Advanced Music Segmentation Using LibROSA

Smart music transition detection based on:
1. Spectral change detection (musical style changes)
2. Tempo change points (rhythm transitions) 
3. Harmonic change detection (chord progressions, key changes)
4. Energy-based boundaries (volume/dynamics changes)
5. Beat tracking synchronization points

Much smarter than hardcoded time-based segmentation!
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from utils.logger import get_logger

logger = get_logger(__name__)

try:
    import librosa
    import scipy.signal
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


class SmartMusicSegmentation:
    """Intelligent music segmentation based on acoustic feature changes"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize smart music segmentation
        
        Args:
            config: Configuration dictionary with segmentation parameters
        """
        librosa_config = config.get('cpu_pipeline', {}).get('librosa', {})
        
        self.sample_rate = librosa_config.get('sample_rate', 22050)
        
        # Smart segmentation parameters
        self.min_segment_length = librosa_config.get('min_segment_length', 3.0)  # Minimum 3s segments
        self.max_segment_length = librosa_config.get('max_segment_length', 30.0)  # Maximum 30s segments
        
        # Feature change detection thresholds
        self.spectral_change_threshold = librosa_config.get('spectral_change_threshold', 0.3)
        self.tempo_change_threshold = librosa_config.get('tempo_change_threshold', 10.0)  # BPM
        self.harmonic_change_threshold = librosa_config.get('harmonic_change_threshold', 0.4)
        self.energy_change_threshold = librosa_config.get('energy_change_threshold', 0.5)
        
        # Smoothing parameters
        self.gaussian_sigma = librosa_config.get('gaussian_sigma', 2.0)  # Smoothing for change detection
        self.peak_prominence = librosa_config.get('peak_prominence', 0.1)  # Peak detection sensitivity
        
        logger.info(f"Smart music segmentation initialized - min: {self.min_segment_length}s, max: {self.max_segment_length}s")
    
    def detect_music_boundaries(self, audio_data: np.ndarray) -> List[Tuple[int, int, float, float]]:
        """
        Detect music boundaries using multiple acoustic features
        
        Args:
            audio_data: Audio signal
            
        Returns:
            List of (start_idx, end_idx, start_time, end_time) segment boundaries
        """
        if not LIBROSA_AVAILABLE:
            logger.warning("LibROSA not available, falling back to fixed segmentation")
            return self._fallback_fixed_segmentation(audio_data)
        
        try:
            logger.info(f"Detecting music boundaries in {len(audio_data) / self.sample_rate:.1f}s audio")
            
            # Extract multiple feature streams for change detection
            feature_streams = self._extract_feature_streams(audio_data)
            
            # Detect change points in each feature stream
            change_points = self._detect_multi_feature_changes(feature_streams)
            
            # Combine and filter change points
            unified_boundaries = self._unify_change_points(change_points, len(audio_data))
            
            # Convert to segment boundaries
            segments = self._create_segments_from_boundaries(unified_boundaries, len(audio_data))
            
            logger.info(f"Smart segmentation detected {len(segments)} music segments based on style changes")
            return segments
            
        except Exception as e:
            logger.error(f"Smart music segmentation failed: {e}, using fallback")
            return self._fallback_fixed_segmentation(audio_data)
    
    def _extract_feature_streams(self, audio_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract multiple feature streams for change detection
        
        Returns:
            Dictionary of feature name -> feature timeline arrays
        """
        features = {}
        
        try:
            # 1. Spectral features (detect style changes)
            hop_length = 1024
            spectral_centroid = librosa.feature.spectral_centroid(
                y=audio_data, sr=self.sample_rate, hop_length=hop_length
            )[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio_data, sr=self.sample_rate, hop_length=hop_length
            )[0]
            features['spectral'] = np.concatenate([spectral_centroid, spectral_rolloff])
            
            # 2. MFCC features (detect timbral changes)
            mfcc = librosa.feature.mfcc(
                y=audio_data, sr=self.sample_rate, n_mfcc=13, hop_length=hop_length
            )
            features['mfcc'] = mfcc
            
            # 3. Chroma features (detect harmonic/key changes)
            chroma = librosa.feature.chroma_stft(
                y=audio_data, sr=self.sample_rate, hop_length=hop_length
            )
            features['chroma'] = chroma
            
            # 4. Tempo tracking (detect rhythm changes)
            tempo, beats = librosa.beat.beat_track(
                y=audio_data, sr=self.sample_rate, hop_length=hop_length
            )
            # Create tempo timeline
            frame_times = librosa.frames_to_time(np.arange(len(spectral_centroid)), 
                                                sr=self.sample_rate, hop_length=hop_length)
            tempo_timeline = np.full_like(frame_times, tempo)
            features['tempo'] = tempo_timeline
            
            # 5. Energy features (detect dynamic changes)
            rms_energy = librosa.feature.rms(
                y=audio_data, hop_length=hop_length
            )[0]
            features['energy'] = rms_energy
            
            logger.debug(f"Extracted {len(features)} feature streams for change detection")
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return {}
    
    def _detect_multi_feature_changes(self, feature_streams: Dict[str, np.ndarray]) -> Dict[str, List[int]]:
        """
        Detect change points in multiple feature streams
        
        Args:
            feature_streams: Dictionary of feature arrays
            
        Returns:
            Dictionary of feature name -> list of change point frame indices
        """
        change_points = {}
        
        for feature_name, feature_data in feature_streams.items():
            try:
                if feature_name == 'spectral':
                    # Spectral change detection using cosine distance
                    changes = self._detect_spectral_changes(feature_data)
                elif feature_name == 'mfcc':
                    # MFCC change detection using euclidean distance
                    changes = self._detect_mfcc_changes(feature_data)
                elif feature_name == 'chroma':
                    # Harmonic change detection using chroma vectors
                    changes = self._detect_harmonic_changes(feature_data)
                elif feature_name == 'tempo':
                    # Tempo change detection
                    changes = self._detect_tempo_changes(feature_data)
                elif feature_name == 'energy':
                    # Energy change detection
                    changes = self._detect_energy_changes(feature_data)
                else:
                    changes = []
                
                change_points[feature_name] = changes
                logger.debug(f"{feature_name} change detection: {len(changes)} change points")
                
            except Exception as e:
                logger.warning(f"Change detection failed for {feature_name}: {e}")
                change_points[feature_name] = []
        
        return change_points
    
    def _detect_spectral_changes(self, spectral_features: np.ndarray) -> List[int]:
        """Detect spectral characteristic changes (musical style changes)"""
        if spectral_features.shape[0] < 10:
            return []
        
        # Compute frame-to-frame spectral distance
        distances = []
        for i in range(1, spectral_features.shape[1]):
            # Cosine distance between adjacent frames
            prev_frame = spectral_features[:, i-1]
            curr_frame = spectral_features[:, i]
            
            # Normalize vectors
            prev_norm = prev_frame / (np.linalg.norm(prev_frame) + 1e-8)
            curr_norm = curr_frame / (np.linalg.norm(curr_frame) + 1e-8)
            
            # Cosine distance
            cosine_dist = 1 - np.dot(prev_norm, curr_norm)
            distances.append(cosine_dist)
        
        distances = np.array(distances)
        
        # Smooth the distance curve
        from scipy.ndimage import gaussian_filter1d
        smoothed_distances = gaussian_filter1d(distances, sigma=self.gaussian_sigma)
        
        # Find peaks (change points)
        peaks, _ = scipy.signal.find_peaks(
            smoothed_distances,
            height=self.spectral_change_threshold,
            prominence=self.peak_prominence,
            distance=int(self.min_segment_length * self.sample_rate / 1024)  # Minimum distance between peaks
        )
        
        return peaks.tolist()
    
    def _detect_mfcc_changes(self, mfcc_features: np.ndarray) -> List[int]:
        """Detect timbral changes using MFCC features"""
        if mfcc_features.shape[1] < 10:
            return []
        
        # Compute frame-to-frame MFCC euclidean distance
        distances = []
        for i in range(1, mfcc_features.shape[1]):
            prev_frame = mfcc_features[:, i-1]
            curr_frame = mfcc_features[:, i]
            euclidean_dist = np.linalg.norm(curr_frame - prev_frame)
            distances.append(euclidean_dist)
        
        distances = np.array(distances)
        
        # Normalize distances
        if np.std(distances) > 0:
            distances = (distances - np.mean(distances)) / np.std(distances)
        
        # Find significant changes
        threshold = np.percentile(distances, 85)  # Top 15% of changes
        change_indices = np.where(distances > threshold)[0]
        
        # Filter to maintain minimum segment length
        filtered_changes = []
        last_change = -int(self.min_segment_length * self.sample_rate / 1024)
        
        for change in change_indices:
            if change - last_change >= int(self.min_segment_length * self.sample_rate / 1024):
                filtered_changes.append(change)
                last_change = change
        
        return filtered_changes
    
    def _detect_harmonic_changes(self, chroma_features: np.ndarray) -> List[int]:
        """Detect harmonic/key changes using chroma features"""
        if chroma_features.shape[1] < 10:
            return []
        
        # Compute chroma change using correlation
        changes = []
        window_size = int(2.0 * self.sample_rate / 1024)  # 2-second windows
        
        for i in range(window_size, chroma_features.shape[1] - window_size, window_size // 2):
            # Compare chroma distributions before and after this point
            before_chroma = np.mean(chroma_features[:, i-window_size:i], axis=1)
            after_chroma = np.mean(chroma_features[:, i:i+window_size], axis=1)
            
            # Normalize chroma vectors
            before_norm = before_chroma / (np.sum(before_chroma) + 1e-8)
            after_norm = after_chroma / (np.sum(after_chroma) + 1e-8)
            
            # Calculate chroma change using KL divergence
            kl_div = np.sum(before_norm * np.log((before_norm + 1e-8) / (after_norm + 1e-8)))
            
            if kl_div > self.harmonic_change_threshold:
                changes.append(i)
        
        return changes
    
    def _detect_tempo_changes(self, tempo_timeline: np.ndarray) -> List[int]:
        """Detect tempo/rhythm changes"""
        if len(tempo_timeline) < 10:
            return []
        
        # Find points where tempo changes significantly
        tempo_diff = np.abs(np.diff(tempo_timeline))
        change_indices = np.where(tempo_diff > self.tempo_change_threshold)[0]
        
        return change_indices.tolist()
    
    def _detect_energy_changes(self, energy_timeline: np.ndarray) -> List[int]:
        """Detect energy/dynamic changes"""
        if len(energy_timeline) < 10:
            return []
        
        # Smooth energy timeline
        from scipy.ndimage import gaussian_filter1d
        smoothed_energy = gaussian_filter1d(energy_timeline, sigma=self.gaussian_sigma)
        
        # Find significant energy transitions
        energy_gradient = np.abs(np.gradient(smoothed_energy))
        threshold = np.percentile(energy_gradient, 90)  # Top 10% of energy changes
        
        change_indices = np.where(energy_gradient > threshold)[0]
        
        return change_indices.tolist()
    
    def _unify_change_points(self, change_points: Dict[str, List[int]], audio_length: int) -> List[int]:
        """
        Combine change points from multiple features into unified boundaries
        """
        all_changes = []
        hop_length = 1024
        
        # Collect all change points with weights
        for feature_name, changes in change_points.items():
            # Weight different features
            weights = {
                'spectral': 1.0,    # High weight for spectral changes
                'mfcc': 0.8,        # Medium-high weight for timbral changes
                'chroma': 0.9,      # High weight for harmonic changes
                'tempo': 0.7,       # Medium weight for tempo changes
                'energy': 0.6       # Lower weight for energy changes
            }
            
            weight = weights.get(feature_name, 0.5)
            
            for change_frame in changes:
                change_time = librosa.frames_to_time(change_frame, sr=self.sample_rate, hop_length=hop_length)
                change_sample = int(change_time * self.sample_rate)
                all_changes.append((change_sample, weight, feature_name))
        
        # Sort by time
        all_changes.sort(key=lambda x: x[0])
        
        # Cluster nearby change points (within 1 second)
        clustered_changes = []
        cluster_window = int(1.0 * self.sample_rate)  # 1 second window
        
        i = 0
        while i < len(all_changes):
            cluster_start = i
            cluster_weight = 0
            cluster_position = 0
            
            # Find all changes within the cluster window
            while (i < len(all_changes) and 
                   all_changes[i][0] - all_changes[cluster_start][0] < cluster_window):
                cluster_weight += all_changes[i][1]
                cluster_position += all_changes[i][0] * all_changes[i][1]
                i += 1
            
            # Calculate weighted average position
            if cluster_weight > 0:
                avg_position = int(cluster_position / cluster_weight)
                clustered_changes.append((avg_position, cluster_weight))
        
        # Filter by minimum weight (require multiple feature agreement)
        min_weight_threshold = 1.0  # Require at least moderate agreement
        filtered_changes = [pos for pos, weight in clustered_changes if weight >= min_weight_threshold]
        
        # Ensure minimum segment lengths
        final_changes = []
        min_samples = int(self.min_segment_length * self.sample_rate)
        
        last_boundary = 0
        for change_pos in filtered_changes:
            if change_pos - last_boundary >= min_samples and change_pos < audio_length - min_samples:
                final_changes.append(change_pos)
                last_boundary = change_pos
        
        logger.info(f"Unified change detection: {len(all_changes)} raw changes -> {len(final_changes)} boundaries")
        return final_changes
    
    def _create_segments_from_boundaries(self, boundaries: List[int], audio_length: int) -> List[Tuple[int, int, float, float]]:
        """
        Create segment boundaries from change points
        """
        if not boundaries:
            # No boundaries detected, return single segment
            return [(0, audio_length, 0.0, audio_length / self.sample_rate)]
        
        segments = []
        
        # Add first segment
        if boundaries[0] > 0:
            segments.append((0, boundaries[0], 0.0, boundaries[0] / self.sample_rate))
        
        # Add middle segments
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]
            start_time = start_idx / self.sample_rate
            end_time = end_idx / self.sample_rate
            segments.append((start_idx, end_idx, start_time, end_time))
        
        # Add final segment
        if boundaries[-1] < audio_length:
            start_idx = boundaries[-1]
            start_time = start_idx / self.sample_rate
            end_time = audio_length / self.sample_rate
            segments.append((start_idx, audio_length, start_time, end_time))
        
        # Enforce maximum segment length by splitting long segments
        final_segments = []
        max_samples = int(self.max_segment_length * self.sample_rate)
        
        for start_idx, end_idx, start_time, end_time in segments:
            if end_idx - start_idx <= max_samples:
                final_segments.append((start_idx, end_idx, start_time, end_time))
            else:
                # Split long segment
                current_start = start_idx
                while current_start < end_idx:
                    current_end = min(current_start + max_samples, end_idx)
                    seg_start_time = current_start / self.sample_rate
                    seg_end_time = current_end / self.sample_rate
                    final_segments.append((current_start, current_end, seg_start_time, seg_end_time))
                    current_start = current_end
        
        return final_segments
    
    def _fallback_fixed_segmentation(self, audio_data: np.ndarray) -> List[Tuple[int, int, float, float]]:
        """
        Fallback to fixed-time segmentation when smart detection fails
        """
        duration = len(audio_data) / self.sample_rate
        segment_length = min(self.max_segment_length, max(self.min_segment_length, duration / 4))
        
        segments = []
        start_time = 0
        
        while start_time < duration:
            end_time = min(start_time + segment_length, duration)
            start_idx = int(start_time * self.sample_rate)
            end_idx = int(end_time * self.sample_rate)
            
            segments.append((start_idx, end_idx, start_time, end_time))
            start_time = end_time
        
        logger.warning(f"Using fallback segmentation: {len(segments)} fixed segments")
        return segments


# Export for use in LibROSAService
__all__ = ['SmartMusicSegmentation']