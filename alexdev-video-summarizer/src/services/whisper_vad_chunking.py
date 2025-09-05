"""
Advanced VAD Chunking Strategies for Advertisement Content

This module provides multiple chunking strategies optimized for continuous speech
content like advertisements where traditional gap-based chunking fails.
"""

import numpy as np
from typing import Dict, Any, List
from utils.logger import get_logger

logger = get_logger(__name__)


class AdvancedVADChunking:
    """Advanced VAD chunking strategies for advertisement content"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize chunking strategies with configuration
        
        Args:
            config: Whisper service configuration containing chunking parameters
        """
        whisper_config = config.get('gpu_pipeline', {}).get('whisper', {})
        
        self.chunk_threshold = whisper_config.get('chunk_threshold', 3.0)
        self.chunking_strategy = whisper_config.get('chunking_strategy', 'multi_strategy')
        self.max_chunk_duration = whisper_config.get('max_chunk_duration', 20.0)
        self.min_chunk_duration = whisper_config.get('min_chunk_duration', 5.0)
        self.energy_based_splitting = whisper_config.get('energy_based_splitting', True)
    
    def chunk_by_gaps(self, speech_timestamps: List[Dict], wav: np.ndarray, VAD_SR: int, collect_chunks) -> List[Dict[str, Any]]:
        """
        Traditional gap-based chunking (original approach)
        
        Works well for content with clear pauses between segments.
        """
        logger.debug("Using gap-based chunking strategy")
        
        # Add padding and remove small gaps
        for i, timestamp in enumerate(speech_timestamps):
            timestamp["start"] = max(0, timestamp["start"] - int(0.2 * VAD_SR))
            timestamp["end"] = min(wav.shape[0] - 16, timestamp["end"] + int(1.3 * VAD_SR))
            
            if i > 0 and timestamp["start"] < speech_timestamps[i - 1]["end"]:
                timestamp["start"] = speech_timestamps[i - 1]["end"]
        
        # Group by gaps
        chunk_threshold_samples = int(self.chunk_threshold * VAD_SR)
        vad_chunks = [[]]
        
        for timestamp in speech_timestamps:
            if (len(vad_chunks[-1]) > 0 and 
                timestamp["start"] > vad_chunks[-1][-1]["end"] + chunk_threshold_samples):
                vad_chunks.append([])
            vad_chunks[-1].append(timestamp)
        
        return self._build_processed_chunks(vad_chunks, wav, VAD_SR, collect_chunks, "gap_based")
    
    def chunk_by_duration(self, speech_timestamps: List[Dict], wav: np.ndarray, VAD_SR: int, collect_chunks) -> List[Dict[str, Any]]:
        """
        Duration-based chunking - split based on maximum chunk duration regardless of gaps
        
        Perfect for advertisements where content is continuous and gaps are minimal.
        """
        logger.debug(f"Using duration-based chunking strategy (max: {self.max_chunk_duration}s)")
        
        # Add minimal padding for duration-based chunking
        for i, timestamp in enumerate(speech_timestamps):
            timestamp["start"] = max(0, timestamp["start"] - int(0.1 * VAD_SR))
            timestamp["end"] = min(wav.shape[0] - 16, timestamp["end"] + int(0.2 * VAD_SR))
            
            if i > 0 and timestamp["start"] < speech_timestamps[i - 1]["end"]:
                timestamp["start"] = speech_timestamps[i - 1]["end"]
        
        # Group by duration
        max_chunk_samples = int(self.max_chunk_duration * VAD_SR)
        vad_chunks = [[]]
        
        for timestamp in speech_timestamps:
            # Check if adding this timestamp would exceed max duration
            if vad_chunks[-1]:
                current_start = vad_chunks[-1][0]["start"]
                potential_end = timestamp["end"]
                current_duration = potential_end - current_start
                
                if current_duration > max_chunk_samples:
                    vad_chunks.append([])
            
            vad_chunks[-1].append(timestamp)
        
        return self._build_processed_chunks(vad_chunks, wav, VAD_SR, collect_chunks, "duration_based")
    
    def chunk_multi_strategy(self, speech_timestamps: List[Dict], wav: np.ndarray, VAD_SR: int, collect_chunks) -> List[Dict[str, Any]]:
        """
        Multi-strategy chunking optimized for advertisements:
        1. Energy-based splitting at natural breaks (breath pauses, music breaks)
        2. Duration-based splitting for continuous speech (max 20s chunks)
        3. Minimum duration constraints (min 5s chunks)
        4. Aggressive gap detection (0.5s instead of 3s)
        
        This is the recommended approach for advertisement content.
        """
        logger.debug(f"Using multi-strategy chunking (max: {self.max_chunk_duration}s, min: {self.min_chunk_duration}s)")
        
        # Add minimal padding for multi-strategy
        for i, timestamp in enumerate(speech_timestamps):
            timestamp["start"] = max(0, timestamp["start"] - int(0.1 * VAD_SR))
            timestamp["end"] = min(wav.shape[0] - 16, timestamp["end"] + int(0.2 * VAD_SR))
            
            if i > 0 and timestamp["start"] < speech_timestamps[i - 1]["end"]:
                timestamp["start"] = speech_timestamps[i - 1]["end"]
        
        # Apply multi-strategy chunking
        vad_chunks = [[]]
        max_chunk_samples = int(self.max_chunk_duration * VAD_SR)
        min_chunk_samples = int(self.min_chunk_duration * VAD_SR)
        
        # Use much smaller gap threshold for advertisements (0.5 seconds instead of 3.0)
        ad_gap_threshold_samples = int(0.5 * VAD_SR)
        
        for timestamp in speech_timestamps:
            should_split = False
            
            if vad_chunks[-1]:
                current_start = vad_chunks[-1][0]["start"]
                current_end = vad_chunks[-1][-1]["end"]
                current_duration = current_end - current_start
                
                # Strategy 1: Hard duration limit (prevents chunks over 20s)
                if timestamp["end"] - current_start > max_chunk_samples:
                    should_split = True
                    logger.debug(f"Splitting chunk due to duration limit: {(timestamp['end'] - current_start) / VAD_SR:.1f}s")
                
                # Strategy 2: Energy-based splitting at natural breaks (breath pauses, music transitions)
                elif (self.energy_based_splitting and 
                      current_duration > min_chunk_samples and
                      timestamp["start"] > current_end + ad_gap_threshold_samples):
                    
                    # Check for energy drop (potential natural break)
                    gap_start = int(current_end)
                    gap_end = int(timestamp["start"])
                    
                    if gap_end > gap_start and gap_end - gap_start > int(0.3 * VAD_SR):  # At least 0.3s gap
                        gap_audio = wav[gap_start:gap_end]
                        if len(gap_audio) > 0:
                            gap_energy = np.mean(gap_audio ** 2)
                            current_energy = np.mean(wav[int(current_start):int(current_end)] ** 2)
                            
                            # Split if there's a significant energy drop (natural break)
                            if gap_energy < 0.001 or (current_energy > 0 and gap_energy / current_energy < 0.1):
                                should_split = True
                                logger.debug(f"Splitting chunk due to energy drop: gap_energy={gap_energy:.6f}, current_energy={current_energy:.6f}")
                
                # Strategy 3: Moderate pause splitting (1 second instead of 3)
                elif timestamp["start"] > current_end + int(1.0 * VAD_SR):
                    should_split = True
                    logger.debug(f"Splitting chunk due to pause: {(timestamp['start'] - current_end) / VAD_SR:.1f}s")
            
            if should_split:
                vad_chunks.append([])
            
            vad_chunks[-1].append(timestamp)
        
        # Post-processing: Merge very short chunks if they exist
        processed_chunks = self._build_processed_chunks(vad_chunks, wav, VAD_SR, collect_chunks, "multi_strategy")
        return self._merge_short_chunks(processed_chunks, VAD_SR)
    
    def _merge_short_chunks(self, chunks: List[Dict[str, Any]], VAD_SR: int) -> List[Dict[str, Any]]:
        """
        Merge chunks that are shorter than minimum duration with adjacent chunks
        """
        if len(chunks) <= 1:
            return chunks
        
        min_duration_seconds = self.min_chunk_duration
        merged_chunks = []
        i = 0
        
        while i < len(chunks):
            current_chunk = chunks[i]
            
            # If chunk is too short and not the last chunk, try to merge with next
            if (current_chunk['duration'] < min_duration_seconds and 
                i < len(chunks) - 1):
                
                next_chunk = chunks[i + 1]
                
                # Merge with next chunk
                merged_chunk = {
                    'chunk_id': current_chunk['chunk_id'],
                    'start_seconds': current_chunk['start_seconds'],
                    'end_seconds': next_chunk['end_seconds'],
                    'duration': next_chunk['end_seconds'] - current_chunk['start_seconds'],
                    'audio_data': None,  # Will be regenerated
                    'offset': current_chunk['offset'],
                    'vad_segments': current_chunk['vad_segments'] + next_chunk['vad_segments'],
                    'sampling_rate': VAD_SR,
                    'chunking_strategy': 'multi_strategy_merged'
                }
                
                merged_chunks.append(merged_chunk)
                logger.debug(f"Merged short chunks: {current_chunk['duration']:.1f}s + {next_chunk['duration']:.1f}s = {merged_chunk['duration']:.1f}s")
                i += 2  # Skip the next chunk since we merged it
            else:
                merged_chunks.append(current_chunk)
                i += 1
        
        return merged_chunks
    
    def _build_processed_chunks(self, vad_chunks: List[List[Dict]], wav: np.ndarray, VAD_SR: int, 
                               collect_chunks, strategy: str) -> List[Dict[str, Any]]:
        """
        Build processed chunks from VAD chunk groups
        """
        processed_chunks = []
        
        for chunk_idx, chunk_timestamps in enumerate(vad_chunks):
            if not chunk_timestamps:
                continue
            
            try:
                # Collect audio data for this chunk
                chunk_audio = collect_chunks(chunk_timestamps, wav)
                
                # Calculate chunk timing in whole-file context
                chunk_start_seconds = chunk_timestamps[0]["start"] / VAD_SR
                chunk_end_seconds = chunk_timestamps[-1]["end"] / VAD_SR
                
                # Calculate offset tracking for timestamp reconstruction
                offset = chunk_start_seconds
                for i, ts in enumerate(chunk_timestamps):
                    if i > 0:
                        offset += (ts["start"] - chunk_timestamps[i-1]["end"]) / VAD_SR
                
                processed_chunks.append({
                    'chunk_id': chunk_idx,
                    'start_seconds': chunk_start_seconds,
                    'end_seconds': chunk_end_seconds,
                    'duration': chunk_end_seconds - chunk_start_seconds,
                    'audio_data': chunk_audio,
                    'offset': offset,
                    'vad_segments': [
                        {
                            'start': ts["start"] / VAD_SR,
                            'end': ts["end"] / VAD_SR
                        } for ts in chunk_timestamps
                    ],
                    'sampling_rate': VAD_SR,
                    'chunking_strategy': strategy
                })
                
                logger.debug(f"Built chunk {chunk_idx}: {chunk_start_seconds:.1f}s - {chunk_end_seconds:.1f}s ({chunk_end_seconds - chunk_start_seconds:.1f}s)")
                
            except Exception as e:
                logger.error(f"Failed to build chunk {chunk_idx}: {e}")
                continue
        
        return processed_chunks


# Export for use in WhisperService
__all__ = ['AdvancedVADChunking']