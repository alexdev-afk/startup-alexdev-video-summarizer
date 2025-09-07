"""
Audio Context Generator

Creates LLM-ready audio context summaries from combined audio timelines
for use in VLM frame analysis prompts.
"""

import json
from pathlib import Path
from typing import Dict, Any, List

from utils.logger import get_logger

logger = get_logger(__name__)


class AudioContextGenerator:
    """Generates concise audio context summaries for VLM frame analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def create_audio_context_from_timeline(self, combined_timeline_path: Path, context_output_path: Path) -> str:
        """
        Create audio context summary from combined audio timeline
        
        Args:
            combined_timeline_path: Path to combined_audio_timeline.json
            context_output_path: Path to save audio_context.txt
            
        Returns:
            Audio context summary as string
        """
        if not combined_timeline_path.exists():
            logger.error(f"Combined audio timeline not found: {combined_timeline_path}")
            return ""
            
        # Load combined timeline
        with open(combined_timeline_path, 'r', encoding='utf-8') as f:
            timeline_data = json.load(f)
            
        # Extract key information
        full_transcript = timeline_data.get('global_data', {}).get('full_transcript', '')
        duration = timeline_data.get('global_data', {}).get('duration', 0)
        
        # Collect all events
        all_events = []
        
        # Extract events from timeline_spans
        for span in timeline_data.get('timeline_spans', []):
            if 'events' in span:
                all_events.extend(span['events'])
                
        # Extract direct events if any
        if 'events' in timeline_data:
            all_events.extend(timeline_data['events'])
            
        # Sort by timestamp
        all_events.sort(key=lambda x: x.get('timestamp', 0))
        
        # Create time-based context summary
        context_summary = self._create_contextual_summary(all_events, full_transcript, duration)
        
        # Save to file
        context_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(context_output_path, 'w', encoding='utf-8') as f:
            f.write(context_summary)
            
        logger.info(f"Audio context summary created: {context_output_path}")
        return context_summary
        
    def _create_contextual_summary(self, events: List[Dict], transcript: str, duration: float) -> str:
        """Create contextual summary in exact knowledge generator timeline format"""
        
        # Sort all events by timestamp (same as knowledge generator)
        events.sort(key=lambda x: x.get('timestamp', 0))
        
        summary_lines = [
            f"Full Transcript: {transcript}",
            ""
        ]
        
        # Process events in exact knowledge generator format
        for event in events:
            timestamp = event.get('timestamp', 0.0)
            source = event.get('source', 'unknown')
            description = event.get('description', '')
            details = event.get('details', {})

            # Skip speaker change events (same as knowledge generator)
            if description == "Speaker Change":
                continue
                
            # Skip pyaudio and librosa events
            if source in ['pyaudio_voice', 'pyaudio_music', 'librosa_music']:
                continue

            m, s = divmod(timestamp, 60)
            time_str = f"[{int(m):02d}:{s:05.2f}]"

            line = f"{time_str} - ({source}) "
            
            if description == "Word":
                word = details.get('word', '')
                line = f"{time_str} - {word}"
            elif description == "Emotion Change":
                prev = details.get('previous_emotion', 'unknown')
                new = details.get('emotion', 'unknown')
                line += f"Emotion Change: {prev} -> {new}"
            elif description == "Musical Onset":
                character = details.get('onset_character', 'N/A')
                line += f"Musical Onset: {character}"
            elif description == "Harmonic Shift":
                prev_key = details.get('previous_key', '?')
                new_key = details.get('new_key', '?')
                line += f"Harmonic Shift: {prev_key} -> {new_key}"
            elif description == "Tempo Change":
                prev_tempo = int(details.get('previous_tempo', 0))
                new_tempo = int(details.get('new_tempo', 0))
                line += f"Tempo Change: {prev_tempo} BPM -> {new_tempo} BPM"
            else:
                line += description
                
            summary_lines.append(line)
                
        return '\n'.join(summary_lines)