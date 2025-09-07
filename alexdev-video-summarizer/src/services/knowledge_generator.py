import json
from pathlib import Path
from typing import Dict, Any, List

from utils.logger import get_logger

logger = get_logger(__name__)

class KnowledgeGenerator:
    """
    Generates a minimal, human-readable text timeline by merging audio and video timelines.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the KnowledgeGenerator.
        Args:
            config: The application configuration dictionary.
        """
        self.config = config
        self.output_dir = Path(self.config.get('paths', {}).get('output_dir', 'output'))
        self.output_dir.mkdir(exist_ok=True)
        logger.info("Knowledge Generator initialized.")
        
    def generate_combined_knowledge(self, audio_json_path: Path, video_timeline_sources: Dict[str, Path], video_name: str, output_path: Path):
        """
        Generate a single comprehensive knowledge file combining all timeline sources.
        
        Args:
            audio_json_path: Path to the combined_audio_timeline.json file
            video_timeline_sources: Dict mapping source names to timeline file paths
                                   e.g. {'internvl3': path, 'vid2seq': path}
            video_name: Name of the video for the report title
            output_path: Output path for the combined knowledge file
        """
        # Load audio timeline
        if not audio_json_path.exists():
            logger.error(f"Audio timeline file not found at: {audio_json_path}")
            return
        with open(audio_json_path, 'r', encoding='utf-8') as f:
            audio_timeline = json.load(f)
        
        # Load all available video timelines
        video_timelines = {}
        for source_name, timeline_path in video_timeline_sources.items():
            if timeline_path and timeline_path.exists():
                try:
                    with open(timeline_path, 'r', encoding='utf-8') as f:
                        video_timelines[source_name] = json.load(f)
                        logger.info(f"Loaded {source_name} timeline: {timeline_path}")
                except Exception as e:
                    logger.warning(f"Failed to load {source_name} timeline from {timeline_path}: {e}")
        
        if not video_timelines:
            logger.error("No video timelines could be loaded")
            return
            
        # Generate combined markdown
        markdown_content = self.format_combined_markdown(audio_timeline, video_timelines, video_name)
        
        # Save combined knowledge file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
            
        logger.info(f"Generated combined knowledge file: {output_path}")

    def generate_timeline_from_files(self, audio_json_path: Path, video_json_path: Path, video_name: str, output_path: Path):
        """
        Loads, merges, and formats audio and video timelines into a single markdown file.

        Args:
            audio_json_path: Path to the combined_audio_timeline.json file (audio).
            video_json_path: Path to the video timeline JSON file (VLM).
            video_name: The name of the video for the report title.
            output_path: The full path for the output markdown file.
        """
        # Load Audio Timeline
        if not audio_json_path.exists():
            logger.error(f"Audio timeline file not found at: {audio_json_path}")
            return
        with open(audio_json_path, 'r', encoding='utf-8') as f:
            audio_timeline = json.load(f)

        # Load Video Timeline
        if not video_json_path.exists():
            logger.error(f"Video timeline file not found at: {video_json_path}")
            return
        with open(video_json_path, 'r', encoding='utf-8') as f:
            video_timeline = json.load(f)

        markdown_content = self.format_as_markdown(audio_timeline, video_timeline, video_name)

        # Ensure the output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)

        logger.info(f"Successfully generated merged timeline: {output_path}")

    def format_combined_markdown(self, audio_data: Dict[str, Any], video_timelines: Dict[str, Dict[str, Any]], video_name: str) -> str:
        """
        Format combined timeline data from multiple sources into comprehensive markdown.
        
        Args:
            audio_data: Combined audio timeline data
            video_timelines: Dict of video timeline data by source name
            video_name: Name of the video
        """
        lines = []
        lines.append(f"# Comprehensive Analysis: {video_name}.mp4")
        lines.append("")
        lines.append("*Multi-modal institutional knowledge extraction combining audio analysis with advanced video understanding*")
        lines.append("")
        
        # Analysis sources summary
        lines.append("## Analysis Sources")
        lines.append("")
        lines.append("**Audio Analysis:**")
        lines.append("- Whisper transcription with speaker diarization")
        lines.append("- LibROSA musical analysis (harmonic shifts, tempo changes)")
        lines.append("- Clean audio separation via Demucs")
        lines.append("")
        lines.append("**Video Analysis:**")
        for source_name, timeline_data in video_timelines.items():
            model_info = timeline_data.get('model_info', {})
            if source_name == 'internvl3' or 'contextual' in source_name or 'noncontextual' in source_name:
                lines.append(f"- InternVL3 VLM: Scene-by-scene visual understanding")
            elif source_name == 'vid2seq':
                lines.append(f"- Vid2Seq: Dense video captioning with temporal localization")
        lines.append("")
        
        # Full transcript section
        if 'events' in audio_data:
            lines.append("## Full Transcript")
            lines.append("")
            for event in audio_data['events']:
                if event.get('source') == 'whisper_voice':
                    timestamp = event['timestamp']
                    description = event['description']
                    lines.append(f"**{timestamp:.1f}s**: {description}")
            lines.append("")
        
        # Combined scene analysis
        lines.append("## Scene-by-Scene Analysis")
        lines.append("")
        lines.append("*Integrated analysis combining visual understanding with audio context*")
        lines.append("")
        
        # Merge all video timeline events by timestamp
        all_video_events = []
        for source_name, timeline_data in video_timelines.items():
            if 'events' in timeline_data:
                for event in timeline_data['events']:
                    event['source_name'] = source_name
                    all_video_events.append(event)
            elif 'spans' in timeline_data:  # Handle spans from InternVL3
                for span in timeline_data['spans']:
                    # Convert span to event format
                    event = {
                        'timestamp': span['start'],
                        'description': span['description'],
                        'source': span.get('source', source_name),
                        'source_name': source_name,
                        'confidence': span.get('confidence', 0.0)
                    }
                    all_video_events.append(event)
        
        # Sort by timestamp
        all_video_events.sort(key=lambda x: x['timestamp'])
        
        # Group events by approximate time windows (5-second windows)
        current_window_start = 0
        window_size = 5.0
        
        while current_window_start < max([e['timestamp'] for e in all_video_events] + [0]):
            window_end = current_window_start + window_size
            window_events = [e for e in all_video_events 
                           if current_window_start <= e['timestamp'] < window_end]
            
            if window_events:
                lines.append(f"### {current_window_start:.0f}s - {window_end:.0f}s")
                lines.append("")
                
                # Group by source
                internvl3_events = [e for e in window_events if 'internvl3' in e['source_name'] or 'contextual' in e['source_name'] or 'noncontextual' in e['source_name']]
                vid2seq_events = [e for e in window_events if 'vid2seq' in e['source_name']]
                
                if internvl3_events:
                    lines.append("**Visual Scene Analysis (InternVL3):**")
                    for event in internvl3_events[:2]:  # Limit to avoid redundancy
                        lines.append(f"- {event['description']}")
                    lines.append("")
                
                if vid2seq_events:
                    lines.append("**Dense Video Caption (Vid2Seq):**")
                    for event in vid2seq_events[:1]:  # Usually one caption per window
                        lines.append(f"- {event['description']}")
                    lines.append("")
                
                lines.append("")
            
            current_window_start = window_end
        
        # Musical events section
        if 'events' in audio_data:
            musical_events = [e for e in audio_data['events'] if 'Harmonic Shift' in e['description'] or 'Tempo Change' in e['description']]
            if musical_events:
                lines.append("## Musical Analysis")
                lines.append("")
                for event in musical_events:
                    timestamp = event['timestamp'] 
                    description = event['description']
                    lines.append(f"**{timestamp:.1f}s**: {description}")
                lines.append("")
        
        # Technical metadata
        lines.append("## Technical Metadata")
        lines.append("")
        processing_date = json.dumps(audio_data.get('processing_timestamp', 'Unknown'), default=str).strip('"')
        lines.append(f"- **Processing Date**: {processing_date}")
        lines.append(f"- **Total Duration**: {audio_data.get('total_duration', 0):.1f} seconds")
        
        # Video analysis metadata
        for source_name, timeline_data in video_timelines.items():
            model_info = timeline_data.get('model_info', {})
            if model_info:
                lines.append(f"- **{source_name.title()} Events**: {model_info.get('event_count', 0)}")
        
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("*Generated by alexdev-video-summarizer multi-modal pipeline*")
        
        return "\n".join(lines)

    def format_as_markdown(self, audio_data: Dict[str, Any], video_data: Dict[str, Any], video_name: str) -> str:
        """
        Formats the loaded timeline data into the specified markdown structure.
        """
        lines = []
        lines.append(f"# Summary for video: {video_name}.mp4")
        lines.append("")

        # Add the full transcript
        full_transcript = audio_data.get("global_data", {}).get("full_transcript", "")
        if full_transcript:
            lines.append("## Full Transcript")
            lines.append(full_transcript)
            lines.append("")

        lines.append("## Detailed Event Timeline")
        lines.append("")

        # Consolidate all events
        all_events = []
        # Add audio events
        if "timeline_spans" in audio_data:
            for span in audio_data["timeline_spans"]:
                if "events" in span:
                    all_events.extend(span["events"])
        if "events" in audio_data:
            all_events.extend(audio_data["events"])
            
        # Add video events
        if "events" in video_data:
            all_events.extend(video_data["events"])

        # Sort all events by timestamp
        all_events.sort(key=lambda x: x.get('timestamp', 0))

        for event in all_events:
            timestamp = event.get('timestamp', 0.0)
            source = event.get('source', 'unknown')
            description = event.get('description', '')
            details = event.get('details', {})

            # Skip speaker change events
            if description == "Speaker Change":
                continue

            m, s = divmod(timestamp, 60)
            time_str = f"[{int(m):02d}:{s:05.2f}]"

            line = f"{time_str} - ({source}) "
            
            if description == "Word":
                speaker = details.get('speaker', 'UNKNOWN')
                word = details.get('word', '')
                line = f"{time_str} - [{speaker}] {word}"
            elif source == "internvl3":
                line = f"{time_str} - [VLM] {description}"
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

            lines.append(line)

        return "\n".join(lines)