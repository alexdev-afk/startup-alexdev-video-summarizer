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

    def generate_timeline_from_files(self, audio_json_path: Path, video_json_path: Path, video_name: str, output_path: Path):
        """
        Loads, merges, and formats audio and video timelines into a single markdown file.

        Args:
            audio_json_path: Path to the master_timeline.json file (audio).
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