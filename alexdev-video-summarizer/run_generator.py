import sys
import yaml
from pathlib import Path

# Add src to the Python path
sys.path.append(str(Path(__file__).parent / 'src'))

from services.knowledge_generator import KnowledgeGenerator

# Define paths
CONFIG_PATH = Path('config/processing.yaml')
AUDIO_JSON_PATH = Path('build/bonita/audio_timelines/master_timeline.json')
VIDEO_JSON_PATH = Path('build/bonita/video_timelines/InternVL3-5-2B_2025-09-07_01-40_timeline.json')
OUTPUT_PATH = Path('build/bonita/bonita_output.md')
VIDEO_NAME = 'bonita'

# Load config
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# Instantiate and run the generator
generator = KnowledgeGenerator(config)
generator.generate_timeline_from_files(AUDIO_JSON_PATH, VIDEO_JSON_PATH, VIDEO_NAME, OUTPUT_PATH)

print(f"Merged timeline generated for {VIDEO_NAME} at {OUTPUT_PATH}")
