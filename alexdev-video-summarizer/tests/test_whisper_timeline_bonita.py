"""
Test Whisper Timeline Service with bonita.mp4

Generate timeline output from bonita.mp4 and create merged_timeline.json
using the TimelineMergerService integration.
"""

import sys
from pathlib import Path
import yaml
import json

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from services.whisper_timeline_service import WhisperTimelineService
from services.timeline_merger_service import TimelineMergerService
from utils.logger import get_logger

logger = get_logger(__name__)


def load_config():
    """Load configuration from config files"""
    try:
        # Load main processing config
        config_path = Path("config/processing.yaml")
        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            return None
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Ensure development mode for testing
        if 'development' not in config:
            config['development'] = {}
        config['development']['mock_ai_services'] = False  # Use real Whisper
        
        # Add timeline service config
        if 'whisper_timeline' not in config:
            config['whisper_timeline'] = {}
        
        config['whisper_timeline'].update({
            'generate_word_events': True,
            'generate_vad_spans': True,
            'generate_speaker_events': True,
            'word_confidence_threshold': 0.3
        })
        
        return config
        
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return None


def find_bonita_audio():
    """Find the bonita audio file in build directory"""
    build_dir = Path("build")
    if not build_dir.exists():
        logger.error("Build directory not found")
        return None
    
    # Look for any directory containing bonita
    for video_dir in build_dir.iterdir():
        if video_dir.is_dir() and "bonita" in video_dir.name.lower():
            audio_file = video_dir / "audio.wav"
            if audio_file.exists():
                logger.info(f"Found bonita audio: {audio_file}")
                return audio_file
    
    logger.error("bonita audio.wav not found in build directory")
    return None


def test_whisper_timeline():
    """Test Whisper Timeline Service with bonita.mp4"""
    logger.info("=== Testing Whisper Timeline Service with bonita.mp4 ===")
    
    # Load configuration
    config = load_config()
    if not config:
        return False
    
    # Find bonita audio file
    audio_path = find_bonita_audio()
    if not audio_path:
        return False
    
    try:
        # Initialize services
        logger.info("Initializing Whisper Timeline Service...")
        whisper_timeline_service = WhisperTimelineService(config)
        
        logger.info("Initializing Timeline Merger Service...")
        merger_service = TimelineMergerService(config)
        
        # Generate Whisper timeline
        logger.info("Generating Whisper timeline...")
        whisper_timeline_path = audio_path.parent / "whisper_timeline.json"
        whisper_timeline = whisper_timeline_service.generate_and_save(
            audio_path, 
            str(whisper_timeline_path)
        )
        
        logger.info(f"Whisper timeline generated: {len(whisper_timeline.events)} events, {len(whisper_timeline.spans)} spans")
        
        # Create merged timeline (with only Whisper for now)
        logger.info("Creating merged timeline...")
        merged_timeline_path = audio_path.parent / "merged_timeline.json"
        
        # Get audio duration from timeline
        total_duration = whisper_timeline.total_duration
        
        merged_timeline = merger_service.merge_and_save(
            output_path=str(merged_timeline_path),
            whisper_timeline_path=str(whisper_timeline_path),
            librosa_timeline_path=None,  # Not implemented yet
            pyaudio_timeline_path=None,  # Not implemented yet
            audio_file=audio_path.name,
            total_duration=total_duration
        )
        
        # Display results
        logger.info("=== RESULTS ===")
        logger.info(f"Whisper Timeline: {whisper_timeline_path}")
        logger.info(f"Merged Timeline: {merged_timeline_path}")
        
        # Show timeline summary
        merged_dict = merged_timeline.to_dict()
        logger.info(f"Total segments: {len(merged_dict['merged_timeline'])}")
        logger.info(f"Total duration: {total_duration:.2f}s")
        
        # Print first few segments for preview
        logger.info("\n=== MERGED TIMELINE PREVIEW ===")
        for i, segment in enumerate(merged_dict['merged_timeline'][:5]):
            logger.info(f"Segment {i+1}: {segment['time_range']}")
            for content in segment['content']:
                logger.info(f"  - {content['description']} [{content['source']}]")
        
        if len(merged_dict['merged_timeline']) > 5:
            logger.info(f"... and {len(merged_dict['merged_timeline']) - 5} more segments")
        
        # Save pretty-printed version for easy viewing
        pretty_output_path = audio_path.parent / "merged_timeline_pretty.json"
        with open(pretty_output_path, 'w', encoding='utf-8') as f:
            json.dump(merged_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\nPretty-printed version saved to: {pretty_output_path}")
        logger.info("=== TEST COMPLETE ===")
        
        # Cleanup
        whisper_timeline_service.cleanup()
        merger_service.cleanup()
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_whisper_timeline()
    if success:
        print("✅ Whisper Timeline test completed successfully")
    else:
        print("❌ Whisper Timeline test failed")
        sys.exit(1)