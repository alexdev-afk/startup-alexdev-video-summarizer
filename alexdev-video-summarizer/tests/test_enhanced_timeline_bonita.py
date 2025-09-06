"""
Test Enhanced Whisper Timeline Service with bonita.mp4

Generate clean hierarchical timeline with proper file organization:
- Intermediate analysis in audio_analysis/
- Timeline files in audio_timelines/
- No duplicate/invalid data
- Clean nested structure
"""

import sys
from pathlib import Path
import yaml
import json

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from services.enhanced_whisper_timeline_service import EnhancedWhisperTimelineService
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
        
        # Add enhanced timeline service config
        if 'whisper_timeline' not in config:
            config['whisper_timeline'] = {}
        
        config['whisper_timeline'].update({
            'generate_word_events': True,
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


def test_enhanced_timeline():
    """Test Enhanced Whisper Timeline Service with bonita.mp4"""
    logger.info("=== Testing Enhanced Whisper Timeline Service ===")
    
    # Load configuration
    config = load_config()
    if not config:
        return False
    
    # Find bonita audio file
    audio_path = find_bonita_audio()
    if not audio_path:
        return False
    
    try:
        # Initialize enhanced service
        logger.info("Initializing Enhanced Whisper Timeline Service...")
        enhanced_service = EnhancedWhisperTimelineService(config)
        
        # Generate enhanced timeline with proper file organization
        logger.info("Generating enhanced timeline with clean structure...")
        enhanced_timeline = enhanced_service.generate_and_save(audio_path)
        
        # Display results
        timeline_dict = enhanced_timeline.to_dict()
        
        logger.info("=== ENHANCED TIMELINE RESULTS ===")
        logger.info(f"Duration: {timeline_dict['global_data']['duration']}s")
        logger.info(f"Full transcript: {timeline_dict['global_data']['full_transcript'][:100]}...")
        logger.info(f"Speakers: {timeline_dict['global_data']['speakers']}")
        logger.info(f"Total spans: {timeline_dict['metadata']['total_spans']}")
        logger.info(f"Total events: {timeline_dict['metadata']['total_events']}")
        
        # Show file organization
        build_dir = audio_path.parent
        analysis_dir = build_dir / "audio_analysis"
        timelines_dir = build_dir / "audio_timelines"
        
        logger.info(f"\n=== FILE ORGANIZATION ===")
        logger.info(f"Intermediate analysis: {analysis_dir}")
        if analysis_dir.exists():
            for file in analysis_dir.glob("*.json"):
                logger.info(f"  - {file.name}")
        
        logger.info(f"Timeline files: {timelines_dir}")  
        if timelines_dir.exists():
            for file in timelines_dir.glob("*.json"):
                logger.info(f"  - {file.name}")
        
        # Preview clean structure
        logger.info(f"\n=== CLEAN STRUCTURE PREVIEW ===")
        spans = timeline_dict.get('timeline_spans', [])
        for i, span in enumerate(spans[:3]):  # Show first 3 spans
            logger.info(f"Span {i+1}: {span['start']}-{span['end']}s")
            logger.info(f"  Description: {span['description']}")
            logger.info(f"  Source: {span['source']}")
            if 'events' in span:
                logger.info(f"  Events: {len(span['events'])}")
                # Show first event as example
                if span['events']:
                    event = span['events'][0]
                    logger.info(f"    - {event['timestamp']}s: {event['description']}")
            logger.info("")
        
        if len(spans) > 3:
            logger.info(f"... and {len(spans) - 3} more spans")
        
        # Show the clean JSON structure
        main_timeline_path = timelines_dir / "whisper_timeline.json"
        logger.info(f"\n=== MAIN TIMELINE FILE ===")
        logger.info(f"Path: {main_timeline_path}")
        
        # Validate clean structure
        logger.info(f"\n=== STRUCTURE VALIDATION ===")
        required_sections = ['global_data', 'timeline_spans', 'metadata']
        for section in required_sections:
            if section in timeline_dict:
                logger.info(f"✅ {section}: Present")
            else:
                logger.info(f"❌ {section}: Missing")
        
        # Check for invalid data patterns
        has_invalid_data = False
        for span in spans:
            if span.get('source') != 'whisper':
                logger.info(f"❌ Invalid source in span: {span.get('source')}")
                has_invalid_data = True
            
            events = span.get('events', [])
            for event in events:
                if 'start' in event or 'end' in event:
                    logger.info(f"❌ Event has start/end times (should only have timestamp)")
                    has_invalid_data = True
        
        if not has_invalid_data:
            logger.info("✅ No invalid data patterns detected")
        
        logger.info("=== ENHANCED TIMELINE TEST COMPLETE ===")
        
        # Cleanup
        enhanced_service.cleanup()
        
        return True
        
    except Exception as e:
        logger.error(f"Enhanced timeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_enhanced_timeline()
    if success:
        print("Enhanced Timeline test completed successfully")
    else:
        print("Enhanced Timeline test failed")
        sys.exit(1)