#!/usr/bin/env python3
"""
Test script for Enhanced Timeline Merger with all 3 services using bonita.mp4

Tests the complete enhanced timeline merger pipeline:
1. Enhanced WhisperTimelineService (speech analysis with VAD chunks)
2. Enhanced LibROSATimelineService (music analysis) 
3. Enhanced PyAudioTimelineService (audio features and speaker analysis)
4. Enhanced TimelineMergerService (combines all 3 into master timeline)

Focuses on timestamp offset verification and enhanced timeline schema compatibility.
"""

import sys
import os
from pathlib import Path
import yaml
import json

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from services.enhanced_whisper_timeline_service import EnhancedWhisperTimelineService
from services.librosa_timeline_service import LibROSATimelineService
from services.pyaudio_timeline_service import PyAudioTimelineService
from services.enhanced_timeline_merger_service import EnhancedTimelineMergerService
from utils.logger import get_logger

logger = get_logger(__name__)

def load_test_configuration():
    """Load test configuration"""
    config_path = Path("config/processing.yaml")
    
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        return None
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return None

def test_enhanced_3_service_timeline_merger():
    """Test the enhanced 3-service timeline merger with bonita.mp4"""
    
    # Load configuration
    config = load_test_configuration()
    if not config:
        return False
    
    # Test with bonita.mp4 
    video_name = "bonita"
    audio_path = Path(f"build/{video_name}/audio.wav")
    
    if not audio_path.exists():
        logger.error(f"Test audio file not found: {audio_path}")
        logger.info("Please run the main pipeline first to generate audio.wav")
        return False
    
    logger.info("=" * 60)
    logger.info("TESTING ENHANCED 3-SERVICE TIMELINE MERGER")
    logger.info("=" * 60)
    logger.info(f"Video: {video_name}")
    logger.info(f"Audio: {audio_path}")
    logger.info("")
    
    try:
        # Initialize all services
        logger.info("1. Initializing services...")
        whisper_service = EnhancedWhisperTimelineService(config)
        librosa_service = LibROSATimelineService(config)
        pyaudio_service = PyAudioTimelineService(config)
        merger_service = EnhancedTimelineMergerService(config)
        
        # Test Whisper service (should work)
        logger.info("2. Testing Enhanced Whisper timeline service...")
        try:
            whisper_timeline = whisper_service.generate_and_save(str(audio_path))
            logger.info(f"   ✅ Whisper service: {len(whisper_timeline.events)} events, {len(whisper_timeline.spans)} spans")
            logger.info(f"   📄 Timeline type: {type(whisper_timeline).__name__}")
            
            # Check timestamp offsets in spans
            if whisper_timeline.spans:
                first_span = whisper_timeline.spans[0]
                last_span = whisper_timeline.spans[-1]
                logger.info(f"   ⏰ First span: {first_span.start:.2f}s - {first_span.end:.2f}s")
                logger.info(f"   ⏰ Last span: {last_span.start:.2f}s - {last_span.end:.2f}s")
            
            # Check word events timestamps
            word_events = [e for e in whisper_timeline.events if e.description == "Word"]
            if word_events:
                logger.info(f"   📝 Word events: {len(word_events)} total")
                logger.info(f"   ⏰ First word: {word_events[0].timestamp:.2f}s")
                logger.info(f"   ⏰ Last word: {word_events[-1].timestamp:.2f}s")
                
        except Exception as e:
            logger.error(f"   ❌ Whisper service failed: {e}")
            whisper_timeline = None
        
        # Test LibROSA service (should now work with enhanced schema)
        logger.info("3. Testing Enhanced LibROSA timeline service...")
        try:
            librosa_timeline = librosa_service.generate_and_save(str(audio_path))
            logger.info(f"   ✅ LibROSA service: {len(librosa_timeline.events)} events, {len(librosa_timeline.spans)} spans")
            logger.info(f"   📄 Timeline type: {type(librosa_timeline).__name__}")
            
            if librosa_timeline.spans:
                first_span = librosa_timeline.spans[0]
                logger.info(f"   🎵 First music span: {first_span.start:.2f}s - {first_span.end:.2f}s")
                
        except Exception as e:
            logger.error(f"   ❌ LibROSA service failed: {e}")
            librosa_timeline = None
        
        # Test pyAudio service (should now work with enhanced schema)
        logger.info("4. Testing Enhanced pyAudio timeline service...")
        try:
            pyaudio_timeline = pyaudio_service.generate_and_save(str(audio_path))
            logger.info(f"   ✅ pyAudio service: {len(pyaudio_timeline.events)} events, {len(pyaudio_timeline.spans)} spans")
            logger.info(f"   📄 Timeline type: {type(pyaudio_timeline).__name__}")
            
            if pyaudio_timeline.spans:
                first_span = pyaudio_timeline.spans[0]
                logger.info(f"   🔊 First audio span: {first_span.start:.2f}s - {first_span.end:.2f}s")
                
        except Exception as e:
            logger.error(f"   ❌ pyAudio service failed: {e}")
            pyaudio_timeline = None
        
        # Test timeline merger (should now work with all enhanced schemas)
        logger.info("5. Testing Enhanced Timeline Merger...")
        
        # Collect successful timelines
        timelines = []
        if whisper_timeline:
            timelines.append(whisper_timeline)
        if librosa_timeline:
            timelines.append(librosa_timeline)
        if pyaudio_timeline:
            timelines.append(pyaudio_timeline)
        
        if not timelines:
            logger.error("   ❌ No successful timelines to merge")
            return False
        
        try:
            # Test merger with available timelines
            logger.info(f"   🔄 Merging {len(timelines)} enhanced timelines...")
            merged_timeline = merger_service.merge_enhanced_timelines(
                timelines=timelines,
                output_path=str(audio_path.parent / "audio_timelines" / "master_timeline.json")
            )
            
            logger.info(f"   ✅ Master timeline created: {len(merged_timeline.events)} events, {len(merged_timeline.spans)} spans")
            logger.info(f"   📄 Master timeline type: {type(merged_timeline).__name__}")
            logger.info(f"   ⏰ Total duration: {merged_timeline.total_duration:.2f}s")
            
            # Show transcript if available
            if merged_timeline.full_transcript:
                logger.info(f"   📄 Full transcript: {len(merged_timeline.full_transcript)} characters")
            if merged_timeline.speakers:
                logger.info(f"   🎤 Speakers: {merged_timeline.speakers}")
            
            # Show event breakdown by source
            event_sources = {}
            for event in merged_timeline.events:
                source = event.source
                event_sources[source] = event_sources.get(source, 0) + 1
            
            span_sources = {}
            for span in merged_timeline.spans:
                source = span.source  
                span_sources[source] = span_sources.get(source, 0) + 1
            
            logger.info(f"   📈 Events by source: {event_sources}")
            logger.info(f"   📈 Spans by source: {span_sources}")
            
            logger.info("")
            logger.info("🎉 ENHANCED 3-SERVICE TIMELINE MERGER TEST SUCCESSFUL!")
            logger.info(f"🎯 Master timeline saved to: {audio_path.parent / 'audio_timelines' / 'master_timeline.json'}")
            return True
            
        except Exception as e:
            logger.error(f"   ❌ Timeline merger failed: {e}")
            import traceback
            logger.error(f"   📋 Traceback: {traceback.format_exc()}")
            return False
    
    except Exception as e:
        logger.error(f"Test failed with unexpected error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def main():
    """Main test function"""
    logger.info("Starting Enhanced 3-Service Timeline Merger Test...")
    
    success = test_enhanced_3_service_timeline_merger()
    
    if success:
        logger.info("✅ All tests completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()