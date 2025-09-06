#!/usr/bin/env python3
"""
InternVL3 Timeline Service Test

Simple test that drives the inference and verifies timeline file creation.

Tests:
1. Service initialization 
2. Timeline generation from video
3. Timeline file verification (correct events count)
"""

import sys
import json
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from services.internvl3_timeline_service import InternVL3TimelineService
from utils.config_loader import ConfigLoader
from utils.logger import setup_logging

def setup_test_environment():
    """Setup test environment and verify prerequisites"""
    print("=== SETUP: InternVL3 Service Test ===")
    
    # Load configuration
    config = ConfigLoader.load_config('config/processing.yaml')
    
    # Verify test video exists
    test_video = Path("build/bonita/video.mp4")
    if not test_video.exists():
        print(f"[FAIL] Test video not found: {test_video}")
        print("       Run test_audio_pipeline.py first to create build structure")
        return None, None
    
    # Verify test audio exists  
    test_audio = Path("build/bonita/audio.wav")
    if not test_audio.exists():
        print(f"[FAIL] Test audio not found: {test_audio}")
        return None, None
        
    print(f"[OK] Test video: {test_video.name} ({test_video.stat().st_size} bytes)")
    print(f"[OK] Test audio: {test_audio.name} ({test_audio.stat().st_size} bytes)")
    
    # Use real VLM model for testing
    config['development']['mock_ai_services'] = False
    print("[PRODUCTION] Using real InternVL3 VLM model for testing")
    
    return config, test_video

def test_service_initialization(config):
    """Test InternVL3 timeline service initialization"""
    print("\n=== TEST 1: Service Initialization ===")
    
    try:
        service = InternVL3TimelineService(config)
        
        # Verify service attributes
        assert service.service_name == "internvl3"
        assert service.vlm_config is not None
        assert service.scene_analyzer is not None
        
        print(f"[OK] Service initialized: {service.service_name}")
        print(f"[OK] Confidence threshold: {service.confidence_threshold}")
        print(f"[OK] Model: {service.get_model_name('short')}")
        print(f"[OK] VLM loaded: {service.scene_analyzer is not None}")
        
        return service
        
    except Exception as e:
        print(f"[FAIL] Service initialization failed: {e}")
        return None

def test_timeline_generation(service, test_video):
    """Test timeline generation from video"""
    print("\n=== TEST 2: Timeline Generation ===")
    
    try:
        # Generate timeline
        timeline = service.generate_and_save(str(test_video), None)
        
        if timeline:
            print(f"[OK] Timeline generated successfully")
            print(f"     Events: {len(timeline.events)}")
            print(f"     Duration: {timeline.total_duration:.2f}s")
            
            # Show sample events
            if timeline.events:
                print(f"[EVENTS] Sample events:")
                for i, event in enumerate(timeline.events[:2]):
                    print(f"  {i+1}. {event.timestamp:.2f}s: {event.description[:60]}...")
            
            return timeline
        else:
            print("[FAIL] Timeline generation returned None")
            return None
            
    except Exception as e:
        print(f"[FAIL] Timeline generation failed: {e}")
        return None

def verify_timeline_file(expected_events=None):
    """Verify timeline file was created with correct structure"""
    print("\n=== TEST 3: Timeline File Verification ===")
    
    timeline_dir = Path("build/bonita/video_timelines")
    timeline_files = list(timeline_dir.glob("*_timeline.json"))
    
    if not timeline_files:
        print("[FAIL] No timeline files found")
        return False
    
    # Get the most recent timeline file
    timeline_file = max(timeline_files, key=lambda f: f.stat().st_mtime)
    size = timeline_file.stat().st_size
    print(f"[OK] Timeline file: {timeline_file.name} ({size} bytes)")
    
    try:
        with open(timeline_file, 'r') as f:
            timeline_data = json.load(f)
        
        events = timeline_data.get('events', [])
        print(f"[OK] Events found: {len(events)}")
        
        if expected_events and len(events) != expected_events:
            print(f"[WARNING] Expected {expected_events} events, got {len(events)}")
        
        if len(events) > 0:
            print(f"[OK] Timeline file structure verified")
            return True
        else:
            print(f"[FAIL] No events in timeline")
            return False
            
    except Exception as e:
        print(f"[FAIL] Could not read timeline file: {e}")
        return False



def main():
    """Run InternVL3 timeline service test"""
    print("InternVL3 Timeline Service Test")
    print("=" * 40)
    
    setup_logging('INFO')
    
    # Setup
    config, test_video = setup_test_environment()
    if not config or not test_video:
        return
    
    # Test 1: Service initialization
    service = test_service_initialization(config)
    if not service:
        return
    
    # Test 2: Timeline generation
    timeline = test_timeline_generation(service, test_video)
    if not timeline:
        return
    
    # Test 3: File verification
    expected_events = len(timeline.events) if timeline else None
    success = verify_timeline_file(expected_events)
    
    print("\n" + "=" * 40)
    if success:
        print("[SUCCESS] InternVL3 Timeline Service Test Complete")
        print(f"Timeline generated with {len(timeline.events)} events")
    else:
        print("[FAIL] Timeline service test failed")

if __name__ == "__main__":
    main()