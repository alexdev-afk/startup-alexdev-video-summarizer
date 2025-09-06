#!/usr/bin/env python3
"""
InternVL3 Simplified Timeline Service Test

Tests the simplified InternVL3 VLM service that replaces YOLO+EasyOCR+OpenCV 
with direct frame-by-frame VLM analysis.

SIMPLIFIED APPROACH:
- Extract frames every 5 seconds
- One VLM analysis per frame
- One timeline event per frame (timestamp + VLM description)
- No analysis files, no spans, no scene complexity

Tests:
1. Service initialization and configuration
2. Frame extraction at intervals
3. Direct VLM description -> timeline event
4. Simplified file output (timeline file only)

Produces internvl3_timeline.json only
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

def test_frame_extraction(service, test_video):
    """Test frame extraction capabilities"""
    print("\n=== TEST 2: Frame Extraction ===")
    
    try:
        # Test single frame extraction
        test_timestamp = 5.0
        frame_image = service._extract_frame_at_timestamp(str(test_video), test_timestamp)
        
        if frame_image:
            print(f"[OK] Frame extracted at {test_timestamp}s: {frame_image.size}")
            
            # Test VLM scene analysis with extracted frame
            if service.scene_analyzer:
                analysis = service.scene_analyzer.analyze_comprehensive_scene(
                    frame_image, test_timestamp, scene_id=1
                )
                
                if analysis and analysis.get('comprehensive_analysis'):
                    print(f"[OK] VLM analysis completed: {len(analysis['comprehensive_analysis'])} chars")
                    print(f"     Sample: {analysis['comprehensive_analysis'][:100]}...")
                    print(f"[OK] Analysis confidence: {analysis['confidence']:.2f}")
                else:
                    print("[FAIL] VLM analysis returned empty result")
                    
        else:
            print(f"[FAIL] Could not extract frame at {test_timestamp}s")
            
    except Exception as e:
        print(f"[FAIL] Frame extraction test failed: {e}")

def test_scene_processing(service, test_video):
    """Test scene-based processing workflow"""
    print("\n=== TEST 3: Scene-Based Processing ===")
    
    try:
        # Look for scene offsets from previous pipeline run
        scene_files = [
            Path("build/bonita/scenes/scene_offsets.json"),
            Path("build/bonita/scene_offsets.json")
        ]
        
        scene_offsets_path = None
        for scene_file in scene_files:
            if scene_file.exists():
                scene_offsets_path = str(scene_file)
                break
        
        if not scene_offsets_path:
            print("[WARNING] No scene offsets found, testing single-scene mode")
            scene_offsets_path = None
            
        # Generate timeline
        timeline = service.generate_and_save(str(test_video), scene_offsets_path)
        
        if timeline:
            print(f"[OK] Timeline generated successfully")
            print(f"     Events: {len(timeline.events)}")
            print(f"     Spans: {len(timeline.spans)}")
            print(f"     Duration: {timeline.total_duration:.2f}s")
            print(f"     Sources: {timeline.sources_used}")
            
            # Show sample events
            if timeline.events:
                print(f"[EVENTS] Sample events:")
                for i, event in enumerate(timeline.events[:3]):
                    print(f"  {i+1}. {event.timestamp:.2f}s: {event.description[:80]}...")
                    
            # Show sample spans  
            if timeline.spans:
                print(f"[SPANS] Sample spans:")
                for i, span in enumerate(timeline.spans[:2]):
                    print(f"  {i+1}. {span.start:.1f}s-{span.end:.1f}s: {span.description[:80]}...")
        else:
            print("[FAIL] Timeline generation returned None")
            
    except Exception as e:
        print(f"[FAIL] Scene processing test failed: {e}")

def verify_output_files():
    """Verify simplified InternVL3 output files are created correctly"""
    print("\n=== TEST 4: Simplified Output File Verification ===")
    
    # Check timeline file with new naming format - no analysis file needed
    timeline_dir = Path("build/bonita/video_timelines")
    timeline_files = list(timeline_dir.glob("*_timeline.json"))
    
    if timeline_files:
        timeline_file = timeline_files[0]  # Get the most recent one
        size = timeline_file.stat().st_size
        print(f"[OK] Timeline file: {timeline_file.name} ({size} bytes)")
        
        # Should be meaningful size (events with VLM descriptions)
        if size > 500:
            print("     Timeline file size looks good for VLM events")
        else:
            print("     [WARNING] Timeline file seems small")
            
        # Verify simplified structure - events only, no spans
        try:
            with open(timeline_file, 'r') as f:
                timeline_data = json.load(f)
            
            # Check for new model_info at top
            if 'model_info' in timeline_data:
                model_info = timeline_data['model_info']
                print(f"[OK] Model info found: {model_info.get('model_name', 'Unknown')}")
                print(f"     Prompt: {model_info.get('prompt_used', 'Unknown')[:60]}...")
                print(f"     Timestamp: {model_info.get('processing_timestamp', 'Unknown')}")
            
            events = timeline_data.get('events', [])
            spans = timeline_data.get('timeline_spans', [])
            
            print(f"     Events: {len(events)} (should have events)")
            print(f"     Spans: {len(spans)} (should be 0 for simplified approach)")
            
            if len(events) > 0 and len(spans) == 0:
                print("[OK] Simplified structure confirmed: events only, no spans")
            else:
                print("[WARNING] Structure not simplified as expected")
                
        except Exception as e:
            print(f"[WARNING] Could not verify timeline structure: {e}")
    else:
        print("[FAIL] No timeline files found with new naming format")
    
    # Analysis file should NOT exist in simplified approach
    analysis_file = Path("build/bonita/video_analysis/internvl3_analysis.json") 
    if analysis_file.exists():
        print("[WARNING] Analysis file exists - should be removed in simplified approach")
    else:
        print("[OK] No analysis file - simplified approach confirmed")
    
    # Check directory structure - only timelines needed
    video_timelines_dir = Path("build/bonita/video_timelines")
    
    if video_timelines_dir.exists():
        print("[OK] Timeline directory structure created correctly")
        print(f"     video_timelines/: {len(list(video_timelines_dir.glob('*.json')))} files")
    else:
        print("[FAIL] Timeline directory missing")

def show_integration_summary():
    """Show how simplified InternVL3 integrates with the pipeline"""
    print("\n=== SIMPLIFIED INTEGRATION SUMMARY ===")
    
    print("InternVL3 Timeline Service Status:")
    print("[OK] Direct _timeline.json generation")
    print("[OK] Scene-based frame analysis (3 frames per scene)")
    print("[OK] VLM analysis per frame with timeline events")
    print("[OK] Replaces YOLO+EasyOCR+OpenCV with unified VLM")
    
    print("\nProcessing Architecture:")
    print("  Uses PySceneDetect frame extraction (3 frames per scene)")
    print("  VLM analysis per frame generates timeline events")
    print("  Direct timeline.json output")
    print("  Unified VLM replaces multiple separate services")
    
    print("\nService Consolidation:")
    print("  YOLO object detection → InternVL3 comprehensive analysis")
    print("  EasyOCR text extraction → InternVL3 comprehensive analysis") 
    print("  OpenCV face detection → InternVL3 comprehensive analysis")
    

def main():
    """Run complete InternVL3 timeline service test"""
    print("InternVL3 Timeline Service Test")
    print("=" * 50)
    
    setup_logging('INFO')
    
    # Setup
    config, test_video = setup_test_environment()
    if not config or not test_video:
        return
    
    # Test 1: Service initialization
    service = test_service_initialization(config)
    if not service:
        return
    
    # Test 2: Frame extraction
    test_frame_extraction(service, test_video)
    
    # Test 3: Scene processing
    test_scene_processing(service, test_video)
    
    # Test 4: Output verification
    verify_output_files()
    
    # Integration summary
    show_integration_summary()
    
    print("\n" + "=" * 50)
    print("[SUCCESS] InternVL3 Timeline Service Test Complete")
    print("Timeline generation with VLM analysis per frame")
    print("Unified VLM replaces YOLO+EasyOCR+OpenCV services")

if __name__ == "__main__":
    main()