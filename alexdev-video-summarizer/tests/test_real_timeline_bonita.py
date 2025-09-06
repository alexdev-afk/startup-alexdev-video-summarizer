#!/usr/bin/env python3
"""
Test real timeline services on bonita.mp4
"""

import json
import sys
from pathlib import Path
import yaml

# Add src to path
sys.path.insert(0, 'src')

def load_config():
    """Load configuration"""
    config_path = Path("config/processing.yaml")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def test_real_timeline_services():
    """Test real timeline services on bonita.mp4"""
    
    # Check if we have audio file (from FFmpeg extraction)
    audio_file = Path("build/bonita/audio.wav")
    if not audio_file.exists():
        print(f"ERROR: Audio file not found: {audio_file}")
        print("Run main processing first to extract audio.wav")
        return
    
    print(f"FOUND: Audio file: {audio_file}")
    
    # Load config
    config = load_config()
    
    # Test LibROSA Timeline Service
    print("\n=== Testing LibROSA Timeline Service ===")
    try:
        from services.librosa_timeline_service import LibROSATimelineService
        
        librosa_service = LibROSATimelineService(config)
        librosa_timeline = librosa_service.generate_timeline(str(audio_file))
        
        print(f"LibROSA Timeline Results:")
        print(f"  - Total duration: {librosa_timeline.total_duration}s")
        print(f"  - Events: {len(librosa_timeline.events)}")
        print(f"  - Spans: {len(librosa_timeline.spans)}")
        
        # Show first few events and spans
        if librosa_timeline.events:
            print("  - First few events:")
            for i, event in enumerate(librosa_timeline.events[:3]):
                print(f"    {i+1}. {event.timestamp}s - {event.description}")
        
        if librosa_timeline.spans:
            print("  - All spans:")
            for i, span in enumerate(librosa_timeline.spans):
                print(f"    {i+1}. {span.start}s-{span.end}s ({span.duration}s) - {span.description}")
        
    except Exception as e:
        print(f"ERROR: LibROSA timeline failed: {e}")
    
    # Test pyAudioAnalysis Timeline Service  
    print("\n=== Testing pyAudioAnalysis Timeline Service ===")
    try:
        from services.pyaudio_timeline_service import PyAudioTimelineService
        
        pyaudio_service = PyAudioTimelineService(config)
        pyaudio_timeline = pyaudio_service.generate_timeline(str(audio_file))
        
        print(f"pyAudioAnalysis Timeline Results:")
        print(f"  - Total duration: {pyaudio_timeline.total_duration}s") 
        print(f"  - Events: {len(pyaudio_timeline.events)}")
        print(f"  - Spans: {len(pyaudio_timeline.spans)}")
        
        # Show all events and spans
        if pyaudio_timeline.events:
            print("  - All events:")
            for i, event in enumerate(pyaudio_timeline.events):
                print(f"    {i+1}. {event.timestamp}s - {event.description}")
        
        if pyaudio_timeline.spans:
            print("  - All spans:")
            for i, span in enumerate(pyaudio_timeline.spans):
                print(f"    {i+1}. {span.start}s-{span.end}s ({span.duration}s) - {span.description}")
                
    except Exception as e:
        print(f"ERROR: pyAudioAnalysis timeline failed: {e}")
    
    # Test Multi-Pass Service
    print("\n=== Testing Multi-Pass Timeline Service ===")
    try:
        from services.multipass_audio_timeline_service import MultiPassAudioTimelineService
        
        multipass_service = MultiPassAudioTimelineService(config)
        multipass_timeline = multipass_service.generate_timeline(str(audio_file))
        
        print(f"Multi-Pass Timeline Results:")
        print(f"  - Total duration: {multipass_timeline.total_duration}s")
        print(f"  - Events: {len(multipass_timeline.events)}")
        print(f"  - Spans: {len(multipass_timeline.spans)}")
        
        # Show all spans (the key improvement)
        if multipass_timeline.spans:
            print("  - Intelligent spans:")
            for i, span in enumerate(multipass_timeline.spans):
                print(f"    {i+1}. {span.start:.1f}s-{span.end:.1f}s ({span.duration:.1f}s)")
                print(f"       {span.description}")
                print(f"       Category: {span.category}, Confidence: {span.confidence}")
        
        # Show event summary by source
        librosa_events = [e for e in multipass_timeline.events if e.details.get("original_source") == "librosa"]
        pyaudio_events = [e for e in multipass_timeline.events if e.details.get("original_source") == "pyaudio"]
        
        print(f"  - Events by source: {len(librosa_events)} LibROSA, {len(pyaudio_events)} pyAudio")
        
        # Show pyAudio events (these drive the boundaries)
        if pyaudio_events:
            print("  - Boundary-driving events (pyAudio):")
            for i, event in enumerate(pyaudio_events):
                clean_desc = event.description.replace("[pyAudio] ", "")
                print(f"    {i+1}. {event.timestamp:.1f}s - {clean_desc}")
        
    except Exception as e:
        print(f"ERROR: Multi-pass timeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_real_timeline_services()