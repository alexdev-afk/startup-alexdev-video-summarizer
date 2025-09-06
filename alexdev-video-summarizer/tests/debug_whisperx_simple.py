#!/usr/bin/env python3
"""
Simple WhisperX test to isolate the transcription issue
"""

import sys
from pathlib import Path
import yaml
import json

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from services.whisper_service import WhisperService
from utils.logger import setup_logging

def test_whisperx_transcription():
    """Test WhisperX transcription directly"""
    
    setup_logging('INFO')
    
    print("=== WhisperX Direct Test ===")
    
    # Load config
    with open('config/processing.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Check test audio file
    audio_file = Path("build/bonita/audio.wav")
    if not audio_file.exists():
        print(f"[FAIL] Audio file not found: {audio_file}")
        return
    
    print(f"[OK] Test audio file: {audio_file} ({audio_file.stat().st_size} bytes)")
    
    # Initialize WhisperService
    try:
        whisper = WhisperService(config)
        print(f"[OK] WhisperService initialized")
        print(f"  Device: {whisper.device}")
        print(f"  Model: {whisper.model_size}")
        print(f"  Mock mode: {config.get('development', {}).get('mock_ai_services', False)}")
        
        # Load models
        print("[INFO] Loading models...")
        whisper._load_model()
        print(f"[OK] Models loaded: {whisper.model_loaded}")
        
        # Test transcription
        print("[INFO] Starting transcription...")
        result = whisper.transcribe_audio(audio_file)
        
        print(f"[OK] Transcription complete!")
        print(f"  Type: {type(result)}")
        print(f"  Keys: {list(result.keys())}")
        
        # Show basic info
        if 'transcript' in result:
            transcript = result['transcript'][:100] + "..." if len(result.get('transcript', '')) > 100 else result.get('transcript', '')
            print(f"  Transcript preview: {transcript}")
        
        if 'segments' in result:
            print(f"  Segments: {len(result['segments'])}")
            if result['segments']:
                first_seg = result['segments'][0]
                print(f"  First segment: {first_seg.get('start', 0):.2f}s - {first_seg.get('end', 0):.2f}s")
                print(f"  First text: {first_seg.get('text', '')[:50]}...")
        
        # Check if file was saved
        analysis_dir = audio_file.parent / "audio_analysis"
        transcription_file = analysis_dir / "whisper_transcription.json"
        
        if transcription_file.exists():
            print(f"[OK] Transcription saved: {transcription_file}")
            with open(transcription_file, 'r') as f:
                saved_data = json.load(f)
            print(f"  Saved segments: {len(saved_data.get('segments', []))}")
        else:
            print(f"[WARN] Transcription not saved to: {transcription_file}")
            if analysis_dir.exists():
                print(f"  Audio analysis dir exists but no transcription file")
                files = list(analysis_dir.glob("*"))
                print(f"  Files in audio_analysis: {[f.name for f in files]}")
            else:
                print(f"  Audio analysis dir not created")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] WhisperX test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    test_whisperx_transcription()