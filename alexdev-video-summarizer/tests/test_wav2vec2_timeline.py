#!/usr/bin/env python3
"""
Isolated Wav2Vec2 Timeline Service Test

Tests the Wav2Vec2 emotion detection timeline service independently.
Assumes vocals.wav already exists from a previous Demucs run.

Prerequisites:
- build/{video_name}/demucs_separated/vocals.wav (from previous pipeline run)
- Transformers and dependencies installed

Output:
- build/{video_name}/audio_timelines/wav2vec2_voice_timeline.json
"""

import sys
import os
import yaml
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from services.wav2vec2_timeline_service import Wav2Vec2TimelineService

def load_config():
    """Load configuration from processing.yaml"""
    config_path = Path(__file__).parent / "config" / "processing.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def find_vocals_files(build_dir: Path):
    """Find all vocals.wav files from previous Demucs runs"""
    vocals_files = []
    
    # Search pattern: build/{video_name}/vocals.wav (Demucs files are directly in video dir)
    for video_dir in build_dir.iterdir():
        if video_dir.is_dir():
            vocals_file = video_dir / "vocals.wav"
            
            if vocals_file.exists():
                vocals_files.append({
                    "video_name": video_dir.name,
                    "vocals_path": vocals_file,
                    "audio_timelines_dir": video_dir / "audio_timelines"
                })
    
    return vocals_files

def test_wav2vec2_timeline_service():
    """Test Wav2Vec2 timeline service with existing vocals.wav files"""
    print("=" * 70)
    print("Isolated Wav2Vec2 Timeline Service Test")
    print("=" * 70)
    
    try:
        # Load configuration
        config = load_config()
        
        # Find build directory
        build_dir = Path(__file__).parent / "build"
        if not build_dir.exists():
            print("❌ Build directory not found. Run main pipeline first to generate vocals.wav")
            return False
        
        # Find vocals.wav files from previous runs
        vocals_files = find_vocals_files(build_dir)
        if not vocals_files:
            print("[ERROR] No vocals.wav files found in build directory.")
            print("   Expected: build/{video_name}/vocals.wav")
            print("   Run main pipeline with Demucs separation first.")
            return False
        
        print(f"Found {len(vocals_files)} vocals.wav files to test:")
        for i, file_info in enumerate(vocals_files, 1):
            print(f"  {i}. {file_info['video_name']} - {file_info['vocals_path']}")
        print()
        
        # Initialize Wav2Vec2 timeline service
        print("Initializing Wav2Vec2TimelineService...")
        wav2vec2_service = Wav2Vec2TimelineService(config)
        print("[SUCCESS] Service initialized")
        print()
        
        # Test each vocals.wav file
        results = []
        
        for file_info in vocals_files:
            video_name = file_info['video_name']
            vocals_path = file_info['vocals_path']
            
            print(f"Processing: {video_name}")
            print(f"  Input: {vocals_path}")
            
            try:
                # Create audio_timelines directory
                file_info['audio_timelines_dir'].mkdir(exist_ok=True)
                
                start_time = time.time()
                
                # Generate timeline using Wav2Vec2 service
                timeline = wav2vec2_service.generate_and_save(
                    audio_path=str(vocals_path),
                    source_tag="wav2vec2_voice",
                    optimization={"analyze_emotion_changes": True}
                )
                
                processing_time = time.time() - start_time
                
                # Results summary
                output_file = file_info['audio_timelines_dir'] / "wav2vec2_voice_timeline.json"
                
                results.append({
                    "video_name": video_name,
                    "success": True,
                    "events_count": len(timeline.events),
                    "processing_time": processing_time,
                    "output_file": output_file
                })
                
                print(f"  [SUCCESS] {len(timeline.events)} emotion events in {processing_time:.2f}s")
                print(f"  [OUTPUT] {output_file}")
                print()
                
            except Exception as e:
                results.append({
                    "video_name": video_name,
                    "success": False,
                    "error": str(e)
                })
                print(f"  [ERROR] {e}")
                print()
        
        # Cleanup service
        wav2vec2_service.cleanup()
        
        # Summary
        print("=" * 70)
        print("Wav2Vec2 Timeline Test Results:")
        print("=" * 70)
        
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]
        
        print(f"[SUCCESS] Successful: {len(successful)}")
        print(f"[FAILED] Failed: {len(failed)}")
        print()
        
        if successful:
            print("Successful Tests:")
            for result in successful:
                print(f"  • {result['video_name']}: {result['events_count']} events ({result['processing_time']:.2f}s)")
                print(f"    Timeline: {result['output_file']}")
        
        if failed:
            print("Failed Tests:")
            for result in failed:
                print(f"  • {result['video_name']}: {result['error']}")
        
        print()
        print("[INFO] Test focuses on emotion detection only (vocals.wav processing)")
        print("[INFO] Timeline JSON files ready for integration with main pipeline")
        
        return len(failed) == 0
        
    except Exception as e:
        print(f"[ERROR] Wav2Vec2 timeline test failed: {e}")
        return False

def main():
    """Run isolated Wav2Vec2 timeline test"""
    print("Isolated ML Timeline Service Tests")
    print("==================================")
    print("Testing: Wav2Vec2 Emotion Detection Timeline Service")
    print()
    
    success = test_wav2vec2_timeline_service()
    
    if success:
        print("\n[SUCCESS] Wav2Vec2 timeline test completed successfully!")
        print("   Timeline JSON files generated and ready for use.")
    else:
        print("\n[WARNING] Some tests failed. Check error messages above.")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())