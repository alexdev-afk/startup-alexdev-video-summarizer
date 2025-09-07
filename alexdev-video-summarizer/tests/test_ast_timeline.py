#!/usr/bin/env python3
"""
Isolated AST Timeline Service Test

Tests the AST audio event classification timeline service independently.
Assumes no_vocals.wav already exists from a previous Demucs run.

Prerequisites:
- build/{video_name}/demucs_separated/no_vocals.wav (from previous pipeline run)
- Transformers and dependencies installed

Output:
- build/{video_name}/audio_timelines/ast_music_timeline.json
"""

import sys
import os
import yaml
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from services.ast_timeline_service import ASTTimelineService

def load_config():
    """Load configuration from processing.yaml"""
    config_path = Path(__file__).parent / "config" / "processing.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def find_no_vocals_files(build_dir: Path):
    """Find all no_vocals.wav files from previous Demucs runs"""
    no_vocals_files = []
    
    # Search pattern: build/{video_name}/no_vocals.wav (Demucs files are directly in video dir)
    for video_dir in build_dir.iterdir():
        if video_dir.is_dir():
            no_vocals_file = video_dir / "no_vocals.wav"
            
            if no_vocals_file.exists():
                no_vocals_files.append({
                    "video_name": video_dir.name,
                    "no_vocals_path": no_vocals_file,
                    "audio_timelines_dir": video_dir / "audio_timelines"
                })
    
    return no_vocals_files

def test_ast_timeline_service():
    """Test AST timeline service with existing no_vocals.wav files"""
    print("=" * 70)
    print("Isolated AST Timeline Service Test")
    print("=" * 70)
    
    try:
        # Load configuration
        config = load_config()
        
        # Find build directory
        build_dir = Path(__file__).parent / "build"
        if not build_dir.exists():
            print("[ERROR] Build directory not found. Run main pipeline first to generate no_vocals.wav")
            return False
        
        # Find no_vocals.wav files from previous runs
        no_vocals_files = find_no_vocals_files(build_dir)
        if not no_vocals_files:
            print("[ERROR] No no_vocals.wav files found in build directory.")
            print("   Expected: build/{video_name}/no_vocals.wav")
            print("   Run main pipeline with Demucs separation first.")
            return False
        
        print(f"Found {len(no_vocals_files)} no_vocals.wav files to test:")
        for i, file_info in enumerate(no_vocals_files, 1):
            print(f"  {i}. {file_info['video_name']} - {file_info['no_vocals_path']}")
        print()
        
        # Initialize AST timeline service
        print("Initializing ASTTimelineService...")
        ast_service = ASTTimelineService(config)
        print("[SUCCESS] Service initialized")
        print()
        
        # Test each no_vocals.wav file
        results = []
        
        for file_info in no_vocals_files:
            video_name = file_info['video_name']
            no_vocals_path = file_info['no_vocals_path']
            
            print(f"Processing: {video_name}")
            print(f"  Input: {no_vocals_path}")
            
            try:
                # Create audio_timelines directory
                file_info['audio_timelines_dir'].mkdir(exist_ok=True)
                
                start_time = time.time()
                
                # Generate timeline using AST service
                timeline = ast_service.generate_and_save(
                    audio_path=str(no_vocals_path),
                    source_tag="ast_music",
                    optimization={"analyze_sound_effects": True}
                )
                
                processing_time = time.time() - start_time
                
                # Results summary
                output_file = file_info['audio_timelines_dir'] / "ast_music_timeline.json"
                
                results.append({
                    "video_name": video_name,
                    "success": True,
                    "events_count": len(timeline.events),
                    "processing_time": processing_time,
                    "output_file": output_file
                })
                
                print(f"  [SUCCESS] {len(timeline.events)} sound effect events in {processing_time:.2f}s")
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
        ast_service.cleanup()
        
        # Summary
        print("=" * 70)
        print("AST Timeline Test Results:")
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
        print("[INFO] Test focuses on sound effects detection (527 AudioSet classes)")
        print("[INFO] Timeline JSON files ready for integration with main pipeline")
        
        return len(failed) == 0
        
    except Exception as e:
        print(f"[ERROR] AST timeline test failed: {e}")
        return False

def main():
    """Run isolated AST timeline test"""
    print("Isolated ML Timeline Service Tests")
    print("==================================")
    print("Testing: AST Audio Event Classification Timeline Service")
    print()
    
    success = test_ast_timeline_service()
    
    if success:
        print("\n[SUCCESS] AST timeline test completed successfully!")
        print("   Timeline JSON files generated and ready for use.")
    else:
        print("\n[WARNING] Some tests failed. Check error messages above.")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())