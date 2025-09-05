#!/usr/bin/env python3
"""
Simple debug test for build directory creation
"""

import sys
from pathlib import Path
import shutil

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from services.ffmpeg_service import FFmpegService
from utils.config_loader import ConfigLoader
from utils.logger import setup_logging

def main():
    print("Debug Build Directory Creation")
    print("=" * 40)
    
    # Clean previous test data
    if Path("build").exists():
        shutil.rmtree("build")
        print("[CLEAN] Removed previous build directory")
    
    # Setup
    setup_logging('DEBUG')
    config = ConfigLoader.load_config('config/processing.yaml')
    
    # Initialize FFmpeg service
    ffmpeg = FFmpegService(config)
    
    # Test video
    video_path = Path("input/Consider 3 key points when ordering your automated laundry rack.mp4")
    
    if not video_path.exists():
        print(f"[FAIL] Test video not found: {video_path}")
        return
    
    print(f"[OK] Test video found: {video_path}")
    
    # Extract streams
    try:
        result = ffmpeg.extract_streams(video_path)
        print(f"[OK] FFmpeg extraction completed")
        print(f"Result: {result}")
        
        # Check build directory
        build_dir = Path("build")
        if build_dir.exists():
            print(f"[OK] Build directory exists: {build_dir}")
            
            print("\nBuild directory contents:")
            for item in build_dir.rglob("*"):
                if item.is_file():
                    size = item.stat().st_size
                    print(f"  FILE: {item} ({size} bytes)")
                else:
                    print(f"  DIR:  {item}")
        else:
            print("[FAIL] Build directory not created")
        
    except Exception as e:
        print(f"[FAIL] FFmpeg extraction failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()