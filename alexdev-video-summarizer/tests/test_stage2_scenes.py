#!/usr/bin/env python3
"""
Stage 2 Test: Scene Detection and Scene File Creation
"""

import sys
from pathlib import Path
import shutil

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from services.ffmpeg_service import FFmpegService
from services.scene_detection_service import SceneDetectionService
from utils.config_loader import ConfigLoader
from utils.logger import setup_logging

def main():
    print("Stage 2: Scene Detection Test")
    print("=" * 40)
    
    # Setup
    setup_logging('INFO')
    config = ConfigLoader.load_config('config/processing.yaml')
    
    # Ensure we have FFmpeg output from Stage 1
    build_dir = Path("build/Consider 3 key points when ordering your automated laundry rack")
    video_file = build_dir / "video.mp4"
    
    if not video_file.exists():
        print("[SETUP] Running Stage 1 first...")
        
        # Clean and run Stage 1
        if Path("build").exists():
            shutil.rmtree("build")
        
        ffmpeg = FFmpegService(config)
        video_path = Path("input/Consider 3 key points when ordering your automated laundry rack.mp4")
        
        if not video_path.exists():
            print("[FAIL] Test video not found")
            return
        
        result = ffmpeg.extract_streams(video_path)
        print(f"[OK] Stage 1 complete: {result}")
    
    print(f"[OK] Video file ready: {video_file}")
    print(f"Video file size: {video_file.stat().st_size} bytes")
    
    # Initialize scene detection service
    scene_service = SceneDetectionService(config)
    print("[OK] SceneDetectionService initialized")
    
    try:
        # Run scene detection
        print(f"\n[TEST] Running scene detection on: {video_file}")
        result = scene_service.analyze_video_scenes(video_file)
        print(f"[OK] Scene detection complete: {result['scene_count']} scenes detected")
        
        # Run scene splitting to create individual scene files
        if result['boundaries']:
            print(f"\n[TEST] Running scene splitting...")
            ffmpeg_service = FFmpegService(config)
            scene_files = scene_service.coordinate_scene_splitting(
                video_file, 
                result['boundaries'], 
                ffmpeg_service
            )
            print(f"[OK] Scene splitting complete: {len(scene_files)} scene files created")
        
        # Check build directory after scene detection
        print(f"\nBuild directory contents after scene detection:")
        for item in build_dir.rglob("*"):
            if item.is_file():
                size = item.stat().st_size
                print(f"  FILE: {item.relative_to(build_dir)} ({size} bytes)")
            else:
                print(f"  DIR:  {item.relative_to(build_dir)}")
        
        # Specifically check for scene files
        scenes_dir = build_dir / "scenes"
        if scenes_dir.exists():
            print(f"\n[OK] Scenes directory exists: {scenes_dir}")
            
            scene_files = list(scenes_dir.glob("scene_*.mp4"))
            print(f"[INFO] Found {len(scene_files)} scene files:")
            for scene_file in scene_files:
                size = scene_file.stat().st_size
                print(f"  - {scene_file.name}: {size} bytes")
            
            # Check scene_offsets.json
            offsets_file = scenes_dir / "scene_offsets.json"
            if offsets_file.exists():
                print(f"[OK] Scene offsets file exists: {offsets_file.name}")
                
                import json
                with open(offsets_file, 'r') as f:
                    offsets_data = json.load(f)
                    scenes_count = len(offsets_data.get('scenes', {}))
                    total_duration = offsets_data.get('total_duration', 0)
                    print(f"  Scenes: {scenes_count}")
                    print(f"  Duration: {total_duration:.1f}s")
            else:
                print("[FAIL] Scene offsets file not found")
        else:
            print("[FAIL] Scenes directory not created")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Scene detection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()