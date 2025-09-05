#!/usr/bin/env python3
"""
Pipeline Stage Testing - Debug build directory persistence

Test each stage of the pipeline individually to identify where 
data persistence fails.
"""

import sys
import os
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from services.ffmpeg_service import FFmpegService
from services.scene_detection_service import SceneDetectionService
from services.orchestrator import VideoProcessingOrchestrator
from utils.config_loader import ConfigLoader
from utils.logger import setup_logging

def test_stage_1_ffmpeg():
    """Stage 1: Test FFmpeg extraction and build directory creation"""
    print("=== STAGE 1: FFmpeg Extraction Test ===")
    
    # Setup
    setup_logging('DEBUG')
    config = ConfigLoader.load_config('../config/processing.yaml')
    
    # Initialize FFmpeg service
    ffmpeg = FFmpegService(config)
    
    # Test video
    video_path = Path("../input/Consider 3 key points when ordering your automated laundry rack.mp4")
    
    if not video_path.exists():
        print(f"[FAIL] Test video not found: {video_path}")
        return False
    
    print(f"[OK] Test video found: {video_path}")
    
    # Extract streams
    try:
        result = ffmpeg.extract_streams(str(video_path))
        print(f"[OK] FFmpeg extraction result: {result}")
        
        # Check build directory
        build_dir = Path("../build")
        if build_dir.exists():
            print(f"[OK] Build directory exists: {build_dir}")
            
            # List contents
            for item in build_dir.rglob("*"):
                print(f"  - {item}")
        else:
            print("[FAIL] Build directory not created")
            return False
            
        return True
        
    except Exception as e:
        print(f"[FAIL] FFmpeg extraction failed: {e}")
        return False

def test_stage_2_scene_detection():
    """Stage 2: Test scene detection and scene file creation"""
    print("\n=== STAGE 2: Scene Detection Test ===")
    
    # Initialize scene detection service
    config = ConfigLoader.load_config('../config/processing.yaml')
    scene_service = SceneDetectionService(config)
    
    # Check for FFmpeg output first
    video_file = Path("build").glob("*/video.mp4")
    video_file = next(video_file, None)
    
    if not video_file:
        print("❌ No video.mp4 found from FFmpeg stage")
        return False
    
    print(f"✓ Found video file: {video_file}")
    
    try:
        # Run scene detection
        result = scene_service.detect_scenes(str(video_file))
        print(f"✓ Scene detection result: {result}")
        
        # Check for scene files
        build_dir = video_file.parent
        scene_files = list(build_dir.glob("scenes/scene_*.mp4"))
        
        print(f"✓ Found {len(scene_files)} scene files:")
        for scene_file in scene_files[:5]:  # Show first 5
            print(f"  - {scene_file}")
        
        # Check for scene_offsets.json
        offsets_file = build_dir / "scenes" / "scene_offsets.json"
        if offsets_file.exists():
            print(f"✓ Scene offsets file exists: {offsets_file}")
            with open(offsets_file, 'r') as f:
                offsets = json.load(f)
                print(f"  Scenes data: {len(offsets.get('scenes', {}))} scenes")
        else:
            print("❌ Scene offsets file not found")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Scene detection failed: {e}")
        return False

def test_stage_3_analysis_persistence():
    """Stage 3: Test analysis data persistence"""
    print("\n=== STAGE 3: Analysis Data Persistence Test ===")
    
    # Check for build directory structure
    build_dirs = list(Path("build").glob("*/"))
    if not build_dirs:
        print("❌ No build directories found")
        return False
    
    build_dir = build_dirs[0]
    print(f"✓ Using build directory: {build_dir}")
    
    # Check for analysis files
    analysis_files = list(build_dir.glob("analysis/*.json"))
    print(f"Found {len(analysis_files)} analysis files:")
    
    for analysis_file in analysis_files[:3]:  # Show first 3
        print(f"  - {analysis_file}")
        
        # Check file content
        try:
            with open(analysis_file, 'r') as f:
                data = json.load(f)
                print(f"    Content keys: {list(data.keys())}")
        except Exception as e:
            print(f"    ❌ Error reading file: {e}")
    
    if analysis_files:
        print("✓ Analysis files found")
        return True
    else:
        print("❌ No analysis files found")
        return False

def test_stage_4_knowledge_output():
    """Stage 4: Test knowledge generation output"""
    print("\n=== STAGE 4: Knowledge Generation Output Test ===")
    
    # Check output directory
    output_dir = Path("output")
    if not output_dir.exists():
        print("❌ Output directory doesn't exist")
        return False
    
    # Check for knowledge base files
    kb_files = list(output_dir.glob("*.md"))
    print(f"Found {len(kb_files)} knowledge base files:")
    
    for kb_file in kb_files:
        print(f"  - {kb_file}")
        
        # Check file size
        size = kb_file.stat().st_size
        print(f"    Size: {size} bytes")
        
        if size > 0:
            print("    ✓ File has content")
        else:
            print("    ❌ File is empty")
    
    # Check INDEX.md
    index_file = output_dir / "INDEX.md"
    if index_file.exists():
        print(f"✓ Master index exists: {index_file}")
    else:
        print("❌ Master index not found")
    
    return len(kb_files) > 0

def main():
    """Run all pipeline stage tests"""
    print("Pipeline Stage Testing - Build Directory Debug")
    print("=" * 50)
    
    # Clean previous test data
    import shutil
    if Path("build").exists():
        shutil.rmtree("build")
        print("🧹 Cleaned previous build directory")
    
    if Path("output").exists():
        shutil.rmtree("output")
        print("🧹 Cleaned previous output directory")
    
    # Run tests sequentially
    tests = [
        ("Stage 1: FFmpeg Extraction", test_stage_1_ffmpeg),
        ("Stage 2: Scene Detection", test_stage_2_scene_detection),
        ("Stage 3: Analysis Persistence", test_stage_3_analysis_persistence),
        ("Stage 4: Knowledge Output", test_stage_4_knowledge_output),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results[test_name] = False
            # Don't continue if early stage fails
            if "Stage 1" in test_name or "Stage 2" in test_name:
                break
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY:")
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    # Build directory final state
    print(f"\nFinal build directory state:")
    if Path("build").exists():
        for item in Path("build").rglob("*"):
            print(f"  {item}")
    else:
        print("  (empty)")

if __name__ == "__main__":
    main()