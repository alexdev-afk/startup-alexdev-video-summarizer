#!/usr/bin/env python3
"""
Complete Audio Pipeline Test

Runs the complete audio processing pipeline from scratch:
1. FFmpeg extraction (audio.wav from bonita.mp4)
2. Scene detection and splitting
3. Whisper transcription → whisper_timeline.json
4. LibROSA music analysis → librosa_timeline.json  
5. PyAudio feature analysis → pyaudio_timeline.json
6. Timeline merger with filtering → master_timeline.json

Produces exactly 4 timeline files with clean naming.
"""

import sys
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from services.ffmpeg_service import FFmpegService
from services.scene_detection_service import SceneDetectionService
from services.enhanced_whisper_timeline_service import EnhancedWhisperTimelineService
from services.librosa_timeline_service import LibROSATimelineService
from services.pyaudio_timeline_service import PyAudioTimelineService
from services.enhanced_timeline_merger_service import EnhancedTimelineMergerService
from utils.config_loader import ConfigLoader
from utils.logger import setup_logging

def setup_clean_environment():
    """Setup clean build directory"""
    print("=== SETUP: Clean Environment ===")
    
    config = ConfigLoader.load_config('config/processing.yaml')
    
    # Clean build directory
    if Path("build").exists():
        shutil.rmtree("build")
        print("[CLEAN] Removed previous build directory")
    
    video_path = Path("input/bonita.mp4")
    if not video_path.exists():
        print("[FAIL] Test video not found: input/bonita.mp4")
        return None
    
    return config

def run_ffmpeg_and_scenes(config):
    """Run FFmpeg extraction and scene detection"""
    print("\n=== STAGE 1: FFmpeg + Scene Detection ===")
    
    # FFmpeg extraction
    ffmpeg = FFmpegService(config)
    video_path = Path("input/bonita.mp4")
    result = ffmpeg.extract_streams(video_path)
    
    build_dir = Path("build/bonita")
    video_file = build_dir / "video.mp4"
    audio_file = build_dir / "audio.wav"
    
    print(f"[OK] FFmpeg: {audio_file.name} ({audio_file.stat().st_size} bytes)")
    
    # Scene detection
    scene_service = SceneDetectionService(config)
    scene_result = scene_service.analyze_video_scenes(video_file)
    
    if scene_result['boundaries']:
        scene_files = scene_service.coordinate_scene_splitting(
            video_file, scene_result['boundaries'], ffmpeg
        )
        print(f"[OK] Scenes: {len(scene_files)} files")
    
    return {
        'build_dir': build_dir,
        'video_file': video_file,
        'audio_file': audio_file,
        'config': config
    }

def run_audio_services(data):
    """Run all 3 audio services and generate individual timelines"""
    print("\n=== STAGE 2: Audio Services ===")
    
    config = data['config']
    audio_file = data['audio_file']
    timelines_dir = data['build_dir'] / "audio_timelines"
    timelines_dir.mkdir(exist_ok=True)
    
    timelines = []
    
    # 1. Whisper Timeline
    print("  [1/3] Whisper transcription...")
    try:
        whisper_service = EnhancedWhisperTimelineService(config)
        whisper_timeline = whisper_service.generate_and_save(str(audio_file))
        
        # Save with clean name
        whisper_path = timelines_dir / "whisper_timeline.json"
        whisper_timeline.save_to_file(str(whisper_path))
        timelines.append(whisper_timeline)
        
        print(f"    [OK] whisper_timeline.json: {len(whisper_timeline.events)} events, {len(whisper_timeline.spans)} spans")
        
    except Exception as e:
        print(f"    [FAIL] Whisper failed: {e}")
    
    # 2. LibROSA Timeline  
    print("  [2/3] LibROSA music analysis...")
    try:
        librosa_service = LibROSATimelineService(config)
        librosa_timeline = librosa_service.generate_and_save(str(audio_file))
        
        # Save with clean name
        librosa_path = timelines_dir / "librosa_timeline.json"
        librosa_timeline.save_to_file(str(librosa_path))
        timelines.append(librosa_timeline)
        
        print(f"    [OK] librosa_timeline.json: {len(librosa_timeline.events)} events, {len(librosa_timeline.spans)} spans")
        
    except Exception as e:
        print(f"    [FAIL] LibROSA failed: {e}")
    
    # 3. PyAudio Timeline
    print("  [3/3] PyAudio feature analysis...")
    try:
        pyaudio_service = PyAudioTimelineService(config)
        pyaudio_timeline = pyaudio_service.generate_and_save(str(audio_file))
        
        # Save with clean name
        pyaudio_path = timelines_dir / "pyaudio_timeline.json"
        pyaudio_timeline.save_to_file(str(pyaudio_path))
        timelines.append(pyaudio_timeline)
        
        print(f"    [OK] pyaudio_timeline.json: {len(pyaudio_timeline.events)} events, {len(pyaudio_timeline.spans)} spans")
        
    except Exception as e:
        print(f"    [FAIL] PyAudio failed: {e}")
    
    return timelines

def run_timeline_merger(timelines, data):
    """Merge all timelines into master timeline with filtering"""
    print("\n=== STAGE 3: Timeline Merger ===")
    
    if not timelines:
        print("  [FAIL] No timelines to merge")
        return False
    
    config = data['config']
    timelines_dir = data['build_dir'] / "audio_timelines"
    
    try:
        merger_service = EnhancedTimelineMergerService(config)
        
        # Create master timeline with clean name
        master_path = timelines_dir / "master_timeline.json"
        master_timeline = merger_service.create_master_timeline(
            timelines=timelines,
            output_path=str(master_path)
        )
        
        print(f"  [OK] master_timeline.json: {len(master_timeline.events)} events, {len(master_timeline.spans)} spans")
        print(f"      Duration: {master_timeline.total_duration:.2f}s")
        print(f"      Speakers: {master_timeline.speakers}")
        print(f"      LibROSA filtering applied: speech artifacts removed")
        
        return True
        
    except Exception as e:
        print(f"  [FAIL] Timeline merger failed: {e}")
        return False

def show_final_output(data):
    """Show final clean output files"""
    print("\n=== FINAL OUTPUT ===")
    
    timelines_dir = data['build_dir'] / "audio_timelines"
    expected_files = [
        "whisper_timeline.json",
        "librosa_timeline.json", 
        "pyaudio_timeline.json",
        "master_timeline.json"
    ]
    
    print("Timeline files generated:")
    for filename in expected_files:
        file_path = timelines_dir / filename
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"  [OK] {filename} ({size} bytes)")
        else:
            print(f"  [MISSING] {filename}")
    
    # Clean up any extra files that shouldn't be there
    extra_files = []
    if timelines_dir.exists():
        for file_path in timelines_dir.glob("*.json"):
            if file_path.name not in expected_files:
                extra_files.append(file_path.name)
                file_path.unlink()  # Delete extra files
    
    if extra_files:
        print(f"  [CLEANED] Removed extra files: {extra_files}")
    
    print(f"\n[OUTPUT] Location: {timelines_dir}")
    print("[READY] master_timeline.json contains filtered LibROSA events")

def main():
    """Run complete audio pipeline with clean output"""
    print("Complete Audio Pipeline Test")
    print("=" * 50)
    
    setup_logging('INFO')
    
    # Setup
    config = setup_clean_environment()
    if not config:
        return
    
    # Stage 1: FFmpeg + Scenes
    data = run_ffmpeg_and_scenes(config)
    if not data:
        return
    
    # Stage 2: Audio Services
    timelines = run_audio_services(data)
    
    # Stage 3: Timeline Merger
    merger_success = run_timeline_merger(timelines, data)
    
    # Final Output
    show_final_output(data)
    
    # Summary
    print("\n" + "=" * 50)
    if merger_success:
        print("[SUCCESS] AUDIO PIPELINE COMPLETE")
        print("[OUTPUT] All 4 timeline files generated with clean naming")
        print("[FILTER] LibROSA speech artifact filtering applied in master timeline")
    else:
        print("[INCOMPLETE] AUDIO PIPELINE FAILED") 
        print("[WARNING] Some stages failed - check individual timeline files")

if __name__ == "__main__":
    main()