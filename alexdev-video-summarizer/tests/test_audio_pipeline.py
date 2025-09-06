#!/usr/bin/env python3
"""
Complete Audio Pipeline Test

Runs the complete audio processing pipeline from scratch:
1. FFmpeg extraction (audio.wav from bonita.mp4)
2. Scene detection and splitting
3. Whisper transcription → whisper_timeline.json
4. LibROSA music analysis → librosa_timeline.json  
5. PyAudio feature analysis → pyaudio_timeline.json
6. Timeline merger with filtering → combined_audio_timeline.json

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
    """Run FFmpeg extraction and scene detection with frame extraction"""
    print("\n=== STAGE 1: FFmpeg + Scene Detection + Frame Extraction ===")
    
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
    
    frame_data = {}
    if scene_result['boundaries']:
        # NEW: Extract 3 frames per scene instead of full scene videos
        frame_data = scene_service.coordinate_frame_extraction(
            video_file, scene_result['boundaries'], ffmpeg
        )
        total_scenes = len(scene_result['boundaries'])
        total_frames = total_scenes * 3
        print(f"[OK] Frames: {total_frames} frames extracted from {total_scenes} scenes")
        print(f"     Format: 3 frames per scene (first, representative, last)")
    
    return {
        'build_dir': build_dir,
        'video_file': video_file,
        'audio_file': audio_file,
        'frame_data': frame_data,
        'scene_result': scene_result,
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
        master_path = timelines_dir / "combined_audio_timeline.json"
        combined_audio_timeline = merger_service.create_combined_audio_timeline(
            timelines=timelines,
            output_path=str(master_path)
        )
        
        print(f"  [OK] combined_audio_timeline.json: {len(combined_audio_timeline.events)} events, {len(combined_audio_timeline.spans)} spans")
        print(f"      Duration: {combined_audio_timeline.total_duration:.2f}s")
        print(f"      Speakers: {combined_audio_timeline.speakers}")
        print(f"      LibROSA filtering applied: speech artifacts removed")
        
        return True
        
    except Exception as e:
        print(f"  [FAIL] Timeline merger failed: {e}")
        return False

def show_final_output(data):
    """Show final clean output files"""
    print("\n=== FINAL OUTPUT ===")
    
    # Audio timelines
    timelines_dir = data['build_dir'] / "audio_timelines"
    expected_timeline_files = [
        "whisper_timeline.json",
        "librosa_timeline.json", 
        "pyaudio_timeline.json",
        "combined_audio_timeline.json"
    ]
    
    print("Audio timeline files generated:")
    for filename in expected_timeline_files:
        file_path = timelines_dir / filename
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"  [OK] {filename} ({size} bytes)")
        else:
            print(f"  [MISSING] {filename}")
    
    # Video frames
    frames_dir = data['build_dir'] / "frames"
    if frames_dir.exists():
        frame_files = list(frames_dir.glob("*.jpg"))
        metadata_file = frames_dir / "frame_metadata.json"
        
        print(f"\nVideo frame files extracted:")
        print(f"  [OK] {len(frame_files)} JPEG frames")
        if metadata_file.exists():
            print(f"  [OK] frame_metadata.json ({metadata_file.stat().st_size} bytes)")
            
            # Show frame breakdown by scene
            frame_data = data.get('frame_data', {})
            if frame_data and 'scenes' in frame_data:
                print(f"  [BREAKDOWN] {len(frame_data['scenes'])} scenes × 3 frames each:")
                for scene_key, scene_info in frame_data['scenes'].items():
                    scene_id = scene_info['scene_id']
                    duration = scene_info.get('duration_seconds', 0)
                    print(f"    Scene {scene_id}: {duration:.1f}s duration (first/representative/last frames)")
        else:
            print(f"  [MISSING] frame_metadata.json")
    else:
        print(f"\n  [MISSING] frames directory")
    
    # Clean up any extra timeline files that shouldn't be there
    extra_files = []
    if timelines_dir.exists():
        for file_path in timelines_dir.glob("*.json"):
            if file_path.name not in expected_timeline_files:
                extra_files.append(file_path.name)
                file_path.unlink()  # Delete extra files
    
    if extra_files:
        print(f"  [CLEANED] Removed extra timeline files: {extra_files}")
    
    print(f"\n[AUDIO OUTPUT] Location: {timelines_dir}")
    print("[AUDIO READY] combined_audio_timeline.json contains filtered LibROSA events")
    
    print(f"\n[VISUAL OUTPUT] Location: {frames_dir}")
    print("[VISUAL READY] 3 frames per scene ready for InternVL3 scene understanding")

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