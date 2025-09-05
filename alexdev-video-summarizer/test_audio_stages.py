#!/usr/bin/env python3
"""
Test Audio Pipeline Stages 1 by 1

Test Whisper -> LibROSA -> pyAudioAnalysis individually to see 
which stage might be cleaning up the build directory.
"""

import sys
from pathlib import Path
import shutil

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from services.ffmpeg_service import FFmpegService
from services.scene_detection_service import SceneDetectionService
from services.whisper_service import WhisperService
from services.librosa_service import LibROSAService
from services.pyaudioanalysis_service import PyAudioAnalysisService
from utils.config_loader import ConfigLoader
from utils.logger import setup_logging

def setup_test_data():
    """Setup build directory with FFmpeg output and scene files"""
    print("=== SETUP: Creating test data ===")
    
    config = ConfigLoader.load_config('config/processing.yaml')
    
    # Clean and create build directory
    if Path("build").exists():
        shutil.rmtree("build")
        print("[CLEAN] Removed previous build directory")
    
    # Run FFmpeg extraction
    ffmpeg = FFmpegService(config)
    video_path = Path("input/bonita.mp4")
    
    if not video_path.exists():
        print("[FAIL] Test video not found")
        return None
    
    result = ffmpeg.extract_streams(video_path)
    build_dir = Path("build/bonita")
    video_file = build_dir / "video.mp4"
    audio_file = build_dir / "audio.wav"
    
    print(f"[OK] FFmpeg extraction complete")
    print(f"  Video: {video_file} ({video_file.stat().st_size} bytes)")
    print(f"  Audio: {audio_file} ({audio_file.stat().st_size} bytes)")
    
    # Run scene detection and splitting
    scene_service = SceneDetectionService(config)
    scene_result = scene_service.analyze_video_scenes(video_file)
    
    if scene_result['boundaries']:
        scene_files = scene_service.coordinate_scene_splitting(
            video_file, scene_result['boundaries'], ffmpeg
        )
        print(f"[OK] Scene splitting complete: {len(scene_files)} files")
    
    return {
        'build_dir': build_dir,
        'video_file': video_file,
        'audio_file': audio_file,
        'config': config
    }

def check_build_directory(stage_name, build_dir):
    """Check build directory contents after each stage"""
    print(f"\n--- Build Directory After {stage_name} ---")
    
    if not build_dir.exists():
        print("[FAIL] Build directory deleted!")
        return False
    
    file_count = 0
    for item in build_dir.rglob("*"):
        if item.is_file():
            size = item.stat().st_size
            print(f"  FILE: {item.relative_to(build_dir)} ({size} bytes)")
            file_count += 1
        else:
            print(f"  DIR:  {item.relative_to(build_dir)}")
    
    print(f"[INFO] Total files: {file_count}")
    return True

def test_whisper_stage(test_data):
    """Test Whisper transcription stage"""
    print("\n=== AUDIO STAGE 1: Whisper Transcription ===")
    
    try:
        whisper = WhisperService(test_data['config'])
        print(f"[OK] WhisperService initialized - device: {whisper.device}")
        
        # Test transcription
        audio_file = test_data['audio_file']
        print(f"[TEST] Transcribing: {audio_file.name}")
        
        result = whisper.transcribe_audio(audio_file)
        print(f"[OK] Transcription complete: {type(result)}")
        
        # Check build directory
        return check_build_directory("Whisper", test_data['build_dir'])
        
    except Exception as e:
        print(f"[FAIL] Whisper stage failed: {e}")
        check_build_directory("Whisper (FAILED)", test_data['build_dir'])
        return False

def test_librosa_stage(test_data):
    """Test LibROSA music analysis stage"""
    print("\n=== AUDIO STAGE 2: LibROSA Music Analysis ===")
    
    try:
        librosa = LibROSAService(test_data['config'])
        from services.librosa_service import LIBROSA_AVAILABLE
        print(f"[OK] LibROSAService initialized - available: {LIBROSA_AVAILABLE}")
        
        # Test music analysis
        audio_file = test_data['audio_file']
        print(f"[TEST] Analyzing music: {audio_file.name}")
        
        result = librosa.analyze_audio_segment(str(audio_file))
        print(f"[OK] Music analysis complete: {type(result)}")
        
        # Check build directory
        return check_build_directory("LibROSA", test_data['build_dir'])
        
    except Exception as e:
        print(f"[FAIL] LibROSA stage failed: {e}")
        check_build_directory("LibROSA (FAILED)", test_data['build_dir'])
        return False

def test_pyaudioanalysis_stage(test_data):
    """Test pyAudioAnalysis interpretive analysis stage"""
    print("\n=== AUDIO STAGE 3: pyAudioAnalysis Interpretive Analysis ===")
    
    try:
        pyaudio = PyAudioAnalysisService(test_data['config'])
        from services.pyaudioanalysis_service import PYAUDIOANALYSIS_AVAILABLE
        print(f"[OK] PyAudioAnalysisService initialized - available: {PYAUDIOANALYSIS_AVAILABLE}")
        
        # Test interpretive analysis with Whisper alignment
        audio_file = test_data['audio_file']
        print(f"[TEST] Running interpretive analysis with Whisper alignment: {audio_file.name}")
        
        # Load Whisper results for alignment
        whisper_file = audio_file.parent / "audio_analysis" / "whisper_transcription.json"
        if whisper_file.exists():
            import json
            with open(whisper_file, 'r') as f:
                whisper_result = json.load(f)
            
            print(f"[INFO] Found {len(whisper_result.get('segments', []))} Whisper segments for alignment")
            
            # Run new interpretive analysis
            result = pyaudio.analyze_whisper_segments(str(audio_file), whisper_result)
            print(f"[OK] Interpretive analysis complete: {type(result)}")
            
            # Show sample interpretive output
            if result.get('segment_analyses'):
                print(f"[INFO] Generated {len(result['segment_analyses'])} interpretive segment analyses")
                first_segment = result['segment_analyses'][0]
                print(f"[SAMPLE] Voice: {first_segment.get('voice_characteristics', {}).get('vocal_clarity', 'N/A')[:60]}...")
        else:
            print("[WARN] No Whisper results found - falling back to basic analysis")
            result = pyaudio.analyze_audio_segment(str(audio_file))
            print(f"[OK] Basic feature extraction complete: {type(result)}")
        
        # Check build directory
        return check_build_directory("pyAudioAnalysis", test_data['build_dir'])
        
    except Exception as e:
        print(f"[FAIL] pyAudioAnalysis stage failed: {e}")
        check_build_directory("pyAudioAnalysis (FAILED)", test_data['build_dir'])
        return False

def main():
    """Test audio pipeline stages individually"""
    print("Audio Pipeline Stages Test")
    print("=" * 50)
    
    setup_logging('INFO')
    
    # Setup test data
    test_data = setup_test_data()
    if not test_data:
        return
    
    check_build_directory("SETUP", test_data['build_dir'])
    
    # Test each audio stage
    stages = [
        ("Whisper Transcription", test_whisper_stage),
        ("LibROSA Music Analysis", test_librosa_stage),
        ("pyAudioAnalysis Features", test_pyaudioanalysis_stage),
    ]
    
    results = {}
    
    for stage_name, test_func in stages:
        print(f"\n{'='*20} TESTING: {stage_name} {'='*20}")
        
        try:
            results[stage_name] = test_func(test_data)
        except Exception as e:
            print(f"[FAIL] {stage_name} crashed: {e}")
            results[stage_name] = False
            # Check if build directory still exists
            check_build_directory(f"{stage_name} (CRASHED)", test_data['build_dir'])
    
    # Final summary
    print("\n" + "=" * 50)
    print("AUDIO STAGES TEST SUMMARY:")
    
    for stage_name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status}: {stage_name}")
    
    # Final build directory check
    print(f"\nFINAL BUILD DIRECTORY STATE:")
    if test_data['build_dir'].exists():
        check_build_directory("FINAL", test_data['build_dir'])
    else:
        print("[FAIL] Build directory completely deleted!")

if __name__ == "__main__":
    main()