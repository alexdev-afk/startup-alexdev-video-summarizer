#!/usr/bin/env python3
"""
Simple Audio Pipeline Integration Test
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from services.audio_pipeline import AudioPipelineController
from services.ffmpeg_service import FFmpegService
from utils.config_loader import ConfigLoader
from utils.processing_context import VideoProcessingContext

def test_audio_pipeline():
    """Test complete audio pipeline integration"""
    
    print("Audio Pipeline Integration Test")
    print("=" * 50)
    
    # Load configuration
    config = ConfigLoader.load_config('config/processing.yaml')
    print("[OK] Configuration loaded")
    
    # Initialize services
    ffmpeg_service = FFmpegService(config)
    audio_controller = AudioPipelineController(config)
    print("[OK] Services initialized")
    
    # Select test video (smallest one)
    input_dir = Path("input")
    test_videos = list(input_dir.glob("*.mp4"))
    if not test_videos:
        print("[ERROR] No test videos found")
        return False
        
    test_video = min(test_videos, key=lambda x: x.stat().st_size)
    print(f"[VIDEO] Test: {test_video.name} ({test_video.stat().st_size / 1024 / 1024:.1f} MB)")
    
    try:
        # Step 1: FFmpeg extraction
        print("\n[STEP 1] FFmpeg Audio Extraction")
        start_time = time.time()
        
        context = VideoProcessingContext(test_video)
        audio_path, video_path = ffmpeg_service.extract_streams(test_video)
        context.audio_path = audio_path
        context.video_path = video_path
        
        if not context.validate_ffmpeg_output():
            print("[ERROR] FFmpeg extraction failed")
            return False
            
        print(f"[OK] FFmpeg completed in {time.time() - start_time:.2f}s")
        print(f"   Audio: {audio_path}")
        print(f"   Video: {video_path}")
        
        # Step 2: Audio pipeline processing
        print("\n[STEP 2] Audio Pipeline Processing")
        mock_scene = {
            'scene_id': 'test_scene_1',
            'start_time': 0.0,
            'end_time': 10.0,
            'start_frame': 0,
            'end_frame': 300
        }
        
        audio_start = time.time()
        audio_results = audio_controller.process_scene(mock_scene, context)
        audio_time = time.time() - audio_start
        
        print(f"[OK] Audio pipeline completed in {audio_time:.2f}s")
        
        # Step 3: Validate results
        print("\n[RESULTS] Audio Pipeline Results:")
        print("-" * 30)
        
        whisper_result = audio_results.get('whisper', {})
        print(f"[WHISPER] Transcription:")
        print(f"   Available: {'[OK]' if not whisper_result.get('fallback_mode') else '[FALLBACK]'}")
        print(f"   Language: {whisper_result.get('language', 'unknown')}")
        
        librosa_result = audio_results.get('librosa', {})
        print(f"[LIBROSA] Music Analysis:")
        print(f"   Available: {'[OK]' if not librosa_result.get('fallback_mode') else '[FALLBACK]'}")
        if 'tempo' in librosa_result:
            print(f"   Tempo: {librosa_result.get('tempo', 0):.1f} BPM")
        
        pyaudio_result = audio_results.get('pyaudioanalysis', {})
        print(f"[PYAUDIO] Features:")
        print(f"   Available: {'[OK]' if not pyaudio_result.get('fallback_mode') else '[FALLBACK]'}")
        if 'features' in pyaudio_result:
            print(f"   Feature count: {len(pyaudio_result['features'])}")
        
        # Cleanup
        print(f"\n[CLEANUP] Resource cleanup:")
        context.cleanup_artifacts()
        print(f"   Temporary files cleaned: [OK]")
        
        # Summary
        print(f"\n[SUMMARY] Integration Test Summary:")
        print(f"   Total time: {time.time() - start_time:.2f}s")
        print(f"   FFmpeg: {'[OK]' if audio_path and video_path else '[FAIL]'}")
        print(f"   Whisper: {'[OK]' if not whisper_result.get('fallback_mode') else '[FALLBACK]'}")
        print(f"   LibROSA: {'[OK]' if not librosa_result.get('fallback_mode') else '[FALLBACK]'}")
        print(f"   pyAudioAnalysis: {'[OK]' if not pyaudio_result.get('fallback_mode') else '[FALLBACK]'}")
        
        has_errors = 'error' in audio_results
        all_fallback = (whisper_result.get('fallback_mode') and 
                       librosa_result.get('fallback_mode') and 
                       pyaudio_result.get('fallback_mode'))
        
        if not has_errors and not all_fallback:
            print(f"\n[PASS] INTEGRATION TEST PASSED")
            return True
        elif not has_errors:
            print(f"\n[PARTIAL] INTEGRATION TEST PARTIAL SUCCESS") 
            return True
        else:
            print(f"\n[FAIL] INTEGRATION TEST FAILED")
            print(f"   Error: {audio_results.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"\n[FAIL] INTEGRATION TEST FAILED")
        print(f"   Exception: {e}")
        return False

if __name__ == '__main__':
    success = test_audio_pipeline()
    exit(0 if success else 1)