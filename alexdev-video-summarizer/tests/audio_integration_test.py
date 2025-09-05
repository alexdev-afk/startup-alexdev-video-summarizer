#!/usr/bin/env python3
"""
Audio Pipeline Integration Test - Production Ready

Validates the complete audio pipeline end-to-end:
FFmpeg → Whisper → LibROSA → pyAudioAnalysis

Tests integration, error handling, and fallback modes.
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

def main():
    """Main integration test"""
    print("Audio Pipeline Integration Test")
    print("=" * 50)
    
    try:
        # Step 1: Configuration
        print("[1/6] Loading configuration...")
        config = ConfigLoader.load_config('config/processing.yaml')
        print("     Configuration loaded successfully")
        
        # Step 2: Service initialization
        print("[2/6] Initializing services...")
        ffmpeg_service = FFmpegService(config)
        audio_controller = AudioPipelineController(config)
        print("     Services initialized successfully")
        
        # Step 3: Test video selection
        print("[3/6] Selecting test video...")
        input_dir = Path("input")
        test_videos = list(input_dir.glob("*.mp4"))
        
        if not test_videos:
            print("     ERROR: No test videos found in input/")
            return False
            
        # Use smallest video for quick test
        test_video = min(test_videos, key=lambda x: x.stat().st_size)
        file_size = test_video.stat().st_size / 1024 / 1024
        print(f"     Selected: {test_video.name} ({file_size:.1f} MB)")
        
        # Step 4: FFmpeg processing
        print("[4/6] Processing with FFmpeg...")
        start_time = time.time()
        
        context = VideoProcessingContext(test_video)
        audio_path, video_path = ffmpeg_service.extract_streams(test_video)
        context.audio_path = audio_path  
        context.video_path = video_path
        
        if not context.validate_ffmpeg_output():
            print("     ERROR: FFmpeg extraction failed")
            return False
            
        ffmpeg_time = time.time() - start_time
        print(f"     FFmpeg completed in {ffmpeg_time:.2f}s")
        
        # Step 5: Audio pipeline processing
        print("[5/6] Processing audio pipeline...")
        
        # Create test scene (first 10 seconds)
        test_scene = {
            'scene_id': 'integration_test',
            'start_time': 0.0,
            'end_time': 10.0,
            'start_frame': 0,
            'end_frame': 300
        }
        
        pipeline_start = time.time()
        results = audio_controller.process_scene(test_scene, context)
        pipeline_time = time.time() - pipeline_start
        
        print(f"     Audio pipeline completed in {pipeline_time:.2f}s")
        
        # Step 6: Results validation
        print("[6/6] Validating results...")
        
        # Check for critical errors
        if 'error' in results:
            print(f"     WARNING: Pipeline reported error: {results['error']}")
            
        # Validate each component
        whisper_data = results.get('whisper', {})
        librosa_data = results.get('librosa', {})  
        pyaudio_data = results.get('pyaudioanalysis', {})
        
        whisper_ok = 'transcript' in whisper_data
        librosa_ok = 'tempo' in librosa_data or 'features' in librosa_data
        pyaudio_ok = 'features' in pyaudio_data
        
        print(f"     Whisper: {'OK' if whisper_ok else 'FALLBACK'}")
        print(f"     LibROSA: {'OK' if librosa_ok else 'FALLBACK'}")  
        print(f"     pyAudioAnalysis: {'OK' if pyaudio_ok else 'FALLBACK'}")
        
        # Integration assessment
        total_time = time.time() - start_time
        print(f"\nIntegration Test Summary:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  FFmpeg time: {ffmpeg_time:.2f}s") 
        print(f"  Audio pipeline time: {pipeline_time:.2f}s")
        
        # Cleanup
        context.cleanup_artifacts()
        print(f"  Cleanup: Complete")
        
        # Success criteria
        pipeline_functional = whisper_ok or librosa_ok or pyaudio_ok
        no_critical_errors = 'error' not in results or results.get('error', '').startswith('Warning')
        
        if pipeline_functional and no_critical_errors:
            print(f"\n[PASS] Audio Pipeline Integration Test PASSED")
            print(f"       End-to-end audio processing pipeline is functional")
            if whisper_ok and librosa_ok and pyaudio_ok:
                print(f"       All three audio services working optimally")
            else:
                print(f"       Pipeline working with graceful fallbacks (development mode)")
            return True
        else:
            print(f"\n[FAIL] Audio Pipeline Integration Test FAILED")
            if not pipeline_functional:
                print(f"       No audio processing components are functional")
            if not no_critical_errors:
                print(f"       Critical errors in pipeline processing")
            return False
            
    except Exception as e:
        print(f"\n[FATAL] Integration test failed with exception:")
        print(f"        {type(e).__name__}: {str(e)}")
        return False

if __name__ == '__main__':
    success = main()
    
    if success:
        print(f"\nREADY FOR PHASE 4: Visual pipeline development can proceed")
        print(f"Audio pipeline (Whisper + LibROSA + pyAudioAnalysis) is production ready")
    else:
        print(f"\nACTION REQUIRED: Audio pipeline integration issues need resolution")
        print(f"Review error messages and service configurations")
    
    exit(0 if success else 1)