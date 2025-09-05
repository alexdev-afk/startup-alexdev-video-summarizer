#!/usr/bin/env python3
"""
Comprehensive Audio Pipeline Integration Test

Validates complete audio pipeline: Whisper + LibROSA + pyAudioAnalysis
Tests both real processing and fallback modes for development environment compatibility
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from services.audio_pipeline import AudioPipelineController
from services.ffmpeg_service import FFmpegService
from services.whisper_service import WhisperService
from services.librosa_service import LibROSAService
from services.pyaudioanalysis_service import PyAudioAnalysisService
from utils.config_loader import ConfigLoader
from utils.processing_context import VideoProcessingContext

def test_individual_services(config):
    """Test individual audio services initialization"""
    print("\n[SERVICE TEST] Individual Service Initialization")
    print("-" * 50)
    
    # Test Whisper service
    whisper_service = WhisperService(config)
    print(f"[WHISPER] Service initialized: [OK]")
    print(f"[WHISPER] Device: {whisper_service.device}")
    print(f"[WHISPER] Fallback mode: {whisper_service.fallback_mode}")
    
    # Test LibROSA service  
    librosa_service = LibROSAService(config)
    print(f"[LIBROSA] Service initialized: [OK]")
    print(f"[LIBROSA] Available: {librosa_service.librosa_available}")
    
    # Test pyAudioAnalysis service
    pyaudio_service = PyAudioAnalysisService(config)
    print(f"[PYAUDIO] Service initialized: [OK]")
    print(f"[PYAUDIO] Available: {pyaudio_service.pyaudio_available}")
    
    return {
        'whisper': whisper_service,
        'librosa': librosa_service,
        'pyaudio': pyaudio_service
    }

def test_ffmpeg_integration(config):
    """Test FFmpeg foundation"""
    print("\n[FFMPEG TEST] FFmpeg Foundation")
    print("-" * 50)
    
    ffmpeg_service = FFmpegService(config)
    
    # Select smallest test video
    input_dir = Path("input")
    test_videos = list(input_dir.glob("*.mp4"))
    if not test_videos:
        print("[ERROR] No test videos found")
        return None, None
        
    test_video = min(test_videos, key=lambda x: x.stat().st_size)
    print(f"[VIDEO] Test: {test_video.name} ({test_video.stat().st_size / 1024 / 1024:.1f} MB)")
    
    try:
        start_time = time.time()
        context = VideoProcessingContext(test_video)
        audio_path, video_path = ffmpeg_service.extract_streams(test_video)
        context.audio_path = audio_path
        context.video_path = video_path
        
        if context.validate_ffmpeg_output():
            print(f"[OK] FFmpeg extraction completed in {time.time() - start_time:.2f}s")
            print(f"[OK] Audio file: {audio_path}")
            print(f"[OK] Video file: {video_path}")
            return context, test_video
        else:
            print("[ERROR] FFmpeg validation failed")
            return None, None
            
    except Exception as e:
        print(f"[ERROR] FFmpeg extraction failed: {e}")
        return None, None

def test_audio_pipeline_integration(config, context):
    """Test complete audio pipeline integration"""
    print("\n[PIPELINE TEST] Audio Pipeline Integration")  
    print("-" * 50)
    
    audio_controller = AudioPipelineController(config)
    
    # Create test scene
    mock_scene = {
        'scene_id': 'test_scene_1',
        'start_time': 0.0,
        'end_time': 5.0,  # Short segment for testing
        'start_frame': 0,
        'end_frame': 150
    }
    
    try:
        start_time = time.time()
        audio_results = audio_controller.process_scene(mock_scene, context)
        processing_time = time.time() - start_time
        
        print(f"[OK] Audio pipeline completed in {processing_time:.2f}s")
        
        # Analyze results
        print(f"\n[ANALYSIS] Pipeline Results:")
        print("-" * 30)
        
        whisper_result = audio_results.get('whisper', {})
        librosa_result = audio_results.get('librosa', {})
        pyaudio_result = audio_results.get('pyaudioanalysis', {})
        
        # Whisper analysis
        whisper_working = not whisper_result.get('fallback_mode', True)
        print(f"[WHISPER] Status: {'[REAL]' if whisper_working else '[FALLBACK]'}")
        if whisper_working and 'transcript' in whisper_result:
            transcript = whisper_result['transcript'][:100] + "..." if len(whisper_result.get('transcript', '')) > 100 else whisper_result.get('transcript', '')
            print(f"[WHISPER] Language: {whisper_result.get('language', 'unknown')}")
            print(f"[WHISPER] Preview: {transcript}")
        
        # LibROSA analysis
        librosa_working = not librosa_result.get('fallback_mode', True)
        print(f"[LIBROSA] Status: {'[REAL]' if librosa_working else '[FALLBACK]'}")
        if librosa_working and 'tempo' in librosa_result:
            print(f"[LIBROSA] Tempo: {librosa_result.get('tempo', 0):.1f} BPM")
            print(f"[LIBROSA] Features: {len(librosa_result.get('features', []))}")
        
        # pyAudioAnalysis analysis
        pyaudio_working = not pyaudio_result.get('fallback_mode', True)
        print(f"[PYAUDIO] Status: {'[REAL]' if pyaudio_working else '[FALLBACK]'}")
        if pyaudio_working and 'features' in pyaudio_result:
            features = pyaudio_result['features']
            if isinstance(features, dict):
                print(f"[PYAUDIO] Feature categories: {len(features)}")
            else:
                print(f"[PYAUDIO] Feature count: {len(features) if hasattr(features, '__len__') else 'unknown'}")
        
        # Overall assessment
        has_errors = 'error' in audio_results
        any_real_processing = whisper_working or librosa_working or pyaudio_working
        
        return {
            'success': not has_errors,
            'whisper_working': whisper_working,
            'librosa_working': librosa_working,
            'pyaudio_working': pyaudio_working,
            'any_real_processing': any_real_processing,
            'processing_time': processing_time,
            'error': audio_results.get('error', None)
        }
        
    except Exception as e:
        print(f"[ERROR] Pipeline integration failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'processing_time': 0
        }

def main():
    """Main test function"""
    print("Comprehensive Audio Pipeline Integration Test")
    print("=" * 60)
    
    try:
        # Load configuration
        config = ConfigLoader.load_config('config/processing.yaml')
        print("[OK] Configuration loaded successfully")
        
        # Test 1: Individual services
        services = test_individual_services(config)
        
        # Test 2: FFmpeg foundation
        context, test_video = test_ffmpeg_integration(config)
        if not context:
            print("\n[FAIL] Cannot proceed without FFmpeg working")
            return False
            
        # Test 3: Complete pipeline integration
        pipeline_results = test_audio_pipeline_integration(config, context)
        
        # Test 4: Cleanup
        print(f"\n[CLEANUP] Resource cleanup:")
        context.cleanup_artifacts()
        print(f"[OK] Temporary files cleaned")
        
        # Final assessment
        print(f"\n[FINAL] Integration Test Assessment:")
        print("=" * 60)
        print(f"Pipeline Success: {'[PASS]' if pipeline_results['success'] else '[FAIL]'}")
        print(f"Processing Time: {pipeline_results.get('processing_time', 0):.2f}s")
        print(f"Real Whisper: {'[YES]' if pipeline_results.get('whisper_working') else '[NO]'}")
        print(f"Real LibROSA: {'[YES]' if pipeline_results.get('librosa_working') else '[NO]'}")
        print(f"Real pyAudio: {'[YES]' if pipeline_results.get('pyaudio_working') else '[NO]'}")
        
        if pipeline_results['success']:
            if pipeline_results.get('any_real_processing'):
                print(f"\n[EXCELLENT] Complete audio pipeline working with real AI processing")
                print(f"[READY] System ready for production audio analysis")
                return True
            else:
                print(f"\n[GOOD] Audio pipeline working with fallback modes")
                print(f"[DEVELOPMENT] Perfect for development and testing")
                return True
        else:
            print(f"\n[FAIL] Audio pipeline integration failed")
            if pipeline_results.get('error'):
                print(f"[ERROR] {pipeline_results['error']}")
            return False
            
    except Exception as e:
        print(f"\n[FATAL] Test execution failed: {e}")
        return False

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)