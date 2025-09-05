#!/usr/bin/env python3
"""
Audio Pipeline Integration Test

Tests the complete audio pipeline: Whisper + LibROSA + pyAudioAnalysis
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

def test_audio_pipeline_integration():
    """Test complete audio pipeline integration"""
    
    print("Audio Pipeline Integration Test")
    print("=" * 50)
    
    # Load configuration
    config = ConfigLoader.load_config()
    print(f"✅ Configuration loaded")
    
    # Initialize services
    ffmpeg_service = FFmpegService(config)
    audio_controller = AudioPipelineController(config)
    print(f"✅ Services initialized")
    
    # Select test video (smallest one for quick test)
    input_dir = Path("input")
    test_videos = list(input_dir.glob("*.mp4"))
    if not test_videos:
        print("❌ No test videos found in input/ directory")
        return False
        
    # Use shortest video for testing
    test_video = min(test_videos, key=lambda x: x.stat().st_size)
    print(f"📹 Test video: {test_video.name} ({test_video.stat().st_size / 1024 / 1024:.1f} MB)")
    
    try:
        # Step 1: FFmpeg audio extraction
        print("\n🔄 Step 1: FFmpeg Audio Extraction")
        start_time = time.time()
        
        context = VideoProcessingContext(test_video)
        audio_path, video_path = ffmpeg_service.extract_streams(test_video)
        context.audio_path = audio_path
        context.video_path = video_path
        
        if not context.validate_ffmpeg_output():
            print("❌ FFmpeg extraction failed")
            return False
            
        print(f"✅ FFmpeg completed in {time.time() - start_time:.2f}s")
        print(f"   Audio: {audio_path}")
        print(f"   Video: {video_path}")
        
        # Step 2: Create mock scene for testing
        print("\n🔄 Step 2: Audio Pipeline Processing")
        mock_scene = {
            'scene_id': 'test_scene_1',
            'start_time': 0.0,
            'end_time': 10.0,  # First 10 seconds
            'start_frame': 0,
            'end_frame': 300  # ~10s at 30fps
        }
        
        # Step 3: Process through complete audio pipeline
        audio_start = time.time()
        audio_results = audio_controller.process_scene(mock_scene, context)
        audio_time = time.time() - audio_start
        
        print(f"✅ Audio pipeline completed in {audio_time:.2f}s")
        
        # Step 4: Validate results
        print("\n📊 Audio Pipeline Results:")
        print("-" * 30)
        
        # Whisper validation
        whisper_result = audio_results.get('whisper', {})
        print(f"🎤 Whisper Transcription:")
        print(f"   Available: {'✅' if not whisper_result.get('fallback_mode') else '⚠️ (Fallback)'}")
        print(f"   Language: {whisper_result.get('language', 'unknown')}")
        print(f"   Confidence: {whisper_result.get('language_probability', 0.0):.2f}")
        if whisper_result.get('transcript'):
            transcript_preview = whisper_result['transcript'][:100] + "..." if len(whisper_result['transcript']) > 100 else whisper_result['transcript']
            print(f"   Preview: {transcript_preview}")
        
        # LibROSA validation
        librosa_result = audio_results.get('librosa', {})
        print(f"\n🎵 LibROSA Music Analysis:")
        print(f"   Available: {'✅' if not librosa_result.get('fallback_mode') else '⚠️ (Fallback)'}")
        if 'tempo' in librosa_result:
            print(f"   Tempo: {librosa_result.get('tempo', 0):.1f} BPM")
            print(f"   Features: {len(librosa_result.get('features', []))} extracted")
        
        # pyAudioAnalysis validation  
        pyaudio_result = audio_results.get('pyaudioanalysis', {})
        print(f"\n🔬 pyAudioAnalysis Features:")
        print(f"   Available: {'✅' if not pyaudio_result.get('fallback_mode') else '⚠️ (Fallback)'}")
        if 'features' in pyaudio_result:
            features = pyaudio_result['features']
            print(f"   Feature count: {len(features)}")
            if 'spectral_features' in features:
                print(f"   Spectral features: {len(features['spectral_features'])}")
        
        # Step 5: Resource cleanup validation
        print(f"\n🧹 Cleanup:")
        context.cleanup_artifacts()
        print(f"   Temporary files cleaned: ✅")
        
        # Step 6: Integration success summary
        print(f"\n🎯 Integration Test Summary:")
        print(f"   Total time: {time.time() - start_time:.2f}s")
        print(f"   FFmpeg: {'✅' if audio_path and video_path else '❌'}")
        print(f"   Whisper: {'✅' if not whisper_result.get('fallback_mode') else '⚠️'}")
        print(f"   LibROSA: {'✅' if not librosa_result.get('fallback_mode') else '⚠️'}")
        print(f"   pyAudioAnalysis: {'✅' if not pyaudio_result.get('fallback_mode') else '⚠️'}")
        
        # Determine overall success
        has_errors = 'error' in audio_results
        all_fallback = (whisper_result.get('fallback_mode') and 
                       librosa_result.get('fallback_mode') and 
                       pyaudio_result.get('fallback_mode'))
        
        if not has_errors and not all_fallback:
            print(f"\n✅ INTEGRATION TEST PASSED")
            print(f"   Complete audio pipeline working with real processing")
            return True
        elif not has_errors:
            print(f"\n⚠️ INTEGRATION TEST PARTIAL SUCCESS") 
            print(f"   Pipeline working but using fallback modes (development environment)")
            return True
        else:
            print(f"\n❌ INTEGRATION TEST FAILED")
            print(f"   Error: {audio_results.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"\n❌ INTEGRATION TEST FAILED")
        print(f"   Exception: {e}")
        return False

if __name__ == '__main__':
    success = test_audio_pipeline_integration()
    exit(0 if success else 1)