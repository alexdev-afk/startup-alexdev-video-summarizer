#!/usr/bin/env python3
"""
Test Enhanced Audio Pipeline Stages

Tests the new advanced audio processing pipeline:
1. WhisperX + Silero VAD with multi-strategy chunking and speaker diarization
2. LibROSA with smart music segmentation based on acoustic feature changes  
3. pyAudioAnalysis with interpretive analysis and Whisper alignment

Verifies all our recent improvements work correctly and build directory integrity.
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
    """Test Whisper transcription stage with advanced VAD chunking and diarization"""
    print("\n=== AUDIO STAGE 1: WhisperX + Silero VAD Transcription ===")
    
    try:
        whisper = WhisperService(test_data['config'])
        print(f"[OK] WhisperService initialized")
        print(f"  Device: {whisper.device}")
        print(f"  Chunking Strategy: {whisper.chunking_strategy}")
        print(f"  Max Chunk Duration: {whisper.max_chunk_duration}s")
        print(f"  Diarization Enabled: {whisper.enable_diarization}")
        
        # Test transcription with advanced features
        audio_file = test_data['audio_file']
        print(f"[TEST] Transcribing with advanced VAD chunking: {audio_file.name}")
        
        result = whisper.transcribe_audio(audio_file)
        
        # Display advanced results
        print(f"[OK] Transcription complete: {type(result)}")
        print(f"  Total Segments: {len(result.get('segments', []))}")
        print(f"  Speakers Detected: {len(result.get('speakers', []))}")
        print(f"  Processing Time: {result.get('processing_time', 0):.2f}s")
        
        # Show VAD analysis
        vad_info = result.get('vad_analysis', {})
        if vad_info:
            print(f"  VAD Chunks: {vad_info.get('total_chunks', 0)}")
            print(f"  Speech Duration: {vad_info.get('total_speech_duration', 0):.2f}s")
            print(f"  VAD Threshold: {vad_info.get('vad_threshold', 'N/A')}")
        
        # Show model info
        model_info = result.get('model_info', {})
        if model_info:
            print(f"  WhisperX Available: {model_info.get('whisperx_enabled', False)}")
            print(f"  Silero VAD Available: {model_info.get('silero_vad_enabled', False)}")
            print(f"  Diarization Available: {model_info.get('diarization_enabled', False)}")
        
        # Show sample segment with speaker info
        segments = result.get('segments', [])
        if segments:
            sample_seg = segments[0]
            print(f"  Sample: '{sample_seg.get('text', '')[:50]}...' - {sample_seg.get('speaker', 'Unknown')}")
        
        # Check build directory
        return check_build_directory("Whisper", test_data['build_dir'])
        
    except Exception as e:
        print(f"[FAIL] Whisper stage failed: {e}")
        check_build_directory("Whisper (FAILED)", test_data['build_dir'])
        return False

def test_librosa_stage(test_data):
    """Test LibROSA smart music segmentation and analysis"""
    print("\n=== AUDIO STAGE 2: LibROSA Smart Music Segmentation ===")
    
    try:
        librosa = LibROSAService(test_data['config'])
        from services.librosa_service import LIBROSA_AVAILABLE
        print(f"[OK] LibROSAService initialized")
        print(f"  LibROSA Available: {LIBROSA_AVAILABLE}")
        print(f"  Music Segmentation: {librosa.enable_music_segmentation}")
        print(f"  Smart Segmentation: {librosa.adaptive_segmentation}")
        
        if librosa.smart_segmentation:
            print(f"  Min Segment Length: {librosa.smart_segmentation.min_segment_length}s")
            print(f"  Max Segment Length: {librosa.smart_segmentation.max_segment_length}s")
        else:
            print(f"  Fixed Segment Length: {librosa.segment_length}s")
            print(f"  Segment Overlap: {librosa.segment_overlap}s")
        
        # Test smart music analysis
        audio_file = test_data['audio_file']
        print(f"[TEST] Smart music analysis: {audio_file.name}")
        
        result = librosa.analyze_audio_segment(str(audio_file))
        
        # Display smart segmentation results
        print(f"[OK] Music analysis complete: {type(result)}")
        print(f"  Segmentation Enabled: {result.get('segmentation_enabled', False)}")
        print(f"  Total Segments: {result.get('total_segments', 1)}")
        print(f"  Processing Time: {result.get('processing_time', 0):.2f}s")
        
        # Show music segmentation info
        music_seg_info = result.get('music_segmentation', {})
        if music_seg_info:
            print(f"  Segmentation Method: {'Smart' if music_seg_info.get('enabled') else 'Single'}")
            print(f"  Detected Segments: {music_seg_info.get('segment_count', 1)}")
            print(f"  Total Duration: {music_seg_info.get('total_duration', 0):.2f}s")
        
        # Show aggregated tempo analysis
        tempo_info = result.get('tempo_analysis', {})
        if tempo_info:
            print(f"  Detected Tempo: {tempo_info.get('tempo', 'N/A'):.1f} BPM")
            if 'tempo_variation' in tempo_info:
                print(f"  Tempo Variation: {tempo_info.get('tempo_variation', 0):.2f}")
                tempo_range = tempo_info.get('tempo_range', [])
                if len(tempo_range) == 2:
                    print(f"  Tempo Range: {tempo_range[0]:.1f} - {tempo_range[1]:.1f} BPM")
        
        # Show audio classification
        audio_class = result.get('audio_classification', {})
        if audio_class:
            print(f"  Audio Type: {audio_class.get('type', 'unknown')}")
            if 'type_confidence' in audio_class:
                print(f"  Type Confidence: {audio_class.get('type_confidence', 0):.2f}")
                type_dist = audio_class.get('type_distribution', {})
                if type_dist:
                    print(f"  Type Distribution: {type_dist}")
        
        # Show sample segment analysis
        segment_analyses = result.get('segment_analysis', [])
        if segment_analyses:
            print(f"  Sample Segment: {segment_analyses[0].get('segment_timing', {})}")
            sample_tempo = segment_analyses[0].get('tempo_analysis', {}).get('tempo', 'N/A')
            print(f"  Sample Segment Tempo: {sample_tempo}")
        
        # Check build directory
        return check_build_directory("LibROSA", test_data['build_dir'])
        
    except Exception as e:
        print(f"[FAIL] LibROSA stage failed: {e}")
        import traceback
        print(traceback.format_exc())
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

def test_vad_chunking_strategies(test_data):
    """Test the new VAD chunking strategies with real audio file"""
    print("\n=== BONUS TEST: Advanced VAD Chunking Strategies ===")
    
    try:
        # Test the new VAD chunking module with real audio
        from services.whisper_vad_chunking import AdvancedVADChunking
        import torch
        import torchaudio
        
        config = test_data['config']
        chunking = AdvancedVADChunking(config)
        
        print(f"[OK] AdvancedVADChunking initialized")
        print(f"  Chunking Strategy: {chunking.chunking_strategy}")
        print(f"  Max Chunk Duration: {chunking.max_chunk_duration}s")
        print(f"  Min Chunk Duration: {chunking.min_chunk_duration}s")
        print(f"  Energy-Based Splitting: {chunking.energy_based_splitting}")
        
        # Load real audio file
        audio_file = test_data['audio_file']
        print(f"[TEST] Loading real audio file: {audio_file.name}")
        
        # Load audio with torchaudio for VAD processing
        wav, sr = torchaudio.load(str(audio_file))
        if wav.shape[0] > 1:
            wav = wav[0]  # Take first channel
        wav = wav.squeeze()
        
        print(f"  Audio loaded: {len(wav)} samples at {sr}Hz ({len(wav)/sr:.2f}s)")
        
        # Get VAD timestamps from real audio
        if hasattr(chunking, 'vad_model'):
            print(f"[TEST] Running Silero VAD on real audio")
            speech_timestamps = chunking.vad_model(wav, sr)
            print(f"  VAD detected {len(speech_timestamps)} speech segments")
            
            if speech_timestamps:
                # Test chunking strategies with real VAD data
                def collect_chunks_real(timestamps, wav_data):
                    return wav_data  # Return actual audio segment
                
                strategies = ['gap_based', 'duration_based', 'multi_strategy']
                
                for strategy in strategies:
                    original_strategy = chunking.chunking_strategy
                    chunking.chunking_strategy = strategy
                    print(f"\n[TEST] Testing {strategy} chunking with real VAD data:")
                    
                    if strategy == 'gap_based':
                        chunks = chunking.chunk_by_gaps(speech_timestamps, wav, sr, collect_chunks_real)
                    elif strategy == 'duration_based':
                        chunks = chunking.chunk_by_duration(speech_timestamps, wav, sr, collect_chunks_real)
                    else:  # multi_strategy
                        chunks = chunking.chunk_multi_strategy(speech_timestamps, wav, sr, collect_chunks_real)
                    
                    print(f"  Generated {len(chunks)} chunks with {strategy}")
                    for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
                        print(f"    Chunk {i}: {chunk['start_seconds']:.2f}s - {chunk['end_seconds']:.2f}s ({chunk['duration']:.2f}s)")
                    
                    if len(chunks) > 3:
                        print(f"    ... and {len(chunks) - 3} more chunks")
                    
                    chunking.chunking_strategy = original_strategy
            else:
                print("[WARN] No speech detected by VAD - cannot test chunking strategies")
        else:
            print("[WARN] VAD model not available - cannot test chunking strategies")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] VAD chunking test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def test_smart_music_segmentation(test_data):
    """Test the smart music segmentation with real audio file"""
    print("\n=== BONUS TEST: Smart Music Segmentation ===")
    
    try:
        # Test the new smart music segmentation with real audio
        from services.music_segmentation import SmartMusicSegmentation
        import librosa
        
        config = test_data['config']
        smart_seg = SmartMusicSegmentation(config)
        
        print(f"[OK] SmartMusicSegmentation initialized")
        print(f"  Min Segment Length: {smart_seg.min_segment_length}s")
        print(f"  Max Segment Length: {smart_seg.max_segment_length}s")
        print(f"  Spectral Change Threshold: {smart_seg.spectral_change_threshold}")
        print(f"  Tempo Change Threshold: {smart_seg.tempo_change_threshold} BPM")
        
        # Load real audio file
        audio_file = test_data['audio_file']
        print(f"[TEST] Loading real audio file: {audio_file.name}")
        
        # Load audio with librosa for music analysis
        y, sr = librosa.load(str(audio_file), sr=smart_seg.sample_rate)
        duration = len(y) / sr
        
        print(f"  Audio loaded: {len(y)} samples at {sr}Hz ({duration:.2f}s)")
        print(f"[TEST] Testing smart boundary detection on real {duration:.1f}s audio")
        
        boundaries = smart_seg.detect_music_boundaries(y)
        
        print(f"[OK] Smart segmentation detected {len(boundaries)} segments:")
        for i, (start_idx, end_idx, start_time, end_time) in enumerate(boundaries):
            print(f"  Segment {i}: {start_time:.2f}s - {end_time:.2f}s ({end_time - start_time:.2f}s)")
        
        # Validate boundaries make sense
        total_boundary_duration = sum(end_time - start_time for _, _, start_time, end_time in boundaries)
        print(f"  Total boundary duration: {total_boundary_duration:.2f}s (vs {duration:.2f}s original)")
        
        # Check for reasonable segment lengths
        segment_durations = [end_time - start_time for _, _, start_time, end_time in boundaries]
        if segment_durations:
            avg_duration = sum(segment_durations) / len(segment_durations)
            min_duration = min(segment_durations)
            max_duration = max(segment_durations)
            print(f"  Segment lengths: min={min_duration:.2f}s, avg={avg_duration:.2f}s, max={max_duration:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Smart music segmentation test failed: {e}")
        import traceback
        print(traceback.format_exc())
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
    
    # Test bonus features first (now using real audio files)
    print(f"\n{'='*20} TESTING: Bonus Advanced Features {'='*20}")
    bonus_tests = [
        ("Advanced VAD Chunking", test_vad_chunking_strategies),
        ("Smart Music Segmentation", test_smart_music_segmentation),
    ]
    
    bonus_results = {}
    for test_name, test_func in bonus_tests:
        try:
            bonus_results[test_name] = test_func(test_data)
        except Exception as e:
            print(f"[FAIL] {test_name} crashed: {e}")
            bonus_results[test_name] = False
    
    # Test main audio pipeline stages
    stages = [
        ("WhisperX + Silero VAD", test_whisper_stage),
        ("LibROSA Smart Segmentation", test_librosa_stage),
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
    print("\n" + "=" * 70)
    print("ENHANCED AUDIO PIPELINE TEST SUMMARY:")
    
    print("\nBonus Advanced Features:")
    for test_name, passed in bonus_results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status}: {test_name}")
    
    print("\nMain Audio Pipeline Stages:")
    for stage_name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status}: {stage_name}")
    
    # Overall success rate
    all_results = {**bonus_results, **results}
    passed_count = sum(1 for passed in all_results.values() if passed)
    total_count = len(all_results)
    success_rate = (passed_count / total_count) * 100 if total_count > 0 else 0
    
    print(f"\nOverall Success Rate: {passed_count}/{total_count} ({success_rate:.1f}%)")
    
    if success_rate == 100:
        print("[SUCCESS] All enhanced audio features working perfectly!")
    elif success_rate >= 80:
        print("[GOOD] Most features working - minor issues detected")
    else:
        print("[WARNING] Multiple issues detected - needs attention")
    
    # Final build directory check
    print(f"\nFINAL BUILD DIRECTORY STATE:")
    if test_data['build_dir'].exists():
        check_build_directory("FINAL", test_data['build_dir'])
    else:
        print("[FAIL] Build directory completely deleted!")

if __name__ == "__main__":
    main()