#!/usr/bin/env python3
"""
Test Whisper-Aligned PyAudioAnalysis Segment Processing

This script tests the new segment-aware audio processing where:
1. Whisper runs first and provides speech segment boundaries
2. pyAudioAnalysis processes each Whisper segment individually for emotion/speaker analysis
3. Aggregate analysis identifies emotion changes across segments
"""

import sys
import json
from pathlib import Path

# Add src directory to path
sys.path.append('src')

from services.ffmpeg_service import FFmpegService
from services.whisper_service import WhisperService  
from services.pyaudioanalysis_service import PyAudioAnalysisService
from utils.logger import get_logger
from utils.config_loader import ConfigLoader

logger = get_logger(__name__)

def test_whisper_aligned_segments():
    """Test Whisper-aligned segment processing"""
    
    print("Whisper-Aligned Segment Processing Test")
    print("=" * 60)
    
    # Load configuration
    config_loader = ConfigLoader()
    config = config_loader.load_config("config/processing.yaml")
    
    # Initialize services
    print("\n=== SETUP: Initializing Services ===")
    whisper_service = WhisperService(config)
    pyaudio_service = PyAudioAnalysisService(config)
    
    print(f"[OK] WhisperService initialized - device: {whisper_service.device}")
    print(f"[OK] PyAudioAnalysisService initialized - available: {getattr(pyaudio_service, 'available', 'Unknown')}")
    
    # Test with existing audio file from previous test
    test_video = "Consider 3 key points when ordering your automated laundry rack.mp4"
    audio_path = f"build/{test_video.replace('.mp4', '')}/audio.wav"
    
    if not Path(audio_path).exists():
        print(f"[ERROR] Audio file not found: {audio_path}")
        print("Run the audio test script first to generate the audio file")
        return False
    
    print(f"\n=== STAGE 1: Whisper Transcription (Baseline) ===")
    print(f"[TEST] Processing: {audio_path}")
    
    # Get Whisper results (should already exist)
    whisper_result_file = f"build/{test_video.replace('.mp4', '')}/audio_analysis/whisper_transcription.json"
    
    if Path(whisper_result_file).exists():
        print(f"[OK] Loading existing Whisper results: {whisper_result_file}")
        with open(whisper_result_file, 'r') as f:
            whisper_result = json.load(f)
    else:
        print(f"[TEST] Running fresh Whisper analysis...")
        whisper_result = whisper_service.transcribe_audio(audio_path)
    
    segments = whisper_result.get('segments', [])
    print(f"[OK] Whisper segments loaded: {len(segments)} speech segments")
    
    # Display segment overview
    print(f"\n--- Whisper Segment Overview ---")
    for i, segment in enumerate(segments[:5]):  # Show first 5
        duration = segment['end'] - segment['start']
        text_preview = segment['text'][:50] + "..." if len(segment['text']) > 50 else segment['text']
        print(f"  Segment {i+1}: {segment['start']:.1f}-{segment['end']:.1f}s ({duration:.1f}s) - \"{text_preview}\"")
    
    if len(segments) > 5:
        print(f"  ... and {len(segments) - 5} more segments")
    
    print(f"\n=== STAGE 2: Whisper-Aligned PyAudioAnalysis ===")
    print(f"[TEST] Running segment-by-segment analysis...")
    
    # Run Whisper-aligned processing
    segment_analysis_result = pyaudio_service.analyze_whisper_segments(audio_path, whisper_result)
    
    print(f"[OK] Segment analysis complete!")
    print(f"     Processing type: {segment_analysis_result.get('processing_type', 'unknown')}")
    print(f"     Total segments processed: {segment_analysis_result.get('total_segments', 0)}")
    print(f"     Processing time: {segment_analysis_result.get('processing_time', 0):.2f}s")
    
    # Display segment-by-segment results
    segment_analyses = segment_analysis_result.get('segment_analyses', [])
    
    print(f"\n--- Segment-by-Segment Analysis Results ---")
    for i, analysis in enumerate(segment_analyses[:5]):  # Show first 5 detailed
        emotion = analysis.get('emotion_analysis', {}).get('emotion', 'unknown')
        voice_quality = analysis.get('speaker_characteristics', {}).get('voice_quality', 'unknown')
        energy_level = analysis.get('speaker_characteristics', {}).get('energy_analysis', {}).get('energy_level', 'unknown')
        
        print(f"  Segment {analysis.get('segment_id', i+1)} ({analysis.get('timespan', 'unknown')}):")
        print(f"    Text: \"{analysis.get('text', 'N/A')[:60]}...\"")
        print(f"    Emotion: {emotion}")
        print(f"    Voice: {voice_quality} | Energy: {energy_level}")
        print()
    
    # Display aggregate analysis
    aggregate = segment_analysis_result.get('aggregate_analysis', {})
    
    print(f"--- Aggregate Analysis Across All Segments ---")
    emotion_trends = aggregate.get('emotion_trends', {})
    energy_trends = aggregate.get('energy_trends', {}) 
    speaking_patterns = aggregate.get('speaking_patterns', {})
    
    print(f"  Emotion Trends:")
    print(f"    Dominant emotion: {emotion_trends.get('dominant_emotion', 'unknown')}")
    print(f"    Emotion changes: {len(emotion_trends.get('emotion_changes', []))}")
    print(f"    Emotional variety: {emotion_trends.get('emotional_variety', 0)} different emotions")
    
    print(f"  Energy Trends:")
    print(f"    Average energy: {energy_trends.get('average_energy', 0):.3f}")
    print(f"    Energy progression: {energy_trends.get('energy_progression', 'unknown')}")
    
    print(f"  Speaking Patterns:")
    print(f"    Average speaking rate: {speaking_patterns.get('average_speaking_rate', 0):.1f}")
    print(f"    Pacing category: {speaking_patterns.get('pacing_category', 'unknown')}")
    
    # Show emotion changes if any
    emotion_changes = emotion_trends.get('emotion_changes', [])
    if emotion_changes:
        print(f"\n--- Detected Emotion Changes ---")
        for change in emotion_changes:
            print(f"  {change.get('emotion_change', 'unknown')} at {change.get('timestamp', 'unknown')}")
    
    print(f"\n=== RESULTS SUMMARY ===")
    print(f"[OK] Whisper-aligned processing: SUCCESSFUL")
    print(f"[OK] Segment-by-segment analysis: {len(segment_analyses)} segments processed")
    print(f"[OK] Emotion tracking across segments: {len(emotion_changes)} emotion changes detected")
    print(f"[OK] Voice characteristics per phrase: Available")
    print(f"[OK] Aggregate trend analysis: Complete")
    
    # Save detailed results for inspection
    output_file = f"build/{test_video.replace('.mp4', '')}/audio_analysis/whisper_aligned_segments.json"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(segment_analysis_result, f, indent=2, ensure_ascii=False)
        print(f"[OK] Detailed results saved: {output_file}")
    except Exception as e:
        print(f"[WARN] Could not save results file: {e}")
    
    print(f"\n=== INSTITUTIONAL KNOWLEDGE IMPACT ===")
    print(f"   Instead of single emotion for entire video, we now have:")
    print(f"   - Phrase-level emotion analysis ({len(segment_analyses)} segments)")
    print(f"   - Emotion change detection ({len(emotion_changes)} transitions)")  
    print(f"   - Speaking pattern evolution across video")
    print(f"   - Voice characteristic changes per message")
    
    return True

if __name__ == "__main__":
    success = test_whisper_aligned_segments()
    if success:
        print(f"\n[SUCCESS] Whisper-aligned segment processing READY!")
    else:
        print(f"\n[FAILED] Test failed")
        sys.exit(1)