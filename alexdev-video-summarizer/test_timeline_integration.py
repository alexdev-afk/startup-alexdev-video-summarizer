"""
Test Timeline Integration

Test the complete timeline generation and merging pipeline with the new
LibROSA and pyAudioAnalysis timeline services.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from services.librosa_timeline_service import LibROSATimelineService
from services.pyaudio_timeline_service import PyAudioTimelineService  
from services.timeline_merger_service import TimelineMergerService
from utils.timeline_schema import ServiceTimeline
import yaml


def load_test_config():
    """Load configuration for testing"""
    return {
        'cpu_pipeline': {
            'librosa': {
                'sample_rate': 22050,
                'tempo_change_threshold': 5.0,
                'analysis_window': 5.0,
                'overlap_window': 2.5
            },
            'pyaudioanalysis': {
                'window_size': 0.050,
                'step_size': 0.025,
                'emotion_change_threshold': 0.3,
                'analysis_window': 3.0,
                'overlap_window': 1.5
            }
        },
        'timeline_merger': {
            'priority_order': ['whisper', 'librosa', 'pyaudio'],
            'confidence_threshold': 0.5,
            'overlap_tolerance': 0.1
        }
    }


def create_test_audio_file():
    """Create a test audio file for testing"""
    import numpy as np
    import scipy.io.wavfile as wavfile
    
    # Generate test audio: sine wave with changing frequency
    sample_rate = 22050
    duration = 15.0  # 15 seconds
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Create audio with changing characteristics for event detection
    audio = np.zeros_like(t)
    
    # First 5 seconds: low frequency (speech-like)
    mask1 = t < 5.0
    audio[mask1] = 0.3 * np.sin(2 * np.pi * 200 * t[mask1])
    
    # Next 5 seconds: higher frequency (music-like) 
    mask2 = (t >= 5.0) & (t < 10.0)
    audio[mask2] = 0.5 * np.sin(2 * np.pi * 400 * t[mask2])
    
    # Final 5 seconds: mixed frequencies
    mask3 = t >= 10.0
    audio[mask3] = 0.2 * np.sin(2 * np.pi * 300 * t[mask3]) + 0.1 * np.sin(2 * np.pi * 600 * t[mask3])
    
    # Add some noise for realism
    noise = 0.05 * np.random.normal(0, 1, len(audio))
    audio = audio + noise
    
    # Normalize to int16
    audio_int16 = (audio * 32767).astype(np.int16)
    
    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    wavfile.write(temp_file.name, sample_rate, audio_int16)
    temp_file.close()
    
    return temp_file.name, duration


def test_librosa_timeline_service():
    """Test LibROSA timeline service"""
    print("\n=== Testing LibROSA Timeline Service ===")
    
    config = load_test_config()
    service = LibROSATimelineService(config)
    
    # Create test audio
    audio_file, duration = create_test_audio_file()
    
    try:
        # Generate timeline
        timeline = service.generate_timeline(audio_file)
        
        print(f"=== LibROSA timeline generated successfully ===")
        print(f"   - Total duration: {timeline.total_duration:.2f}s")
        print(f"   - Events: {len(timeline.events)}")
        print(f"   - Spans: {len(timeline.spans)}")
        
        # Show some events
        for i, event in enumerate(timeline.events[:3]):
            print(f"   - Event {i+1}: {event.timestamp:.2f}s - {event.description}")
        
        # Show some spans
        for i, span in enumerate(timeline.spans[:2]):
            print(f"   - Span {i+1}: {span.start:.2f}s-{span.end:.2f}s - {span.description}")
        
        return timeline, audio_file, duration
        
    except Exception as e:
        print(f"=== LibROSA timeline failed: {e} ===")
        return None, audio_file, duration


def test_pyaudio_timeline_service(audio_file, duration):
    """Test pyAudioAnalysis timeline service"""
    print("\n=== Testing pyAudioAnalysis Timeline Service ===")
    
    config = load_test_config()
    service = PyAudioTimelineService(config)
    
    try:
        # Generate timeline
        timeline = service.generate_timeline(audio_file)
        
        print(f"=== pyAudioAnalysis timeline generated successfully ===")
        print(f"   - Total duration: {timeline.total_duration:.2f}s")
        print(f"   - Events: {len(timeline.events)}")
        print(f"   - Spans: {len(timeline.spans)}")
        
        # Show some events
        for i, event in enumerate(timeline.events[:3]):
            print(f"   - Event {i+1}: {event.timestamp:.2f}s - {event.description}")
        
        # Show some spans
        for i, span in enumerate(timeline.spans[:2]):
            print(f"   - Span {i+1}: {span.start:.2f}s-{span.end:.2f}s - {span.description}")
        
        return timeline
        
    except Exception as e:
        print(f"=== pyAudioAnalysis timeline failed: {e} ===")
        return None


def create_mock_whisper_timeline(audio_file, duration):
    """Create a mock Whisper timeline for testing"""
    from utils.timeline_schema import ServiceTimeline, TimelineEvent, TimelineSpan
    
    timeline = ServiceTimeline(
        source="whisper",
        audio_file=audio_file,
        total_duration=duration
    )
    
    # Add mock speech segments
    timeline.add_span(TimelineSpan(
        start=1.0,
        end=6.0,
        description="Male narrator: 'Welcome to our amazing product demonstration'",
        category="speech",
        source="whisper",
        confidence=0.95,
        details={"speaker": "SPEAKER_00", "language": "en"}
    ))
    
    timeline.add_span(TimelineSpan(
        start=8.0,
        end=13.0,
        description="Female voice: 'This revolutionary technology will change everything'", 
        category="speech",
        source="whisper",
        confidence=0.92,
        details={"speaker": "SPEAKER_01", "language": "en"}
    ))
    
    timeline.add_event(TimelineEvent(
        timestamp=7.0,
        description="Speaker change detected",
        category="speech",
        source="whisper",
        confidence=0.88,
        details={"speaker_transition": "SPEAKER_00 to SPEAKER_01"}
    ))
    
    return timeline


def test_timeline_merger(librosa_timeline, pyaudio_timeline, whisper_timeline, audio_file, duration):
    """Test timeline merger service"""
    print("\n=== Testing Timeline Merger Service ===")
    
    config = load_test_config()
    service = TimelineMergerService(config)
    
    try:
        # Save timelines to temporary files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            
            # Save individual timelines
            whisper_file = temp_dir / "whisper_timeline.json"
            librosa_file = temp_dir / "librosa_timeline.json"
            pyaudio_file = temp_dir / "pyaudio_timeline.json"
            
            if whisper_timeline:
                whisper_timeline.save_to_file(str(whisper_file))
            if librosa_timeline:
                librosa_timeline.save_to_file(str(librosa_file))
            if pyaudio_timeline:
                pyaudio_timeline.save_to_file(str(pyaudio_file))
            
            # Merge timelines
            merged_timeline = service.merge_timelines(
                whisper_timeline_path=str(whisper_file) if whisper_timeline else None,
                librosa_timeline_path=str(librosa_file) if librosa_timeline else None,
                pyaudio_timeline_path=str(pyaudio_file) if pyaudio_timeline else None,
                audio_file=audio_file,
                total_duration=duration
            )
            
            print(f"=== Timeline merge completed successfully ===")
            print(f"   - Total segments: {len(merged_timeline.timeline_segments)}")
            print(f"   - Sources used: {[k for k, v in merged_timeline.source_files.items() if v]}")
            
            # Show merged timeline segments
            for i, segment in enumerate(merged_timeline.timeline_segments[:5]):
                print(f"   - Segment {i+1}: {segment['time_range']}")
                for content in segment['content'][:2]:  # Show first 2 content items
                    print(f"     * [{content['source']}] {content['description']}")
                if len(segment['content']) > 2:
                    print(f"     ... and {len(segment['content']) - 2} more items")
            
            # Save merged timeline
            output_file = temp_dir / "final_timeline.json"
            service.save_merged_timeline(merged_timeline, str(output_file))
            
            print(f"   - Merged timeline saved to: {output_file}")
            
            # Test utility functions
            from services.timeline_merger_service import extract_llm_ready_text, get_timeline_summary
            
            timeline_dict = merged_timeline.to_dict()
            summary = get_timeline_summary(timeline_dict)
            llm_text = extract_llm_ready_text(timeline_dict)
            
            print(f"   - Summary: {summary}")
            print(f"   - LLM text length: {len(llm_text)} characters")
            
            return merged_timeline
            
    except Exception as e:
        print(f"=== Timeline merger failed: {e} ===")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run timeline integration tests"""
    print("=== Starting Timeline Integration Tests ===")
    
    try:
        # Test LibROSA timeline service
        librosa_timeline, audio_file, duration = test_librosa_timeline_service()
        
        # Test pyAudioAnalysis timeline service
        pyaudio_timeline = test_pyaudio_timeline_service(audio_file, duration)
        
        # Create mock Whisper timeline
        whisper_timeline = create_mock_whisper_timeline(audio_file, duration)
        print(f"\n=== Mock Whisper timeline created with {len(whisper_timeline.events)} events and {len(whisper_timeline.spans)} spans ===")
        
        # Test timeline merger
        merged_timeline = test_timeline_merger(
            librosa_timeline, 
            pyaudio_timeline, 
            whisper_timeline, 
            audio_file, 
            duration
        )
        
        if merged_timeline:
            print(f"\n=== Timeline integration test completed successfully! ===")
            print(f"   Final timeline has {len(merged_timeline.timeline_segments)} segments")
            
            # Display final timeline structure
            print(f"\n=== Final Timeline Structure ===")
            timeline_dict = merged_timeline.to_dict()
            for segment in timeline_dict['merged_timeline'][:3]:
                print(f"   {segment['time_range']}:")
                for content in segment['content']:
                    print(f"     - {content['description']} [{content['source']}]")
        else:
            print(f"\n=== Timeline integration test failed ===")
        
    except Exception as e:
        print(f"=== Timeline integration test crashed: {e} ===")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup test audio file
        try:
            if 'audio_file' in locals():
                os.unlink(audio_file)
                print(f"=== Cleaned up test audio file ===")
        except:
            pass


if __name__ == "__main__":
    main()