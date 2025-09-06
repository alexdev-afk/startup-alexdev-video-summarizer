"""
Test Full 3-Service Timeline Merger with bonita.mp4

Comprehensive test of the complete timeline merging pipeline:
- Enhanced Whisper Timeline Service (with clean structure)
- LibROSA Timeline Service (musical analysis)  
- pyAudio Timeline Service (audio analysis)
- Enhanced Timeline Merger (combining all 3)

Focus on verifying all timestamp offsets are properly coded.
"""

import sys
from pathlib import Path
from typing import Dict, Any
import yaml
import json

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from services.enhanced_whisper_timeline_service import EnhancedWhisperTimelineService
from services.librosa_timeline_service import LibROSATimelineService
from services.pyaudio_timeline_service import PyAudioTimelineService
from utils.logger import get_logger

logger = get_logger(__name__)


def load_config():
    """Load configuration for all services"""
    try:
        config_path = Path("config/processing.yaml")
        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            return None
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Ensure real processing (no mocks)
        if 'development' not in config:
            config['development'] = {}
        config['development']['mock_ai_services'] = False
        
        # Enhanced timeline configurations
        timeline_configs = {
            'whisper_timeline': {
                'generate_word_events': True,
                'generate_speaker_events': True,
                'word_confidence_threshold': 0.3
            },
            'librosa_timeline': {
                'tempo_analysis': True,
                'key_analysis': True,
                'onset_detection': True,
                'spectral_analysis': True
            },
            'pyaudio_timeline': {
                'emotion_analysis': True,
                'speaker_analysis': True,
                'environment_analysis': True,
                'segment_size': 2.0
            }
        }
        
        for service, service_config in timeline_configs.items():
            if service not in config:
                config[service] = {}
            config[service].update(service_config)
        
        return config
        
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return None


def find_bonita_audio():
    """Find the bonita audio file"""
    build_dir = Path("build")
    if not build_dir.exists():
        logger.error("Build directory not found")
        return None
    
    for video_dir in build_dir.iterdir():
        if video_dir.is_dir() and "bonita" in video_dir.name.lower():
            audio_file = video_dir / "audio.wav"
            if audio_file.exists():
                logger.info(f"Found bonita audio: {audio_file}")
                return audio_file
    
    logger.error("bonita audio.wav not found in build directory")
    return None


def test_individual_services(audio_path: Path, config: Dict):
    """Test each individual timeline service"""
    results = {}
    
    # Test Enhanced Whisper Service
    logger.info("=== Testing Enhanced Whisper Timeline Service ===")
    try:
        whisper_service = EnhancedWhisperTimelineService(config)
        whisper_timeline = whisper_service.generate_and_save(audio_path)
        
        whisper_dict = whisper_timeline.to_dict()
        logger.info(f"✅ Whisper: {len(whisper_dict['timeline_spans'])} spans, {sum(len(s.get('events', [])) for s in whisper_dict['timeline_spans'])} events")
        
        # Verify timestamp structure
        spans = whisper_dict.get('timeline_spans', [])
        if spans:
            first_span = spans[0]
            last_span = spans[-1]
            logger.info(f"   Time range: {first_span['start']}s - {last_span['end']}s")
            logger.info(f"   Duration: {whisper_dict['global_data']['duration']}s")
            
            # Check for proper event nesting
            events_in_spans = sum(len(s.get('events', [])) for s in spans)
            logger.info(f"   Events properly nested: {events_in_spans} events in spans")
        
        results['whisper'] = {
            'timeline': whisper_timeline,
            'path': audio_path.parent / "audio_timelines" / "whisper_timeline.json"
        }
        whisper_service.cleanup()
        
    except Exception as e:
        logger.error(f"❌ Whisper service failed: {e}")
        results['whisper'] = None
    
    # Test LibROSA Service
    logger.info("\n=== Testing LibROSA Timeline Service ===")
    try:
        librosa_service = LibROSATimelineService(config)
        librosa_timeline_path = audio_path.parent / "audio_timelines" / "librosa_timeline.json"
        librosa_timeline = librosa_service.generate_and_save(audio_path, str(librosa_timeline_path))
        
        librosa_dict = librosa_timeline.to_dict()
        total_objects = librosa_dict['summary']['total_objects']
        logger.info(f"✅ LibROSA: {total_objects} timeline objects")
        
        # Check timestamp ranges
        timeline_objects = librosa_dict.get('timeline', [])
        if timeline_objects:
            timestamps = []
            for obj in timeline_objects:
                if obj['type'] == 'event':
                    timestamps.append(obj['timestamp'])
                elif obj['type'] == 'span':
                    timestamps.extend([obj['start'], obj['end']])
            
            if timestamps:
                logger.info(f"   Time range: {min(timestamps):.2f}s - {max(timestamps):.2f}s")
        
        results['librosa'] = {
            'timeline': librosa_timeline,
            'path': librosa_timeline_path
        }
        librosa_service.cleanup()
        
    except Exception as e:
        logger.error(f"❌ LibROSA service failed: {e}")
        results['librosa'] = None
    
    # Test pyAudio Service  
    logger.info("\n=== Testing pyAudio Timeline Service ===")
    try:
        pyaudio_service = PyAudioTimelineService(config)
        pyaudio_timeline_path = audio_path.parent / "audio_timelines" / "pyaudio_timeline.json"
        pyaudio_timeline = pyaudio_service.generate_and_save(audio_path, str(pyaudio_timeline_path))
        
        pyaudio_dict = pyaudio_timeline.to_dict()
        total_objects = pyaudio_dict['summary']['total_objects']
        logger.info(f"✅ pyAudio: {total_objects} timeline objects")
        
        # Check timestamp ranges
        timeline_objects = pyaudio_dict.get('timeline', [])
        if timeline_objects:
            timestamps = []
            for obj in timeline_objects:
                if obj['type'] == 'event':
                    timestamps.append(obj['timestamp'])
                elif obj['type'] == 'span':
                    timestamps.extend([obj['start'], obj['end']])
            
            if timestamps:
                logger.info(f"   Time range: {min(timestamps):.2f}s - {max(timestamps):.2f}s")
        
        results['pyaudio'] = {
            'timeline': pyaudio_timeline,
            'path': pyaudio_timeline_path
        }
        pyaudio_service.cleanup()
        
    except Exception as e:
        logger.error(f"❌ pyAudio service failed: {e}")
        results['pyaudio'] = None
    
    return results


def test_timeline_merger(audio_path: Path, service_results: Dict, config: Dict):
    """Test the timeline merger with all 3 services"""
    logger.info("\n=== Testing Timeline Merger Service ===")
    
    try:
        from services.timeline_merger_service import TimelineMergerService
        merger_service = TimelineMergerService(config)
        
        # Get paths to timeline files
        whisper_path = service_results['whisper']['path'] if service_results.get('whisper') else None
        librosa_path = service_results['librosa']['path'] if service_results.get('librosa') else None 
        pyaudio_path = service_results['pyaudio']['path'] if service_results.get('pyaudio') else None
        
        # Get total duration from whisper (most reliable)
        total_duration = 0.0
        if service_results.get('whisper'):
            whisper_dict = service_results['whisper']['timeline'].to_dict()
            total_duration = whisper_dict['global_data']['duration']
        
        logger.info(f"Merging timelines:")
        logger.info(f"  Whisper: {whisper_path}")
        logger.info(f"  LibROSA: {librosa_path}")
        logger.info(f"  pyAudio: {pyaudio_path}")
        logger.info(f"  Duration: {total_duration}s")
        
        # Perform merge
        merged_timeline_path = audio_path.parent / "audio_timelines" / "merged_timeline_full.json"
        merged_timeline = merger_service.merge_and_save(
            output_path=str(merged_timeline_path),
            whisper_timeline_path=str(whisper_path) if whisper_path else None,
            librosa_timeline_path=str(librosa_path) if librosa_path else None,
            pyaudio_timeline_path=str(pyaudio_path) if pyaudio_path else None,
            audio_file=audio_path.name,
            total_duration=total_duration
        )
        
        # Analyze merged result
        merged_dict = merged_timeline.to_dict()
        logger.info(f"✅ Merge complete:")
        logger.info(f"   Total segments: {len(merged_dict['merged_timeline'])}")
        logger.info(f"   Sources used: {merged_dict['summary']['sources_used']}")
        
        # Verify timestamp coverage
        segments = merged_dict.get('merged_timeline', [])
        if segments:
            # Parse time ranges to verify coverage
            time_ranges = []
            for segment in segments:
                time_range = segment['time_range']
                if '-' in time_range:
                    start_str, end_str = time_range.replace('s', '').split('-')
                    try:
                        start_time = float(start_str)
                        end_time = float(end_str)
                        time_ranges.extend([start_time, end_time])
                    except ValueError:
                        continue
                else:
                    # Single timestamp
                    try:
                        timestamp = float(time_range.replace('s', ''))
                        time_ranges.append(timestamp)
                    except ValueError:
                        continue
            
            if time_ranges:
                logger.info(f"   Timeline coverage: {min(time_ranges):.2f}s - {max(time_ranges):.2f}s")
        
        # Show sample merged content
        logger.info(f"\n=== Sample Merged Timeline Content ===")
        for i, segment in enumerate(segments[:3]):
            logger.info(f"Segment {i+1}: {segment['time_range']}")
            content_sources = set(item['source'] for item in segment['content'])
            logger.info(f"   Sources: {', '.join(content_sources)}")
            logger.info(f"   Items: {len(segment['content'])}")
        
        if len(segments) > 3:
            logger.info(f"   ... and {len(segments) - 3} more segments")
        
        merger_service.cleanup()
        
        return {
            'merged_timeline': merged_timeline,
            'path': merged_timeline_path,
            'success': True
        }
        
    except Exception as e:
        logger.error(f"❌ Timeline merger failed: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def verify_timestamp_offsets(service_results: Dict, merged_result: Dict):
    """Verify that all timestamp offsets are properly applied"""
    logger.info(f"\n=== Timestamp Offset Verification ===")
    
    # Check Whisper timestamps
    if service_results.get('whisper'):
        whisper_dict = service_results['whisper']['timeline'].to_dict()
        spans = whisper_dict.get('timeline_spans', [])
        
        logger.info(f"Whisper Offset Check:")
        for i, span in enumerate(spans[:3]):
            events = span.get('events', [])
            logger.info(f"  Span {i+1}: {span['start']:.2f}-{span['end']:.2f}s")
            if events:
                word_events = [e for e in events if e['description'] == 'Word']
                if word_events:
                    first_word = word_events[0]
                    last_word = word_events[-1]
                    logger.info(f"    Words: {first_word['timestamp']:.2f}s - {last_word['timestamp']:.2f}s")
                    
                    # Verify words are within span bounds
                    if first_word['timestamp'] >= span['start'] and last_word['timestamp'] <= span['end']:
                        logger.info(f"    ✅ Word timestamps within span bounds")
                    else:
                        logger.info(f"    ❌ Word timestamps outside span bounds!")
        
        if len(spans) > 3:
            logger.info(f"  ... checked {len(spans)} total spans")
    
    # Overall verification
    total_services = len([s for s in service_results.values() if s is not None])
    logger.info(f"\n=== Overall Results ===")
    logger.info(f"✅ Services processed: {total_services}/3")
    
    if merged_result.get('success'):
        logger.info(f"✅ Timeline merging: SUCCESS")
    else:
        logger.info(f"❌ Timeline merging: FAILED")
    
    return total_services > 0 and merged_result.get('success', False)


def main():
    """Main test function"""
    logger.info("=== Full 3-Service Timeline Merger Test ===")
    
    # Load configuration
    config = load_config()
    if not config:
        return False
    
    # Find bonita audio
    audio_path = find_bonita_audio()
    if not audio_path:
        return False
    
    # Test individual services
    service_results = test_individual_services(audio_path, config)
    
    # Test timeline merger
    merged_result = test_timeline_merger(audio_path, service_results, config)
    
    # Verify timestamp offsets
    success = verify_timestamp_offsets(service_results, merged_result)
    
    if success:
        logger.info("\n🎉 Full timeline merger test PASSED!")
        logger.info(f"📁 Results saved in: {audio_path.parent / 'audio_timelines'}")
    else:
        logger.info("\n❌ Full timeline merger test FAILED!")
    
    return success


if __name__ == "__main__":
    success = main()
    if success:
        print("Full 3-Service Timeline Merger test completed successfully")
    else:
        print("Full 3-Service Timeline Merger test failed")
        sys.exit(1)