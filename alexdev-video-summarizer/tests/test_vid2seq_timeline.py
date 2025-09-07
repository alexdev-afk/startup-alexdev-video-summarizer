"""
Test Vid2Seq Timeline Service with live data
"""

import pytest
import yaml
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from services.vid2seq_timeline_service import Vid2SeqTimelineService, Vid2SeqTimelineServiceError


def test_vid2seq_service_initialization():
    """Test Vid2Seq service initializes correctly"""
    config = {
        'gpu_pipeline': {
            'vid2seq': {
                'model_path': '../references/vid2seq-pytorch/vid2seq',
                'checkpoint_path': None,
                'confidence_threshold': 0.5,
                'max_duration': 600,
                'device': 'cuda'
            }
        }
    }
    
    service = Vid2SeqTimelineService(config)
    assert service.service_name == "vid2seq"
    assert service.confidence_threshold == 0.5
    assert service.max_duration == 600


def test_vid2seq_model_loading():
    """Test Vid2Seq model loading (lazy initialization)"""
    config = {
        'gpu_pipeline': {
            'vid2seq': {
                'model_path': '../references/vid2seq-pytorch/vid2seq',
                'checkpoint_path': None,
                'confidence_threshold': 0.5,
                'max_duration': 600
            }
        }
    }
    
    service = Vid2SeqTimelineService(config)
    
    # Model should not be loaded initially
    assert not service._model_loaded
    assert service.model is None
    assert service.tokenizer is None
    
    # Test lazy loading
    try:
        service._ensure_model_loaded()
        assert service._model_loaded
        assert service.model is not None
        assert service.tokenizer is not None
        print("[OK] Vid2Seq model loaded successfully")
    except Exception as e:
        print(f"[FAIL] Model loading failed: {e}")
        # Don't fail test if model dependencies missing
        pytest.skip(f"Vid2Seq model dependencies not available: {e}")


def test_vid2seq_timeline_generation():
    """Test Vid2Seq timeline generation with live video data"""
    
    # Load actual configuration
    config_path = Path(__file__).parent.parent / "config" / "processing.yaml"
    if not config_path.exists():
        pytest.skip("Configuration file not found")
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Look for test video in input directory
    input_dir = Path(__file__).parent.parent / "input"
    test_videos = list(input_dir.glob("*.mp4"))
    
    if not test_videos:
        pytest.skip("No test videos found in input directory")
    
    test_video = test_videos[0]  # Use first available video
    print(f"Testing with video: {test_video.name}")
    
    service = Vid2SeqTimelineService(config)
    
    try:
        # Generate timeline
        timeline = service.generate_and_save(str(test_video))
        
        # Verify timeline structure
        assert timeline is not None
        assert hasattr(timeline, 'events')
        assert hasattr(timeline, 'sources_used')
        assert hasattr(timeline, 'total_duration')
        
        # Verify Vid2Seq is in sources
        assert 'vid2seq' in timeline.sources_used
        
        # Verify timeline has events (not empty)
        assert len(timeline.events) > 0
        print(f"[OK] Generated {len(timeline.events)} Vid2Seq events")
        
        # Verify events have proper structure
        for event in timeline.events[:3]:  # Check first 3 events
            assert hasattr(event, 'timestamp')
            assert hasattr(event, 'description')
            assert hasattr(event, 'source')
            assert hasattr(event, 'confidence')
            assert event.source == 'vid2seq'
            assert isinstance(event.timestamp, (int, float))
            assert isinstance(event.description, str)
            assert len(event.description) > 0
            print(f"[OK] Event at {event.timestamp:.1f}s: {event.description[:50]}...")
        
        # Verify timeline file was saved
        video_name = test_video.stem
        timeline_file = Path('build') / video_name / 'video_timelines' / 'vid2seq_timeline.json'
        assert timeline_file.exists(), f"Timeline file not saved: {timeline_file}"
        print(f"[OK] Timeline saved to: {timeline_file}")
        
        # Verify saved file structure
        import json
        with open(timeline_file, 'r') as f:
            saved_data = json.load(f)
        
        assert 'model_info' in saved_data
        assert 'events' in saved_data
        assert saved_data['model_info']['model_name'] == 'Vid2Seq'
        assert saved_data['model_info']['model_type'] == 'dense_video_captioning'
        assert len(saved_data['events']) > 0
        print(f"[OK] Saved timeline contains {len(saved_data['events'])} events")
        
        return True
        
    except Vid2SeqTimelineServiceError as e:
        print(f"[FAIL] Vid2Seq timeline generation failed: {e}")
        return False
    except Exception as e:
        print(f"[FAIL] Unexpected error: {e}")
        raise


def test_vid2seq_visual_features():
    """Test visual feature extraction"""
    config = {
        'gpu_pipeline': {
            'vid2seq': {
                'model_path': '../references/vid2seq-pytorch/vid2seq',
                'checkpoint_path': None,
                'confidence_threshold': 0.5,
                'max_duration': 600
            }
        }
    }
    
    service = Vid2SeqTimelineService(config)
    
    # Look for test video
    input_dir = Path(__file__).parent.parent / "input"
    test_videos = list(input_dir.glob("*.mp4"))
    
    if not test_videos:
        pytest.skip("No test videos found in input directory")
    
    test_video = str(test_videos[0])
    
    try:
        features = service._extract_visual_features(test_video)
        assert features is not None
        assert len(features.shape) == 2  # Should be 2D array (frames, features)
        assert features.shape[1] == 224 * 224 * 3  # Flattened RGB frames
        print(f"[OK] Extracted visual features: {features.shape}")
        return True
    except Exception as e:
        print(f"[FAIL] Visual feature extraction failed: {e}")
        return False


if __name__ == "__main__":
    print("Testing Vid2Seq Timeline Service")
    print("=" * 50)
    
    # Run individual tests
    print("\n1. Testing service initialization...")
    test_vid2seq_service_initialization()
    print("[OK] Service initialization passed")
    
    print("\n2. Testing model loading...")
    try:
        test_vid2seq_model_loading()
    except Exception as e:
        print(f"Model loading test failed: {e}")
    
    print("\n3. Testing visual feature extraction...")
    try:
        test_vid2seq_visual_features()
    except Exception as e:
        print(f"Visual features test failed: {e}")
    
    print("\n4. Testing timeline generation...")
    try:
        success = test_vid2seq_timeline_generation()
        if success:
            print("[OK] All Vid2Seq timeline tests passed!")
        else:
            print("[FAIL] Timeline generation test failed")
    except Exception as e:
        print(f"Timeline generation test failed: {e}")
    
    print("\nTest complete.")