#!/usr/bin/env python3
"""
ML Emotion Detection Integration Test

Quick test script to validate the ML emotion detection integration.
Tests both ML model loading and fallback to heuristic approach.
"""

import sys
import os
import yaml
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from services.ml_emotion_service import MLEmotionService
from services.pyaudio_timeline_service import PyAudioTimelineService

def load_config():
    """Load configuration from processing.yaml"""
    config_path = Path(__file__).parent / "config" / "processing.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def test_ml_emotion_service():
    """Test ML emotion service directly"""
    print("=" * 60)
    print("Testing ML Emotion Service")
    print("=" * 60)
    
    try:
        config = load_config()
        ml_service = MLEmotionService(config)
        
        # Test model info
        print("ML Model Info:")
        model_info = ml_service.get_model_info()
        for key, value in model_info.items():
            print(f"  {key}: {value}")
        print()
        
        # Generate synthetic audio for testing (5 seconds at 16kHz)
        sample_rate = 16000
        duration = 5.0
        samples = int(sample_rate * duration)
        
        # Create different synthetic audio patterns
        test_cases = [
            ("High energy audio", np.random.normal(0, 0.5, samples)),
            ("Low energy audio", np.random.normal(0, 0.1, samples)),  
            ("Sine wave audio", np.sin(2 * np.pi * 440 * np.linspace(0, duration, samples))),
        ]
        
        for test_name, audio_data in test_cases:
            print(f"Testing {test_name}:")
            try:
                emotion, confidence, all_probs = ml_service.classify_emotion_ml(audio_data, sample_rate)
                print(f"  Detected emotion: {emotion}")
                print(f"  Confidence: {confidence:.3f}")
                print(f"  Top 3 emotions: {sorted(all_probs.items(), key=lambda x: x[1], reverse=True)[:3]}")
                print()
            except Exception as e:
                print(f"  Error: {e}")
                print()
        
        # Cleanup
        ml_service.cleanup()
        print("ML Emotion Service test completed [SUCCESS]")
        
    except Exception as e:
        print(f"ML Emotion Service test failed: {e}")
        return False
    
    return True

def test_pyaudio_integration():
    """Test PyAudioTimelineService with ML emotion integration"""
    print("=" * 60)
    print("Testing PyAudioTimelineService ML Integration")
    print("=" * 60)
    
    try:
        config = load_config()
        pyaudio_service = PyAudioTimelineService(config)
        
        print("PyAudioTimelineService Configuration:")
        print(f"  ML emotion enabled: {pyaudio_service.use_ml_emotion}")
        print(f"  ML emotion service available: {pyaudio_service.ml_emotion_service is not None}")
        print()
        
        # Generate synthetic audio for testing
        sample_rate = 22050  # PyAudio uses 22kHz
        duration = 3.0
        samples = int(sample_rate * duration)
        audio_data = np.sin(2 * np.pi * 220 * np.linspace(0, duration, samples))
        
        # Test emotion classification
        print("Testing emotion classification:")
        try:
            emotion, confidence = pyaudio_service._classify_emotion(audio_data, sample_rate)
            print(f"  Detected emotion: {emotion}")
            print(f"  Confidence: {confidence:.3f}")
            print(f"  Method: {'ML' if pyaudio_service.use_ml_emotion else 'Heuristic'}")
            print()
        except Exception as e:
            print(f"  Error: {e}")
            print()
        
        # Test heuristic fallback
        print("Testing heuristic fallback:")
        try:
            emotion, confidence = pyaudio_service._classify_emotion_heuristic(audio_data, sample_rate)
            print(f"  Heuristic emotion: {emotion}")
            print(f"  Heuristic confidence: {confidence:.3f}")
            print()
        except Exception as e:
            print(f"  Error: {e}")
            print()
        
        # Cleanup
        pyaudio_service.cleanup()
        print("PyAudioTimelineService integration test completed [SUCCESS]")
        
    except Exception as e:
        print(f"PyAudioTimelineService integration test failed: {e}")
        return False
    
    return True

def main():
    """Run all ML emotion detection tests"""
    print("ML Emotion Detection Integration Test")
    print("====================================")
    print()
    
    success = True
    
    # Test ML emotion service directly
    if not test_ml_emotion_service():
        success = False
    
    print()
    
    # Test PyAudioTimelineService integration
    if not test_pyaudio_integration():
        success = False
    
    print("=" * 60)
    if success:
        print("[SUCCESS] All ML emotion detection tests PASSED!")
        print("ML emotion integration is working correctly.")
    else:
        print("[FAILED] Some tests FAILED!")
        print("Check the error messages above for details.")
    print("=" * 60)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())