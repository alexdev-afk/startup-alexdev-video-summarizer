#!/usr/bin/env python3
"""
Debug File Save Test
Test the individual save methods to see why files aren't being created.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from services.whisper_service import WhisperService
from services.librosa_service import LibROSAService  
from services.pyaudioanalysis_service import PyAudioAnalysisService
from utils.config_loader import ConfigLoader
from utils.logger import setup_logging

def test_file_saving():
    print("=== DEBUG FILE SAVE TEST ===")
    
    setup_logging('INFO')
    config = ConfigLoader.load_config('config/processing.yaml')
    
    # Create a fake audio path in the build directory structure
    audio_path = Path("build/Consider 3 key points when ordering your automated laundry rack/audio.wav")
    
    print(f"Testing file save with audio_path: {audio_path}")
    print(f"Audio path exists: {audio_path.exists()}")
    
    # Test 1: Whisper service save method directly
    print("\n--- Testing Whisper Save Method ---")
    whisper = WhisperService(config)
    
    fake_result = {
        'transcript': 'Test transcript',
        'language': 'en',
        'processing_time': 1.0
    }
    
    try:
        whisper._save_analysis_to_file(audio_path, fake_result)
        print("Whisper save method completed")
    except Exception as e:
        print(f"Whisper save method failed: {e}")
    
    # Test 2: LibROSA service save method directly  
    print("\n--- Testing LibROSA Save Method ---")
    librosa = LibROSAService(config)
    
    fake_result = {
        'tempo_analysis': {'tempo': 120.0},
        'processing_time': 1.0
    }
    
    try:
        librosa._save_analysis_to_file(str(audio_path), fake_result)
        print("LibROSA save method completed")
    except Exception as e:
        print(f"LibROSA save method failed: {e}")
        
    # Test 3: PyAudioAnalysis service save method directly
    print("\n--- Testing PyAudioAnalysis Save Method ---")
    pyaudio = PyAudioAnalysisService(config)
    
    fake_result = {
        'features_68': [1, 2, 3],
        'processing_time': 1.0
    }
    
    try:
        pyaudio._save_analysis_to_file(str(audio_path), fake_result)
        print("PyAudioAnalysis save method completed")
    except Exception as e:
        print(f"PyAudioAnalysis save method failed: {e}")
    
    # Check what files were created
    print("\n--- Checking Created Files ---")
    build_dir = audio_path.parent
    analysis_dir = build_dir / "audio_analysis"
    
    if analysis_dir.exists():
        print(f"Analysis directory exists: {analysis_dir}")
        for file in analysis_dir.iterdir():
            print(f"  - {file.name} ({file.stat().st_size} bytes)")
    else:
        print(f"Analysis directory does not exist: {analysis_dir}")

if __name__ == "__main__":
    test_file_saving()