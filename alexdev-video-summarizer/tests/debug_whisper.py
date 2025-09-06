#!/usr/bin/env python3
"""
Debug Whisper Access
Test if Whisper can access the audio file directly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from utils.logger import setup_logging

def test_whisper_access():
    print("=== DEBUG WHISPER ACCESS ===")
    
    setup_logging('INFO')
    
    # Test file access
    audio_path = Path("build/Consider 3 key points when ordering your automated laundry rack/audio.wav")
    
    print(f"Testing audio path: {audio_path}")
    print(f"Absolute path: {audio_path.absolute()}")
    print(f"File exists: {audio_path.exists()}")
    
    if audio_path.exists():
        print(f"File size: {audio_path.stat().st_size} bytes")
        
        # Test basic Whisper import and model loading
        try:
            # Setup FFmpeg path first
            import os
            import platform
            
            if platform.system() == 'Windows':
                # WinGet installation path
                winget_path = Path.home() / "AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-8.0-full_build/bin/ffmpeg.exe"
                if winget_path.exists():
                    # Add the directory containing ffmpeg.exe to PATH
                    ffmpeg_dir = str(winget_path.parent)
                    if ffmpeg_dir not in os.environ.get('PATH', ''):
                        os.environ['PATH'] = ffmpeg_dir + os.pathsep + os.environ.get('PATH', '')
                        print(f"Added FFmpeg directory to PATH: {ffmpeg_dir}")
            
            import whisper
            print("Whisper import successful")
            
            # Test loading small model first
            print("Loading Whisper base model for testing...")
            model = whisper.load_model("base")
            print("Base model loaded successfully")
            
            # Test transcription with absolute path
            print(f"Testing transcription with absolute path...")
            abs_path = str(audio_path.absolute())
            print(f"Absolute path: {abs_path}")
            
            result = model.transcribe(abs_path, task='transcribe', verbose=False)
            print("Transcription successful!")
            print(f"Result type: {type(result)}")
            print(f"Text preview: {result['text'][:100]}...")
            
        except Exception as e:
            print(f"Whisper test failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Audio file not found - need to run the test setup first")

if __name__ == "__main__":
    test_whisper_access()