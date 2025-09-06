#!/usr/bin/env python3
"""
Test WhisperX with correct FFmpeg PATH
"""

import os
import sys
from pathlib import Path
import yaml

# Add FFmpeg to PATH before importing anything else
ffmpeg_path = r"C:\Users\noise\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0-full_build\bin"
if ffmpeg_path not in os.environ.get('PATH', ''):
    os.environ['PATH'] = ffmpeg_path + os.pathsep + os.environ.get('PATH', '')
    print(f"[INFO] Added FFmpeg to PATH: {ffmpeg_path}")

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from utils.logger import setup_logging

def test_whisperx_with_path():
    """Test WhisperX with correct FFmpeg PATH"""
    
    setup_logging('INFO')
    
    print("=== WhisperX with FFmpeg PATH Test ===")
    
    # Verify FFmpeg is accessible
    import subprocess
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        print(f"[OK] FFmpeg accessible via PATH")
        print(f"  Version: {result.stdout.split()[2]}")
    except FileNotFoundError:
        print(f"[FAIL] FFmpeg still not found in PATH")
        return False
    
    # Check test audio file
    audio_file = Path("build/bonita/audio.wav")
    if not audio_file.exists():
        print(f"[FAIL] Audio file not found: {audio_file}")
        return
    
    print(f"[OK] Test audio file: {audio_file} ({audio_file.stat().st_size} bytes)")
    
    try:
        import whisperx
        import torch
        
        print(f"[OK] WhisperX and PyTorch imported")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        
        # Direct WhisperX usage with FFmpeg PATH fixed
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        compute_type = 'float16' if device == 'cuda' else 'int8'
        
        print(f"[INFO] Loading WhisperX model...")
        print(f"  Device: {device}")
        print(f"  Compute type: {compute_type}")
        
        model = whisperx.load_model("base", device=device, compute_type=compute_type)
        print(f"[OK] Model loaded: {type(model)}")
        
        # Direct transcription
        print(f"[INFO] Transcribing...")
        result = model.transcribe(str(audio_file), language="en")
        
        print(f"[OK] Transcription complete!")
        print(f"  Result type: {type(result)}")
        print(f"  Result keys: {list(result.keys())}")
        
        if 'segments' in result:
            print(f"  Segments: {len(result['segments'])}")
            if result['segments']:
                first_seg = result['segments'][0]
                print(f"  First segment: {first_seg.get('start', 0):.2f}s - {first_seg.get('end', 0):.2f}s")
                text = first_seg.get('text', '')[:100] + "..." if len(first_seg.get('text', '')) > 100 else first_seg.get('text', '')
                print(f"  First text: {text}")
        
        # Test saving
        import json
        output_dir = audio_file.parent / "test_output"
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / "working_transcription.json"
        
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"[OK] Transcription saved to: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    test_whisperx_with_path()