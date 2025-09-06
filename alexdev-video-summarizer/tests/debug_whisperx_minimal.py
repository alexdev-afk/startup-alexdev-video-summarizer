#!/usr/bin/env python3
"""
Minimal WhisperX test bypassing VAD
"""

import sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from utils.logger import setup_logging

def minimal_whisperx_test():
    """Minimal WhisperX test with direct transcription"""
    
    setup_logging('INFO')
    
    print("=== Minimal WhisperX Test ===")
    
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
        
        # Direct WhisperX usage without our wrapper
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        compute_type = 'float16' if device == 'cuda' else 'int8'
        
        print(f"[INFO] Loading WhisperX model directly...")
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
                print(f"  First text: {first_seg.get('text', '')}")
        
        # Test saving
        import json
        output_dir = audio_file.parent / "test_output"
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / "minimal_transcription.json"
        
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"[OK] Transcription saved to: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Minimal test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    minimal_whisperx_test()