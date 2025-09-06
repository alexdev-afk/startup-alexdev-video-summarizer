#!/usr/bin/env python3
"""
Debug CUDA compatibility issues with WhisperX
"""

import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent / 'src'))

def debug_cuda_compatibility():
    """Check CUDA compatibility step by step"""
    
    print("=== CUDA Compatibility Debug ===")
    
    # Basic CUDA info
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name()}")
        print(f"GPU compute capability: {torch.cuda.get_device_capability()}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    # Test basic CUDA operations
    print("\n=== Basic CUDA Test ===")
    try:
        # Simple tensor operation
        if torch.cuda.is_available():
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.mm(x, y)
            print(f"✅ Basic CUDA tensor operations working")
            print(f"   Result shape: {z.shape}")
            print(f"   GPU memory after test: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")
            torch.cuda.empty_cache()
        else:
            print("❌ CUDA not available for basic test")
    except Exception as e:
        print(f"❌ Basic CUDA test failed: {e}")
    
    # Test WhisperX with simpler model
    print("\n=== WhisperX Small Model Test ===")
    try:
        import whisperx
        print("✅ WhisperX imported")
        
        # Try loading smallest model on CUDA
        print("Loading tiny model on CUDA...")
        model = whisperx.load_model("tiny", device="cuda", compute_type="float16")
        print("✅ WhisperX tiny model loaded on CUDA")
        
        # Check memory usage
        if torch.cuda.is_available():
            print(f"   GPU memory after model: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")
        
        del model
        torch.cuda.empty_cache()
        print("✅ Model cleaned up successfully")
        
    except Exception as e:
        print(f"❌ WhisperX tiny model failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        
        # Try CPU fallback
        try:
            print("Trying CPU fallback...")
            model = whisperx.load_model("tiny", device="cpu", compute_type="int8")
            print("✅ WhisperX tiny model loaded on CPU")
            del model
        except Exception as e2:
            print(f"❌ CPU fallback also failed: {e2}")
    
    # Check CUDNN specifically
    print("\n=== CUDNN Check ===")
    try:
        print(f"CUDNN version: {torch.backends.cudnn.version()}")
        print(f"CUDNN enabled: {torch.backends.cudnn.enabled}")
        if torch.backends.cudnn.is_available():
            print("✅ CUDNN available")
        else:
            print("❌ CUDNN not available")
    except Exception as e:
        print(f"❌ CUDNN check failed: {e}")

if __name__ == '__main__':
    debug_cuda_compatibility()