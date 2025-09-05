#!/usr/bin/env python3
"""
Debug WhisperX GPU acceleration issue
"""

import torch
import gc
import time

def test_whisperx_gpu():
    """Test WhisperX GPU acceleration directly"""
    
    print("=== CUDA Status ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
        print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        print(f"Memory reserved: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
    
    print("\n=== WhisperX Import ===")
    try:
        import whisperx
        print("✅ WhisperX imported successfully")
    except ImportError as e:
        print(f"❌ WhisperX import failed: {e}")
        return
    
    print("\n=== Model Loading Test ===")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test different model sizes and compute types
    test_configs = [
        ('base', 'float16'),
        ('base', 'float32'),
        ('base', 'int8'),
        ('small', 'float16'),
        ('medium', 'float16'),
    ]
    
    for model_size, compute_type in test_configs:
        print(f"\n--- Testing {model_size} with {compute_type} on {device} ---")
        try:
            start_time = time.time()
            model = whisperx.load_model(
                model_size,
                device=device,
                compute_type=compute_type
            )
            load_time = time.time() - start_time
            print(f"✅ Model loaded in {load_time:.2f}s")
            
            # Check if model is actually on GPU
            if hasattr(model, 'model') and hasattr(model.model, 'device'):
                print(f"   Model device: {model.model.device}")
            
            # Clean up
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ Failed: {e}")
    
    print("\n=== Compute Type Compatibility Test ===")
    # Test with explicit device configuration
    try:
        print(f"Testing large-v2 model with auto compute type detection...")
        
        # Auto-detect compute type based on device
        if device == 'cuda':
            compute_type = 'float16'
        else:
            compute_type = 'int8'
        
        print(f"Using compute_type: {compute_type}")
        
        model = whisperx.load_model(
            'large-v2',
            device=device,
            compute_type=compute_type
        )
        print(f"✅ large-v2 model loaded successfully")
        
        # Test transcription with dummy audio if possible
        print("Testing transcription capabilities...")
        # (Would need actual audio file for full test)
        
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"❌ large-v2 model failed: {e}")
        print(f"Error type: {type(e).__name__}")
        
        # Try fallback options
        print("\nTrying fallback options...")
        try:
            model = whisperx.load_model(
                'base',
                device='cpu',  # Force CPU
                compute_type='int8'
            )
            print("✅ Fallback CPU model loaded")
            del model
            gc.collect()
        except Exception as e2:
            print(f"❌ CPU fallback also failed: {e2}")

if __name__ == '__main__':
    test_whisperx_gpu()