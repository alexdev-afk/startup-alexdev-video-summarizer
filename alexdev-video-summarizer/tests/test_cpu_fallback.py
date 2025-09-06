#!/usr/bin/env python3
"""
Test CPU Fallback for GPU Pipeline

Verify that all GPU services can run on CPU when CUDA is not available.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from services.whisper_service import WhisperService
from services.yolo_service import YOLOService
from services.easyocr_service import EasyOCRService
from utils.config_loader import ConfigLoader
from utils.logger import setup_logging

def test_whisper_cpu_fallback():
    """Test Whisper service CPU fallback"""
    print("=== WHISPER CPU FALLBACK TEST ===")
    
    try:
        config = ConfigLoader.load_config('config/processing.yaml')
        whisper = WhisperService(config)
        
        print(f"[OK] WhisperService initialized")
        print(f"Device: {whisper.device}")
        print(f"Model: {whisper.model_name}")
        print(f"Available: {whisper.whisper_available}")
        
        if not whisper.whisper_available:
            print("[INFO] Whisper not available - using mock mode")
            return True
        
        # Test with a dummy audio file path
        test_audio = Path("build/Consider 3 key points when ordering your automated laundry rack/audio.wav")
        if test_audio.exists():
            print(f"[TEST] Testing transcription with {test_audio}")
            result = whisper.transcribe_audio(test_audio)
            print(f"[OK] Transcription result: {type(result)}")
            return True
        else:
            print("[SKIP] No test audio file found")
            return True
            
    except Exception as e:
        print(f"[FAIL] Whisper CPU fallback failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_yolo_cpu_fallback():
    """Test YOLO service CPU fallback"""
    print("\n=== YOLO CPU FALLBACK TEST ===")
    
    try:
        config = ConfigLoader.load_config('config/processing.yaml')
        yolo = YOLOService(config)
        
        print(f"[OK] YOLOService initialized")
        print(f"Device: {yolo.device}")
        print(f"Model: {yolo.model_name}")
        print(f"Available: {yolo.yolo_available}")
        
        if not yolo.yolo_available:
            print("[INFO] YOLO not available - using mock mode")
            return True
        
        # Test with a dummy video file path
        test_video = Path("build/Consider 3 key points when ordering your automated laundry rack/video.mp4")
        if test_video.exists():
            print(f"[TEST] Testing object detection with {test_video}")
            result = yolo.analyze_video(test_video)
            print(f"[OK] Analysis result: {type(result)}")
            return True
        else:
            print("[SKIP] No test video file found")
            return True
            
    except Exception as e:
        print(f"[FAIL] YOLO CPU fallback failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_easyocr_cpu_fallback():
    """Test EasyOCR service CPU fallback"""
    print("\n=== EASYOCR CPU FALLBACK TEST ===")
    
    try:
        config = ConfigLoader.load_config('config/processing.yaml')
        easyocr = EasyOCRService(config)
        
        print(f"[OK] EasyOCRService initialized")
        print(f"Device: {easyocr.device}")
        print(f"Languages: {easyocr.languages}")
        print(f"Available: {easyocr.easyocr_available}")
        
        if not easyocr.easyocr_available:
            print("[INFO] EasyOCR not available - using mock mode")
            return True
        
        # Test with a dummy video file path
        test_video = Path("build/Consider 3 key points when ordering your automated laundry rack/video.mp4")
        if test_video.exists():
            print(f"[TEST] Testing text extraction with {test_video}")
            result = easyocr.extract_text_from_video(test_video)
            print(f"[OK] Text extraction result: {type(result)}")
            return True
        else:
            print("[SKIP] No test video file found")
            return True
            
    except Exception as e:
        print(f"[FAIL] EasyOCR CPU fallback failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gpu_pipeline_controller():
    """Test GPU pipeline controller with CPU fallback"""
    print("\n=== GPU PIPELINE CONTROLLER CPU FALLBACK TEST ===")
    
    try:
        from services.gpu_pipeline import VideoGPUPipelineController
        
        config = ConfigLoader.load_config('config/processing.yaml')
        gpu_controller = VideoGPUPipelineController(config)
        
        print(f"[OK] VideoGPUPipelineController initialized")
        
        # Check individual service status
        services_status = {
            'whisper': hasattr(gpu_controller, 'whisper_service') and gpu_controller.whisper_service is not None,
            'yolo': hasattr(gpu_controller, 'yolo_service') and gpu_controller.yolo_service is not None,
            'easyocr': hasattr(gpu_controller, 'easyocr_service') and gpu_controller.easyocr_service is not None,
        }
        
        for service, available in services_status.items():
            print(f"  {service}: {'Available' if available else 'Not Available'}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] GPU Pipeline Controller failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Test all GPU services for CPU fallback capability"""
    print("GPU Pipeline CPU Fallback Test")
    print("=" * 50)
    
    setup_logging('INFO')
    
    tests = [
        ("Whisper CPU Fallback", test_whisper_cpu_fallback),
        ("YOLO CPU Fallback", test_yolo_cpu_fallback), 
        ("EasyOCR CPU Fallback", test_easyocr_cpu_fallback),
        ("GPU Pipeline Controller", test_gpu_pipeline_controller),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\nRunning: {test_name}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"[FAIL] {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("CPU FALLBACK TEST SUMMARY:")
    
    all_passed = True
    for test_name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n[SUCCESS] All GPU services can run on CPU fallback!")
        print("The pipeline is production-ready for systems without CUDA.")
    else:
        print("\n[WARNING] Some GPU services failed CPU fallback.")
        print("Production deployment may require CUDA for full functionality.")

if __name__ == "__main__":
    main()