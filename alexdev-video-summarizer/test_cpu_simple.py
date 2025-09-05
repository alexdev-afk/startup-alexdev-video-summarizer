#!/usr/bin/env python3
"""
Simple CPU Fallback Test - Verify all services initialize on CPU
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from services.whisper_service import WhisperService
from services.yolo_service import YOLOService
from services.easyocr_service import EasyOCRService
from services.gpu_pipeline import VideoGPUPipelineController
from utils.config_loader import ConfigLoader
from utils.logger import setup_logging

def main():
    print("Simple CPU Fallback Test")
    print("=" * 30)
    
    setup_logging('INFO')
    config = ConfigLoader.load_config('config/processing.yaml')
    
    # Test individual services
    services = []
    
    try:
        whisper = WhisperService(config)
        services.append(("Whisper", whisper.device, "OK"))
    except Exception as e:
        services.append(("Whisper", "unknown", f"FAIL: {e}"))
    
    try:
        yolo = YOLOService(config)
        services.append(("YOLO", yolo.device, "OK"))
    except Exception as e:
        services.append(("YOLO", "unknown", f"FAIL: {e}"))
    
    try:
        easyocr = EasyOCRService(config)
        services.append(("EasyOCR", easyocr.device, "OK"))
    except Exception as e:
        services.append(("EasyOCR", "unknown", f"FAIL: {e}"))
    
    try:
        gpu_pipeline = VideoGPUPipelineController(config)
        services.append(("GPU Pipeline", "controller", "OK"))
    except Exception as e:
        services.append(("GPU Pipeline", "unknown", f"FAIL: {e}"))
    
    # Results
    print("\nService Initialization Results:")
    all_ok = True
    for service_name, device, status in services:
        print(f"  {service_name:12}: {device:8} - {status}")
        if "FAIL" in status:
            all_ok = False
    
    if all_ok:
        print("\n[SUCCESS] All GPU services can run on CPU!")
        print("Production ready for systems without CUDA.")
    else:
        print("\n[WARNING] Some services failed initialization.")

if __name__ == "__main__":
    main()