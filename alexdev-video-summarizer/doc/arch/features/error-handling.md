# Error Handling & Circuit Breaker System

**Priority**: Phase 5 - Production Readiness  
**Risk**: LOW - Error handling implementation  
**Dependencies**: All processing services

---

## **Purpose**
Implement comprehensive error handling with fail-fast per video and circuit breaker for batch processing, ensuring reliable operation across 100+ video libraries.

## **Core Functions**

### **1. Fail-Fast Per Video Strategy**
```python
class VideoProcessingOrchestrator:
    def process_video(self, video_path):
        """All-or-nothing processing per video"""
        try:
            # Initialize processing context
            context = VideoProcessingContext(video_path)
            
            # Step 1: FFmpeg Foundation (CRITICAL)
            context.audio_path, context.video_path = self.ffmpeg_service.extract_streams(video_path)
            if not context.validate_ffmpeg_output():
                raise FFmpegProcessingError("Audio/video extraction failed")
            
            # Step 2: Scene Detection (CRITICAL) 
            context.scene_data = self.scene_service.analyze_video_scenes(context.video_path)
            if context.scene_data['scene_count'] == 0:
                raise SceneDetectionError("No scenes detected")
            
            # Step 3: Per-Scene Processing (FAIL ON ANY TOOL FAILURE)
            for scene in context.scene_data['scenes']:
                self.process_scene_all_tools(scene, context)
            
            # Step 4: Knowledge Base Generation
            knowledge_file = self.knowledge_service.generate_video_knowledge_base(
                video_path, context.get_all_analysis()
            )
            
            return ProcessingResult.success(knowledge_file)
            
        except Exception as e:
            # Fail-fast: Any error fails entire video
            self.cleanup_failed_video(video_path, context)
            return ProcessingResult.failure(video_path, e)
```

### **2. Circuit Breaker for Batch Processing**
```python
class BatchProcessingCircuitBreaker:
    def __init__(self, failure_threshold=3):
        self.failure_threshold = failure_threshold
        self.consecutive_failures = 0
        self.total_processed = 0
        self.total_failures = 0
        
    def process_video_batch(self, video_paths):
        """Process batch with circuit breaker protection"""
        results = []
        
        for video_path in video_paths:
            if self.should_abort_batch():
                self.log_batch_abort(video_path, video_paths)
                break
                
            result = self.process_single_video(video_path)
            results.append(result)
            
            if result.failed:
                self.handle_video_failure(result)
            else:
                self.handle_video_success(result)
                
        return BatchProcessingResults(results)
    
    def should_abort_batch(self):
        """Circuit breaker logic"""
        return self.consecutive_failures >= self.failure_threshold
    
    def handle_video_failure(self, result):
        """Track failures and trigger circuit breaker"""
        self.consecutive_failures += 1
        self.total_failures += 1
        
        if self.should_abort_batch():
            self.trigger_circuit_breaker()
    
    def handle_video_success(self, result):
        """Reset consecutive failure counter on success"""
        self.consecutive_failures = 0
        self.total_processed += 1
```

---

## **Implementation Specification**

### **Tool-Specific Error Handling**

#### **FFmpeg Error Handling**
```python
class FFmpegErrorHandler:
    def handle_ffmpeg_error(self, video_path, error):
        """Comprehensive FFmpeg error recovery"""
        error_type = self.classify_ffmpeg_error(error)
        
        if error_type == FFmpegError.UNSUPPORTED_FORMAT:
            return self.attempt_format_conversion(video_path)
        elif error_type == FFmpegError.CORRUPTED_FILE:
            return self.attempt_repair_or_skip(video_path)
        elif error_type == FFmpegError.INSUFFICIENT_SPACE:
            return self.cleanup_and_retry(video_path)
        elif error_type == FFmpegError.PERMISSION_DENIED:
            return self.request_permission_fix(video_path)
        else:
            raise FFmpegProcessingError(f"Unrecoverable FFmpeg error: {error}")
    
    def attempt_format_conversion(self, video_path):
        """Try alternative codec/format combinations"""
        fallback_configs = [
            {'vcodec': 'libx264', 'acodec': 'aac'},
            {'vcodec': 'mpeg4', 'acodec': 'mp3'},
            {'format': 'avi'}
        ]
        
        for config in fallback_configs:
            try:
                return self.extract_with_config(video_path, config)
            except Exception:
                continue
                
        raise FFmpegProcessingError("All format conversion attempts failed")
```

#### **GPU Tool Error Handling**
```python
class GPUToolErrorHandler:
    def handle_gpu_error(self, tool_name, error, context):
        """GPU-specific error recovery"""
        error_type = self.classify_gpu_error(error)
        
        if error_type == GPUError.OUT_OF_MEMORY:
            return self.attempt_memory_recovery(tool_name, context)
        elif error_type == GPUError.MODEL_LOADING_FAILED:
            return self.attempt_model_reload(tool_name)
        elif error_type == GPUError.CUDA_DRIVER_ERROR:
            return self.attempt_cuda_recovery()
        else:
            raise GPUProcessingError(f"Unrecoverable GPU error in {tool_name}: {error}")
    
    def attempt_memory_recovery(self, tool_name, context):
        """Clear GPU memory and retry with smaller batch"""
        torch.cuda.empty_cache()
        gc.collect()
        
        # Retry with reduced parameters
        if tool_name == 'yolo':
            return self.retry_yolo_reduced_resolution(context)
        elif tool_name == 'easyocr':
            return self.retry_easyocr_cpu_fallback(context)
```

#### **Scene Detection Error Handling**
```python
class SceneDetectionErrorHandler:
    def handle_scene_error(self, video_path, error):
        """Scene detection fallback strategies"""
        error_type = self.classify_scene_error(error)
        
        if error_type == SceneError.NO_SCENES_DETECTED:
            return self.fallback_to_time_based_scenes(video_path)
        elif error_type == SceneError.THRESHOLD_TOO_SENSITIVE:
            return self.retry_with_adjusted_threshold(video_path)
        elif error_type == SceneError.VIDEO_TOO_SHORT:
            return self.treat_as_single_scene(video_path)
        else:
            raise SceneDetectionError(f"Scene detection failed: {error}")
    
    def fallback_to_time_based_scenes(self, video_path):
        """Create time-based scenes when content detection fails"""
        duration = self.get_video_duration(video_path)
        scene_length = 120  # 2-minute scenes as fallback
        
        scenes = []
        for i in range(0, int(duration), scene_length):
            scenes.append({
                'scene_id': len(scenes) + 1,
                'start_seconds': i,
                'end_seconds': min(i + scene_length, duration),
                'fallback_method': 'time_based'
            })
        
        return scenes
```

---

## **Error Recovery Strategies**

### **Graduated Recovery Approach**
1. **Immediate Retry**: Simple transient error recovery
2. **Parameter Adjustment**: Retry with modified settings
3. **Fallback Method**: Use alternative approach  
4. **Graceful Degradation**: Continue with reduced functionality
5. **Fail-Fast**: Mark video as failed, continue batch

### **Recovery Decision Matrix**
| Error Type | Recovery Strategy | Fallback | Final Action |
|------------|------------------|----------|--------------|
| FFmpeg Format | Alternative codecs | Skip video | Continue batch |
| GPU Memory | Clear cache + retry | CPU fallback | Continue batch |
| Scene Detection | Time-based scenes | Single scene | Continue batch |
| Tool Crash | Restart service | Skip tool | Continue batch |
| File Corruption | Repair attempt | Skip video | Continue batch |
| Disk Full | Cleanup + retry | Skip video | Abort batch |

---

## **Progress Reporting During Errors**

### **Error State Visualization**
```
❌ PROCESSING FAILED: Video 47/100

┌─────────────────────────────────────────────────────────────────┐
│ ERROR LOCATION IN PIPELINE                                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ [✓] FFmpeg Foundation  (Audio/video extracted)                 │
│ [✓] Scene Detection    (6 scenes detected)                     │
│                                                                 │
│ [❌] Per-Scene Analysis (FAILED: Scene 3/6)                     │
│     ├─ [✓] YOLO         (2 objects detected)                   │
│     ├─ [❌] EasyOCR      (CUDA out of memory)                   │
│     └─ [⏸] OpenCV       (Skipped due to failure)               │
│                                                                 │
│ Error: GPU memory exhausted during text detection              │
│ Scene: build/video47/scenes/scene_003.mp4                     │
│ Recovery: Attempted cache clear - FAILED                       │
│ Action: Video marked as FAILED, continuing batch               │
│                                                                 │
│ 📊 Batch Status: 46 success, 1 failed, 0 consecutive failures │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### **Circuit Breaker Activation**
```  
🚨 CIRCUIT BREAKER ACTIVATED - BATCH PROCESSING ABORTED

┌─────────────────────────────────────────────────────────────────┐
│ FAILURE PATTERN DETECTED                                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ Videos 47, 48, 49: Consecutive GPU memory failures             │
│ Common Error: CUDA out of memory during EasyOCR processing     │
│                                                                 │
│ 📊 Final Batch Results:                                         │
│     ✅ Successfully processed: 46/100 videos                    │
│     ❌ Failed videos: 3/100 videos                              │
│     ⏸ Remaining videos: 51/100 videos (skipped)                │
│                                                                 │
│ 🔧 Recommended Actions:                                         │
│     1. Check GPU memory usage and available VRAM               │
│     2. Consider reducing EasyOCR batch size in config          │
│     3. Restart batch processing from video 47                  │
│                                                                 │
│ 📁 Successful outputs available in: output/                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## **Logging and Diagnostics**

### **Comprehensive Error Logging**
```python
class ErrorLogger:
    def log_video_failure(self, video_path, error_context):
        """Detailed error logging for diagnostics"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'video_path': video_path,
            'error_stage': error_context['stage'],
            'tool_name': error_context.get('tool'),
            'error_type': error_context['error_type'],
            'error_message': str(error_context['error']),
            'system_state': self.capture_system_state(),
            'recovery_attempts': error_context.get('recovery_attempts', []),
            'final_action': error_context['action']
        }
        
        # Write to structured log file
        with open('build/error_logs/processing_errors.jsonl', 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def capture_system_state(self):
        """Capture system state for debugging"""
        return {
            'gpu_memory_used': torch.cuda.memory_allocated(),
            'gpu_memory_cached': torch.cuda.memory_cached(),
            'cpu_memory_percent': psutil.virtual_memory().percent,
            'disk_space_gb': psutil.disk_usage('.').free // (1024**3)
        }
```

### **Error Pattern Analysis**
```python
class ErrorPatternAnalyzer:
    def analyze_batch_errors(self, error_log):
        """Analyze error patterns for improvement suggestions"""
        patterns = {
            'gpu_memory_errors': [],
            'file_format_errors': [],
            'scene_detection_errors': [],
            'tool_crash_errors': []
        }
        
        for error in error_log:
            self.categorize_error(error, patterns)
        
        return self.generate_improvement_suggestions(patterns)
    
    def generate_improvement_suggestions(self, patterns):
        """Generate actionable suggestions based on error patterns"""
        suggestions = []
        
        if len(patterns['gpu_memory_errors']) > 2:
            suggestions.append({
                'issue': 'Frequent GPU memory errors',
                'solution': 'Reduce batch size or enable CPU fallback for EasyOCR',
                'config_change': 'gpu.memory_limit = 0.8'
            })
        
        return suggestions
```

---

## **Success Criteria**
- Reliable fail-fast processing prevents corrupted outputs
- Circuit breaker protects against systematic failures  
- Comprehensive error logging enables rapid debugging
- Clear error reporting guides user remediation
- Graceful degradation maintains batch processing continuity

---

**Integration Point**: All processing services implement error handling protocols with consistent reporting and recovery strategies