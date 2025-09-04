# System Design Overview

Complete system architecture for alexdev-video-summarizer scene-based institutional knowledge extraction.

## Architecture Principles

### Core Design Philosophy
- **Scene-Based Processing**: 70x performance improvement through representative frame analysis
- **Service Isolation**: Each AI tool runs in isolated environment to prevent conflicts
- **Sequential CUDA**: GPU tools process one at a time to prevent memory conflicts
- **Fail-Fast Design**: All-or-nothing processing per video with circuit breaker protection
- **Local Processing**: Complete independence from cloud services

### Quality Attributes
- **Reliability**: Circuit breaker prevents cascade failures
- **Performance**: 10-minute processing target per video for rich analysis
- **Scalability**: 100+ video batch processing with overnight execution
- **Maintainability**: Modular service architecture with clear separation of concerns
- **Security**: Local processing with no external data transmission

## System Context

```
┌─────────────────────────────────────────────────────────────────┐
│                    SYSTEM BOUNDARY                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  INPUT VIDEOS  →  alexdev-video-summarizer  →  KNOWLEDGE BASE   │
│                                                                 │
│  • MP4, AVI, MOV        8-Tool AI Pipeline        • Scene-by-   │
│  • Local files          Service Architecture        scene .md   │
│  • 100+ videos         GPU + CPU coordination      • Cross-ref │
│                                                     • Search    │
└─────────────────────────────────────────────────────────────────┘
```

### External Dependencies
- **FFmpeg**: Video/audio processing foundation
- **AI Models**: YOLOv8, Whisper, EasyOCR (downloaded on first run)
- **GPU Drivers**: NVIDIA CUDA for optimal performance
- **Claude**: Manual synthesis workflow integration

## Component Architecture

### High-Level Components

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLI INTERFACE                           │
├─────────────────────────────────────────────────────────────────┤
│  Launch Screen  │  Processing Screen  │  Completion Screen      │
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                   ORCHESTRATION LAYER                          │
├─────────────────────────────────────────────────────────────────┤
│  VideoProcessingOrchestrator  │  CircuitBreaker  │  Progress    │
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                     SERVICE LAYER                              │
├─────────────────────────────────────────────────────────────────┤
│  FFmpeg    │  Scene     │  GPU Pipeline  │  CPU Pipeline      │
│  Service   │  Detection │  Controller    │  Controller        │
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                      AI TOOL SERVICES                          │
├─────────────────────────────────────────────────────────────────┤
│  Whisper  │  YOLO  │  EasyOCR  │  OpenCV  │  LibROSA  │ Audio │
│  (venv1)  │ (venv2) │  (venv3)  │  (venv4) │  (venv5)  │(venv6)│
└─────────────────────────────────────────────────────────────────┘
```

### Processing Pipeline

```
INPUT VIDEO → FFmpeg Foundation → Scene Detection → Dual Pipeline → Knowledge Base

1. FFmpeg Foundation
   ├── Audio Extraction (audio.wav)
   └── Video Extraction (video.mp4)

2. Scene Detection  
   ├── PySceneDetect Analysis
   └── Scene Boundary Creation

3. Per-Scene Processing
   ├── GPU Pipeline (Sequential)
   │   ├── Whisper Transcription
   │   ├── YOLO Object Detection  
   │   └── EasyOCR Text Extraction
   └── CPU Pipeline (Parallel)
       ├── OpenCV Face Detection
       ├── LibROSA Audio Features
       └── pyAudioAnalysis

4. Knowledge Generation
   ├── Scene Analysis Aggregation
   ├── Cross-Reference Generation
   └── Searchable Markdown Output
```

## Data Architecture

### Processing Data Flow

```
input/video.mp4
       │
       ▼
build/video-name/
├── audio.wav              ← FFmpeg extraction
├── video.mp4              ← FFmpeg extraction
├── scenes/                ← Scene detection + splitting
│   ├── scene_001.mp4
│   ├── scene_002.mp4
│   └── scene_metadata.json
├── analysis/              ← AI tool results
│   ├── whisper_results.json
│   ├── yolo_results.json
│   ├── easyocr_results.json
│   ├── opencv_results.json
│   ├── librosa_results.json
│   └── pyaudioanalysis_results.json
└── logs/
    ├── processing.log
    └── errors.log
       │
       ▼
output/video-name.md       ← Final knowledge base
```

### Data Models

#### Scene Context
```python
@dataclass
class SceneContext:
    scene_id: int
    start_seconds: float
    end_seconds: float
    duration: float
    representative_frame_path: Path
    audio_segment_path: Path
    analysis_results: Dict[str, Any]
```

#### Processing Result
```python
@dataclass  
class ProcessingResult:
    video_path: Path
    success: bool
    knowledge_file: Optional[Path]
    error: Optional[str]
    processing_time: float
    scenes_processed: int
```

## Service Architecture Details

### Service Isolation Strategy

```python
# Each service runs in isolated environment
class BaseAnalysisService:
    def __init__(self, service_name, venv_path):
        self.service_name = service_name
        self.venv_path = venv_path
        self.model = None
        
    def setup_isolated_environment(self):
        # Activate specific venv for this service
        # Verify dependencies
        # Load models in isolation
```

### GPU Pipeline Coordination

```python
class GPUPipelineController:
    def __init__(self):
        self.gpu_lock = threading.Lock()
        
    def process_gpu_pipeline(self, scene, context):
        with self.gpu_lock:
            # 1. Whisper (if audio)
            results['whisper'] = self.execute_gpu_service('whisper', scene)
            self.cleanup_gpu_memory()
            
            # 2. YOLO Object Detection
            results['yolo'] = self.execute_gpu_service('yolo', scene)  
            self.cleanup_gpu_memory()
            
            # 3. EasyOCR Text Extraction
            results['easyocr'] = self.execute_gpu_service('easyocr', scene)
            self.cleanup_gpu_memory()
```

### CPU Pipeline Coordination

```python
class CPUPipelineController:
    def process_cpu_pipeline(self, scene, context):
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                'opencv': executor.submit(self.opencv_service.process_scene, scene),
                'librosa': executor.submit(self.librosa_service.process_audio, scene),
                'pyaudioanalysis': executor.submit(self.audio_service.process_audio, scene)
            }
            # Collect results with timeout handling
```

## Error Handling Architecture

### Circuit Breaker Pattern

```python
class CircuitBreaker:
    def __init__(self, failure_threshold=3):
        self.failure_threshold = failure_threshold
        self.consecutive_failures = 0
        
    def record_failure(self):
        self.consecutive_failures += 1
        
    def should_trip(self) -> bool:
        return self.consecutive_failures >= self.failure_threshold
```

### Fail-Fast Strategy
- **Per Video**: Any tool failure fails entire video processing
- **Batch Level**: Circuit breaker stops after 3 consecutive video failures
- **Resource Cleanup**: Automatic cleanup on failures to prevent resource leaks

### Error Recovery Hierarchy
1. **Immediate Retry**: Simple transient errors
2. **Parameter Adjustment**: Retry with modified settings  
3. **Fallback Method**: Alternative processing approach
4. **Graceful Degradation**: Continue with reduced functionality
5. **Fail-Fast**: Mark video as failed, continue batch

## Performance Architecture

### Scene-Based Optimization

```
Traditional Approach:
Video (1000 frames) → Process all frames → 1000x processing time

Scene-Based Approach:  
Video → Detect 4 scenes → Process 4 representative frames → 70x improvement
```

### Resource Management
- **GPU Memory**: Sequential processing prevents CUDA conflicts
- **CPU Utilization**: Parallel CPU pipeline during GPU processing
- **Memory Cleanup**: Forced garbage collection between videos
- **Storage**: Automatic cleanup of intermediate artifacts

### Performance Targets
- **Per Video**: 10 minutes average processing time
- **Batch Processing**: 100 videos = ~17 hours (overnight)
- **Memory Usage**: 4-8GB VRAM (sequential), 16-32GB RAM
- **Storage**: 500MB-1GB per video (temporary artifacts)

## Security Architecture

### Data Protection
- **Local Processing**: All video content remains on local workstation
- **No Cloud Dependencies**: Complete independence from external services
- **File System Security**: Proper permissions for processing directories
- **Content Privacy**: Internal video content never leaves local environment

### System Security
- **Service Isolation**: Each tool runs in isolated Python environment
- **Resource Management**: Controlled GPU and CPU resource allocation
- **Error Containment**: Tool failures isolated to prevent system compromise
- **Clean Artifacts**: Automatic cleanup of temporary files

### Access Control
- **File Permissions**: Restricted access to processing directories
- **Process Isolation**: Services cannot access each other's data
- **Resource Limits**: Controlled memory and GPU usage per service

## Scalability Architecture

### Horizontal Scaling Potential
- **Multiple GPU Support**: Architecture supports multiple GPU coordination
- **Distributed Processing**: Scene-based architecture enables distribution
- **Batch Optimization**: Efficient resource utilization across large libraries

### Vertical Scaling
- **GPU Utilization**: Sequential processing maximizes single GPU efficiency  
- **CPU Parallelization**: Multi-core utilization for CPU-intensive tasks
- **Memory Management**: Efficient memory usage patterns

### Load Balancing
- **GPU/CPU Coordination**: Parallel CPU processing during GPU tasks
- **Scene Distribution**: Even workload distribution across scenes
- **Resource Allocation**: Dynamic resource allocation based on content

## Integration Architecture

### External Tool Integration
- **FFmpeg**: Standard command-line interface with error handling
- **AI Models**: Standardized model loading and inference patterns
- **File System**: Consistent directory structure and naming conventions

### Claude Integration
- **Manual Handoff**: CLI completion triggers manual Claude workflow
- **Structured Input**: Organized build/ directory for Claude consumption
- **Output Format**: Standardized analysis format for synthesis

### Configuration Management
- **YAML Configuration**: Human-readable configuration files
- **Environment Overrides**: Local configuration overrides
- **Runtime Settings**: Dynamic configuration updates

## Deployment Architecture

### Development Environment
- **Virtual Environment**: Isolated Python dependencies
- **Local Models**: AI models cached locally
- **Development Tools**: Testing and debugging utilities

### Production Environment  
- **System Requirements**: GPU-enabled workstation
- **Batch Processing**: Unattended overnight processing
- **Monitoring**: Progress tracking and error reporting

### Maintenance
- **Log Management**: Automated log rotation and cleanup
- **Model Updates**: Versioned model management
- **Configuration Updates**: Hot-reloadable configuration

## Quality Attributes Implementation

### Reliability
- **Circuit Breaker**: Prevents cascade failures
- **Error Recovery**: Multiple recovery strategies  
- **Resource Cleanup**: Automatic cleanup on failures
- **Progress Persistence**: Resumable processing

### Performance
- **Scene Optimization**: 70x improvement through representative frames
- **GPU Efficiency**: Sequential processing prevents conflicts
- **Parallel Processing**: CPU tasks run parallel to GPU
- **Memory Management**: Efficient resource utilization

### Maintainability
- **Service Architecture**: Clear separation of concerns
- **Configuration Management**: External configuration files
- **Logging**: Comprehensive logging and debugging
- **Documentation**: Complete technical documentation

### Usability
- **CLI Interface**: Simple 3-screen workflow
- **Progress Display**: Real-time progress visualization
- **Error Reporting**: Clear error messages and recovery guidance
- **Configuration**: YAML-based human-readable settings

This system design provides a robust, scalable, and maintainable foundation for scene-based institutional knowledge extraction from video libraries.