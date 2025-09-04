# Scene Detection System - PySceneDetect Integration

**Priority**: Phase 2 - Scene Architecture  
**Risk**: MEDIUM - Scene accuracy and FFmpeg coordination  
**Dependencies**: FFmpeg foundation (video.mp4 files)

---

## **Purpose**
PySceneDetect analyzes video content to identify scene boundaries, enabling 70x performance improvement through representative frame analysis instead of frame-by-frame processing.

## **Core Functions**

### **1. Content-Aware Scene Detection**
```python
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

def detect_scenes(video_path):
    """Detect scene boundaries using content analysis"""
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    
    # ContentDetector with optimized threshold
    scene_manager.add_detector(ContentDetector(threshold=27.0))
    
    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    
    scene_list = scene_manager.get_scene_list()
    fps = video_manager.get_framerate()
    video_manager.release()
    
    return scene_list, fps
```

### **2. Scene Boundary Processing**
```python
def process_scene_boundaries(scene_list, fps):
    """Convert scene timecodes to processing boundaries"""
    boundaries = []
    for i, scene in enumerate(scene_list):
        start_time = scene[0]
        end_time = scene[1] if i < len(scene_list)-1 else None
        
        boundaries.append({
            'scene_id': i + 1,
            'start_seconds': start_time.get_seconds(), 
            'end_seconds': end_time.get_seconds() if end_time else None,
            'start_frame': start_time.get_frames(),
            'end_frame': end_time.get_frames() if end_time else None,
            'duration': (end_time - start_time).get_seconds() if end_time else None
        })
    
    return boundaries
```

### **3. Representative Frame Extraction**
```python
def extract_representative_frame(scene_boundary):
    """Extract middle frame of scene for 70x performance improvement"""
    start_frame = scene_boundary['start_frame']
    end_frame = scene_boundary['end_frame']
    
    # Middle frame provides best scene representation
    middle_frame = int(start_frame + (end_frame - start_frame) / 2)
    
    return {
        'scene_id': scene_boundary['scene_id'],
        'representative_frame': middle_frame,
        'frame_timestamp': middle_frame / fps,
        'scene_context': scene_boundary
    }
```

---

## **Implementation Specification**

### **Service Architecture**
```python
class SceneDetectionService:
    def __init__(self, config):
        self.config = config
        self.threshold = config.get('scene_threshold', 27.0)
        self.downscale_factor = config.get('downscale_factor', 1)
    
    def analyze_video_scenes(self, video_path):
        """Complete scene analysis workflow"""
        try:
            # 1. Detect scene boundaries
            scene_list, fps = self.detect_scenes(video_path)
            
            # 2. Process boundaries into usable format
            boundaries = self.process_scene_boundaries(scene_list, fps)
            
            # 3. Extract representative frames
            representative_frames = []
            for boundary in boundaries:
                frame_info = self.extract_representative_frame(boundary)
                representative_frames.append(frame_info)
            
            # 4. Trigger FFmpeg scene splitting
            scene_files = self.coordinate_scene_splitting(video_path, boundaries)
            
            return {
                'scene_count': len(boundaries),
                'boundaries': boundaries,
                'representative_frames': representative_frames,
                'scene_files': scene_files,
                'fps': fps
            }
            
        except Exception as e:
            self.handle_scene_detection_error(video_path, e)
            raise
```

### **FFmpeg Coordination**
```python
def coordinate_scene_splitting(self, video_path, boundaries):
    """Trigger FFmpeg to create individual scene files"""
    scene_files = []
    
    for boundary in boundaries:
        scene_file = self.ffmpeg_service.extract_scene(
            video_path=video_path,
            start_seconds=boundary['start_seconds'],
            end_seconds=boundary['end_seconds'], 
            scene_id=boundary['scene_id']
        )
        scene_files.append(scene_file)
    
    return scene_files
```

---

## **Performance Optimization**

### **70x Performance Improvement Strategy**
- **Representative Frame Analysis**: Process middle frame per scene vs. all frames
- **Content-Aware Boundaries**: Smart scene detection vs. arbitrary time intervals  
- **Selective Processing**: Only analyze frames with significant content changes
- **Batch Scene Processing**: Process multiple scenes efficiently

### **Scene Detection Optimization**
```python
# Optimized scene detection parameters
SCENE_DETECTION_CONFIG = {
    'threshold': 27.0,          # Balanced sensitivity
    'min_scene_len': 2.0,       # Minimum 2-second scenes  
    'downscale_factor': 1,      # Full resolution analysis
    'frame_skip': 0             # Analyze every frame for accuracy
}
```

---

## **Integration with Dual Pipelines**

### **GPU Pipeline Integration**
```python
def process_scenes_gpu_pipeline(scene_data):
    """Process each scene through GPU tools sequentially"""
    for scene in scene_data['representative_frames']:
        # Extract frame for GPU processing
        frame = extract_frame_at_timestamp(
            video_path=scene_data['video_path'],
            timestamp=scene['frame_timestamp']  
        )
        
        # Sequential GPU processing per scene
        yolo_results = yolo_service.analyze_frame(frame, scene['scene_context'])
        easyocr_results = easyocr_service.analyze_frame(frame, scene['scene_context'])
        
        # Store results with scene context
        store_scene_analysis(scene['scene_id'], {
            'objects': yolo_results,
            'text': easyocr_results,
            'scene_context': scene['scene_context']
        })
```

### **CPU Pipeline Integration**  
```python
def process_scenes_cpu_pipeline(scene_data):
    """Process scene audio segments through CPU tools"""
    for boundary in scene_data['boundaries']:
        # Extract audio segment for this scene
        audio_segment = extract_audio_segment(
            audio_path=scene_data['audio_path'],
            start_seconds=boundary['start_seconds'],
            end_seconds=boundary['end_seconds']
        )
        
        # Parallel CPU processing per scene
        opencv_results = opencv_service.analyze_scene_video(boundary)  
        librosa_results = librosa_service.analyze_audio_segment(audio_segment)
        
        store_scene_audio_analysis(boundary['scene_id'], {
            'faces': opencv_results,
            'audio_features': librosa_results
        })
```

---

## **Output Structure**

### **Scene Metadata**
```json
{
  "video_path": "input/presentation.mp4",
  "scene_count": 4,
  "total_duration": 450.0,
  "fps": 30.0,
  "scenes": [
    {
      "scene_id": 1,
      "start_seconds": 0.0,
      "end_seconds": 120.5,
      "duration": 120.5,
      "representative_frame": 1815,
      "representative_timestamp": 60.25,
      "scene_file": "build/presentation/scenes/scene_001.mp4"
    }
  ]
}
```

### **Representative Frame Analysis**
```json
{
  "scene_1": {
    "representative_frame": 1815,
    "timestamp": 60.25,
    "scene_context": "Conference room discussion",
    "visual_analysis": {
      "objects": ["person", "laptop", "whiteboard"],
      "text_overlays": ["Q4 Budget Review"],
      "faces_detected": 2
    },
    "audio_context": {
      "speakers": ["John", "Sarah"],
      "audio_type": "speech",
      "background_noise": "minimal"
    }
  }
}
```

---

## **Error Handling**

### **Scene Detection Failures**
- **Video Format Issues**: Fallback to simpler detection methods
- **Content Analysis Failures**: Use time-based scene splitting as backup
- **Memory Issues**: Reduce downscale factor and retry
- **Threshold Adjustment**: Auto-adjust threshold based on content type

### **FFmpeg Coordination Errors**
- **Scene Splitting Failures**: Mark individual scenes as failed
- **File Access Issues**: Retry with different file paths
- **Disk Space Problems**: Clean temporary files and retry
- **Permission Errors**: Clear error reporting for user action

---

## **Quality Assurance**

### **Scene Boundary Accuracy**
- **Content Change Validation**: Verify boundaries align with visual changes
- **Minimum Scene Length**: Prevent overly short scenes (2+ seconds)
- **Maximum Scene Length**: Split very long scenes (10+ minutes)
- **Representative Frame Quality**: Ensure middle frames are high-quality

### **Performance Validation**
- **Processing Speed**: Verify 70x improvement vs. frame-by-frame
- **Memory Usage**: Monitor PySceneDetect memory consumption
- **GPU Coordination**: Ensure clean handoff to visual analysis tools
- **File Management**: Validate scene file creation and cleanup

---

## **Success Criteria**
- Accurate scene boundary detection for diverse video content
- Successful coordination with FFmpeg for scene file creation
- 70x performance improvement through representative frame analysis
- Clean integration with dual pipeline processing
- Reliable error handling for edge cases

---

**Next Integration**: Dual-pipeline coordination for per-scene processing with GPU and CPU pipelines