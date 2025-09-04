# FFmpeg Foundation - Video/Audio Separation

**Priority**: Phase 1 - Foundation  
**Risk**: LOW - Mature technology  
**Dependencies**: None (foundation service)

---

## **Purpose**
FFmpeg handles all video file preparation, providing clean video and audio streams for downstream AI tools. This is the foundation that all other processing depends on.

## **Core Functions**

### **1. Media Stream Separation**
```python
# Extract audio stream for transcription tools
ffmpeg -i input/video.mp4 -vn -acodec pcm_s16le -ar 22050 -ac 2 build/video-name/audio.wav

# Extract video stream for visual analysis tools  
ffmpeg -i input/video.mp4 -an -vcodec libx264 build/video-name/video.mp4
```

### **2. Format Standardization**
- **Audio Output**: 22.050 kHz, 16-bit PCM WAV (Whisper optimized)
- **Video Output**: H.264 MP4, maintain original resolution
- **Compatibility**: Handle all common video formats (MP4, AVI, MOV, MKV, WebM)

### **3. Scene-Based Video Splitting** (Phase 2)
```python
# After PySceneDetect provides scene boundaries
ffmpeg -i build/video-name/video.mp4 -ss 00:00:30 -to 00:02:15 build/video-name/scenes/scene_001.mp4
ffmpeg -i build/video-name/video.mp4 -ss 00:02:15 -to 00:05:45 build/video-name/scenes/scene_002.mp4
```

---

## **Implementation Specification**

### **Service Architecture**
```python
class FFmpegService:
    def __init__(self, config):
        self.config = config
        self.verify_ffmpeg_availability()
    
    def extract_streams(self, video_path):
        """Phase 1: Extract audio.wav and video.mp4"""
        output_dir = self.get_build_directory(video_path)
        
        # Audio extraction for Whisper/LibROSA/pyAudioAnalysis
        audio_path = self.extract_audio(video_path, output_dir)
        
        # Video extraction for YOLO/EasyOCR/OpenCV  
        video_path = self.extract_video(video_path, output_dir)
        
        return audio_path, video_path
    
    def split_by_scenes(self, video_path, scene_boundaries):
        """Phase 2: Create individual scene files"""
        scene_files = []
        for i, (start, end) in enumerate(scene_boundaries):
            scene_file = self.extract_scene(video_path, start, end, i+1)
            scene_files.append(scene_file)
        return scene_files
```

### **Error Handling**
- **File Format Validation**: Check input format compatibility
- **Codec Support**: Verify required codecs available
- **Disk Space**: Ensure sufficient storage for extraction
- **Permission Checks**: Validate read/write access to directories

### **Output Structure**
```
build/video-name/
├── audio.wav           # For audio analysis tools
├── video.mp4           # For visual analysis tools
├── scenes/             # Phase 2: Per-scene files
│   ├── scene_001.mp4
│   ├── scene_002.mp4
│   └── scene_N.mp4
├── metadata.json       # Processing metadata
└── ffmpeg_logs.txt     # Processing logs
```

---

## **Integration Points**

### **Downstream Tool Requirements**
- **Whisper**: Requires `audio.wav` (22.050 kHz PCM)
- **YOLO**: Requires `video.mp4` or `scene_N.mp4` files  
- **EasyOCR**: Requires video files or extracted frames
- **LibROSA**: Requires `audio.wav` for feature extraction
- **pyAudioAnalysis**: Requires `audio.wav` for comprehensive analysis
- **OpenCV**: Requires video files for face detection

### **PySceneDetect Integration** (Phase 2)
```python
# PySceneDetect analyzes video.mp4 → provides scene boundaries
# FFmpeg uses boundaries to create individual scene files
def coordinate_scene_splitting(video_path):
    scene_boundaries = pyscenedetect.detect_scenes(video_path)
    scene_files = ffmpeg_service.split_by_scenes(video_path, scene_boundaries)
    return scene_files, scene_boundaries
```

---

## **Configuration Options**

### **Audio Settings**
- **Sample Rate**: 22050 Hz (Whisper optimized)
- **Bit Depth**: 16-bit PCM
- **Channels**: 2 (stereo preserved)
- **Format**: WAV uncompressed

### **Video Settings**  
- **Codec**: H.264 (maximum compatibility)
- **Resolution**: Preserve original
- **Frame Rate**: Preserve original
- **Quality**: High quality for AI analysis

### **Performance Settings**
- **Hardware Acceleration**: Auto-detect GPU acceleration
- **Thread Count**: CPU core optimization
- **Memory Usage**: Balanced for concurrent AI tool usage

---

## **Quality Assurance**

### **Validation Checks**
- **Audio Quality**: Verify sample rate and bit depth
- **Video Quality**: Ensure no frame corruption
- **Synchronization**: Maintain A/V sync for scene splitting
- **File Integrity**: Validate output file completeness

### **Error Recovery**
- **Retry Logic**: Attempt with different codec options
- **Fallback Formats**: Alternative codec/format combinations
- **User Notification**: Clear error reporting for unsupported formats
- **Graceful Failure**: Mark video as failed, continue batch processing

---

## **Success Criteria**
- Clean audio.wav files compatible with all audio analysis tools
- High-quality video.mp4 files for visual analysis
- Reliable scene splitting based on PySceneDetect boundaries  
- Robust error handling for diverse video formats
- Foundation for all downstream AI tool processing

---

**Next Integration**: Whisper transcription and YOLO object detection using FFmpeg-prepared files