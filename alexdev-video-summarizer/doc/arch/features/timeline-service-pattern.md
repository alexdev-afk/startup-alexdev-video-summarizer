# Timeline Service Pattern: Raw Analysis → LLM-Ready Format

*Authoritative specification for creating semantic timeline services from raw AI analysis data*

---

## **Core Concept: The 3-Stage Integration Process**

The timeline service pattern transforms raw AI tool output into LLM-ready semantic events through a proven 3-stage architecture:

1. **Stage 1**: `build/[type]_analysis/` - Raw structured data export with metadata only
2. **Stage 2**: `build/[type]_timelines/` - Semantic timeline events with temporal coordination  
3. **Stage 3**: `master_timeline.json` - Multi-service coordination with intelligent filtering

---

## **Stage 1: Raw Analysis Files (`_analysis.json`)**

**Purpose**: ALL raw processing data for validation and parameter tuning - enables humans to validate semantic interpretations in Stage 2

### **Required Structure**
```json
{
  "metadata": {
    "source": "service_name",
    "processing_timestamp": "2025-09-06T19:41:21.027235",
    "total_duration": 30.33,
    "configuration": {
      "model_name": "yolov8x.pt",
      "device": "cuda", 
      "confidence_threshold": 0.75
    }
  },
  "processing_summary": {
    "detected_events": 25,
    "detected_spans": 4,
    "processing_time": 21.63,
    "analysis_windows": {
      "window_size": 5.0,
      "overlap": 2.5
    },
    "thresholds": {
      "significance_threshold": 0.5,
      "change_threshold": 10.0
    }
  },
  "scene_context": {
    // Scene boundary information if applicable
  },
  "raw_detections_by_scene": {
    "scene_001": [
      {
        "timestamp": 1.2,
        "class": "person",
        "class_id": 0,
        "confidence": 0.87,
        "bounding_box": [150, 200, 350, 400],
        "area": 40000,
        "keyframe_source": true
      }
    ]
  },
  "raw_keyframes_analyzed": {
    "scene_001": [
      {
        "timestamp": 1.2,
        "keyframe_type": "motion_based",
        "motion_score": 0.8
      }
    ]
  },
  "raw_semantic_transformations": {
    "scene_001": {
      "input_raw_detections": 5,
      "output_semantic_events": 2,
      "semantic_events": [
        {
          "description": "Person enters scene from left",
          "confidence": 0.85,
          "transformation_logic": "object_appearance_from_raw_detection"
        }
      ]
    }
  }
}
```

### **Key Principles**
- **Size**: Large files (50K-200K bytes) containing ALL raw processing data
- **Content**: Configuration, thresholds, processing stats, scene context, AND complete raw results
- **VALIDATION**: All raw detections, keyframes, and semantic transformations for human validation
- **Purpose**: Enable parameter tuning, threshold adjustment, and semantic interpretation validation

---

## **Stage 2: Timeline Files (`_timeline.json`)**

**Purpose**: Full semantic timeline with LLM-ready event descriptions and temporal coordination

### **Required Structure**
```json
{
  "global_data": {
    "audio_file": "build/bonita/audio.wav",
    "duration": 30.33,
    "sources_used": ["service_name"]
  },
  "events": [
    {
      "timestamp": 2.45,
      "description": "Person enters scene from left",
      "source": "yolo",
      "confidence": 0.87,
      "details": {
        "analysis_type": "scene_change",
        "change_magnitude": 0.6,
        "previous_state": "empty_scene",
        "new_state": "single_person"
      }
    }
  ],
  "timeline_spans": [
    {
      "start": 0.0,
      "end": 5.2,
      "description": "Static scene composition",
      "source": "yolo", 
      "confidence": 0.75,
      "details": {
        "dominant_objects": ["person", "chair"],
        "scene_stability": "high",
        "analysis_type": "scene_segment"
      }
    }
  ],
  "metadata": {
    "total_events": 25,
    "total_spans": 4,
    "processing_notes": [
      "Motion-aware keyframe sampling with 8-12 frames per scene",
      "Semantic change detection with 0.5 significance threshold"
    ]
  }
}
```

### **Key Principles**
- **Size**: Large files (10K-50K bytes) containing full processed data
- **Content**: Semantic events, timeline spans, detailed analysis
- **LLM-Ready**: Human-readable descriptions with confidence scoring
- **Temporal**: Precise timestamps and duration spans for coordination

---

## **Raw Data → Semantic Transformation Patterns**

### **Pattern 1: Change Detection**
Transform raw consecutive measurements into semantic change events:

**Raw Data**: 
```
detections: [
  {bbox: [100,200,300,400], class: "person", confidence: 0.85, timestamp: 1.2},
  {bbox: [150,200,350,400], class: "person", confidence: 0.87, timestamp: 2.1}  
]
```

**Semantic Event**:
```json
{
  "timestamp": 2.1,
  "description": "Person moves right across scene",
  "confidence": 0.86,
  "details": {
    "analysis_type": "motion_detection",
    "movement_direction": "right",
    "displacement": 50,
    "movement_confidence": 0.9
  }
}
```

### **Pattern 2: Threshold-Based Significance**
Only create events when changes exceed meaningful thresholds:

**Configuration Thresholds**:
```yaml
yolo:
  scene_change_threshold: 0.3      # 30% bounding box change
  new_object_confidence: 0.7       # Minimum confidence for "enters scene"
  object_disappear_frames: 3       # Frames missing before "exits scene"
  motion_significance: 25          # Pixel movement threshold
```

### **Pattern 3: Contextual State Tracking**
Maintain scene context to create meaningful semantic descriptions:

```python
class SceneState:
    def __init__(self):
        self.active_objects = {}     # Object ID -> last detection
        self.scene_composition = []  # Current objects in scene
        self.interaction_zones = {}  # Proximity tracking
    
    def detect_semantic_changes(self, new_detections):
        events = []
        
        # New object enters scene
        for obj in new_detections:
            if obj.id not in self.active_objects:
                events.append({
                    "description": f"{obj.class_name} enters scene from {self._detect_entry_direction(obj)}",
                    "analysis_type": "object_appearance"
                })
        
        # Object exits scene  
        for obj_id, last_detection in self.active_objects.items():
            if obj_id not in [d.id for d in new_detections]:
                events.append({
                    "description": f"{last_detection.class_name} exits scene",
                    "analysis_type": "object_disappearance"
                })
                
        return events
```

### **Pattern 4: Multi-Frame Coordination**
Use motion-aware keyframes to create richer semantic context:

```python
def analyze_keyframe_sequence(self, keyframes):
    """Analyze sequence of motion-aware keyframes for semantic patterns"""
    
    events = []
    
    for i in range(1, len(keyframes)):
        prev_frame = keyframes[i-1]
        curr_frame = keyframes[i]
        
        # Detect interactions between objects
        interactions = self._detect_object_interactions(prev_frame, curr_frame)
        for interaction in interactions:
            events.append({
                "timestamp": curr_frame.timestamp,
                "description": f"{interaction.obj1} and {interaction.obj2} {interaction.interaction_type}",
                "details": {
                    "analysis_type": "object_interaction",
                    "interaction_confidence": interaction.confidence,
                    "spatial_relationship": interaction.proximity
                }
            })
    
    return events
```

---

## **Service Implementation Requirements**

### **1. Timeline Service Class Structure**
```python
class [Service]TimelineService:
    def __init__(self, config: Dict[str, Any]):
        # Load thresholds from config
        self.significance_threshold = config.get('significance_threshold', 0.5)
        self.change_threshold = config.get('change_threshold', 0.3)
        
    def generate_and_save(self, input_path: str, scene_offsets: Optional[str]) -> EnhancedTimeline:
        # 1. Process raw service data
        raw_results = self._process_with_service(input_path, scene_offsets)
        
        # 2. Transform to semantic events
        semantic_events = self._create_semantic_events(raw_results)
        
        # 3. Create timeline spans  
        timeline_spans = self._create_timeline_spans(raw_results, semantic_events)
        
        # 4. Save both analysis and timeline files
        self._save_intermediate_analysis(raw_results, input_path)
        return self._save_timeline(semantic_events, timeline_spans, input_path)
```

### **2. Motion-Aware Integration**
For visual services, integrate with `MotionAwareSampler`:

```python
def _process_with_motion_analysis(self, video_path, scene_info):
    # Get motion-aware keyframes instead of single representative frame
    keyframes = self.motion_sampler.extract_motion_keyframes(video_path, scene_info)
    
    all_detections = []
    for keyframe in keyframes:
        # Process each keyframe with service
        detections = self._analyze_keyframe(keyframe)
        detections['keyframe_type'] = keyframe.get('keyframe_type', 'motion_based')
        detections['motion_score'] = keyframe.get('motion_score', 0.5)
        all_detections.append(detections)
    
    # Create semantic events from keyframe sequence
    return self._create_semantic_events_from_sequence(all_detections)
```

### **3. Required Methods**
Every timeline service must implement:

- `_create_semantic_events(raw_data)` - Transform raw analysis into LLM-ready events
- `_create_timeline_spans(raw_data, events)` - Create temporal coordination spans  
- `_save_intermediate_analysis(raw_data, path)` - Save metadata-only analysis file
- `_save_timeline(events, spans, path)` - Save full semantic timeline file

---

## **Quality Criteria for LLM-Ready Events**

### **1. Semantic Descriptions**
✅ **Good**: `"Person enters scene from left"`  
❌ **Bad**: `"Person detected at coordinates [150, 200, 350, 400]"`

✅ **Good**: `"Two people begin conversation"`  
❌ **Bad**: `"Person confidence 0.85, Person confidence 0.73"`

### **2. Confidence Scoring**
Base confidence on semantic meaning, not just technical accuracy:
- **Detection confidence**: Raw model output (0.85)
- **Semantic confidence**: Confidence in interpretation (0.75 for "begins conversation")
- **Change confidence**: Confidence in detected change (0.90 for clear movement)

### **3. Actionable Details**
Include contextual information useful for LLM understanding:
```json
{
  "details": {
    "analysis_type": "scene_change",
    "change_magnitude": 0.6,
    "previous_state": "empty_room", 
    "new_state": "person_present",
    "spatial_context": "center_of_scene",
    "movement_direction": "left_to_right"
  }
}
```

### **4. Temporal Precision**
- **Events**: Precise timestamps for moments of change
- **Spans**: Duration ranges for stable states or ongoing activities
- **Coordination**: Timestamps align across all services for master timeline integration

---

## **Testing and Validation**

### **File Size Validation**
- **Analysis files**: 50K-200K bytes (all raw data for validation)
- **Timeline files**: 10K-50K bytes (semantic events and spans only)
- **Ratio**: Analysis should be 5-10x larger than timeline (raw data vs semantic summaries)

### **Content Validation**
- **Events**: Human-readable descriptions with confidence > 0.5
- **Spans**: Meaningful temporal segments with contextual details  
- **Metadata**: Processing configuration and statistics preserved

### **LLM Integration Test**
Timeline events should be directly usable in prompts:
```
"Based on the video timeline, describe what happens:
- At 2.1s: Person enters scene from left (confidence: 0.86)
- At 5.3s: Person and chair interaction detected (confidence: 0.72)
- At 8.7s: Scene composition stabilizes (confidence: 0.81)"
```

---

## **Common Anti-Patterns to Avoid**

❌ **Raw Data Dump**: Including bounding boxes, model outputs, or technical coordinates  
❌ **No Semantic Layer**: Events like "Detection 1", "Detection 2" without meaning  
❌ **Missing Confidence**: Events without reliability scoring  
❌ **No Temporal Context**: Events without proper timeline coordination  
❌ **Oversized Analysis Files**: Putting actual data in metadata-only analysis files  
❌ **Undersized Timeline Files**: Not including full semantic processing in timeline files

---

## **Success Metrics**

### **Technical Metrics**
- Analysis file size: 500-2000 bytes
- Timeline file size: 10-50K bytes  
- Event density: 0.5-2.0 events per second of content
- Confidence distribution: Mean > 0.7, Min > 0.5

### **Semantic Quality**
- **Human Readability**: Non-technical person can understand event descriptions
- **LLM Usability**: Events can be directly incorporated into LLM prompts
- **Temporal Coherence**: Events and spans create coherent narrative of content
- **Contextual Richness**: Details provide actionable context for downstream processing

---

**This pattern has been proven successful with audio services (Whisper, LibROSA, pyAudioAnalysis) and provides the foundation for visual service implementation (YOLO, EasyOCR, OpenCV).**