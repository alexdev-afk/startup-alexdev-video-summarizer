# Scene-Based Processing Pipeline Architecture
*Created: 2025-09-04*
*Final Architecture Decision: Rich Analysis with Scene-Based Processing*

## FINALIZED ARCHITECTURE DECISIONS

### ✅ USER DECISIONS IMPLEMENTED
1. **🎯 Audio Analysis**: pyAudioAnalysis (expert speaker diarization)
2. **🔄 Scene Detection**: PySceneDetect (replace FFmpeg entirely)
3. **⚡ GPU Scheduling**: Sequential processing (RTX 5070 safety)
4. **🏢 Brand Detection**: YOLOv8 general objects (no custom models)
5. **🎬 Processing Strategy**: Scene-based pipeline (eliminate merge conflicts)

## SCENE-BASED PROCESSING ARCHITECTURE

### DETAILED PIPELINE FLOW

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: SCENE DETECTION & VIDEO SPLITTING                                         │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  Input: marketing_video.mp4 (15.2MB, 3:12 duration)                                │
│       │                                                                             │
│       ▼                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐ │
│  │ PySceneDetect Analysis                                                          │ │
│  │ • Threshold: 0.2 (proven from reference analysis)                              │ │
│  │ • Content detection: histogram-based scene changes                             │ │
│  │ • Generate scene list with timestamps                                          │ │
│  │                                                                                 │ │
│  │ Example Output:                                                                 │ │
│  │ Scene 01: 00:00:00 - 00:00:15 (Intro/Logo)                                     │ │
│  │ Scene 02: 00:00:15 - 00:01:30 (Presenter speaking)                             │ │
│  │ Scene 03: 00:01:30 - 00:02:45 (Product demo)                                   │ │
│  │ Scene 04: 00:02:45 - 00:03:12 (Call to action)                                 │ │
│  └─────────────────────────────────────────────────────────────────────────────────┘ │
│       │                                                                             │
│       ▼                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐ │
│  │ FFmpeg Video Splitting                                                          │ │
│  │ • Split video file based on PySceneDetect timestamps                           │ │
│  │ • Generate individual scene files                                              │ │
│  │                                                                                 │ │
│  │ Output Files:                                                                   │ │
│  │ ├── temp/scenes/scene_01.mp4 (15s)                                             │ │
│  │ ├── temp/scenes/scene_02.mp4 (75s)                                             │ │
│  │ ├── temp/scenes/scene_03.mp4 (75s)                                             │ │
│  │ └── temp/scenes/scene_04.mp4 (27s)                                             │ │
│  └─────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ PHASE 2: PER-SCENE VISUAL ANALYSIS (Sequential GPU Processing)                     │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  FOR EACH SCENE FILE:                                                              │
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐ │
│  │ SCENE PROCESSING LOOP                                                           │ │
│  │                                                                                 │ │
│  │ Current: scene_01.mp4 (Intro/Logo scene)                                       │ │
│  │                                                                                 │ │
│  │ Step 1: YOLOv8 Object Detection                                                 │ │
│  │ ┌─────────────────────────────────────────────────────────────────────────────┐ │ │
│  │ │ • Load YOLOv8 model to GPU                                                  │ │ │
│  │ │ • Extract key frames from scene (every 1-2 seconds)                        │ │ │
│  │ │ • Detect: persons, objects, logos, text regions                            │ │ │
│  │ │ • Confidence thresholds: >0.5 for persons, >0.3 for objects               │ │ │
│  │ │                                                                             │ │ │
│  │ │ Example Output:                                                             │ │ │
│  │ │ Frame 00:00:02: [person: 0.89, logo: 0.72, text: 0.65]                    │ │ │
│  │ │ Frame 00:00:05: [person: 0.91, logo: 0.78]                                 │ │ │
│  │ │                                                                             │ │ │
│  │ │ • Unload YOLOv8, clear GPU memory                                           │ │ │
│  │ └─────────────────────────────────────────────────────────────────────────────┘ │ │
│  │                                                                                 │ │
│  │ Step 2: EasyOCR Text Detection                                                  │ │
│  │ ┌─────────────────────────────────────────────────────────────────────────────┐ │ │
│  │ │ • Load EasyOCR model to GPU                                                 │ │ │
│  │ │ • Extract text from same key frames                                        │ │ │
│  │ │ • Languages: English + common marketing languages                          │ │ │
│  │ │ • Confidence threshold: >0.4 for marketing text                            │ │ │
│  │ │                                                                             │ │ │
│  │ │ Example Output:                                                             │ │ │
│  │ │ Frame 00:00:02: ["ACME PRODUCTS", "New Launch 2025"]                       │ │ │
│  │ │ Frame 00:00:05: ["ACME PRODUCTS"]                                          │ │ │
│  │ │                                                                             │ │ │
│  │ │ • Unload EasyOCR, clear GPU memory                                          │ │ │
│  │ └─────────────────────────────────────────────────────────────────────────────┘ │ │
│  │                                                                                 │ │
│  │ Step 3: OpenCV Analysis (CPU-based)                                            │ │
│  │ ┌─────────────────────────────────────────────────────────────────────────────┐ │ │
│  │ │ • Face detection within person bounding boxes                              │ │ │
│  │ │ • Motion analysis: camera movement, object movement                        │ │ │
│  │ │ • Color palette analysis: dominant colors, mood                            │ │ │
│  │ │ • Frame quality assessment                                                  │ │ │
│  │ │                                                                             │ │ │
│  │ │ Example Output:                                                             │ │ │
│  │ │ Faces: 1 detected, confidence: 0.87                                        │ │ │
│  │ │ Motion: minimal camera movement, static presenter                           │ │ │
│  │ │ Colors: [#1a365d: 35%, #ffffff: 25%, #e53e3e: 15%]                        │ │ │
│  │ │ Quality: Sharp, well-lit, professional                                     │ │ │
│  │ └─────────────────────────────────────────────────────────────────────────────┘ │ │
│  │                                                                                 │ │
│  │ Result: scene_01_visual_analysis.json                                          │ │
│  └─────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                     │
│  REPEAT FOR: scene_02.mp4, scene_03.mp4, scene_04.mp4...                          │
│                                                                                     │
│  Final Visual Output:                                                              │
│  ├── scene_01_visual_analysis.json                                                 │
│  ├── scene_02_visual_analysis.json                                                 │ │
│  ├── scene_03_visual_analysis.json                                                 │
│  └── scene_04_visual_analysis.json                                                 │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ PHASE 3: FULL-VIDEO AUDIO ANALYSIS (Parallel Processing)                          │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  Input: Original marketing_video.mp4 (full audio track)                            │
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐ │
│  │ Audio Processing Pipeline (CPU-based, can run parallel to scene analysis)      │ │
│  │                                                                                 │ │
│  │ Step 1: Whisper Large Transcription                                            │ │
│  │ ┌─────────────────────────────────────────────────────────────────────────────┐ │ │
│  │ │ • Enhanced Whisper with timestamp precision                                 │ │ │
│  │ │ • Speaker change detection (basic)                                          │ │ │
│  │ │ • Audio event tags: [Music], [Applause], etc.                              │ │ │
│  │ │                                                                             │ │ │
│  │ │ Example Output:                                                             │ │ │
│  │ │ 00:00:00-00:00:15: [Music] Welcome to ACME Products                        │ │ │
│  │ │ 00:00:15-00:01:30: Hi everyone, I'm Sarah from marketing...                │ │ │
│  │ │ 00:01:30-00:02:45: Let me show you our new features...                     │ │ │
│  │ └─────────────────────────────────────────────────────────────────────────────┘ │ │
│  │                                                                                 │ │
│  │ Step 2: LibROSA Music Analysis                                                  │ │
│  │ ┌─────────────────────────────────────────────────────────────────────────────┐ │ │
│  │ │ • Extract music segments (separate from speech)                             │ │ │
│  │ │ • Tempo, key, energy level analysis                                        │ │ │
│  │ │ • Genre classification, mood detection                                      │ │ │
│  │ │ • Background vs foreground music identification                             │ │ │
│  │ │                                                                             │ │ │
│  │ │ Example Output:                                                             │ │ │
│  │ │ 00:00:00-00:00:15: Upbeat corporate, 120 BPM, major key, energetic        │ │ │
│  │ │ 00:02:45-00:03:12: Soft outro music, 90 BPM, resolved feeling             │ │ │
│  │ └─────────────────────────────────────────────────────────────────────────────┘ │ │
│  │                                                                                 │ │
│  │ Step 3: pyAudioAnalysis Speaker Diarization                                    │ │
│  │ ┌─────────────────────────────────────────────────────────────────────────────┐ │ │
│  │ │ • Advanced speaker identification and separation                             │ │ │
│  │ │ • Gender detection, emotional tone analysis                                 │ │ │
│  │ │ • Speaking pace, clarity assessment                                         │ │ │
│  │ │ • Audio quality indicators                                                  │ │ │
│  │ │                                                                             │ │ │
│  │ │ Example Output:                                                             │ │ │
│  │ │ Speaker 1 (Sarah): Female, confident tone, clear speech, 00:00:15-02:45    │ │ │
│  │ │ Audio Quality: Professional microphone, minimal background noise           │ │ │
│  │ └─────────────────────────────────────────────────────────────────────────────┘ │ │
│  │                                                                                 │ │
│  │ Result: full_audio_analysis.json (with precise timestamps)                     │ │
│  └─────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ PHASE 4: TIMELINE INTEGRATION & RICH METADATA GENERATION                          │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐ │
│  │ INTELLIGENT TIMELINE MERGER                                                     │ │
│  │                                                                                 │ │
│  │ Input Sources:                                                                  │ │
│  │ • Scene boundaries + visual analysis per scene                                 │ │
│  │ • Full-video audio analysis with timestamps                                    │ │
│  │ • Original frame extraction from PySceneDetect                                 │ │
│  │                                                                                 │ │
│  │ Merge Logic:                                                                    │ │
│  │ ┌─────────────────────────────────────────────────────────────────────────────┐ │ │
│  │ │ 1. Map audio timeline to scene boundaries                                   │ │ │
│  │ │ 2. Combine visual analysis with corresponding audio segments                │ │ │
│  │ │ 3. Generate scene-by-scene rich descriptions                                │ │ │
│  │ │ 4. Create searchable metadata structure                                     │ │ │
│  │ │ 5. Flag potential duplicates based on visual + audio similarity            │ │ │
│  │ └─────────────────────────────────────────────────────────────────────────────┘ │ │
│  │                                                                                 │ │
│  │ Example Rich Output:                                                            │ │
│  │ ┌─────────────────────────────────────────────────────────────────────────────┐ │ │
│  │ │ Scene 01 (00:00:00-00:00:15): "Brand Introduction"                         │ │ │
│  │ │ • Visual: ACME logo prominent, professional lighting, corporate colors     │ │ │
│  │ │ • Audio: Upbeat corporate music (120 BPM), no speech                       │ │ │
│  │ │ • Text: "ACME PRODUCTS", "New Launch 2025"                                 │ │ │
│  │ │ • Objects: Logo, text graphics, branded background                         │ │ │
│  │ │ • Mood: Professional, energetic, modern                                    │ │ │
│  │ │                                                                             │ │ │
│  │ │ Scene 02 (00:00:15-00:01:30): "Presenter Introduction"                     │ │ │
│  │ │ • Visual: 1 person (female), professional attire, minimal camera movement │ │ │
│  │ │ • Audio: Speaker "Sarah" (confident, clear), background music fades       │ │ │
│  │ │ • Speech: "Hi everyone, I'm Sarah from marketing..."                       │ │ │
│  │ │ • Objects: Person, corporate background, subtle branding                   │ │ │
│  │ │ • Mood: Friendly, professional, welcoming                                  │ │ │
│  │ └─────────────────────────────────────────────────────────────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## ENHANCED OUTPUT STRUCTURE

### Per-Video Output Package
```
output/marketing_video/
├── transcript.txt                    # Enhanced with rich scene-by-scene metadata
├── scenes/
│   ├── scene_01.png                 # Representative frame from each scene
│   ├── scene_02.png
│   ├── scene_03.png
│   └── scene_04.png
├── analysis/
│   ├── full_audio_analysis.json     # Complete audio timeline
│   ├── scene_01_visual_analysis.json
│   ├── scene_02_visual_analysis.json
│   ├── scene_03_visual_analysis.json
│   └── scene_04_visual_analysis.json
├── rich_metadata.json               # Merged timeline with searchable data
└── processing_log.txt               # Detailed processing information
```

### Enhanced Transcript Header Example
```
# RICH VIDEO ANALYSIS REPORT
# Generated: 2025-09-04 15:45:22
# Video: marketing_video.mp4 (15.2MB, 3:12 duration)
# Processing Time: 12 minutes 34 seconds
# Analysis Confidence: 91% average

## PROCESSING SUMMARY
- Scenes Detected: 4 (PySceneDetect with 0.2 threshold)
- Visual Analysis: YOLOv8 + EasyOCR + OpenCV per scene
- Audio Analysis: Whisper Large + LibROSA + pyAudioAnalysis
- Objects Detected: 15 total (logos, people, text, graphics)
- Speakers Identified: 1 (Sarah - female, confident tone)
- Music Analysis: Corporate upbeat intro + soft outro
- Text Overlays: 3 detected ("ACME PRODUCTS", "New Launch 2025", etc.)

## SCENE-BY-SCENE BREAKDOWN
[Scene 01] 00:00:00-00:00:15: Brand introduction with logo and corporate music
[Scene 02] 00:00:15-00:01:30: Sarah introduces herself, professional presentation style
[Scene 03] 00:01:30-00:02:45: Product demonstration with screen sharing and graphics
[Scene 04] 00:02:45-00:03:12: Call to action with contact information and soft outro music

## MARKETING INSIGHTS
- Brand Presence: Strong (logo visible 78% of video duration)
- Speaker Credibility: High (clear speech, professional presentation)
- Visual Style: Modern corporate (blue/white color scheme, clean graphics)
- Audio Quality: Professional (studio microphone, balanced music levels)
- Content Type: Product launch announcement with demonstration

## SEARCHABLE KEYWORDS
Brand: ACME, Products, Launch, 2025
People: Sarah, Marketing
Objects: Logo, Graphics, Text overlays, Professional background
Audio: Corporate music, Clear speech, Background music
Mood: Professional, Energetic, Welcoming, Modern

---

[Full transcript with timestamps follows...]
```

## PROCESSING PERFORMANCE ESTIMATES

**Scene-Based Processing Benefits:**
- **GPU Memory**: Never exceeds 6GB (sequential per scene)
- **Processing Time**: ~8-12 minutes per 3-minute video
- **Analysis Depth**: 10-15x richer than basic transcription
- **Conflict Resolution**: Zero conflicts (scene isolation)
- **Marketing Value**: Scene-by-scene searchable insights

**This architecture provides the rich institutional knowledge marketing teams need while maintaining technical feasibility with your RTX 5070 setup.**

Ready to proceed with updating the wireframes to support this enhanced scene-based processing approach?