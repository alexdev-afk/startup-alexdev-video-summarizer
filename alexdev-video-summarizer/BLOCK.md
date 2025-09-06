# Video Processing Architecture - Proposed with Demucs Integration

**PROPOSED BLOCK DIAGRAM** with Demucs audio separation and clean source routing:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        VIDEO PROCESSING ORCHESTRATOR                           │
│                              (PROPOSED WITH DEMUCS)                            │
└─────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                   input/*.mp4
                                       │
                                       ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                           STEP 1: FFmpeg Foundation                        │
    │                                                                             │
    │    video.mp4 ──► FFmpegService ──┬─► build/bonita/audio.wav               │
    │                                  └─► build/bonita/video.mp4               │
    └─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                        STEP 2: Scene Detection + Frame Extraction         │
    │                                                                             │
    │    build/bonita/video.mp4 ──► SceneDetectionService ──┐                   │
    │                              (PySceneDetect)          │                   │
    │                                                        │                   │
    │    OUTPUTS:                                            │                   │
    │    ├─► scene_boundaries + scene_count                  │                   │
    │    └─► build/bonita/frames/ (3 frames per scene)      │                   │
    └─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                       STEP 2.5: Demucs Audio Separation                   │
    │                              (NEW INTEGRATION)                             │
    │                                                                             │
    │    build/bonita/audio.wav ──► DemucsService ──┬─► vocals.wav              │
    │                              (htdemucs model)  └─► no_vocals.wav          │
    │                                                                             │
    │    Clean separation eliminates need for complex heuristics                │
    └─────────────────────────────────────────────────────────────────────────────┘
                                       │
                              ┌────────┴────────┐
                              │                 │
                              ▼                 ▼
    ┌─────────────────────────────────┐   ┌─────────────────────────────────┐
    │        STEP 3A: Video Processing      │        STEP 3B: Audio Processing     │
    │                                 │   │                                 │
    │  FOR EACH SCENE:                │   │  CLEAN SOURCE ROUTING:         │
    │    FOR EACH FRAME:              │   │                                 │
    │      frame.jpg ──► InternVL3    │   │  ┌─ audio.wav ──► Whisper       │
    │                    Timeline     │   │  │   = whisper_voice             │
    │                    Service      │   │  │   (transcription + speakers)  │
    │                                 │   │  │                               │
    │  OUTPUT:                        │   │  ├─ no_vocals.wav ──► LibROSA    │
    │  └─► video_timeline.json        │   │  │   = librosa_music             │
    │                                 │   │  │   (tempo, key, instruments)   │
    │                                 │   │  │                               │
    │                                 │   │  ├─ no_vocals.wav ──► pyAudio    │
    │                                 │   │  │   = pyaudio_music             │
    │                                 │   │  │   (energy, genre)             │
    │                                 │   │  │                               │
    │                                 │   │  └─ vocals.wav ──► pyAudio       │
    │                                 │   │      = pyaudio_voice             │
    │                                 │   │      (emotion, energy)           │
    │                                 │   │                                   │
    │                                 │   │  OUTPUT:                          │
    │                                 │   │  └─► master_timeline.json         │
    │                                 │   │      (4 clean analysis sources)  │
    └─────────────────────────────────┘   └─────────────────────────────────┘
                              │                 │
                              └────────┬────────┘
                                       │
                                       ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                    STEP 4: Knowledge Base Generation                       │
    │                                                                             │
    │    video_timeline.json + master_timeline.json ──► KnowledgeGenerator      │
    │    (with 4 clean labeled sources: whisper_voice, librosa_music,           │
    │     pyaudio_music, pyaudio_voice)                ──► output.md            │
    │                                                                             │
    └─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                              CLEAN AUDIO SOURCES                               │
│                                                                                 │
│  🎤 whisper_voice:   audio.wav → Whisper → transcription + speakers          │
│  🎵 librosa_music:   no_vocals.wav → LibROSA → tempo, key, instruments       │
│  🎶 pyaudio_music:   no_vocals.wav → pyAudio → energy, genre analysis        │
│  🗣️  pyaudio_voice:   vocals.wav → pyAudio → emotion, speech energy           │
│                                                                                 │
│  ✅ BENEFITS:                                                                  │
│  - Perfect transcription (no music interference)                              │
│  - Pure music analysis (no speech artifacts)                                  │
│  - Clean emotion detection (isolated vocals)                                  │
│  - Separate music/voice energy profiles                                       │
│  - No complex heuristics needed                                               │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Implementation Strategy

**Corrected Audio Routing:**
1. ✅ **whisper_voice**: `audio.wav` → Whisper (perfect transcription)
2. ✅ **librosa_music**: `no_vocals.wav` → LibROSA (pure music analysis)  
3. ✅ **pyaudio_music**: `no_vocals.wav` → pyAudio (music energy + genre)
4. ✅ **pyaudio_voice**: `vocals.wav` → pyAudio (emotion + speech energy)

**Key Improvements:**
- **Demucs Breakthrough**: Clean vocals/music separation eliminates complex heuristics
- **Hybrid Transcription**: Keep Whisper on full audio for perfect quality
- **Clean Analysis**: Each AI tool gets the optimal audio source
- **4 Labeled Sources**: Clear source attribution in timeline output
- **No Heuristics**: Eliminates 358-line MultipassAudioTimelineService
- **Simplified Pipeline**: Linear flow with proven components

**Integration Points:**
1. Add DemucsService call after FFmpeg extraction
2. Route audio services to appropriate separated tracks
3. Coordinate audio processing in orchestrator with source labeling
4. Maintain existing video and knowledge generation workflows

This architecture provides the **best of both worlds**: perfect transcription quality + clean separated analysis without complex filtering logic.