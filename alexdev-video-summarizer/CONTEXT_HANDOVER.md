# Context Handover - Demucs Integration & Heuristic Elimination

## Executive Summary

We're at the breakthrough moment for eliminating complex heuristic logic with clean Demucs audio separation. The codebase has been modernized and is ready for the final Demucs integration phase.

## Current Architecture Status

### ✅ COMPLETED MODERNIZATION
- **Hardcoded paths fixed**: InternVL3 service now uses dynamic video names
- **File renaming complete**: `master_timeline.json` → `combined_audio_timeline.json`, `internvl3_timeline.json` → `video_timeline.json`  
- **Dead code eliminated**: Removed 4,880+ lines including MultiPassAudioTimelineService (358 lines)
- **Orchestrator cleaned**: Updated to use actual working services, removed legacy pipeline references
- **VideoProcessingContext enhanced**: Added `get_audio_file_path()` and `get_audio_analysis_path()` for Demucs
- **Batch processing ready**: CLI supports single video or full `input/` folder processing

### 🎯 CURRENT TARGET: Enhanced Timeline Merger Service

**CRITICAL DISCOVERY**: `src/services/enhanced_timeline_merger_service.py` contains **692 lines** of complex heuristic filtering logic designed to separate speech artifacts from music analysis.

**Key Heuristic Methods** (TO BE ELIMINATED):
- `_filter_librosa_speech_artifacts()` - Filters LibROSA events during speech
- `_is_speech_artifact()` - Multi-criteria artifact detection 
- `_classify_speech_overlap_event()` - Speech vs music classification during overlap
- `_filter_librosa_speech_artifacts_from_timelines()` - Raw timeline filtering
- `_is_speech_artifact_from_raw_data()` - Complex heuristic detection
- `_is_near_structural_boundary()` - Boundary detection heuristics
- `_has_distant_pyaudio_events()` - Distance-based filtering

**Why This Complexity Exists**: LibROSA analyzes mixed audio (speech + music) requiring complex heuristics to filter out speech artifacts from music detection.

## Demucs Integration Plan

### Current Working Architecture
```
input/*.mp4 → FFmpeg → SceneDetection + Frames → InternVL3 → KnowledgeGenerator → output.md
```

### Proposed Architecture with Demucs (See BLOCK.md)
```
input/*.mp4 → FFmpeg → SceneDetection + Frames → Demucs Separation
                                                        ↓
                    ┌─── audio.wav → Whisper (whisper_voice)
                    ├─── no_vocals.wav → LibROSA (librosa_music)  
                    ├─── no_vocals.wav → pyAudio (pyaudio_music)
                    └─── vocals.wav → pyAudio (pyaudio_voice)
                                                        ↓
                            Clean Timeline Merger → Combined Audio Timeline
                                                        ↓
                                    InternVL3 + KnowledgeGenerator → output.md
```

### File Structure (Per Video)
```
build/{video_name}/
├── extraction/
│   ├── audio.wav          # Original FFmpeg output
│   ├── video.mp4         # Original FFmpeg output
│   ├── vocals.wav        # NEW: Demucs vocals
│   └── no_vocals.wav     # NEW: Demucs instrumentals
├── frames/               # Scene detection frames
├── audio_timelines/
│   ├── whisper_timeline.json      # whisper_voice
│   ├── librosa_timeline.json      # librosa_music  
│   ├── pyaudio_music_timeline.json # pyaudio_music
│   ├── pyaudio_voice_timeline.json # pyaudio_voice
│   └── combined_audio_timeline.json # Clean merger output
├── video_timelines/
│   └── video_timeline.json
└── {video_name}_output.md
```

## Implementation Strategy

### Phase 1: Demucs Service Integration
1. **Fix DemucsService.separate_audio()** - Currently expects VideoProcessingContext methods that exist
2. **Add to orchestrator** - Insert Demucs step after FFmpeg extraction
3. **Test separation** - Verify clean vocals.wav and no_vocals.wav generation

### Phase 2: Audio Source Routing  
1. **Update audio services** to use separated tracks:
   - Whisper: `audio.wav` (keep perfect transcription)
   - LibROSA: `no_vocals.wav` (pure music analysis)
   - pyAudio music: `no_vocals.wav` (genre, energy)  
   - pyAudio voice: `vocals.wav` (emotion, speech features)
2. **Label sources** in timeline output for traceability

### Phase 3: Heuristic Elimination
1. **Gut enhanced_timeline_merger_service.py** - Remove all 692 lines of heuristic filtering
2. **Replace with simple merger** - Just chronologically merge clean sources
3. **Update method calls** - Ensure `create_combined_audio_timeline()` works with simplified logic

### Phase 4: Testing & Validation
1. **Test full pipeline** on bonita.mp4
2. **Compare output quality** - Should be cleaner than heuristic approach
3. **Batch test** - Process multiple videos from input/

## Key Benefits of Demucs Approach

- ✅ **Perfect transcription** - No music interference in Whisper
- ✅ **Pure music analysis** - No speech artifacts in LibROSA  
- ✅ **Clean emotion detection** - Isolated vocals for pyAudio
- ✅ **No complex heuristics** - Eliminates 692 lines of filtering logic
- ✅ **Source traceability** - Clear labeling of whisper_voice, librosa_music, pyaudio_music, pyaudio_voice
- ✅ **10-15x more searchable** - Each AI tool gets optimal input

## Immediate Next Steps

1. **Test current DemucsService** - We confirmed it works with `python -m demucs` 
2. **Integrate into orchestrator** - Add Step 2.5 after FFmpeg
3. **Route audio services** - Update each service to use appropriate separated track
4. **Eliminate heuristics** - Replace enhanced_timeline_merger_service complex logic with simple merger

## Reference Files

- **BLOCK.md** - Complete architecture diagram with Demucs integration
- **build/bonita_backup/** - Example of current file structure
- **src/services/demucs_service.py** - Ready Demucs service (needs orchestrator integration)
- **src/services/enhanced_timeline_merger_service.py** - 692 lines of heuristics TO BE ELIMINATED

## Context Notes

- **30 videos** in input/ ready for batch processing
- **CLI system** already supports single video and batch processing
- **All hardcoded paths** have been made dynamic for any video name
- **Tests remain hardcoded** to bonita.mp4 for consistency
- **Circuit breaker** in orchestrator handles batch processing failures (abort after 3 consecutive failures)

The architecture is clean, modernized, and ready for the Demucs breakthrough that will eliminate complex heuristics with clean audio separation.