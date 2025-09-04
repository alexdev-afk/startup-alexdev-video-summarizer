# Architecture Decisions Framework - Task 1.5 Completion Report

## Overview
Interactive architecture decision-making session completed. Technical blueprint established using copy-first, adapt-next, extend-last approach to minimize development cost and risk.

## Copy Decisions (Direct Adoption)
**From FFmpeg Napoleon Scripts:**
- Scene detection + frame extraction: `ffmpeg -i input.mp4 -vf "select='gt(scene,0.2)',showinfo" -vsync vfr frames/scene_%04d.png`
- Shell script error handling and validation patterns
- File organization structure (input/output directories)

**From Whisper-Transcriber:**
- Python batch processing workflow architecture
- Device management for RTX 5070 GPU utilization
- File processing queue and status tracking
- Output formatting and metadata structure

**From Batch-Whisper:**
- Error recovery and retry mechanisms
- Progress reporting and logging patterns
- Batch optimization strategies
- Memory management approaches

## Deviation Decisions
- **No Configuration Files**: Static configurations embedded in scripts (vs config-heavy approaches)
- **Guided Workflow**: Step-by-step runbook-driven process (vs assume-expertise approach)
- **Zero Setup**: Everything pre-configured and ready to run (vs requiring user configuration)
- **No Frame Review**: Automated processing without user intervention (transcription provides rich data)

## Extension Decisions
**Enhanced Transcript Headers:**
- Basic Metadata: Filename, video length, file size, processing date
- Processing Quality Indicators: Number of frames extracted, transcript confidence scores, processing duration
- Asset Location Mapping: References to extracted frames directory

**Master Index System:**
- Simple .md file with ASCII art/tables for monospace readability
- Lightweight navigation to detailed transcript info
- No duplicate metadata - single source of truth per video

**Duplicate Detection:**
- Content-based comparison: transcript similarity, video length, file size
- Filename pattern recognition: similar names, version numbers, dates
- Index flagging: Clear indicators for potential duplicates
- Human decision support: Let marketing teams decide on flagged duplicates

## Technology Stack Decisions
- **Python**: 3.11+ (proven Whisper/FFmpeg compatibility)
- **FFmpeg**: Latest stable (scene detection + frame extraction)
- **Whisper**: Large model (quality-first approach, RTX 5070 capable)
- **Platform**: Windows RTX 5070 GPU acceleration
- **Output Formats**:
  - Transcripts: .txt with enhanced headers
  - Frames: .png files in subdirectories
  - Master Index: .md file with ASCII formatting

## Integration Strategy
- **Completely Local**: No cloud services, no external APIs beyond Whisper
- **Manual File Management**: Drop videos in input, find results in output
- **Self-Contained**: All processing on local RTX 5070 machine

## Scalability Plan
**Serial Processing Approach:**
- One video at a time: Predictable GPU memory usage
- Batch queue: Multiple videos processed sequentially
- Progress reporting: Clear status of current/remaining videos
- Duplicate check: After each video, before index update
- Scales to 100+ video libraries via overnight/low-usage processing

## Workflow Architecture
1. **Single FFmpeg Pass**: Combined scene detection + frame extraction
2. **Whisper Transcription**: Large model processing with RTX 5070 acceleration
3. **Duplicate Detection**: Compare against existing library during processing
4. **Index Update**: Add to master .md index with duplicate flags

## Decision Documentation
Each major decision documented with rationale, trade-offs, risks, and alternatives considered. Architecture supports business model (eliminate tribal knowledge dependency) and scales to projected user base (marketing teams with 100+ video libraries).

## Validation Criteria Met
✅ Complete technical architecture blueprint established
✅ Copy-first approach minimizes development risk
✅ Extensions provide genuine value for institutional knowledge
✅ Technology choices support business model and user needs
✅ Scalability plan handles projected usage patterns

## Next Phase
Phase 1: Concept & Validation is now complete. Ready to proceed to Phase 2: Foundation & Planning, starting with Task 2.1: ASCII Wireframes & User Flow Design.