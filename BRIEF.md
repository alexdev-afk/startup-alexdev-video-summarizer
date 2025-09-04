# Project Brief: alexdev-video-summarizer

## Problem Statement
Campaign content organizers need preprocessed video data (transcriptions + visual frames + duplicate detection) to enable AI-powered metadata tagging and strategic content analysis, but manual video processing is time-intensive and inconsistent.

## Solution Overview
AI-powered video processing pipeline that automatically converts raw video files into structured data suitable for downstream campaign content organization and strategic analysis.

## Core Requirements

### 1. Environment Setup
- Install ffmpeg for video processing capabilities
- Install Python + openai-whisper for audio transcription
- Setup robust directory structure for batch processing
- Handle common video format inputs (MP4, MOV, AVI, etc.)

### 2. Video Processing Pipeline
**Input:** `/input/` folder with mixed video files
**Output:** `/output/[video-name]/` structure containing:
```
/output/video-name/
├── transcript.txt          # Whisper-generated transcription
├── frames/                 # Key frames extracted via ffmpeg
│   ├── frame_001.png      # Systematic frame sampling
│   ├── frame_005.png
│   └── frame_010.png
├── metadata.json           # Video technical details
└── processing_log.txt      # Processing status/errors
```

### 3. Duplicate Detection System
- Compare transcripts using content similarity algorithms
- Flag potential duplicates (renamed/recompressed same content)
- Generate comprehensive duplicate analysis report
- Handle edge cases: different resolutions, compression, minor edits

### 4. Integration-Ready Output
- Clean, structured output format for campaign-content-organizer consumption
- JSON metadata schema for strategic tagging pipeline integration
- Batch processing capabilities for large content libraries
- Error handling and recovery mechanisms

## Technical Decisions to Address

### Frame Extraction Strategy
- **Option A:** Fixed interval sampling (every N seconds)
- **Option B:** Scene change detection for key moments
- **Option C:** Hybrid approach with minimum/maximum frame limits
- **Question:** What sampling rate optimizes content analysis vs processing time?

### Duplicate Detection Threshold
- **Text similarity threshold:** What % match indicates potential duplicate?
- **Handling variations:** How to detect same content with different intro/outro?
- **Performance consideration:** Balance accuracy vs false positives
- **Question:** Should visual similarity complement text similarity?

### Batch Processing Approach
- **Sequential processing:** One video at a time with detailed logging
- **Parallel processing:** Multiple videos simultaneously (resource management)
- **Incremental processing:** Skip already-processed files
- **Question:** How to handle processing failures and resume operations?

### Integration Format
- **Metadata schema:** What JSON structure works best for campaign organizer?
- **File organization:** Flat structure vs hierarchical output?
- **Status tracking:** How to communicate processing progress/completion?
- **Question:** What data format enables seamless handoff to strategic analysis?

## Success Criteria
- ✅ Automated video processing without manual intervention
- ✅ High-quality transcriptions suitable for content categorization
- ✅ Reliable duplicate detection with minimal false positives
- ✅ Scalable processing for hundreds of campaign videos
- ✅ Integration-ready output for strategic content analysis pipeline

## Constraints
- Must work with local processing (no cloud dependencies for core features)
- Handle common video formats without conversion requirements
- Process videos ranging from 30 seconds to 2+ hours
- Maintain privacy (no external API requirements for sensitive content)

## Dependencies
- ffmpeg (video processing)
- Python + openai-whisper (transcription)
- Standard libraries for file operations and similarity detection

## Target Integration
This tool serves as preprocessing step for **alexdev-campaign-content-organizer** strategic content analysis and gap identification system.