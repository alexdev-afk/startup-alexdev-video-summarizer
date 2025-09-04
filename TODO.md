# Breaking Discovery Protocol - Active Context Management
*Status: IN PROGRESS - Scene-Based Rich Analysis Pipeline Reset*
*Context Reset Needed: High priority*

## BREAKING DISCOVERY STATUS

**Discovery Trigger**: Enhanced audio/video analysis requirements discovered during architecture validation - basic Whisper transcription insufficient for institutional knowledge needs.

**Pipeline Reset**: Complete architecture overhaul from simple FFmpeg+Whisper to scene-based rich analysis pipeline with 7 integrated tools.

## COMPLETED WORK (Re-executed)

### ✅ Task 1.1: Brief Ideation - UPDATED
- **Problem**: Internal team video library chaos (100 videos)
- **Solution**: Scene-by-scene rich analysis pipeline  
- **Value**: Transform video chaos into searchable institutional knowledge
- **Model**: Internal tool for team productivity
- **Status**: reports/startup/concept-validation-1.1.md UPDATED as single source of truth

### ✅ Task 1.2: Technical Feasibility - UPDATED  
- **Architecture**: Service-based with venv isolation per tool
- **Processing**: Sequential CUDA, 10 min per video, ASCII progress charts
- **Success Probability**: 92% confidence  
- **Circuit Breaker**: Stop after 3 consecutive failures
- **Status**: reports/startup/technical-feasibility-1.2.md UPDATED as single source of truth

## PENDING RE-EXECUTION

### 🔄 Task 1.3: Study Similar Codebases - NEEDS SCENE-BASED FOCUS
- **New Requirement**: Research scene-based analysis tools, service architectures
- **Tools**: PySceneDetect, YOLOv8, EasyOCR, OpenCV, LibROSA, pyAudioAnalysis  
- **Focus**: Service integration patterns, GPU lifecycle management

### 🔄 Task 1.4: Similarities/Differences - NEEDS RICH ANALYSIS POSITIONING
- **New Positioning**: Against rich analysis platforms vs basic transcription tools
- **Value Prop**: Scene-by-scene institutional knowledge vs simple text output

### 🔄 Task 1.5: Architecture Decisions - NEEDS FINALIZATION
- **Service Architecture**: venv isolation, sequential CUDA confirmed
- **Tool Selection**: Final integration strategy for 7-tool pipeline

### 🔄 Task 2.1: ASCII Wireframes - NEEDS SCENE-BASED UI
- **Progress Display**: ASCII pipeline charts with full granularity
- **Error Reporting**: Pipeline failure visualization  
- **Configuration**: Service architecture options

## KEY ARCHITECTURE DECISIONS MADE

- **PySceneDetect** → Split videos into scenes (eliminate merge conflicts)
- **Sequential GPU Processing** → YOLOv8, EasyOCR per scene (CUDA safety)
- **Service Architecture** → venv per tool (dependency isolation)
- **10 minutes per video** → Acceptable for rich institutional knowledge
- **ASCII Progress** → Full granularity pipeline visualization
- **Circuit Breaker** → Stop after 3 consecutive video failures
- **input/build/output/** → Clean separation of processing stages

## NEXT CONTEXT PRIORITIES

1. **Task 1.3** - Study scene-based analysis codebases and service architectures
2. **Task 1.4** - Position against rich analysis tools (not basic transcription)  
3. **Task 1.5** - Finalize 7-tool integration strategy
4. **Task 2.1** - Design ASCII progress displays for scene-based processing

## CRITICAL FILES FOR CONTEXT CONTINUITY

- `scene-based-pipeline-architecture.md` - Complete 4-phase pipeline design
- `tool-overlap-matrix-analysis.md` - Tool selection decisions with ASCII matrices
- `reports/startup/concept-validation-1.1.md` - Updated concept (internal 100-video library)
- `reports/startup/technical-feasibility-1.2.md` - Updated feasibility (92% confidence, service architecture)

---

**Context Note**: Breaking discovery protocol requires systematic re-execution of all affected roadmap items. Tasks 1.1-1.2 completed with enhanced architecture. Continue with scene-based analysis focus for remaining items.