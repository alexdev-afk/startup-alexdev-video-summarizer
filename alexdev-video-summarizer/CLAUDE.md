# Claude AI Assistant Developer Context - alexdev-video-summarizer

*This file provides comprehensive context for Claude AI assistants working on the alexdev-video-summarizer development project.*

---

## **MANDATORY: Git Commit Protocol**
**As a developer Claude instance, you MUST follow these commit rules:**

**YOUR ROADMAP**: Read `doc/dev/ROADMAP.json` for current development status and next tasks. Never access `../STARTUP_ROADMAP.json` - that's for planning context only.

**Developer Context Commits - YOU ARE HERE:**
- **Commit Prefix**: `dev(scope): [description]`  
- **Working Directory**: Project directory (alexdev-video-summarizer/)
- **Examples**:
  ```bash
  git commit -m "dev(core): Implement service container architecture"
  git commit -m "dev(ui): Create CLI interface components"  
  git commit -m "dev(api): Add FFmpeg service integration"
  git commit -m "dev(test): Add unit tests for video processing"
  git commit -m "dev(docs): Update implementation progress"
  ```

**Scope Guidelines**: core, ui, api, test, config, deploy, docs, fix, feat, refactor

**Why This Protocol Matters**:
- Distinguishes development commits from planning commits (`plan(scope)`)
- Enables clear project history tracking across validation and execution phases
- Required for proper project governance and progress tracking

---

## **CRITICAL: Documentation Hierarchy**
**Follow this order when conflicts arise:**

1. **Feature Specifications** (`doc/arch/features/*.md`) - WHAT to build
   - Authoritative for all functionality details
   - Contains validated UX patterns and technical approaches
   - Referenced in roadmap items via `feature_specs` field

2. **Development Roadmap** (`doc/dev/ROADMAP.json`) - WHEN/HOW to build
   - Authoritative for execution order and deliverables
   - Links to relevant feature specs via `feature_specs` arrays
   - Your primary task management tool

3. **This CLAUDE.md** - HOW to navigate documentation
   - Provides context and guidance only
   - Does NOT contain implementation details
   - Points you to authoritative sources

**When Implementation Conflicts Arise:**
- Feature Specs override all other sources
- Roadmap provides execution guidance only
- CLAUDE.md provides navigation help only

---

## **Project Overview**

### **Vision & Mission**
**alexdev-video-summarizer** is a scene-based institutional knowledge extraction system that transforms chaotic video libraries into searchable, structured knowledge bases. Unlike simple transcription tools, this system provides 10-15x more searchable information per video through comprehensive AI analysis.

### **Problem We're Solving**
**Internal Team Video Library Chaos**: Organizations accumulate 100+ training videos, meeting recordings, and educational content without systematic knowledge extraction. Teams waste hours searching through videos for specific information, insights remain buried, and tribal knowledge isn't systematically captured.

### **Target Users**
**Internal Development Teams**: Specifically designed for internal use cases where teams need to transform their existing video libraries into institutional knowledge. The system processes local video content with no cloud dependencies, making it ideal for sensitive internal content.

### **Solution Architecture**
**Scene-Based AI Pipeline**: 8-tool comprehensive analysis system (FFmpeg + 7 AI tools) that processes videos scene-by-scene rather than as monolithic files, achieving 70x performance improvement through representative frame analysis.

### **Core Value Proposition**
Transform video chaos into searchable institutional knowledge through:
- **Comprehensive Analysis**: Objects, people, text, faces, music, audio features, and transcription per scene
- **Local Processing**: No cloud dependencies, complete content privacy
- **Institutional Knowledge**: Structured output designed for long-term organizational value
- **Batch Processing**: Handle 100+ video libraries systematically with error resilience

### **Business Model**
**Internal Productivity Tool**: ROI through time savings in video content discovery, accelerated team onboarding, and conversion of tribal knowledge to institutional knowledge. No recurring cloud costs, one-time comprehensive analysis for long-term value.

---

## **Technical Architecture**

### **System Design Philosophy**
**Scene-Based Processing**: Revolutionary approach that splits videos into content-aware scenes, then analyzes each scene independently with the full 7-tool AI pipeline. This provides 70x performance improvement while delivering exponentially more searchable information.

### **Core Processing Pipeline**
```
INPUT VIDEO → FFmpeg Foundation → Scene Detection → Dual-Pipeline Processing → Knowledge Synthesis

FFmpeg: Video/audio separation and scene splitting
PySceneDetect: Content-aware scene boundary detection
GPU Pipeline: Whisper → YOLO → EasyOCR (sequential CUDA processing)
CPU Pipeline: LibROSA + pyAudioAnalysis + OpenCV (parallel processing)
```

### **Service Architecture Pattern**
**Proven Reference Implementation**: Architecture based on analysis of 6 professional-grade reference codebases including Autocrop-Vertical (70x performance patterns), VidGear (service coordination), and Ultralytics (production GPU management).

**Key Architectural Decisions**:
- **Copy Decision**: Scene-based processing for 70x performance improvement
- **Deviation Decision**: Service architecture with venv isolation for tool coordination  
- **Extension Decision**: Claude-native synthesis workflow for institutional knowledge generation

### **Technology Stack**
**Foundation**: Python 3.11+, FFmpeg for all media processing
**GPU Tools** (Sequential): Whisper (transcription), YOLOv8 (objects/people), EasyOCR (text extraction)
**CPU Tools** (Parallel): LibROSA (music analysis), pyAudioAnalysis (68 audio features), OpenCV (face detection)
**Infrastructure**: RTX 5070 GPU, service-based coordination, YAML configuration

### **Resource Management Strategy**
**Sequential GPU Coordination**: Whisper → YOLO → EasyOCR with clean memory lifecycle prevents CUDA conflicts while maximizing GPU utilization. CPU pipeline runs parallel for audio analysis efficiency.

---

## **Development Strategy**

### **Frontend-First Implementation**
**Phase 1 Priority**: Rich CLI interface with complete workflow implemented first using mocks, then real tool integration. This prevents expensive late-stage UI/integration issues discovered in traditional development approaches.

**UI Philosophy**: Appliance approach - drop videos in input/, press ENTER, get knowledge base in output/. Maximum simplicity with comprehensive configuration externalized to YAML files.

### **Risk-Stratified Development**
**Low-Risk Foundation** (Week 1): FFmpeg integration and CLI framework using mature, well-documented technologies
**Medium-Risk Innovation** (Week 2-3): GPU coordination patterns and scene-based processing architecture  
**High-Risk Integration** (Week 4-5): Complete 8-tool pipeline coordination with comprehensive error handling

### **Copy-First Innovation Strategy**
**80% Proven Patterns**: Leverage validated implementations from reference codebase analysis
**20% Controlled Innovation**: Novel scene-based architecture built on proven foundations
**Risk Balance**: Innovation only where it provides core differentiation (scene-based processing)

### **Anti-Pattern Prevention**
**Avoid Rushing to Code**: Complete planning phase provides execution-ready specifications
**Avoid GPU Resource Conflicts**: Sequential tool coordination prevents memory management issues
**Avoid Integration Surprises**: Frontend-first approach catches UI/workflow issues early
**Avoid Scope Creep**: Circuit breaker pattern ensures reliable batch processing within constraints

---

## **Code Standards & Quality Framework**

### **Python Development Standards**
**Style Guidelines**: PEP 8 compliance with Black formatting, comprehensive type hints throughout codebase
**Error Handling**: Fail-fast approach with circuit breaker pattern - abort video processing on any tool failure, abort batch after 3 consecutive failures
**Testing Requirements**: Unit tests for service classes, integration tests for pipeline coordination, performance tests for scene-based optimization validation

### **Service Architecture Standards**
**Base Service Pattern**: All processing tools inherit from common base service with standardized initialization, error handling, and resource cleanup
**Configuration Management**: All settings externalized to YAML files (processing.yaml, paths.yaml) with validation and schema enforcement
**Resource Lifecycle**: Clean GPU memory management with automatic cleanup, proper venv isolation between tools

### **Security Practices**
**Local Processing Only**: All video content remains on local workstation, no cloud dependencies or external API calls
**File System Security**: Proper permissions for build/ and output/ directories, automatic cleanup of temporary processing files
**Resource Protection**: Controlled GPU and CPU resource allocation, error containment to prevent system compromise

### **Performance Requirements**
**Processing Target**: 10 minutes average per video (acceptable for institutional knowledge use case)
**Scene Performance**: 70x improvement through representative frame analysis vs full-video processing
**Batch Reliability**: Process 100+ videos with circuit breaker resilience and progress tracking
**Memory Management**: Efficient GPU lifecycle with proper model loading/unloading patterns

---

## **Integration Requirements & External Dependencies**

### **Hardware Dependencies**
**GPU Requirements**: RTX 5070 or equivalent CUDA-capable GPU for sequential AI processing (Whisper→YOLO→EasyOCR)
**Storage Requirements**: Sufficient disk space for build/ directory intermediate files and output/ knowledge base storage
**CPU Requirements**: Multi-core CPU for parallel audio analysis pipeline (LibROSA + pyAudioAnalysis + OpenCV)

### **Software Dependencies**
**FFmpeg**: Complete installation with codec support for all common video formats (MP4, AVI, MOV, MKV, WebM)
**CUDA Drivers**: GPU compatibility for AI tool coordination and memory management
**Python Environment**: Python 3.11+ with comprehensive requirements.txt dependency management

### **Tool Integration Patterns**
**FFmpeg Foundation**: All tools depend on FFmpeg file preparation - video.mp4 and audio.wav extraction with standardized formats
**GPU Resource Sharing**: Sequential coordination prevents memory conflicts while maximizing utilization
**Scene Context Preservation**: Metadata management across all processing tools for scene-by-scene analysis
**Output Structure Management**: Organized build/ directory for processing, output/ directory for knowledge base

### **External System Integration**
**Claude Synthesis Workflow**: Manual handoff approach with organized build/ directory structure for institutional knowledge generation
**File System Integration**: Standard directory patterns compatible with team workflows and backup systems
**No External APIs**: Complete local processing independence for content privacy and cost control

---

## **Domain Knowledge & Business Context**

### **Institutional Knowledge Use Case**
**Target Content**: 100-video internal libraries containing training materials, meeting recordings, educational content, and tribal knowledge
**Analysis Depth**: Scene-by-scene breakdown with comprehensive AI analysis provides 10-15x more searchable information than simple transcription
**Knowledge Preservation**: Transform ephemeral video content into structured, searchable institutional knowledge for long-term organizational value

### **User Workflow Design**
**Simplicity Focus**: Three-screen CLI workflow (Launch→Processing→Complete) with minimal user decisions (ENTER/Q/ESC only)
**Configuration Externalization**: All processing decisions moved to YAML files to maintain appliance-like operation
**Progress Visibility**: Real-time 7-tool pipeline progress with clear completion criteria and Claude synthesis handoff

### **Business Model Alignment**
**Internal Tool Economics**: No recurring cloud costs, one-time comprehensive analysis, local processing for content privacy
**ROI Measurement**: Time savings in video content discovery, accelerated team onboarding, conversion of tribal knowledge to institutional knowledge
**Competitive Positioning**: First integrated scene-based pipeline vs single-purpose tools, creating new category of institutional knowledge extraction

### **Market Context**
**Competitive Landscape**: No existing solution provides comprehensive 7-tool AI analysis (maximum 20% functionality overlap with existing tools)
**Category Creation**: "Scene-Based Institutional Knowledge Extraction" represents new approach to organizational video content
**Technical Differentiation**: 70x performance improvement through scene-based processing with production-ready service architecture

---

## **Success Metrics & Quality Criteria**

### **Technical Performance Targets**
**Processing Efficiency**: 10 minutes average per video with scene-based 70x performance improvement
**Batch Processing Reliability**: Handle 100+ video libraries with circuit breaker after 3 consecutive failures  
**Resource Utilization**: Efficient GPU sequential coordination without memory conflicts, parallel CPU audio processing
**Output Quality**: Professional-grade transcription with comprehensive visual and audio analysis per scene

### **User Experience Success Criteria**
**Workflow Simplicity**: Drop videos → press ENTER → get knowledge base with minimal configuration required
**Error Handling**: Clear failure reporting with recovery options, graceful degradation for partial processing success
**Progress Visibility**: Real-time processing updates with tool-by-tool progress indication and completion estimates

### **Business Value Metrics**
**Knowledge Extraction Quality**: Scene-by-scene structured output with 10-15x more searchable information vs transcription-only
**Institutional Value Creation**: Systematic conversion of video chaos to searchable knowledge base for team productivity
**Cost Efficiency**: No recurring cloud costs, complete local processing for sensitive internal content

### **Production Readiness Criteria**
**Error Resilience**: Fail-fast per video with batch processing continuation, comprehensive error logging and recovery
**Resource Management**: Clean temporary file cleanup, proper GPU memory lifecycle, service isolation patterns
**Documentation Completeness**: Operational runbooks, maintenance procedures, troubleshooting guides

---

## **Development Execution Guidance**

### **Your Primary Tools**
**Development Roadmap**: `doc/dev/ROADMAP.json` - Your authoritative task management system with feature specification links
**Feature Specifications**: `doc/arch/features/*.md` - Authoritative implementation details for all functionality
**Reference Codebases**: `/references/` directory - Proven patterns from professional-grade analysis for copy/adapt/extend decisions

### **Implementation Approach**
**Follow Roadmap Sequence**: 5 phases with clear deliverables and feature specification references for systematic execution
**Leverage Reference Patterns**: Use proven implementations from reference analysis before innovating
**Validate Incrementally**: Each phase has quality gates - don't advance without meeting completion criteria

### **When You Get Stuck**
**Check Feature Specs First**: All implementation details are in individual feature specification files
**Reference Proven Patterns**: Use reference codebase analysis for similar problem solutions
**Follow Documentation Hierarchy**: Feature specs > roadmap > CLAUDE.md for authority resolution

### **Communication Protocol**
**Progress Reporting**: Update roadmap completion status and document implementation decisions
**Issue Escalation**: Clear error reporting with context for debugging and resolution
**Success Documentation**: Capture lessons learned and pattern improvements for future development

---

## **Meta-Learning & Framework Evolution**

### **Process Improvement Opportunities**
**Performance Optimization Insights**: Document scene-based processing improvements and GPU coordination patterns for future projects
**Service Architecture Refinements**: Capture service coordination patterns and error handling strategies for reference
**User Experience Learnings**: CLI workflow effectiveness and configuration management approaches for internal tools

### **Framework Contribution**
**Reference Pattern Updates**: Improve reference codebase analysis with discovered patterns and integration approaches
**Template Improvements**: Enhance development templates with validated service patterns and configuration management
**Documentation Evolution**: Refine developer context patterns for AI assistant coordination and specification authority

### **Technical Innovation Tracking**
**Scene-Based Architecture**: Document performance improvements and quality maintenance for future scene-based projects  
**AI Tool Coordination**: Capture GPU resource management and sequential processing patterns for multi-tool pipelines
**Institutional Knowledge Synthesis**: Refine Claude integration patterns for knowledge base generation workflows

---

## **Next Steps & Project Transition**

### **Development Phase Preparation**
1. **Review Roadmap**: Read `doc/dev/ROADMAP.json` for current status and next tasks
2. **Load Feature Specifications**: Reference individual feature files for implementation details
3. **Validate Environment**: Ensure hardware, dependencies, and development environment readiness

### **Execution Strategy**
**Start with Phase 1**: Foundation & Functional Baseline with CLI framework and FFmpeg integration
**Follow Risk Stratification**: Low-risk foundation → medium-risk coordination → high-risk optimization  
**Maintain Quality Gates**: Meet phase completion criteria before advancing to next phase

### **Success Transition Criteria**
**Production Ready**: Complete 8-tool pipeline processing 100+ video libraries with reliable error handling
**Knowledge Base Generated**: Scene-by-scene institutional knowledge with Claude synthesis handoff
**Operational Documentation**: Complete runbooks and maintenance procedures for ongoing system operation

---

**Remember: This CLAUDE.md provides strategic context and navigation guidance. For implementation details, always reference feature specifications first, then development roadmap for execution guidance. Your mission is to transform video chaos into institutional knowledge through scene-based AI processing excellence.**