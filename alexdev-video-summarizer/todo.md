# Whisper Timeline Service - Multi-Granularity Implementation

## **Objective**
Create WhisperTimelineService that produces 3 granularities of timeline data:
1. **Word-level events** - Individual words with timestamps, emphasis, and emotional metadata
2. **Phrase-level spans** - Meaningful speech phrases/sentences with context
3. **Full script output** - Complete transcript in single unified format

## **Context & Architecture**
Replace current WhisperService output format with ServiceTimeline format to integrate with:
- Multi-pass audio timeline approach (using event boundaries for intelligent spans)
- Timeline merger service for chronological audio event combination
- Event-based institutional knowledge generation

## **Current Whisper Capabilities Analysis**
Whisper provides excellent foundation:
- ✅ Word-level timestamps with confidence scores
- ✅ Speaker diarization (Silero VAD + PyAnnote)  
- ✅ Language detection with confidence
- ✅ Segment-level transcription with precise timing
- ❌ No emphasis detection (stress, volume changes)
- ❌ No emotional metadata (tone, sentiment)
- ❌ No timeline format output (uses dict format)

## **Phase 1: Word-Level Events**

### **1.1 Basic Word Events**
- [ ] Extract word-level timestamps from Whisper segments
- [ ] Convert to TimelineEvent objects with precise timing
- [ ] Include confidence scores and speaker attribution
- [ ] Test word-level granularity with bonita.mp4

### **1.2 Emphasis Detection Enhancement**
- [ ] **Research**: Can we detect word emphasis from Whisper confidence scores?
- [ ] **Research**: Audio amplitude analysis around word timestamps for volume emphasis
- [ ] **Research**: Pitch analysis integration with LibROSA for tonal emphasis
- [ ] **Implementation**: Add emphasis metadata to word events
- [ ] **Testing**: Validate emphasis detection accuracy

### **1.3 Emotional Metadata Integration**
- [ ] **Research**: Prosodic feature analysis for emotional context per word/phrase
- [ ] **Integration**: Leverage pyAudioAnalysis emotion detection aligned with word timing
- [ ] **Enhancement**: Cross-reference emotion events with word-level timeline
- [ ] **Implementation**: Add emotional context to word events
- [ ] **Testing**: Emotional metadata accuracy validation

## **Phase 2: Phrase-Level Spans**

### **2.1 Intelligent Phrase Segmentation**
- [ ] Analyze Whisper segments for natural phrase boundaries (punctuation, pauses)
- [ ] Create TimelineSpan objects for meaningful speech phrases
- [ ] Include speaker context and language metadata per phrase
- [ ] Test phrase segmentation quality with real content

### **2.2 Cross-Modal Context Enhancement**
- [ ] **Integration**: Use pyAudio speaker change events as phrase boundary hints
- [ ] **Enhancement**: Incorporate LibROSA musical accent data for speech rhythm analysis
- [ ] **Context**: Add environmental context (background music, sound effects) to phrase spans
- [ ] **Testing**: Validate phrase boundaries against content meaning

### **2.3 Semantic Phrase Characterization**
- [ ] **Research**: Can we detect question vs statement vs exclamation from prosodics?
- [ ] **Implementation**: Add semantic metadata to phrase spans
- [ ] **Enhancement**: Include speech pace analysis (fast/slow delivery per phrase)
- [ ] **Testing**: Semantic accuracy validation

## **Phase 3: Full Script Output**

### **3.1 Unified Script Generation**
- [ ] Combine all word events into complete chronological transcript
- [ ] Preserve speaker attribution and timing precision
- [ ] Include confidence and quality metrics for full script
- [ ] Generate speaker summary and language analysis

### **3.2 Script Context Enhancement**
- [ ] **Cross-Library Integration**: Include musical/environmental context in script
- [ ] **Timeline Markers**: Add significant audio events as script annotations
- [ ] **Structure**: Identify speech patterns (introductions, transitions, conclusions)
- [ ] **Quality**: Overall speech quality and clarity assessment

### **3.3 Export Format Optimization**
- [ ] **ServiceTimeline Format**: Ensure compatibility with timeline merger
- [ ] **LLM Integration**: Optimize script format for institutional knowledge synthesis
- [ ] **Metadata Preservation**: Maintain all analysis details while keeping script readable
- [ ] **Testing**: Full pipeline integration validation

## **Phase 4: Integration & Testing**

### **4.1 WhisperTimelineService Implementation**
- [ ] Create new WhisperTimelineService class following timeline service pattern
- [ ] Implement generate_timeline() method returning ServiceTimeline object
- [ ] Integrate with existing WhisperService (wrapper approach)
- [ ] Update AudioPipelineController to use timeline service

### **4.2 Multi-Pass Integration**
- [ ] **Timeline Merger**: Update to handle Whisper word-level events
- [ ] **Boundary Detection**: Use Whisper phrase boundaries as additional span cues
- [ ] **Cross-Library**: Ensure Whisper events enhance LibROSA/pyAudio span characterization
- [ ] **Testing**: Full multi-pass pipeline with all three services

### **4.3 Production Validation**
- [ ] **Performance**: Ensure word-level processing doesn't impact performance significantly
- [ ] **Accuracy**: Validate emphasis and emotional metadata accuracy
- [ ] **Integration**: Test with multiple advertisement videos
- [ ] **Quality**: Compare output quality vs current Whisper service

## **Research Questions**

### **Emphasis Detection Approaches**
1. **Confidence Score Analysis**: Do lower confidence scores indicate emphasis or unclear speech?
2. **Audio Amplitude**: Can we analyze RMS energy around word timestamps for volume emphasis?
3. **Pitch Integration**: Using LibROSA pitch analysis aligned with word timing for tonal emphasis?
4. **Cross-Validation**: Combining multiple signals for emphasis confidence?

### **Emotional Metadata Sources**
1. **Prosodic Features**: Pitch variation, speaking rate, pause patterns per word/phrase
2. **pyAudio Integration**: Aligning emotion detection events with word-level timeline
3. **Contextual Analysis**: Emotional arc across phrases and full speech
4. **Accuracy Validation**: How to validate emotional metadata without ground truth?

## **Implementation Priority**
1. **Phase 1.1**: Basic word events (foundation)
2. **Phase 2.1**: Phrase spans (core functionality)
3. **Phase 3.1**: Full script (integration)
4. **Phase 4.1**: Service implementation (production)
5. **Phase 1.2-1.3**: Emphasis/emotion enhancement (advanced features)

## **Success Criteria**
- ✅ Word-level timeline events with precise timing and metadata
- ✅ Phrase-level spans with intelligent boundaries and context
- ✅ Full script output maintaining all detail while being LLM-ready
- ✅ ServiceTimeline format compatibility with multi-pass architecture
- ✅ Performance maintaining current Whisper processing speed
- ✅ Enhanced institutional knowledge generation through granular timeline data

**Goal**: Transform Whisper from good transcription service to comprehensive speech timeline service with multi-granular analysis supporting ultra-rich advertisement audio description.