# Knowledge Base Generation - Scene-by-Scene Output

**Priority**: Phase 5 - Final Output  
**Risk**: LOW - Data aggregation and formatting  
**Dependencies**: All AI tool analysis complete

---

## **Purpose**
Aggregate all scene-based analysis into comprehensive, searchable knowledge base files with scene-by-scene breakdowns for institutional knowledge discovery.

## **Core Functions**

### **1. Scene-by-Scene Knowledge Compilation**
```python
def compile_scene_knowledge(video_analysis_data):
    """Compile all analysis into scene-by-scene knowledge structure"""
    scenes_knowledge = []
    
    for scene in video_analysis_data['scenes']:
        scene_knowledge = {
            'scene_id': scene['scene_id'],
            'timestamp': f"{format_time(scene['start_seconds'])} - {format_time(scene['end_seconds'])}",
            'duration': scene['duration'],
            
            # Visual Analysis Results
            'objects_detected': scene['analysis']['yolo']['objects'],
            'people_count': scene['analysis']['yolo']['people_count'],
            'text_overlays': scene['analysis']['easyocr']['text_content'],
            'faces_detected': scene['analysis']['opencv']['faces'],
            
            # Audio Analysis Results  
            'transcript_segment': scene['analysis']['whisper']['transcript'],
            'speakers': scene['analysis']['whisper']['speakers'],
            'audio_features': scene['analysis']['librosa']['features'],
            'audio_classification': scene['analysis']['pyaudioanalysis']['classification'],
            
            # Scene Context
            'scene_description': generate_scene_description(scene),
            'key_topics': extract_key_topics(scene),
            'searchable_terms': generate_searchable_terms(scene)
        }
        scenes_knowledge.append(scene_knowledge)
    
    return scenes_knowledge
```

### **2. Searchable Knowledge Base Format**
```python
def generate_knowledge_base_md(video_name, scenes_knowledge, metadata):
    """Generate comprehensive .md knowledge base file"""
    
    content = f"""# Video Knowledge Base: {video_name}

## Video Summary
- **Duration**: {format_duration(metadata['total_duration'])}
- **Scenes**: {len(scenes_knowledge)} detected
- **Processing Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
- **Speakers Identified**: {', '.join(metadata['all_speakers'])}

## Quick Navigation
{generate_scene_navigation(scenes_knowledge)}

---

## Scene-by-Scene Analysis

{generate_detailed_scenes(scenes_knowledge)}

---

## Cross-References
{generate_cross_references(scenes_knowledge)}

## Search Index
{generate_search_terms(scenes_knowledge)}
"""
    
    return content
```

### **3. Scene Navigation Generation**
```python
def generate_scene_navigation(scenes_knowledge):
    """Create quick navigation links for each scene"""
    nav_items = []
    
    for scene in scenes_knowledge:
        # Extract key elements for navigation
        key_objects = scene['objects_detected'][:3]  # Top 3 objects
        key_text = scene['text_overlays'][:2] if scene['text_overlays'] else []
        speakers = scene['speakers']
        
        nav_description = f"Scene {scene['scene_id']} ({scene['timestamp']})"
        if key_objects:
            nav_description += f" - Objects: {', '.join(key_objects)}"
        if key_text:
            nav_description += f" - Text: {', '.join(key_text)}"  
        if speakers:
            nav_description += f" - Speakers: {', '.join(speakers)}"
            
        nav_items.append(f"- [{nav_description}](#scene-{scene['scene_id']})")
    
    return '\n'.join(nav_items)
```

---

## **Implementation Specification**

### **Knowledge Base Generator Service**
```python
class KnowledgeBaseGenerator:
    def __init__(self, config):
        self.config = config
        self.output_dir = config['paths']['output_dir']
        
    def generate_video_knowledge_base(self, video_name, analysis_results):
        """Complete knowledge base generation workflow"""
        try:
            # 1. Compile scene-by-scene knowledge
            scenes_knowledge = self.compile_scene_knowledge(analysis_results)
            
            # 2. Generate metadata summary
            metadata = self.generate_metadata_summary(analysis_results)
            
            # 3. Create searchable knowledge base content
            knowledge_content = self.generate_knowledge_base_md(
                video_name, scenes_knowledge, metadata
            )
            
            # 4. Write final knowledge base file
            output_file = os.path.join(self.output_dir, f"{video_name}.md")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(knowledge_content)
            
            # 5. Update master index
            self.update_master_index(video_name, metadata, scenes_knowledge)
            
            return output_file
            
        except Exception as e:
            self.handle_knowledge_generation_error(video_name, e)
            raise
```

### **Detailed Scene Formatting**
```python
def generate_detailed_scenes(scenes_knowledge):
    """Generate detailed scene-by-scene analysis"""
    scenes_content = []
    
    for scene in scenes_knowledge:
        scene_content = f"""
### Scene {scene['scene_id']}: {scene['scene_description']}

**Timeline**: {scene['timestamp']} (Duration: {format_duration(scene['duration'])})

#### Visual Elements
- **Objects Detected**: {', '.join(scene['objects_detected']) if scene['objects_detected'] else 'None'}
- **People Present**: {scene['people_count']} people detected
- **Text Overlays**: {', '.join(scene['text_overlays']) if scene['text_overlays'] else 'None'}  
- **Faces Identified**: {len(scene['faces_detected'])} faces

#### Audio Content
- **Transcript**: 
  ```
  {scene['transcript_segment']}
  ```
- **Speakers**: {', '.join(scene['speakers']) if scene['speakers'] else 'Unknown'}
- **Audio Type**: {scene['audio_classification']['type']}
- **Audio Quality**: {scene['audio_features']['quality_score']}/10

#### Key Topics
{chr(10).join([f"- {topic}" for topic in scene['key_topics']])}

#### Searchable Terms
{', '.join(scene['searchable_terms'])}

---
"""
        scenes_content.append(scene_content)
    
    return '\n'.join(scenes_content)
```

---

## **Output Structure**

### **Individual Video Knowledge Base**
```markdown
# Video Knowledge Base: quarterly_review_2024.mp4

## Video Summary
- **Duration**: 15m 30s
- **Scenes**: 6 detected  
- **Processing Date**: 2025-09-04 14:30
- **Speakers Identified**: John Smith, Sarah Johnson, Mike Chen

## Quick Navigation
- [Scene 1 (00:00 - 02:30) - Objects: person, laptop, whiteboard - Text: "Q4 Budget" - Speakers: John](#{scene-1})
- [Scene 2 (02:30 - 05:15) - Objects: person, document - Speakers: Sarah, Mike](#{scene-2})

## Scene-by-Scene Analysis

### Scene 1: Budget Review Discussion
**Timeline**: 00:00 - 02:30 (Duration: 2m 30s)

#### Visual Elements
- **Objects Detected**: person, laptop, whiteboard, coffee_cup
- **People Present**: 3 people detected
- **Text Overlays**: "Q4 Budget Review", "Revenue Target: $2.5M"
- **Faces Identified**: 2 faces

#### Audio Content
- **Transcript**: 
  ```
  John: "Let's start with the Q4 budget review. As you can see on the whiteboard, 
  we're targeting $2.5M in revenue this quarter..."
  ```
- **Speakers**: John Smith
- **Audio Type**: Clear speech, conference room
- **Audio Quality**: 9/10

#### Key Topics
- Q4 budget planning
- Revenue targets  
- Financial projections

#### Searchable Terms
budget, revenue, Q4, financial, projections, targets, $2.5M
```

### **Master Index Generation**
```markdown
# Video Library Index

## Recently Processed
- [quarterly_review_2024.md](quarterly_review_2024.md) - 6 scenes, 15m 30s
- [product_demo_2024.md](product_demo_2024.md) - 4 scenes, 8m 45s

## Search by Speaker
- **John Smith**: quarterly_review_2024.md (Scenes 1,3,6), product_demo_2024.md (Scene 2)
- **Sarah Johnson**: quarterly_review_2024.md (Scenes 2,4,5)

## Search by Topic
- **Budget/Financial**: quarterly_review_2024.md, budget_meeting_jan.md
- **Product Demo**: product_demo_2024.md, client_presentation_feb.md

## Search by Objects
- **Whiteboard**: quarterly_review_2024.md (Scenes 1,3), team_meeting_mar.md
- **Laptop**: product_demo_2024.md, quarterly_review_2024.md
```

---

## **Cross-Reference Generation**

### **Similar Content Detection**
```python
def generate_cross_references(scenes_knowledge):
    """Generate cross-references to similar content in other videos"""
    cross_refs = []
    
    for scene in scenes_knowledge:
        # Find similar scenes based on:
        # - Common objects
        # - Similar text overlays  
        # - Same speakers
        # - Related topics
        
        similar_scenes = find_similar_content(
            objects=scene['objects_detected'],
            text=scene['text_overlays'],
            speakers=scene['speakers'],
            topics=scene['key_topics']
        )
        
        if similar_scenes:
            cross_refs.append({
                'scene_id': scene['scene_id'],
                'similar_content': similar_scenes
            })
    
    return format_cross_references(cross_refs)
```

### **Searchable Term Extraction**
```python
def generate_searchable_terms(scene):
    """Extract comprehensive searchable terms from scene analysis"""
    terms = set()
    
    # From objects
    terms.update(scene['objects_detected'])
    
    # From text overlays  
    for text in scene['text_overlays']:
        terms.update(extract_keywords(text))
    
    # From transcript
    terms.update(extract_keywords(scene['transcript_segment']))
    
    # From topics
    terms.update(scene['key_topics'])
    
    # From audio classification
    terms.add(scene['audio_classification']['type'])
    
    return list(terms)
```

---

## **Quality Assurance**

### **Content Validation**
- **Scene Completeness**: Verify all scenes have complete analysis data
- **Timestamp Accuracy**: Validate scene timing and duration calculations  
- **Text Quality**: Ensure readable formatting and proper encoding
- **Cross-Reference Accuracy**: Validate similar content detection

### **Search Optimization**
- **Term Extraction**: Comprehensive keyword extraction from all sources
- **Navigation Links**: Functional anchor links to scene sections
- **Index Accuracy**: Master index reflects all processed videos
- **Metadata Integrity**: Consistent speaker names and topic classification

---

## **Success Criteria**
- Comprehensive scene-by-scene knowledge bases for institutional discovery
- Searchable content with cross-references to related videos
- Master index for library navigation and content discovery
- High-quality markdown formatting for readability and search
- Complete integration of all 7-tool analysis results

---

**Integration Point**: Claude synthesis workflow using structured build/ data for final knowledge base creation