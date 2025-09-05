"""
Knowledge Base Generator Service.

Production implementation for Phase 5 - generates comprehensive knowledge base files
from scene-by-scene analysis results for institutional knowledge discovery.
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Set
from datetime import datetime

from utils.logger import get_logger

logger = get_logger(__name__)


class KnowledgeBaseGenerator:
    """Knowledge base generator service"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize knowledge base generator
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.output_config = config.get('output', {})
        self.paths_config = config.get('paths', {})
        
        # Create output directory
        self.output_dir = Path(self.paths_config.get('output_dir', 'output'))
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info("Knowledge base generator initialized (PRODUCTION MODE)")
    
    def generate_video_knowledge_base(self, video_name: str, analysis_data: Dict[str, Any]) -> Path:
        """
        Generate comprehensive knowledge base file from scene analysis
        
        Args:
            video_name: Name of the video
            analysis_data: Complete analysis data from processing context
            
        Returns:
            Path to generated knowledge base file
        """
        logger.info(f"Generating comprehensive knowledge base for: {video_name}")
        
        try:
            # 1. Compile scene-by-scene knowledge
            scenes_knowledge = self.compile_scene_knowledge(analysis_data)
            
            # 2. Generate metadata summary
            metadata = self.generate_metadata_summary(analysis_data, scenes_knowledge)
            
            # 3. Create searchable knowledge base content
            knowledge_content = self.generate_knowledge_base_md(
                video_name, scenes_knowledge, metadata
            )
            
            # 4. Write final knowledge base file
            output_file = self.output_dir / f"{video_name}.md"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(knowledge_content)
            
            # 5. Update master index
            self.update_master_index(video_name, metadata, scenes_knowledge)
            
            logger.info(f"Comprehensive knowledge base generated: {output_file}")
            return output_file
            
        except Exception as e:
            self.handle_knowledge_generation_error(video_name, e)
            raise
    
    def compile_scene_knowledge(self, analysis_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Compile all analysis into scene-by-scene knowledge structure"""
        scenes_knowledge = []
        scene_analysis = analysis_data.get('scene_analysis', {})
        scene_data = analysis_data.get('scene_data', {})
        
        for scene_id, analysis in scene_analysis.items():
            combined = analysis.get('combined_analysis', {})
            audio_analysis = combined.get('audio_analysis', {})
            visual_analysis = combined.get('visual_analysis', {})
            scene_timing = scene_data.get('scenes', {}).get(scene_id, {})
            
            # Extract structured data from each analysis component
            whisper_data = audio_analysis.get('whisper', {})
            yolo_data = visual_analysis.get('yolo', {})
            easyocr_data = visual_analysis.get('easyocr', {})
            opencv_data = visual_analysis.get('opencv', {})
            librosa_data = audio_analysis.get('librosa', {})
            pyaudio_data = audio_analysis.get('pyaudioanalysis', {})
            
            scene_knowledge = {
                'scene_id': scene_id,
                'timestamp': self.format_timestamp_range(scene_timing),
                'duration': scene_timing.get('duration', 0),
                'start_seconds': scene_timing.get('start_seconds', 0),
                'end_seconds': scene_timing.get('end_seconds', 0),
                
                # Visual Analysis Results
                'objects_detected': yolo_data.get('objects', []),
                'people_count': yolo_data.get('people_count', 0),
                'text_overlays': easyocr_data.get('text_content', []),
                'faces_detected': opencv_data.get('faces', []),
                
                # Audio Analysis Results
                'transcript_segment': whisper_data.get('transcript', ''),
                'speakers': whisper_data.get('speakers', []),
                'audio_features': librosa_data.get('features', {}),
                'audio_classification': pyaudio_data.get('classification', {}),
                
                # Enhanced context
                'scene_description': self.generate_scene_description(combined),
                'key_topics': self.extract_key_topics(combined),
                'searchable_terms': self.generate_searchable_terms(combined)
            }
            scenes_knowledge.append(scene_knowledge)
        
        return scenes_knowledge
    
    def generate_knowledge_base_md(self, video_name: str, scenes_knowledge: List[Dict[str, Any]], metadata: Dict[str, Any]) -> str:
        """Generate comprehensive .md knowledge base file"""
        
        content = f"""# Video Knowledge Base: {video_name}

## Video Summary
- **Duration**: {self.format_duration(metadata['total_duration'])}
- **Scenes**: {len(scenes_knowledge)} detected
- **Processing Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
- **Speakers Identified**: {', '.join(metadata['all_speakers'])}

## Quick Navigation
{self.generate_scene_navigation(scenes_knowledge)}

---

## Scene-by-Scene Analysis

{self.generate_detailed_scenes(scenes_knowledge)}

---

## Cross-References
{self.generate_cross_references(scenes_knowledge)}

## Search Index
{self.generate_search_terms(scenes_knowledge)}

---

Generated by alexdev-video-summarizer v1.0.0 - Comprehensive Institutional Knowledge Extraction
"""
        
        return content
    
    def generate_scene_navigation(self, scenes_knowledge: List[Dict[str, Any]]) -> str:
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
    
    def generate_detailed_scenes(self, scenes_knowledge: List[Dict[str, Any]]) -> str:
        """Generate detailed scene-by-scene analysis"""
        scenes_content = []
        
        for scene in scenes_knowledge:
            scene_content = f"""
### Scene {scene['scene_id']}: {scene['scene_description']}

**Timeline**: {scene['timestamp']} (Duration: {self.format_duration(scene['duration'])})

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
- **Audio Type**: {scene['audio_classification'].get('type', 'Unknown')}
- **Audio Quality**: {scene['audio_features'].get('quality_score', 'N/A')}/10

#### Key Topics
{chr(10).join([f"- {topic}" for topic in scene['key_topics']])}

#### Searchable Terms
{', '.join(scene['searchable_terms'])}

---
"""
            scenes_content.append(scene_content)
        
        return '\n'.join(scenes_content)
    
    def update_master_index(self, video_name: str, metadata: Dict[str, Any], scenes_knowledge: List[Dict[str, Any]]):
        """Update master index with comprehensive video information"""
        index_file = self.output_dir / "INDEX.md"
        
        # Create comprehensive master index content
        if index_file.exists():
            with open(index_file, 'r', encoding='utf-8') as f:
                existing_content = f.read()
        else:
            existing_content = self._create_initial_index()
        
        # Add new entry with rich metadata
        new_entry = f"- [{video_name}.md]({video_name}.md) - {len(scenes_knowledge)} scenes, {self.format_duration(metadata['total_duration'])}\n"
        
        # Update recently processed section
        if "## Recently Processed\n" in existing_content:
            parts = existing_content.split("## Recently Processed\n", 1)
            updated_content = f"{parts[0]}## Recently Processed\n{new_entry}{parts[1]}"
        else:
            updated_content = f"{existing_content}\n{new_entry}"
        
        # Update speaker and topic indices
        updated_content = self._update_speaker_index(updated_content, video_name, metadata['all_speakers'], scenes_knowledge)
        updated_content = self._update_topic_index(updated_content, video_name, scenes_knowledge)
        updated_content = self._update_object_index(updated_content, video_name, scenes_knowledge)
        
        with open(index_file, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        logger.info(f"Master index updated with comprehensive data for {video_name}")
    
    def generate_metadata_summary(self, analysis_data: Dict[str, Any], scenes_knowledge: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate metadata summary from analysis data"""
        scene_data = analysis_data.get('scene_data', {})
        
        # Collect all speakers across scenes
        all_speakers = set()
        for scene in scenes_knowledge:
            all_speakers.update(scene['speakers'])
        
        return {
            'total_duration': scene_data.get('total_duration', 0),
            'scene_count': len(scenes_knowledge),
            'all_speakers': list(all_speakers),
            'processing_date': datetime.now().isoformat(),
            'total_scenes': len(scenes_knowledge)
        }
    
    def generate_scene_description(self, combined_analysis: Dict[str, Any]) -> str:
        """Generate descriptive scene summary"""
        audio = combined_analysis.get('audio_analysis', {})
        visual = combined_analysis.get('visual_analysis', {})
        
        # Extract key elements for description
        objects = visual.get('yolo', {}).get('objects', [])[:2]
        speakers = audio.get('whisper', {}).get('speakers', [])
        text_content = visual.get('easyocr', {}).get('text_content', [])[:1]
        
        # Build descriptive title
        description_parts = []
        
        if speakers:
            description_parts.append(f"{', '.join(speakers)} speaking")
        elif audio.get('whisper', {}).get('transcript'):
            description_parts.append("Audio content")
        
        if objects:
            description_parts.append(f"with {', '.join(objects)}")
        
        if text_content:
            description_parts.append(f"showing '{text_content[0]}'")
        
        return ' '.join(description_parts) if description_parts else "Scene content"
    
    def extract_key_topics(self, combined_analysis: Dict[str, Any]) -> List[str]:
        """Extract key topics from scene analysis"""
        topics = []
        
        audio = combined_analysis.get('audio_analysis', {})
        visual = combined_analysis.get('visual_analysis', {})
        
        # Extract from transcript
        transcript = audio.get('whisper', {}).get('transcript', '')
        if transcript:
            # Simple keyword extraction - could be enhanced with NLP
            topic_keywords = self._extract_topic_keywords(transcript)
            topics.extend(topic_keywords)
        
        # Extract from text overlays
        text_overlays = visual.get('easyocr', {}).get('text_content', [])
        for text in text_overlays:
            topic_keywords = self._extract_topic_keywords(text)
            topics.extend(topic_keywords)
        
        # Extract from object context
        objects = visual.get('yolo', {}).get('objects', [])
        if 'laptop' in objects or 'computer' in objects:
            topics.append('Technology/Computing')
        if 'whiteboard' in objects:
            topics.append('Presentation/Meeting')
        if 'document' in objects or 'book' in objects:
            topics.append('Documentation/Reference')
        
        return list(set(topics))[:5]  # Limit to top 5 unique topics
    
    def generate_searchable_terms(self, combined_analysis: Dict[str, Any]) -> List[str]:
        """Extract comprehensive searchable terms from scene analysis"""
        terms = set()
        
        audio = combined_analysis.get('audio_analysis', {})
        visual = combined_analysis.get('visual_analysis', {})
        
        # From objects
        terms.update(visual.get('yolo', {}).get('objects', []))
        
        # From text overlays  
        for text in visual.get('easyocr', {}).get('text_content', []):
            terms.update(self._extract_keywords(text))
        
        # From transcript
        transcript = audio.get('whisper', {}).get('transcript', '')
        terms.update(self._extract_keywords(transcript))
        
        # From speakers
        terms.update(audio.get('whisper', {}).get('speakers', []))
        
        # From audio classification
        audio_type = audio.get('pyaudioanalysis', {}).get('classification', {}).get('type')
        if audio_type:
            terms.add(audio_type)
        
        return list(terms)
    
    def generate_cross_references(self, scenes_knowledge: List[Dict[str, Any]]) -> str:
        """Generate cross-references to similar content (placeholder for future enhancement)"""
        # This is a simplified implementation - could be enhanced with actual similarity detection
        cross_refs = []
        
        for scene in scenes_knowledge:
            if len(scene['objects_detected']) > 2 or len(scene['speakers']) > 1:
                cross_refs.append(f"- Scene {scene['scene_id']}: Rich content available for cross-referencing")
        
        if cross_refs:
            return "### Similar Content Opportunities\n" + '\n'.join(cross_refs)
        else:
            return "### Cross-References\nNo similar content patterns detected in current processing."
    
    def generate_search_terms(self, scenes_knowledge: List[Dict[str, Any]]) -> str:
        """Generate comprehensive search index"""
        all_terms = set()
        
        for scene in scenes_knowledge:
            all_terms.update(scene['searchable_terms'])
        
        # Group terms alphabetically
        sorted_terms = sorted(list(all_terms))
        grouped_terms = {}
        
        for term in sorted_terms:
            if term:  # Skip empty terms
                first_letter = term[0].upper()
                if first_letter not in grouped_terms:
                    grouped_terms[first_letter] = []
                grouped_terms[first_letter].append(term)
        
        # Format as searchable index
        index_content = []
        for letter in sorted(grouped_terms.keys()):
            index_content.append(f"**{letter}**: {', '.join(grouped_terms[letter])}")
        
        return '\n'.join(index_content)
    
    def format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
    
    def format_timestamp_range(self, scene_timing: Dict[str, Any]) -> str:
        """Format scene timestamp range"""
        start = scene_timing.get('start_seconds', 0)
        end = scene_timing.get('end_seconds', 0)
        
        def format_time(seconds):
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins:02d}:{secs:02d}"
        
        return f"{format_time(start)} - {format_time(end)}"
    
    def handle_knowledge_generation_error(self, video_name: str, error: Exception):
        """Handle knowledge generation errors"""
        logger.error(f"Knowledge generation failed for {video_name}: {str(error)}")
        logger.error(f"Error type: {type(error).__name__}")
        
        # Could implement recovery strategies here
        # For now, just log comprehensive error information
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Simple keyword extraction from text"""
        if not text:
            return []
        
        # Simple approach - could be enhanced with NLP libraries
        words = re.findall(r'\b[A-Za-z]{3,}\b', text.lower())
        
        # Filter out common stop words
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'use', 'way', 'she', 'many', 'oil', 'sit', 'word', 'long', 'down', 'side', 'been', 'call', 'came', 'each', 'find', 'have', 'here', 'just', 'like', 'look', 'made', 'make', 'most', 'move', 'must', 'name', 'over', 'said', 'same', 'tell', 'than', 'that', 'them', 'they', 'this', 'time', 'very', 'well', 'were', 'what', 'with', 'would', 'your'}
        
        keywords = [word for word in words if word not in stop_words and len(word) > 3]
        return list(set(keywords))[:10]  # Return unique keywords, max 10
    
    def _extract_topic_keywords(self, text: str) -> List[str]:
        """Extract topic-relevant keywords"""
        if not text:
            return []
        
        # Look for business/technical terms
        business_patterns = {
            r'\b(budget|revenue|profit|cost|financial|quarter|Q\d)\b': 'Financial',
            r'\b(project|deadline|timeline|milestone|deliverable)\b': 'Project Management', 
            r'\b(meeting|presentation|demo|review|discussion)\b': 'Meeting',
            r'\b(product|feature|design|development|engineering)\b': 'Product Development',
            r'\b(client|customer|user|stakeholder|team)\b': 'Stakeholder Management'
        }
        
        topics = []
        text_lower = text.lower()
        
        for pattern, topic in business_patterns.items():
            if re.search(pattern, text_lower):
                topics.append(topic)
        
        return topics
    
    def _create_initial_index(self) -> str:
        """Create initial master index structure"""
        return """# Video Library Index

## Recently Processed

## Search by Speaker

## Search by Topic

## Search by Objects

---
Generated by alexdev-video-summarizer v1.0.0
"""
    
    def _update_speaker_index(self, content: str, video_name: str, speakers: List[str], scenes_knowledge: List[Dict[str, Any]]) -> str:
        """Update speaker index section"""
        # Simple implementation - could be enhanced
        speaker_info = []
        for speaker in speakers:
            scene_ids = [scene['scene_id'] for scene in scenes_knowledge if speaker in scene['speakers']]
            if scene_ids:
                speaker_info.append(f"- **{speaker}**: {video_name}.md (Scenes {', '.join(scene_ids)})")
        
        if speaker_info and "## Search by Speaker" in content:
            speaker_section = '\n'.join(speaker_info)
            # Simple append - production would need smarter merging
            content = content.replace("## Search by Speaker", f"## Search by Speaker\n{speaker_section}")
        
        return content
    
    def _update_topic_index(self, content: str, video_name: str, scenes_knowledge: List[Dict[str, Any]]) -> str:
        """Update topic index section"""
        all_topics = set()
        for scene in scenes_knowledge:
            all_topics.update(scene['key_topics'])
        
        topic_info = []
        for topic in all_topics:
            topic_info.append(f"- **{topic}**: {video_name}.md")
        
        if topic_info and "## Search by Topic" in content:
            topic_section = '\n'.join(topic_info)
            content = content.replace("## Search by Topic", f"## Search by Topic\n{topic_section}")
        
        return content
    
    def _update_object_index(self, content: str, video_name: str, scenes_knowledge: List[Dict[str, Any]]) -> str:
        """Update object index section"""
        all_objects = set()
        for scene in scenes_knowledge:
            all_objects.update(scene['objects_detected'])
        
        object_info = []
        for obj in all_objects:
            if obj:  # Skip empty objects
                object_info.append(f"- **{obj}**: {video_name}.md")
        
        if object_info and "## Search by Objects" in content:
            object_section = '\n'.join(object_info)
            content = content.replace("## Search by Objects", f"## Search by Objects\n{object_section}")
        
        return content