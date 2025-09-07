"""
Vid2Seq Timeline Service

Integrates Vid2Seq dense video captioning model for comprehensive timeline generation.
Vid2Seq processes entire videos and generates temporally localized event descriptions.

Key capabilities:
- Dense video captioning with temporal localization
- Speech + visual input processing
- Single-stage end-to-end timeline generation
- Pre-trained on millions of YouTube videos
"""

import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from utils.logger import get_logger
from utils.enhanced_timeline_schema import EnhancedTimeline, TimelineSpan

logger = get_logger(__name__)


class Vid2SeqTimelineServiceError(Exception):
    """Vid2Seq timeline service error"""
    pass


class Vid2SeqTimelineService:
    """Vid2Seq dense video captioning timeline service"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Vid2Seq timeline service"""
        self.config = config
        self.service_name = "vid2seq"
        self.vid2seq_config = config.get('gpu_pipeline', {}).get('vid2seq', {})
        
        # Model configuration
        self.model_path = self.vid2seq_config.get('model_path', '../references/vidchapters')
        self.checkpoint_path = self.vid2seq_config.get('checkpoint_path', None)
        self.confidence_threshold = self.vid2seq_config.get('confidence_threshold', 0.5)
        self.max_duration = self.vid2seq_config.get('max_duration', 600)  # 10 minutes max
        
        # Model components (lazy loading)
        self.model = None
        self.tokenizer = None
        self._model_loaded = False
        
        logger.info(f"Vid2Seq timeline service initialized")
        
    def _ensure_model_loaded(self):
        """Ensure Vid2Seq model is loaded (lazy loading)"""
        if self._model_loaded:
            return
            
        logger.info("Loading Vid2Seq model on-demand...")
        self._initialize_vid2seq()
        self._model_loaded = True
    
    def _initialize_vid2seq(self):
        """Initialize Vid2Seq model from VidChapters implementation"""
        try:
            import sys
            import torch
            from pathlib import Path
            
            # Add VidChapters to path
            vidchapters_path = Path(self.model_path).resolve()
            if str(vidchapters_path) not in sys.path:
                sys.path.insert(0, str(vidchapters_path))
            
            # Import Vid2Seq model
            from model.vid2seq import Vid2Seq, _get_tokenizer
            from model.vit import VisionTransformer
            
            # Model configuration (from VidChapters defaults)
            t5_path = "t5-base"  # VidChapters uses t5-base instead of t5-v1_1-base
            num_bins = 100  # Time tokens
            
            # Initialize tokenizer with time tokens
            logger.info(f"Loading tokenizer: {t5_path}")
            self.tokenizer = _get_tokenizer(t5_path, num_bins=num_bins)
            
            # Initialize Vid2Seq model
            logger.info(f"Initializing Vid2Seq model")
            self.model = Vid2Seq(
                t5_path=t5_path,
                tokenizer=self.tokenizer,
                num_features=100,  # Visual features per frame
                embed_dim=768,     # ViT embedding dimension  
                depth=12,          # ViT depth
                heads=12,          # ViT attention heads
                mlp_dim=2048,      # ViT MLP dimension
                use_speech=True,   # Use ASR input
                use_video=True,    # Use visual input
                num_bins=num_bins  # Time tokenization
            )
            
            # Load checkpoint if provided
            if self.checkpoint_path and Path(self.checkpoint_path).exists():
                logger.info(f"Loading checkpoint: {self.checkpoint_path}")
                checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
                self.model.load_state_dict(checkpoint['model'], strict=False)
            else:
                logger.warning("No checkpoint provided - using random initialization")
                
            # Set to evaluation mode
            self.model.eval()
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                logger.info("Vid2Seq model loaded on GPU")
            else:
                logger.info("Vid2Seq model loaded on CPU")
                
        except Exception as e:
            logger.error(f"Failed to initialize Vid2Seq: {e}")
            raise Vid2SeqTimelineServiceError(f"Failed to initialize Vid2Seq: {str(e)}") from e
    
    def generate_and_save(self, video_path: str, scene_offsets_path: Optional[str] = None) -> EnhancedTimeline:
        """
        Generate dense video captioning timeline using Vid2Seq
        
        Args:
            video_path: Path to video.mp4 file
            scene_offsets_path: Optional scene context (not used - Vid2Seq is end-to-end)
            
        Returns:
            EnhancedTimeline with Vid2Seq dense captioning events
        """
        start_time = time.time()
        
        try:
            # Get video metadata
            video_metadata = self._get_video_metadata(video_path)
            total_duration = video_metadata.get('duration', 30.0)
            
            # Check duration limits
            if total_duration > self.max_duration:
                logger.warning(f"Video duration {total_duration}s exceeds limit {self.max_duration}s")
                
            # Create timeline
            timeline = EnhancedTimeline(
                audio_file=str(video_path).replace('video.mp4', 'audio.wav'),
                total_duration=total_duration
            )
            
            timeline.sources_used.append(self.service_name)
            timeline.processing_notes.append(f"Vid2Seq dense video captioning")
            timeline.processing_notes.append(f"Processing timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            
            # Process with Vid2Seq (mock for now - need actual implementation)
            self._process_video_with_vid2seq(video_path, timeline, total_duration)
            
            processing_time = time.time() - start_time
            logger.info(f"Vid2Seq timeline: {len(timeline.events)} events in {processing_time:.2f}s")
            
            # Save timeline
            self._save_timeline(timeline, video_path)
            
            return timeline
            
        except Exception as e:
            logger.error(f"Vid2Seq timeline generation failed: {e}")
            return self._create_fallback_timeline(video_path, error=str(e))
    
    def _process_video_with_vid2seq(self, video_path: str, timeline: EnhancedTimeline, total_duration: float):
        """Process video with Vid2Seq model (MOCK IMPLEMENTATION)"""
        
        # MOCK: Generate dense captions for demonstration
        # In real implementation, this would:
        # 1. Extract visual features (CLIP ViT-L/14)  
        # 2. Extract ASR with Whisper
        # 3. Run Vid2Seq inference
        # 4. Parse temporal localization and captions
        
        logger.info("Processing video with Vid2Seq (MOCK IMPLEMENTATION)")
        
        # Mock dense captions with temporal localization
        mock_captions = [
            {"start": 0.0, "end": 5.2, "caption": "Two women stand in a beauty salon preparing for a consultation"},
            {"start": 5.2, "end": 12.8, "caption": "The women sit down and begin discussing hair treatment options"},
            {"start": 12.8, "end": 18.5, "caption": "Beautician examines the client's hair texture and condition"},
            {"start": 18.5, "end": 25.1, "caption": "Discussion about available treatments and products displayed on shelves"},
            {"start": 25.1, "end": 30.0, "caption": "Consultation concludes with the women standing up in the salon"}
        ]
        
        for caption_data in mock_captions:
            span = TimelineSpan(
                start=caption_data["start"],
                end=caption_data["end"],
                description=caption_data['caption'],  # Clean description without time prefix
                source=self.service_name,
                confidence=0.85,  # Mock confidence
                details={
                    'duration': caption_data["end"] - caption_data["start"],
                    'caption_type': 'dense_video_caption',
                    'model': 'vid2seq_mock'
                }
            )
            timeline.add_span(span)
            
        logger.info(f"Generated {len(mock_captions)} dense video captions")
    
    def _get_video_metadata(self, video_path: str) -> Dict[str, Any]:
        """Extract video metadata"""
        try:
            video_file = Path(video_path)
            if not video_file.exists():
                raise Vid2SeqTimelineServiceError(f"Video file not found: {video_path}")
                
            return {
                'file_size': video_file.stat().st_size,
                'duration': 30.0,  # Default fallback
                'path': str(video_file)
            }
        except Exception as e:
            logger.warning(f"Could not extract video metadata: {e}")
            return {'duration': 30.0}
    
    def _save_timeline(self, timeline: EnhancedTimeline, video_path: str):
        """Save Vid2Seq timeline file"""
        
        # Create build directory structure
        video_path_parts = Path(video_path).parts
        if 'build' in video_path_parts:
            video_name = video_path_parts[video_path_parts.index('build') + 1]
        else:
            video_name = Path(video_path).stem.replace('video', '') or 'unknown'
            
        timeline_dir = Path('build') / video_name / 'video_timelines'
        timeline_dir.mkdir(parents=True, exist_ok=True)
        
        # Save Vid2Seq timeline
        timeline_file = timeline_dir / 'vid2seq_timeline.json'
        
        try:
            timeline_dict = timeline.to_dict()
            
            # Add Vid2Seq model information
            model_info = {
                'model_name': 'Vid2Seq',
                'model_type': 'dense_video_captioning', 
                'architecture': 'T5 + ViT + temporal_localization',
                'processing_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
                'event_count': len(timeline_dict.get('events', [])),
                'processing_mode': 'end_to_end_dense_captioning',
                'mock_implementation': True  # Remove when real model integrated
            }
            
            timeline_dict = {'model_info': model_info, **timeline_dict}
            
            with open(timeline_file, 'w') as f:
                json.dump(timeline_dict, f, indent=2, default=str)
                
            logger.info(f"Vid2Seq timeline saved to: {timeline_file}")
            
        except Exception as e:
            logger.error(f"Failed to save Vid2Seq timeline: {e}")
    
    def _create_fallback_timeline(self, video_path: str, error: str = "") -> EnhancedTimeline:
        """Create fallback timeline when Vid2Seq processing fails"""
        
        timeline = EnhancedTimeline(
            audio_file=str(video_path).replace('video.mp4', 'audio.wav'),
            total_duration=30.0
        )
        
        timeline.sources_used.append(self.service_name)
        timeline.processing_notes.append(f"Vid2Seq processing failed: {error}")
        timeline.processing_notes.append("Fallback timeline created")
        
        # Add failure span
        timeline.add_span(TimelineSpan(
            start=0.0,
            end=30.0,  # Full video duration
            description="Vid2Seq dense video captioning failed - no timeline available",
            source=self.service_name,
            confidence=0.1,
            details={
                'analysis_type': 'processing_failure',
                'error': error,
                'fallback_mode': True
            }
        ))
        
        return timeline
    
    def cleanup(self):
        """Cleanup Vid2Seq model to free memory"""
        if self._model_loaded and self.model is not None:
            try:
                logger.info("Unloading Vid2Seq model...")
                del self.model
                del self.tokenizer
                self.model = None
                self.tokenizer = None
                self._model_loaded = False
                
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                logger.info("Vid2Seq model unloaded successfully")
                
            except Exception as e:
                logger.warning(f"Error during Vid2Seq cleanup: {e}")


# Export for easy importing
__all__ = ['Vid2SeqTimelineService']