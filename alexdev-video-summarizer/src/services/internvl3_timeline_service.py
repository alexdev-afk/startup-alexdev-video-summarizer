"""
InternVL3 Timeline Service

Vision-Language Model timeline service following the proven 3-stage integration pattern:
Stage 1: build/video_analysis - Processing metadata and summary statistics only
Stage 2: build/video_timelines - Semantic timeline events with LLM-ready descriptions  
Stage 3: combined_audio_timeline.json - Multi-service coordination with filtering

Replaces YOLO+EasyOCR+OpenCV with a single VLM approach for comprehensive scene understanding:
- "Person interacts with laptop while speaking"
- "Text overlay displays 'Bonita means beautiful'"  
- "Elegant salon environment with modern styling"
- "Customer consultation begins with hair assessment"
"""

import time
import json
import math
import base64
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from io import BytesIO

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
import cv2
import numpy as np

from utils.logger import get_logger
from utils.enhanced_timeline_schema import EnhancedTimeline, TimelineSpan, TimelineEvent

# InternVL3 Constants (copied from streamlit_demo)
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = '<image>'
DEFAULT_IMAGE_PATCH_TOKEN = '<IMG_CONTEXT>'
DEFAULT_IM_START_TOKEN = '<img>'
DEFAULT_IM_END_TOKEN = '</img>'
IMAGE_PLACEHOLDER = '<image-placeholder>'
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

logger = get_logger(__name__)


# ============================================================================
# InternVL3 Core Functions (copied from streamlit_demo/model_worker.py)
# ============================================================================

def load_image_from_base64(image):
    """Load PIL Image from base64 string"""
    return Image.open(BytesIO(base64.b64decode(image)))


def build_transform(input_size):
    """Build image preprocessing transform"""
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Find closest aspect ratio for dynamic preprocessing"""
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    """Dynamic image preprocessing with tiling"""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # Calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # Resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # Split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

# ============================================================================


class InternVL3TimelineServiceError(Exception):
    """InternVL3 timeline service error"""
    pass


class InternVL3SceneAnalyzer:
    """Comprehensive scene analysis using VLM for rich contextual understanding"""
    
    def __init__(self, model, tokenizer, config: Dict[str, Any]):
        """
        Initialize VLM scene analyzer
        
        Args:
            model: InternVL3 model instance
            processor: InternVL3 processor/tokenizer
            config: Configuration dictionary
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.vlm_config = config.get('gpu_pipeline', {}).get('internvl3', {})
        self.max_tokens = self.vlm_config.get('max_tokens', 256)
        self.image_size = getattr(model.config, 'force_image_size', 448) if model else 448
        
        # Simple image description prompt - no timeline confusion
        self.unified_prompt = """Describe what you see in this image. Include the people, setting, objects, and any visible text."""
        
        # Context tracking for change detection
        self.scene_context_history = {}
    
    def analyze_comprehensive_scene(self, image: Image.Image, timestamp: float, scene_id: int) -> Dict[str, Any]:
        """
        Perform unified VLM scene analysis with single optimized prompt
        
        Args:
            image: PIL Image of the scene
            timestamp: Video timestamp
            scene_id: Scene identifier
            
        Returns:
            Dictionary with unified scene analysis
        """
        try:
            # Single comprehensive VLM call
            scene_analysis = self._query_vlm(image, self.unified_prompt)
            
            return {
                'timestamp': timestamp,
                'scene_id': scene_id,
                'comprehensive_analysis': scene_analysis,
                'confidence': self._estimate_analysis_confidence(scene_analysis),
                'analysis_type': 'unified_vlm_scene'
            }
            
        except Exception as e:
            logger.error(f"Unified VLM scene analysis failed at {timestamp:.2f}s: {e}")
            return {
                'timestamp': timestamp,
                'scene_id': scene_id,
                'comprehensive_analysis': "Scene analysis failed due to processing error",
                'confidence': 0.1,
                'analysis_type': 'vlm_failure',
                'error': str(e)
            }
    
    def _query_vlm(self, image: Image.Image, prompt: str) -> str:
        """
        Query the VLM with image and text prompt using real InternVL3 generation
        
        Args:
            image: PIL Image to analyze
            prompt: Text prompt for the VLM
            
        Returns:
            VLM response as string
        """
        try:
            # Check if this is development/mock mode
            if self.config.get('development', {}).get('mock_ai_services', False):
                return self._generate_mock_vlm_response(prompt)
            
            # Check if real model is loaded
            if not hasattr(self, 'model') or self.model is None:
                logger.warning("InternVL3 model not loaded, using mock response")
                return self._generate_mock_vlm_response(prompt)
            
            logger.info(f"Starting VLM inference with prompt: {prompt[:50]}...")
            
            # Real InternVL3 inference (fixed based on streamlit demo model_worker.py)
            import torch
            
            # Convert image to InternVL3 format (following streamlit demo exactly)
            max_input_tiles = self.vlm_config.get('max_input_tiles', 6)
            logger.debug(f"Max input tiles: {max_input_tiles}")
            
            # Create image tile list and num_patches_list (following streamlit demo lines 323-339)
            image_tiles, num_patches_list = [], []
            transform = build_transform(input_size=self.image_size)
            
            # Process single image (simplified from streamlit demo multi-image logic)
            if self.model.config.dynamic_image_size:
                logger.debug("Using dynamic image preprocessing")
                tiles = dynamic_preprocess(
                    image, 
                    image_size=self.image_size, 
                    max_num=max_input_tiles,
                    use_thumbnail=getattr(self.model.config, 'use_thumbnail', False)
                )
            else:
                logger.debug("Using single tile preprocessing")
                tiles = [image]
            
            # Add tiles to image_tiles list and track num_patches per image
            num_patches_list.append(len(tiles))  # Number of tiles for this image
            image_tiles += tiles  # Add tiles to combined list
            
            # Transform all tiles to tensors (following streamlit demo exactly)
            pixel_values = [transform(item) for item in image_tiles]
            pixel_values = torch.stack(pixel_values).to(self.model.device, dtype=torch.bfloat16)
            
            logger.debug(f"Created {len(tiles)} tiles, pixel_values shape: {pixel_values.shape}")
            logger.debug(f"num_patches_list: {num_patches_list}")
            
            # Prepare conversation format with image token (following streamlit demo format)
            question = f"<image>\n{prompt}"  # Add image token before prompt
            history = []  # Empty history for single-turn conversation
            
            logger.debug(f"Formatted question with image token: {question[:100]}...")
            
            # Generation config from YAML configuration
            generation_config = self.vlm_config.get('generation_config', {}).copy()
            
            # Override with dynamic values
            generation_config['max_new_tokens'] = self.max_tokens
            generation_config['max_length'] = getattr(self.model, 'context_len', 8192)
            
            logger.debug(f"Generation config: {generation_config}")
            
            # Use InternVL3's chat method (direct call, no streaming)
            logger.debug("Calling model.chat()...")
            response = self.model.chat(
                tokenizer=self.tokenizer,
                pixel_values=pixel_values,
                num_patches_list=num_patches_list,
                question=question,
                history=history,
                return_history=False,
                generation_config=generation_config
            )
            
            logger.debug(f"Raw response: {response}")
            
            # Handle response format
            if isinstance(response, tuple):
                response = response[0]  # Take first element if tuple
            
            result = str(response).strip()
            logger.info(f"VLM inference successful: {len(result)} chars")
            return result
            
        except Exception as e:
            import traceback
            logger.error(f"InternVL3 inference failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return f"VLM analysis unavailable: {str(e)}"
    
    def _generate_mock_vlm_response(self, prompt: str) -> str:
        """Generate realistic mock VLM responses for development"""
        return """Professional salon setting with two women in consultation. The client (left) and stylist (right) are engaged in discussion about hair services. Modern salon interior with mirrors, professional lighting, and clean aesthetic. Visible text includes "Bonita" branding and salon service information. The atmosphere is welcoming and professional."""
    
    def _estimate_analysis_confidence(self, analysis_text: str) -> float:
        """
        Estimate confidence in VLM analysis based on response quality
        
        Args:
            analysis_text: The VLM response text
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not analysis_text or len(analysis_text) < 20:
            return 0.2
        
        # Basic heuristics for confidence estimation
        confidence = 0.5  # Base confidence
        
        # More detailed responses get higher confidence
        if len(analysis_text) > 100:
            confidence += 0.2
        
        # Specific details mentioned boost confidence
        detail_indicators = ['person', 'object', 'text', 'environment', 'interaction', 'activity']
        detail_count = sum(1 for indicator in detail_indicators if indicator in analysis_text.lower())
        confidence += min(0.3, detail_count * 0.05)
        
        # Error indicators reduce confidence
        if any(error_word in analysis_text.lower() for error_word in ['error', 'failed', 'unavailable', 'unclear']):
            confidence *= 0.5
        
        return min(1.0, max(0.1, confidence))


class InternVL3TimelineService:
    """Semantic VLM timeline service for comprehensive visual scene understanding"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize InternVL3 timeline service
        
        Args:
            config: Configuration dictionary with VLM settings
        """
        self.config = config
        self.service_name = "internvl3"
        self.vlm_config = config.get('gpu_pipeline', {}).get('internvl3', {})
        
        # Validate required model configuration early
        self._validate_model_config()
        
        # Store prompt at service level for access in save methods
        self.unified_prompt = "Describe what you see in this image. Include the people, setting, objects, and any visible text."
        
        # VLM processing configuration
        self.confidence_threshold = self.vlm_config.get('confidence_threshold', 0.7)
        self.max_tokens = self.vlm_config.get('max_tokens', 256)
        self.frame_sample_rate = self.vlm_config.get('frame_sample_rate', 3)  # frames per scene
        
        # Initialize VLM components (lazy loading)
        self.model = None
        self.tokenizer = None
        self.scene_analyzer = None
        self._model_loaded = False
        
        # Skip model loading during init - load on-demand when needed
        logger.info("InternVL3 timeline service initialized (model will load on-demand)")
    
    def _validate_model_config(self):
        """Validate required model configuration for rapid model testing"""
        required_fields = ['model_name', 'model_path']
        missing_fields = []
        
        for field in required_fields:
            if not self.vlm_config.get(field):
                missing_fields.append(f"gpu_pipeline.internvl3.{field}")
        
        if missing_fields:
            error_msg = f"Missing required InternVL3 configuration: {', '.join(missing_fields)}"
            logger.error(error_msg)
            logger.info("Example configuration:")
            logger.info("gpu_pipeline:")
            logger.info("  internvl3:")
            logger.info("    model_name: 'InternVL3_5-2B'")
            logger.info("    model_path: 'OpenGVLab/InternVL3_5-2B'")
            raise ValueError(error_msg)
        
        # Log current model configuration for transparency
        model_name = self.vlm_config.get('model_name')
        model_path = self.vlm_config.get('model_path')
        logger.info(f"Using model: {model_name} from {model_path}")
    
    def get_model_name(self, format_type: str = 'full') -> str:
        """Get model name in different formats for metadata and filenames
        
        Args:
            format_type: 'full', 'clean', or 'short'
                - full: Complete model name (e.g., 'InternVL3_5-2B')
                - clean: Filename-safe version (e.g., 'InternVL3_5-2B') 
                - short: Short version for displays (e.g., 'InternVL3.5-2B')
        """
        model_name = self.vlm_config.get('model_name', 'UnknownModel')
        
        if format_type == 'full':
            return model_name
        elif format_type == 'clean':
            # Remove path separators and clean for filenames
            return model_name.replace('/', '-').replace('OpenGVLab-', '').replace('_', '-')
        elif format_type == 'short':
            # Human-readable short version
            return model_name.replace('_', '.').replace('OpenGVLab/', '')
        else:
            return model_name
    
    def _ensure_model_loaded(self):
        """Ensure model is loaded before processing (lazy loading)"""
        if self._model_loaded:
            return
        
        logger.info("Loading InternVL3 model on-demand...")
        self._initialize_vlm()
        self._model_loaded = True

    def _initialize_vlm(self):
        """Initialize InternVL3 model and processor"""
        try:
            # Check if in development/mock mode
            if self.config.get('development', {}).get('mock_ai_services', False):
                logger.info("InternVL3 running in mock mode for development")
                self.scene_analyzer = InternVL3SceneAnalyzer(None, None, self.config)
                return
            
            # Load model from config (allows rapid model testing)
            model_path = self.vlm_config.get('model_path')
            model_name = self.vlm_config.get('model_name')
            
            if not model_path:
                raise ValueError("model_path must be specified in config under gpu_pipeline.internvl3.model_path")
            if not model_name:
                raise ValueError("model_name must be specified in config under gpu_pipeline.internvl3.model_name")
                
            logger.info(f"Loading InternVL3 model: {model_name} from {model_path}")
            
            try:
                from transformers import AutoTokenizer, AutoModel
                import torch
                
                # Load tokenizer (exact copy from model_worker.py)
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path, 
                    trust_remote_code=True,
                    use_fast=False
                )
                # Remove specific tokens (from model_worker.py line 172-174)
                tokens_to_keep = ['<box>', '</box>', '<ref>', '</ref>']
                tokenizer.additional_special_tokens = [item for item in tokenizer.additional_special_tokens if item not in tokens_to_keep]
                self.tokenizer = tokenizer
                
                # Load model (exact copy from model_worker.py)
                self.model = AutoModel.from_pretrained(
                    model_path,
                    load_in_8bit=False,
                    load_in_4bit=False,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True
                ).eval().cuda()
                
                # Get image size from model config
                self.image_size = self.model.config.force_image_size
                logger.info(f"Model image size: {self.image_size}")
                
                # Initialize scene analyzer with real model
                self.scene_analyzer = InternVL3SceneAnalyzer(self.model, self.tokenizer, self.config)
                logger.info(f"{model_name} model loaded successfully")
                
            except ImportError as e:
                logger.warning(f"InternVL3 dependencies not available: {e}")
                logger.info("Falling back to mock mode - install transformers and torch")
                self.scene_analyzer = InternVL3SceneAnalyzer(None, None, self.config)
            except Exception as e:
                logger.error(f"Failed to load InternVL3 model: {e}")
                logger.info("Falling back to mock mode")
                self.scene_analyzer = InternVL3SceneAnalyzer(None, None, self.config)
            
        except Exception as e:
            logger.error(f"Failed to initialize InternVL3: {e}")
            logger.info("Falling back to mock mode")
            self.scene_analyzer = InternVL3SceneAnalyzer(None, None, self.config)
    
    def generate_and_save(self, video_path: str, scene_offsets_path: Optional[str] = None) -> EnhancedTimeline:
        """
        Generate simplified timeline from video with VLM analysis - one event per frame
        
        Args:
            video_path: Path to video.mp4 file (from FFmpeg)
            scene_offsets_path: Optional scene context (not used in simplified approach)
            
        Returns:
            EnhancedTimeline with VLM events (no spans, no analysis file)
        """
        start_time = time.time()
        
        # Ensure model is loaded before processing (lazy loading)
        self._ensure_model_loaded()
        
        try:
            # Get video duration
            video_metadata = self._get_video_metadata(video_path)
            total_duration = video_metadata.get('duration', 30.0)
            
            # Create simple timeline with model info
            timeline = EnhancedTimeline(
                audio_file=str(video_path).replace('video.mp4', 'audio.wav'),
                total_duration=total_duration
            )
            
            # Add model and prompt information from config
            timeline.sources_used.append(self.service_name)
            timeline.processing_notes.append(f"Model: {self.get_model_name('short')}")
            timeline.processing_notes.append(f"Prompt: {self.unified_prompt}")
            timeline.processing_notes.append(f"Processing timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            
            # Process video frames directly - no complex scene processing
            self._process_video_frames_simple(video_path, timeline, total_duration)
            
            processing_time = time.time() - start_time
            logger.info(f"InternVL3 simple timeline: {len(timeline.events)} events in {processing_time:.2f}s")
            
            # Save only the timeline file - no analysis file needed
            self._save_timeline(timeline, video_path)
            
            return timeline
            
        except Exception as e:
            logger.error(f"InternVL3 timeline generation failed: {e}")
            return self._create_fallback_timeline(video_path, error=str(e))
    
    def _process_video_frames_simple(self, video_path: str, timeline: EnhancedTimeline, total_duration: float):
        """
        Process PySceneDetect extracted frames - load from frames/ directory
        Creates one event per scene using representative frame + VLM description
        """
        try:
            # Extract video name from video_path for dynamic build directory
            video_name = Path(video_path).stem if '/' in video_path or '\\' in video_path else Path(video_path).name.replace('.mp4', '')
            
            # Load frame metadata from PySceneDetect  
            frame_metadata_path = Path("build") / video_name / "frames" / "frame_metadata.json"
            if not frame_metadata_path.exists():
                print("[ERROR] No frame metadata found - run PySceneDetect first")
                logger.error("Frame metadata not found, cannot process scenes")
                return
            
            with open(frame_metadata_path, 'r') as f:
                frame_metadata = json.load(f)
            
            scenes = frame_metadata.get('scenes', {})
            total_scenes = len(scenes)
            
            # Calculate total frames (3 per scene: first, representative, last)
            total_frames = total_scenes * 3
            frame_count = 0
            
            print(f"[VLM] Starting VLM processing: {total_scenes} scenes ({total_frames} frames)")
            print(f"[VLM] Processing ALL frames: first, representative, last")
            logger.info(f"Processing {total_frames} frames from {total_scenes} scenes")
            
            for scene_key, scene_data in scenes.items():
                scene_id = scene_data['scene_id']
                frames = scene_data['frames']
                
                print(f"[VLM] Scene {scene_id}/{total_scenes}: Processing 3 frames")
                
                # Process all 3 frames: first, representative, last
                for frame_type in ['first', 'representative', 'last']:
                    frame_count += 1
                    frame_info = frames[frame_type]
                    timestamp = frame_info['timestamp']
                    frame_path = frame_info['path']
                    
                    print(f"[VLM] Frame {frame_count}/{total_frames}: Loading {frame_type} - {frame_path}")
                    
                    # Load pre-extracted frame
                    try:
                        frame_image = Image.open(frame_path)
                        print(f"[VLM] Frame {frame_count}/{total_frames}: Frame loaded ({frame_image.size})")
                    except Exception as e:
                        print(f"[ERROR] Frame {frame_count}/{total_frames}: Failed to load frame - {e}")
                        continue
                    
                    print(f"[VLM] Frame {frame_count}/{total_frames}: Running VLM analysis...")
                    
                    # Analyze with VLM
                    vlm_description = self._analyze_frame_with_vlm(frame_image, timestamp)
                    
                    print(f"[VLM] Frame {frame_count}/{total_frames}: VLM complete ({len(vlm_description)} chars)")
                    print(f"[VLM]   Description: {vlm_description[:100]}...")
                    
                    # Create timeline event
                    timeline.events.append(TimelineEvent(
                        timestamp=timestamp,
                        description=vlm_description,
                        source=self.service_name,
                        confidence=0.8,
                        details={
                            'analysis_type': 'vlm_frame_analysis',
                            'scene_id': scene_id,
                            'frame_type': frame_type,
                            'scene_duration': scene_data['duration_seconds']
                        }
                    ))
                    
                    print(f"[VLM] Frame {frame_count}/{total_frames}: Added to timeline\n")
                
        except Exception as e:
            print(f"[ERROR] Frame processing failed: {e}")
            logger.error(f"PySceneDetect frame processing failed: {e}")
    
    def _analyze_frame_with_vlm(self, image: Image.Image, timestamp: float) -> str:
        """
        Analyze single frame with VLM and return description
        Simple wrapper around scene analyzer
        """
        try:
            if self.scene_analyzer:
                analysis = self.scene_analyzer.analyze_comprehensive_scene(image, timestamp, scene_id=1)
                if analysis and analysis.get('comprehensive_analysis'):
                    return analysis['comprehensive_analysis']
            
            return f"Frame at {timestamp:.1f}s - VLM analysis unavailable"
            
        except Exception as e:
            logger.error(f"VLM frame analysis failed at {timestamp:.1f}s: {e}")
            return f"Frame at {timestamp:.1f}s - analysis failed"
    
    
    def _get_video_metadata(self, video_path: str) -> Dict[str, Any]:
        """Extract video metadata for timeline creation"""
        try:
            video_file = Path(video_path)
            if not video_file.exists():
                raise InternVL3TimelineServiceError(f"Video file not found: {video_path}")
            
            # Basic metadata from file
            file_size = video_file.stat().st_size
            
            return {
                'file_size': file_size,
                'duration': 30.0,  # Default fallback, will be updated from scene info
                'path': str(video_file)
            }
        except Exception as e:
            logger.warning(f"Could not extract video metadata: {e}")
            return {'duration': 30.0}
    
    
    def _extract_frame_at_timestamp(self, video_path: str, timestamp: float) -> Optional[Image.Image]:
        """Extract single frame at specific timestamp as PIL Image"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"Could not open video: {video_path}")
                return None
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30.0
            
            # Seek to timestamp
            frame_number = int(timestamp * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            # Read frame
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                logger.warning(f"Could not read frame at {timestamp}s")
                return None
            
            # Convert BGR to RGB and create PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            return pil_image
            
        except Exception as e:
            logger.error(f"Frame extraction failed at {timestamp}s: {e}")
            return None
    
    
    
    def _save_timeline(self, timeline: EnhancedTimeline, video_path: str):
        """Save full semantic timeline file following timeline service pattern"""
        
        # Create build directory structure  
        video_path_parts = Path(video_path).parts
        if 'build' in video_path_parts:
            video_name = video_path_parts[video_path_parts.index('build') + 1]
        else:
            video_name = Path(video_path).stem.replace('video', '') or 'unknown'
            
        timeline_dir = Path('build') / video_name / 'video_timelines'
        timeline_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename with clean model name from config and timestamp
        clean_model_name = self.get_model_name('clean')
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
        
        timeline_file = timeline_dir / f'video_timeline.json'
        
        try:
            # Convert timeline to dictionary format
            timeline_dict = timeline.to_dict()
            
            # Add model and prompt information at the top
            prompt_used = getattr(self, 'unified_prompt', 'Describe what you see in this image. Include the people, setting, objects, and any visible text.')
            model_name = self.get_model_name('full')
            model_path = self.vlm_config.get('model_path', model_name)
            
            # Get the actual generation config used (same as inference)
            generation_config = self.vlm_config.get('generation_config', {}).copy()
            generation_config['max_new_tokens'] = self.max_tokens
            generation_config['max_length'] = getattr(self.model, 'context_len', 8192) if hasattr(self, 'model') and self.model else 8192
            
            model_info = {
                'model_name': model_name,
                'model_path': model_path,
                'prompt_used': prompt_used,
                'processing_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
                'frame_count': len(timeline_dict.get('events', [])),
                'processing_mode': 'all_frames',  # vs 'representative_only'
                'generation_config': generation_config
            }
            
            # Insert model info at the beginning of the JSON
            timeline_dict = {'model_info': model_info, **timeline_dict}
            
            with open(timeline_file, 'w') as f:
                json.dump(timeline_dict, f, indent=2, default=str)
            
            logger.info(f"InternVL3 timeline saved to: {timeline_file}")
            
        except Exception as e:
            logger.error(f"Failed to save InternVL3 timeline: {e}")
    
    def _create_fallback_timeline(self, video_path: str, error: str = "") -> EnhancedTimeline:
        """Create fallback timeline when VLM processing fails"""
        
        timeline = EnhancedTimeline(
            audio_file=str(video_path).replace('video.mp4', 'audio.wav'),
            total_duration=30.0
        )
        
        timeline.sources_used.append(self.service_name)
        timeline.processing_notes.append(f"InternVL3 VLM processing failed: {error}")
        timeline.processing_notes.append("Fallback timeline created with minimal data")
        
        # Add a basic failure event
        timeline.events.append(TimelineEvent(
            timestamp=0.0,
            description="InternVL3 VLM analysis failed - no scene understanding available",
            source=self.service_name,
            confidence=0.1,
            details={
                'analysis_type': 'processing_failure',
                'error': error,
                'fallback_mode': True
            }
        ))
        
        return timeline
    
    def unload_model(self):
        """Unload InternVL3 model to free VRAM"""
        if self._model_loaded and self.model is not None:
            try:
                logger.info("Unloading InternVL3 model to free VRAM...")
                
                # Clear model and tokenizer
                del self.model
                del self.tokenizer
                if self.scene_analyzer:
                    del self.scene_analyzer
                
                # Reset state
                self.model = None
                self.tokenizer = None
                self.scene_analyzer = None
                self._model_loaded = False
                
                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                logger.info("InternVL3 model unloaded successfully")
                
            except Exception as e:
                logger.warning(f"Error during InternVL3 model cleanup: {e}")
                
    def cleanup(self):
        """Public cleanup method"""
        self.unload_model()