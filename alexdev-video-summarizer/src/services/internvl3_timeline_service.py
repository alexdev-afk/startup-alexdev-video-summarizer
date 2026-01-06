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

# Alias for backward compatibility
InternVL3TimelineError = InternVL3TimelineServiceError


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
        
        # Simple image description prompt - flowing paragraph format
        self.unified_prompt = """Describe what you see in this image in a single flowing paragraph. Include the people, their positioning and clothing, the setting, lighting, objects, and any visible text or branding. Write as one comprehensive paragraph without bullet points or sections."""
        
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
        Query the VLM with single image and text prompt using real InternVL3 generation
        
        Args:
            image: PIL Image to analyze
            prompt: Text prompt for the VLM
            
        Returns:
            VLM response as string
        """
        try:
            # Check if real model is loaded
            if not hasattr(self, 'model') or self.model is None:
                raise InternVL3TimelineError("InternVL3 model not loaded - cannot perform VLM inference")
            
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
    
    def _query_vlm_dual(self, image1: Image.Image, image2: Image.Image, prompt: str) -> str:
        """
        Query the VLM with two images for comparison using real InternVL3 generation
        
        Args:
            image1: First PIL Image to analyze
            image2: Second PIL Image to analyze
            prompt: Text prompt for dual-image comparison
            
        Returns:
            VLM comparison response as string
        """
        try:
            # Check if real model is loaded
            if not hasattr(self, 'model') or self.model is None:
                raise InternVL3TimelineError("InternVL3 model not loaded - cannot perform VLM inference")
            
            logger.info(f"Starting dual VLM inference with prompt: {prompt[:50]}...")
            
            # Real InternVL3 dual-image inference
            import torch
            
            # Convert both images to InternVL3 format
            max_input_tiles = self.vlm_config.get('max_input_tiles', 6)
            logger.debug(f"Max input tiles per image: {max_input_tiles}")
            
            # Create image tile list and num_patches_list for BOTH images
            image_tiles, num_patches_list = [], []
            transform = build_transform(input_size=self.image_size)
            
            # Process BOTH images (this is the key difference from single-image)
            for i, image in enumerate([image1, image2], 1):
                logger.debug(f"Processing image {i}/2")
                
                if self.model.config.dynamic_image_size:
                    logger.debug(f"Using dynamic preprocessing for image {i}")
                    tiles = dynamic_preprocess(
                        image, 
                        image_size=self.image_size, 
                        max_num=max_input_tiles,
                        use_thumbnail=getattr(self.model.config, 'use_thumbnail', False)
                    )
                else:
                    logger.debug(f"Using single tile preprocessing for image {i}")
                    tiles = [image]
                
                # Add tiles to combined list and track patches per image
                num_patches_list.append(len(tiles))  # Number of tiles for THIS image
                image_tiles += tiles  # Add tiles to combined list
                
                logger.debug(f"Image {i}: {len(tiles)} tiles")
            
            # Transform all tiles to tensors
            pixel_values = [transform(item) for item in image_tiles]
            pixel_values = torch.stack(pixel_values).to(self.model.device, dtype=torch.bfloat16)
            
            logger.debug(f"Total tiles: {len(image_tiles)}, pixel_values shape: {pixel_values.shape}")
            logger.debug(f"num_patches_list: {num_patches_list}")
            
            # Prepare conversation format with TWO image tokens
            question = f"<image><image>\n{prompt}"  # Two image tokens for two images
            history = []  # Empty history for single-turn conversation
            
            logger.debug(f"Formatted dual-image question: {question[:100]}...")
            
            # Generation config from YAML configuration
            generation_config = self.vlm_config.get('generation_config', {}).copy()
            generation_config['max_new_tokens'] = self.max_tokens
            generation_config['max_length'] = getattr(self.model, 'context_len', 8192)
            
            logger.debug(f"Generation config: {generation_config}")
            
            # Use InternVL3's chat method with dual images
            logger.debug("Calling model.chat() with dual images...")
            response = self.model.chat(
                tokenizer=self.tokenizer,
                pixel_values=pixel_values,
                num_patches_list=num_patches_list,  # [tiles_img1, tiles_img2]
                question=question,
                history=history,
                return_history=False,
                generation_config=generation_config
            )
            
            logger.debug(f"Raw dual-image response: {response}")
            
            # Handle response format
            if isinstance(response, tuple):
                response = response[0]  # Take first element if tuple
            
            result = str(response).strip()
            logger.info(f"Dual VLM inference successful: {len(result)} chars")
            return result
            
        except Exception as e:
            import traceback
            logger.error(f"InternVL3 dual inference failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return f"Dual VLM analysis unavailable: {str(e)}"
    
    def _query_vlm_triple(self, image1: Image.Image, image2: Image.Image, image3: Image.Image, prompt: str) -> str:
        """
        Query the VLM with three images for relative comparison
        
        Args:
            image1: First reference image 
            image2: Middle comparison image
            image3: Second reference image
            prompt: Text prompt for triple-image comparison
            
        Returns:
            VLM comparison response as string
        """
        try:
            # Check if real model is loaded
            if not hasattr(self, 'model') or self.model is None:
                raise InternVL3TimelineError("InternVL3 model not loaded - cannot perform VLM inference")
            
            logger.info(f"Starting triple VLM inference with prompt: {prompt[:50]}...")
            
            # Real InternVL3 triple-image inference
            import torch
            
            # Convert all three images to InternVL3 format
            max_input_tiles = self.vlm_config.get('max_input_tiles', 6)
            
            # Create image tile list and num_patches_list for ALL THREE images
            image_tiles, num_patches_list = [], []
            transform = build_transform(input_size=self.image_size)
            
            # Process ALL THREE images
            for i, image in enumerate([image1, image2, image3], 1):
                logger.debug(f"Processing image {i}/3")
                
                if self.model.config.dynamic_image_size:
                    tiles = dynamic_preprocess(
                        image, 
                        image_size=self.image_size, 
                        max_num=max_input_tiles,
                        use_thumbnail=getattr(self.model.config, 'use_thumbnail', False)
                    )
                else:
                    tiles = [image]
                
                # Add tiles to combined list and track patches per image
                num_patches_list.append(len(tiles))  # Number of tiles for THIS image
                image_tiles += tiles  # Add tiles to combined list
            
            # Convert to tensors
            pixel_values = torch.stack([transform(tile) for tile in image_tiles]).to(torch.bfloat16).to(self.model.device)
            
            # Create prompt with THREE image tokens  
            question = f"<image><image><image>\n{prompt}"
            
            # Generation config from YAML configuration
            generation_config = self.vlm_config.get('generation_config', {}).copy()
            generation_config['max_new_tokens'] = self.max_tokens
            generation_config['max_length'] = getattr(self.model, 'context_len', 8192)
            
            logger.debug(f"Generation config: {generation_config}")
            
            # Use InternVL3's chat method with triple images
            logger.debug("Calling model.chat() with triple images...")
            history = []
            response = self.model.chat(
                tokenizer=self.tokenizer,
                pixel_values=pixel_values,
                num_patches_list=num_patches_list,  # [patches_img1, patches_img2, patches_img3]
                question=question,
                history=history,
                return_history=False,
                generation_config=generation_config
            )
            
            result = str(response).strip()
            logger.info(f"Triple VLM inference successful: {len(result)} chars")
            return result
            
        except Exception as e:
            import traceback
            logger.error(f"InternVL3 triple inference failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return f"Triple VLM analysis unavailable: {str(e)}"
    
    
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
        self.unified_prompt = "Describe what you see in this image in a single flowing paragraph. Include the people, their positioning and clothing, the setting, lighting, objects, and any visible text or branding. Write as one comprehensive paragraph without bullet points or sections."
        
        # VLM processing configuration
        self.confidence_threshold = self.vlm_config.get('confidence_threshold', 0.7)
        self.max_tokens = self.vlm_config.get('max_tokens', 256)
        self.frame_sample_rate = self.vlm_config.get('frame_sample_rate', 3)  # frames per scene
        
        # Contextual processing flag and setup - supports "both" mode
        contextual_config = self.vlm_config.get('use_contextual_prompting', False)
        self.use_contextual_prompting = contextual_config
        self.both_mode = contextual_config == "both"
        self.audio_context = None
        self.previous_frame_descriptions = {}  # timestamp -> description mapping
        self.previous_frame_images = {}  # timestamp -> PIL Image mapping for dual-image analysis
        
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
            # InternVL3 requires proper model loading - no fallback modes
            
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
                logger.error(f"InternVL3 dependencies not available: {e}")
                raise InternVL3TimelineError(f"InternVL3 dependencies missing: {str(e)}") from e
            except Exception as e:
                logger.error(f"Failed to load InternVL3 model: {e}")
                raise InternVL3TimelineError(f"Failed to load InternVL3 model: {str(e)}") from e
            
        except Exception as e:
            logger.error(f"Failed to initialize InternVL3: {e}")
            raise InternVL3TimelineError(f"Failed to initialize InternVL3: {str(e)}") from e
    
    def generate_and_save(self, video_path: str, scene_offsets_path: Optional[str] = None) -> EnhancedTimeline:
        """
        Generate simplified timeline from video with VLM analysis - supports both mode for comparison
        
        Args:
            video_path: Path to video.mp4 file (from FFmpeg)
            scene_offsets_path: Optional scene context (not used in simplified approach)
            
        Returns:
            EnhancedTimeline with VLM events (last generated timeline for compatibility)
        """
        start_time = time.time()
        
        # Ensure model is loaded before processing (lazy loading)
        self._ensure_model_loaded()
        
        try:
            # Get video duration
            video_metadata = self._get_video_metadata(video_path)
            total_duration = video_metadata.get('duration', 30.0)
            
            # Determine modes to run
            if self.both_mode:
                modes_to_run = [False, True]  # noncontextual first, then contextual
                logger.info("BOTH MODE: Generating both noncontextual and contextual timelines for comparison")
            else:
                # Single mode (maintain backward compatibility)
                modes_to_run = [bool(self.use_contextual_prompting)]
                
            last_timeline = None
            
            # Process each mode
            for is_contextual in modes_to_run:
                mode_name = "contextual" if is_contextual else "noncontextual"
                logger.info(f"Processing {mode_name} mode...")
                
                # Temporarily set contextual mode for this iteration
                original_contextual = self.use_contextual_prompting
                self.use_contextual_prompting = is_contextual
                
                # Reset previous frame context for each mode
                self.previous_frame_descriptions = {}
                self.previous_frame_images = {}
                self.audio_context = None
                
                # Create timeline for this mode
                timeline = EnhancedTimeline(
                    audio_file=str(video_path).replace('video.mp4', 'audio.wav'),
                    total_duration=total_duration
                )
                
                # Add model and prompt information from config
                timeline.sources_used.append(self.service_name)
                timeline.processing_notes.append(f"Model: {self.get_model_name('short')}")
                timeline.processing_notes.append(f"Mode: {mode_name}")
                timeline.processing_notes.append(f"Prompt: {self.unified_prompt if not is_contextual else 'Contextual prompting with audio context'}")
                timeline.processing_notes.append(f"Processing timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
                
                # Process video frames for this mode
                self._process_video_frames_simple(video_path, timeline, total_duration)
                
                # Save timeline file for this mode
                self._save_timeline(timeline, video_path)
                
                last_timeline = timeline
                
                # Restore original contextual setting
                self.use_contextual_prompting = original_contextual
                
                logger.info(f"{mode_name.capitalize()} timeline: {len(timeline.events)} events generated")
            
            processing_time = time.time() - start_time
            logger.info(f"InternVL3 timeline generation complete: {processing_time:.2f}s")
            
            return last_timeline
            
        except Exception as e:
            logger.error(f"InternVL3 timeline generation failed: {e}")
            return self._create_fallback_timeline(video_path, error=str(e))
    
    def _load_audio_context_if_needed(self, video_name: str):
        """Load audio context file for contextual prompting"""
        if not self.use_contextual_prompting:
            return
            
        audio_context_path = Path("build") / video_name / "audio_context.txt"
        if audio_context_path.exists():
            with open(audio_context_path, 'r', encoding='utf-8') as f:
                self.audio_context = f.read()
            logger.info(f"Loaded audio context for contextual prompting: {len(self.audio_context)} chars")
        else:
            logger.warning(f"Audio context file not found: {audio_context_path} - using non-contextual mode")
            self.use_contextual_prompting = False

    def _create_contextual_prompt(self, timestamp: float, previous_timestamp: float = None, is_dual_image: bool = False) -> str:
        """Create contextual prompt with audio context only - no dual-image processing to avoid hallucinations"""
        if not self.use_contextual_prompting or not self.audio_context:
            return self.unified_prompt
            
        # Same prompt for all frames - single image with audio context
        prompt = f"""<image>
Analyze this video frame at timestamp {timestamp:.2f}s.

Audio context (full timeline):
{self.audio_context}

Describe what you see in a single flowing paragraph. Include the people, their positioning, clothing, the setting, lighting, objects, and any visible text or branding. Write as one comprehensive paragraph without bullet points or sections."""
        
        return prompt
    
    def _process_video_frames_simple(self, video_path: str, timeline: EnhancedTimeline, total_duration: float):
        """
        Process PySceneDetect extracted frames - load from frames/ directory
        Creates one event per scene using representative frame + VLM description
        """
        try:
            # Extract video name from video_path for dynamic build directory
            # For path like "build/bonita/video.mp4", we want "bonita"
            video_path_obj = Path(video_path)
            if 'build' in video_path_obj.parts:
                # Find the part after 'build' - that's the video name
                build_index = video_path_obj.parts.index('build')
                if build_index + 1 < len(video_path_obj.parts):
                    video_name = video_path_obj.parts[build_index + 1]
                else:
                    video_name = video_path_obj.stem
            else:
                video_name = video_path_obj.stem
            
            # Load audio context for contextual prompting if enabled
            self._load_audio_context_if_needed(video_name)
            
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
            
            # Process only representative frames for efficiency (1 per scene)
            total_frames = total_scenes
            frame_count = 0
            
            print(f"[VLM] Starting VLM processing: {total_scenes} scenes ({total_frames} representative frames)")
            print(f"[VLM] Processing REPRESENTATIVE frames only at scene start timestamps")
            logger.info(f"Processing {total_frames} representative frames from {total_scenes} scenes")
            
            for scene_key, scene_data in scenes.items():
                scene_id = scene_data['scene_id']
                frames = scene_data['frames']
                
                frame_count += 1
                print(f"[VLM] Scene {scene_id}/{total_scenes}: Processing representative frame")
                
                # Process only the representative frame
                frame_info = frames['representative']
                # Use the scene start timestamp (first frame) for timeline positioning
                scene_start_timestamp = frames['first']['timestamp'] 
                representative_frame_path = frame_info['path']
                
                print(f"[VLM] Scene {scene_id}/{total_scenes}: Loading representative - {representative_frame_path}")
                print(f"[VLM] Scene {scene_id}/{total_scenes}: Timeline timestamp: {scene_start_timestamp:.2f}s")
                
                # Load pre-extracted representative frame
                try:
                    frame_image = Image.open(representative_frame_path)
                    print(f"[VLM] Scene {scene_id}/{total_scenes}: Frame loaded ({frame_image.size})")
                except Exception as e:
                    print(f"[ERROR] Scene {scene_id}/{total_scenes}: Failed to load frame - {e}")
                    continue
                
                print(f"[VLM] Scene {scene_id}/{total_scenes}: Running VLM analysis...")
                
                # Analyze representative frame with VLM using scene start timestamp
                vlm_description = self._analyze_frame_with_vlm(frame_image, scene_start_timestamp)
                
                print(f"[VLM] Scene {scene_id}/{total_scenes}: VLM complete ({len(vlm_description)} chars)")
                print(f"[VLM]   Description: {vlm_description[:80]}...")
                
                # Create timeline event at scene start with representative frame description
                timeline.events.append(TimelineEvent(
                    timestamp=scene_start_timestamp,  # Scene start timestamp
                    description=vlm_description,      # Representative frame description
                    source=self.service_name,
                    confidence=0.8,
                    details={
                        'analysis_type': 'representative_frame_scene_start',
                        'scene_id': scene_id,
                        'representative_frame_path': representative_frame_path,
                        'representative_frame_timestamp': frame_info['timestamp'],
                        'scene_duration': scene_data['duration_seconds'],
                        'scene_end_timestamp': frames['last']['timestamp']
                    }
                ))
                
                print(f"[VLM] Scene {scene_id}/{total_scenes}: Added to timeline at {scene_start_timestamp:.2f}s\n")
                
        except Exception as e:
            print(f"[ERROR] Frame processing failed: {e}")
            logger.error(f"PySceneDetect frame processing failed: {e}")
    
    def _analyze_frame_with_vlm(self, image: Image.Image, timestamp: float) -> str:
        """
        Analyze representative frame with VLM using contextual dual-image or standard prompting
        Returns semantic description for institutional knowledge extraction
        """
        try:
            if self.scene_analyzer:
                # Check for previous frame for dual-image contextual analysis
                previous_image = None
                previous_timestamp = None
                
                if self.use_contextual_prompting and self.previous_frame_images:
                    # Find the most recent previous frame
                    previous_timestamps = [ts for ts in self.previous_frame_images.keys() if ts < timestamp]
                    if previous_timestamps:
                        previous_timestamp = max(previous_timestamps)
                        previous_image = self.previous_frame_images[previous_timestamp]
                
                # Always use single-image analysis for contextual mode to avoid hallucinations
                description_prompt = self._create_contextual_prompt(timestamp)
                description = self.scene_analyzer._query_vlm(image, description_prompt)
                
                # Store this description for reference (not used in current implementation)
                if self.use_contextual_prompting:
                    self.previous_frame_descriptions[timestamp] = description
                
                return description
            
            return f"Scene at {timestamp:.1f}s - VLM analysis unavailable"
            
        except Exception as e:
            logger.error(f"VLM frame analysis failed at {timestamp:.1f}s: {e}")
            return f"Scene at {timestamp:.1f}s - analysis failed"
    
    
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
        
        # Add contextual/noncontextual suffix based on processing mode
        context_suffix = "_contextual" if self.use_contextual_prompting else "_noncontextual"
        timeline_file = timeline_dir / f'video_timeline{context_suffix}.json'
        
        try:
            # Convert timeline to dictionary format
            timeline_dict = timeline.to_dict()
            
            # Add model and prompt information at the top
            prompt_used = getattr(self, 'unified_prompt', 'Describe what you see in this image in a single flowing paragraph. Include the people, their positioning and clothing, the setting, lighting, objects, and any visible text or branding. Write as one comprehensive paragraph without bullet points or sections.')
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