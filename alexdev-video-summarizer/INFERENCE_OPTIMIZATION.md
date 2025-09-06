# InternVL3 Inference Performance Optimization

## Current Performance Issues

### GPU Utilization Analysis
- **Current GPU Utilization**: ~10-20% (severely underutilized)
- **Expected GPU Utilization**: 80-90% for optimal performance
- **Primary Bottleneck**: Sequential frame processing causing GPU idle time

## Root Cause Analysis

### 1. Sequential Frame Processing (Critical - High Impact)
**Location**: `src/services/internvl3_timeline_service.py:554-594`

**Current Implementation**:
```python
for scene_key, scene_data in scenes.items():
    for frame_type in ['first', 'representative', 'last']:
        frame_image = Image.open(frame_path)          # I/O wait
        vlm_description = self._analyze_frame_with_vlm(frame_image)  # GPU active
        # GPU idle while processing next frame I/O
```

**Problem**: GPU sits idle while loading next frame, causing 70-80% GPU downtime.

### 2. Individual File I/O Operations (Medium Impact)
**Location**: `src/services/internvl3_timeline_service.py:571`

**Current Implementation**:
```python
frame_image = Image.open(frame_path)  # Individual file I/O per frame
```

**Problem**: Disk I/O blocks GPU processing pipeline.

### 3. Memory Allocation Per Frame (Medium Impact)
**Location**: `src/services/internvl3_timeline_service.py:246-247`

**Current Implementation**:
```python
pixel_values = [transform(item) for item in image_tiles]
pixel_values = torch.stack(pixel_values).to(self.model.device, dtype=torch.bfloat16)
```

**Problem**: Memory allocation/deallocation overhead per frame.

### 4. No Memory Management Between Frames (Low-Medium Impact)
**Problem**: GPU memory fragmentation over time, no explicit cleanup.

## Optimization Solutions

### Solution 1: Batch Processing (High Impact - 5-8x Speed Improvement)

**Implementation Strategy**:
```python
def process_frames_batch(self, frames_batch):
    """Process multiple frames in a single GPU call"""
    
    # Pre-load all frames into memory
    images = []
    metadata = []
    
    for frame_info in frames_batch:
        image = Image.open(frame_info['path'])
        images.append(image)
        metadata.append(frame_info)
    
    # Batch transform all images
    pixel_values_batch = []
    for image in images:
        if self.model.config.dynamic_image_size:
            tiles = dynamic_preprocess(image, image_size=self.image_size, max_num=6)
        else:
            tiles = [image]
        
        pixel_values = [transform(tile) for tile in tiles]
        pixel_values_batch.append(torch.stack(pixel_values))
    
    # Stack all batch tensors
    batch_tensors = torch.cat(pixel_values_batch, dim=0).to(
        self.model.device, dtype=torch.bfloat16
    )
    
    # Single GPU inference call for entire batch
    with torch.no_grad():
        results = self.model.chat_batch(
            tokenizer=self.tokenizer,
            pixel_values=batch_tensors,
            questions=[self.unified_prompt] * len(images),
            generation_config=self.generation_config
        )
    
    return results
```

**Expected Performance Gain**: 5-8x faster, 80%+ GPU utilization

### Solution 2: Asynchronous Frame Pre-loading (Medium Impact - 2-3x Speed Improvement)

**Implementation Strategy**:
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def preload_frames_async(self, frame_paths):
    """Asynchronously pre-load all frames while GPU processes previous batch"""
    
    def load_frame(path):
        return Image.open(path)
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(executor, load_frame, path) 
            for path in frame_paths
        ]
        return await asyncio.gather(*tasks)

def process_with_async_loading(self, scenes):
    """Pipeline frame loading with GPU processing"""
    
    # Pre-load first batch
    current_batch = preload_frames_async(get_frame_paths(scenes[:batch_size]))
    
    for batch_idx in range(0, len(scenes), batch_size):
        # Start loading next batch while processing current
        next_batch = preload_frames_async(get_frame_paths(scenes[batch_idx+batch_size:batch_idx+2*batch_size]))
        
        # Process current batch on GPU
        results = process_frames_batch(current_batch)
        
        # Swap batches
        current_batch = next_batch
```

**Expected Performance Gain**: 2-3x faster by overlapping I/O with computation

### Solution 3: Memory-Optimized Inference (Medium Impact - 1.5-2x Speed Improvement)

**Implementation Strategy**:
```python
def optimized_inference(self, pixel_values_batch):
    """Memory-optimized inference with explicit cleanup"""
    
    # Pre-allocate reusable tensors
    if not hasattr(self, '_tensor_cache'):
        self._tensor_cache = {
            'pixel_buffer': torch.empty(
                (32, 3, self.image_size, self.image_size),
                device=self.model.device,
                dtype=torch.bfloat16
            )
        }
    
    try:
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                # Reuse pre-allocated tensors
                batch_size = len(pixel_values_batch)
                self._tensor_cache['pixel_buffer'][:batch_size] = pixel_values_batch
                
                results = self.model.chat(
                    tokenizer=self.tokenizer,
                    pixel_values=self._tensor_cache['pixel_buffer'][:batch_size],
                    generation_config=self.generation_config
                )
                
                return results
    finally:
        # Explicit memory cleanup
        torch.cuda.empty_cache()
```

**Expected Performance Gain**: 1.5-2x faster, reduced memory fragmentation

### Solution 4: Advanced Pipeline Optimization (High Impact - 3-5x Speed Improvement)

**Implementation Strategy**:
```python
class OptimizedVLMPipeline:
    """High-performance VLM inference pipeline"""
    
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # Pre-compute reusable components
        self.transform = build_transform(input_size=config.image_size)
        self.generation_config = self._build_generation_config()
        
        # Pre-allocate GPU memory
        self._allocate_gpu_buffers()
    
    def _allocate_gpu_buffers(self):
        """Pre-allocate reusable GPU memory buffers"""
        max_batch_size = 32
        self.gpu_buffers = {
            'pixel_values': torch.empty(
                (max_batch_size, 3, self.config.image_size, self.config.image_size),
                device=self.model.device,
                dtype=torch.bfloat16
            ),
            'attention_mask': torch.ones(
                (max_batch_size, self.config.max_tokens),
                device=self.model.device,
                dtype=torch.bool
            )
        }
    
    def process_video_optimized(self, frame_metadata):
        """Optimized end-to-end video processing"""
        
        # 1. Pre-load all frames in parallel
        all_frames = self._preload_all_frames_parallel(frame_metadata)
        
        # 2. Process in optimal batch sizes
        optimal_batch_size = self._calculate_optimal_batch_size()
        
        results = []
        for i in range(0, len(all_frames), optimal_batch_size):
            batch = all_frames[i:i + optimal_batch_size]
            
            # 3. Batch inference with memory reuse
            batch_results = self._process_batch_optimized(batch)
            results.extend(batch_results)
        
        return results
    
    def _calculate_optimal_batch_size(self):
        """Calculate optimal batch size based on GPU memory"""
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        available_memory = gpu_memory - torch.cuda.memory_reserved(0)
        
        # Conservative estimate: ~100MB per frame for InternVL3-2B
        frame_memory_mb = 100
        optimal_batch = min(32, available_memory // (frame_memory_mb * 1024 * 1024))
        
        return max(1, optimal_batch)
```

**Expected Performance Gain**: 3-5x faster, near-optimal GPU utilization

## Implementation Priority

### Phase 1: Quick Wins (1-2 hours)
1. **Batch Processing** - Implement basic batching for immediate 5x improvement
2. **Memory Management** - Add torch.cuda.empty_cache() between batches

### Phase 2: Pipeline Improvements (2-4 hours)
1. **Async Frame Loading** - Pre-load frames while GPU processes
2. **Memory Optimization** - Pre-allocated tensors and buffer reuse

### Phase 3: Advanced Optimization (4-8 hours)
1. **Full Pipeline Optimization** - Complete optimized pipeline class
2. **Dynamic Batch Sizing** - Adaptive batch size based on GPU memory
3. **Profiling and Tuning** - Fine-tune based on actual performance metrics

## Expected Performance Improvements

| Optimization Level | GPU Utilization | Speed Improvement | Implementation Time |
|-------------------|-----------------|-------------------|-------------------|
| Current (Baseline) | 10-20% | 1x | - |
| Phase 1 (Batching) | 60-80% | 5-8x | 1-2 hours |
| Phase 2 (Pipeline) | 70-85% | 8-12x | 3-6 hours |
| Phase 3 (Advanced) | 80-95% | 12-20x | 7-14 hours |

## Code Locations to Modify

### Primary Files:
1. **`src/services/internvl3_timeline_service.py`**
   - Line 527: `_process_video_frames_simple()` - Replace with batch processing
   - Line 554: Sequential loops - Replace with batch operations
   - Line 571: Individual frame loading - Replace with batch loading
   - Line 217: `_analyze_frame_with_vlm()` - Add batch support

### Supporting Files:
2. **`config/processing.yaml`**
   - Add batch processing configuration
   - Add memory optimization settings

3. **New Files to Create**:
   - `src/services/vlm_batch_processor.py` - Dedicated batch processing class
   - `src/utils/gpu_memory_optimizer.py` - GPU memory management utilities

## Testing Strategy

### Performance Benchmarks:
1. **Before Optimization**: Process 30-second video (8 scenes = 24 frames)
   - Current time: ~10-15 minutes
   - GPU utilization: 10-20%

2. **After Phase 1**: Same video with batching
   - Expected time: ~2-3 minutes  
   - Expected GPU utilization: 60-80%

3. **After Phase 3**: Same video fully optimized
   - Expected time: <1 minute
   - Expected GPU utilization: 80-95%

### Test Commands:
```bash
# Benchmark current implementation
time python src/main.py --input input/ --output output/ --verbose

# Profile GPU memory usage
nvidia-smi dmon -s u -i 0

# Monitor GPU utilization during processing
watch -n 1 'nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.free --format=csv'
```

## Implementation Notes

### Configuration Changes Required:
```yaml
# Add to config/processing.yaml
gpu_pipeline:
  internvl3:
    # Performance optimization settings
    batch_size: "auto"  # or specific number like 16
    enable_batch_processing: true
    memory_optimization: true
    async_frame_loading: true
    
    # Advanced settings
    max_batch_size: 32
    gpu_memory_fraction: 0.9
    enable_mixed_precision: true
```

### Error Handling Considerations:
- Batch processing failure recovery (process individually if batch fails)
- GPU memory overflow handling (reduce batch size dynamically)
- Frame loading failure handling (skip corrupted frames)

### Monitoring Integration:
- Add batch processing metrics to logs
- Track GPU memory usage during processing
- Monitor inference latency per batch

---

**Priority**: High - This optimization could reduce video processing time from 10+ minutes to <2 minutes while maximizing GPU hardware utilization.

**Risk Level**: Medium - Changes core inference pipeline, requires thorough testing.

**Dependencies**: None - can be implemented with existing codebase structure.