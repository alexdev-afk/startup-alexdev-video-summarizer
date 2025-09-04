# Quality Assurance & Testing Procedures

Comprehensive testing strategies, quality gates, and validation procedures for alexdev-video-summarizer.

## Testing Strategy Overview

### Testing Pyramid
```
                    /\
                   /  \
                  / E2E\         ← End-to-End Tests
                 /______\
                /        \
               / Integration\    ← Integration Tests  
              /______________\
             /                \
            /    Unit Tests    \  ← Unit Tests (Base)
           /____________________\
```

### Test Categories
- **Unit Tests**: Individual service and component testing
- **Integration Tests**: Service-to-service interaction testing  
- **Performance Tests**: Processing speed and resource utilization
- **End-to-End Tests**: Complete pipeline validation
- **Regression Tests**: Ensure new changes don't break existing functionality
- **Load Tests**: System behavior under stress

## Unit Testing Framework

### Test Structure
```
tests/
├── unit/                       # Unit tests
│   ├── services/
│   │   ├── test_ffmpeg_service.py
│   │   ├── test_scene_detection_service.py
│   │   ├── test_orchestrator.py
│   │   ├── test_gpu_pipeline.py
│   │   └── test_cpu_pipeline.py
│   ├── cli/
│   │   └── test_video_processor.py
│   └── utils/
│       ├── test_config_loader.py
│       └── test_progress_display.py
├── integration/                # Integration tests
│   ├── test_pipeline_integration.py
│   ├── test_service_coordination.py
│   └── test_error_handling.py
├── performance/                # Performance tests
│   ├── test_processing_speed.py
│   └── test_memory_usage.py
├── e2e/                       # End-to-end tests
│   └── test_complete_pipeline.py
├── fixtures/                  # Test data
│   ├── videos/               # Sample videos
│   ├── configs/              # Test configurations
│   └── expected_outputs/     # Expected results
└── conftest.py               # Pytest configuration
```

### Unit Test Examples

#### FFmpeg Service Tests
```python
# tests/unit/services/test_ffmpeg_service.py
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from services.ffmpeg_service import FFmpegService
from utils.processing_context import VideoProcessingContext

class TestFFmpegService:
    
    @pytest.fixture
    def ffmpeg_service(self):
        config = {
            'ffmpeg': {
                'audio': {'sample_rate': 22050, 'format': 'wav'},
                'video': {'codec': 'libx264', 'quality': 'high'}
            },
            'paths': {'build_dir': 'build'}
        }
        return FFmpegService(config)
    
    @pytest.fixture
    def sample_video_path(self, tmp_path):
        # Create a temporary video file for testing
        video_file = tmp_path / "test_video.mp4"
        video_file.touch()  # Create empty file for testing
        return video_file
    
    def test_extract_streams_success(self, ffmpeg_service, sample_video_path):
        """Test successful audio/video stream extraction"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            
            audio_path, video_path = ffmpeg_service.extract_streams(sample_video_path)
            
            assert audio_path.exists()
            assert video_path.exists()
            assert audio_path.suffix == '.wav'
            assert video_path.suffix == '.mp4'
            
            # Verify FFmpeg was called with correct parameters
            assert mock_run.call_count == 2  # Audio + Video extraction
            
    def test_extract_streams_failure(self, ffmpeg_service, sample_video_path):
        """Test handling of FFmpeg extraction failure"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 1
            mock_run.return_value.stderr = "FFmpeg error: Invalid file format"
            
            with pytest.raises(Exception, match="FFmpeg extraction failed"):
                ffmpeg_service.extract_streams(sample_video_path)
    
    def test_split_by_scenes(self, ffmpeg_service, sample_video_path):
        """Test scene-based video splitting"""
        scene_boundaries = [
            {'scene_id': 1, 'start_seconds': 0.0, 'end_seconds': 30.0},
            {'scene_id': 2, 'start_seconds': 30.0, 'end_seconds': 60.0}
        ]
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            
            scene_files = ffmpeg_service.split_by_scenes(sample_video_path, scene_boundaries)
            
            assert len(scene_files) == 2
            assert all(isinstance(f, Path) for f in scene_files)
            assert mock_run.call_count == 2  # One call per scene
    
    @pytest.mark.parametrize("video_format,expected_success", [
        ("mp4", True),
        ("avi", True),
        ("mov", True),
        ("mkv", True),
        ("invalid", False)
    ])
    def test_format_support(self, ffmpeg_service, tmp_path, video_format, expected_success):
        """Test support for different video formats"""
        video_file = tmp_path / f"test_video.{video_format}"
        video_file.touch()
        
        if expected_success:
            # Should not raise exception
            assert ffmpeg_service.validate_video_format(video_file)
        else:
            with pytest.raises(ValueError, match="Unsupported video format"):
                ffmpeg_service.validate_video_format(video_file)
```

#### Scene Detection Service Tests
```python
# tests/unit/services/test_scene_detection_service.py
import pytest
from unittest.mock import Mock, patch
from services.scene_detection_service import SceneDetectionService

class TestSceneDetectionService:
    
    @pytest.fixture
    def scene_service(self):
        config = {
            'scene_detection': {
                'threshold': 27.0,
                'min_scene_length': 2.0,
                'fallback_to_time_based': True
            }
        }
        return SceneDetectionService(config)
    
    def test_detect_scenes_success(self, scene_service):
        """Test successful scene detection"""
        mock_scene_list = [
            (Mock(get_seconds=lambda: 0.0, get_frames=lambda: 0), 
             Mock(get_seconds=lambda: 30.0, get_frames=lambda: 900)),
            (Mock(get_seconds=lambda: 30.0, get_frames=lambda: 900), 
             Mock(get_seconds=lambda: 60.0, get_frames=lambda: 1800))
        ]
        
        with patch('scenedetect.VideoManager') as mock_vm, \
             patch('scenedetect.SceneManager') as mock_sm:
            
            mock_sm_instance = Mock()
            mock_sm.return_value = mock_sm_instance
            mock_sm_instance.get_scene_list.return_value = mock_scene_list
            
            mock_vm_instance = Mock()
            mock_vm.return_value = mock_vm_instance
            mock_vm_instance.get_framerate.return_value = 30.0
            
            scene_list, fps = scene_service.detect_scenes("test_video.mp4")
            
            assert len(scene_list) == 2
            assert fps == 30.0
    
    def test_no_scenes_detected_fallback(self, scene_service):
        """Test fallback to time-based scenes when no scenes detected"""
        with patch('scenedetect.VideoManager'), \
             patch('scenedetect.SceneManager') as mock_sm, \
             patch.object(scene_service, 'get_video_duration', return_value=120.0):
            
            mock_sm_instance = Mock()
            mock_sm.return_value = mock_sm_instance
            mock_sm_instance.get_scene_list.return_value = []  # No scenes detected
            
            scene_boundaries = scene_service.fallback_to_time_based_scenes("test_video.mp4")
            
            assert len(scene_boundaries) > 0
            assert all('fallback_method' in scene for scene in scene_boundaries)
    
    def test_representative_frame_extraction(self, scene_service):
        """Test representative frame extraction from scene"""
        scene_boundary = {
            'scene_id': 1,
            'start_frame': 0,
            'end_frame': 900,
            'start_seconds': 0.0,
            'end_seconds': 30.0
        }
        
        frame_info = scene_service.extract_representative_frame(scene_boundary, fps=30.0)
        
        assert frame_info['scene_id'] == 1
        assert frame_info['representative_frame'] == 450  # Middle frame
        assert frame_info['frame_timestamp'] == 15.0  # Middle timestamp
```

#### Orchestrator Integration Tests
```python
# tests/integration/test_orchestrator.py
import pytest
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from services.orchestrator import VideoProcessingOrchestrator

class TestVideoProcessingOrchestrator:
    
    @pytest.fixture
    def orchestrator(self):
        config = {
            'circuit_breaker_threshold': 3,
            'ffmpeg': {'audio': {'sample_rate': 22050}},
            'scene_detection': {'threshold': 27.0},
            'gpu_pipeline': {'sequential_processing': True},
            'cpu_pipeline': {'max_workers': 3}
        }
        return VideoProcessingOrchestrator(config)
    
    @pytest.fixture
    def sample_video(self, tmp_path):
        video_file = tmp_path / "test_video.mp4"
        video_file.write_bytes(b"fake video content")
        return video_file
    
    def test_successful_video_processing(self, orchestrator, sample_video):
        """Test complete successful video processing pipeline"""
        progress_updates = []
        
        def mock_progress_callback(stage, data):
            progress_updates.append((stage, data))
        
        # Mock all services to return success
        with patch.object(orchestrator.ffmpeg_service, 'extract_streams', 
                         return_value=('audio.wav', 'video.mp4')), \
             patch.object(orchestrator.scene_service, 'analyze_video_scenes', 
                         return_value={'scene_count': 2, 'scenes': [{'scene_id': 1}, {'scene_id': 2}]}), \
             patch.object(orchestrator.gpu_pipeline, 'process_scene', 
                         return_value={'yolo': {'objects': ['person']}}), \
             patch.object(orchestrator.cpu_pipeline, 'process_scene', 
                         return_value={'opencv': {'faces': 1}}), \
             patch.object(orchestrator.knowledge_generator, 'generate_video_knowledge_base', 
                         return_value=Path('output/test_video.md')):
            
            result = orchestrator.process_video_with_progress(sample_video, mock_progress_callback)
            
            assert result.success
            assert result.knowledge_file == Path('output/test_video.md')
            assert result.scenes_processed == 2
            assert len(progress_updates) > 0
    
    def test_circuit_breaker_functionality(self, orchestrator, sample_video):
        """Test circuit breaker stops processing after consecutive failures"""
        # Simulate consecutive failures
        with patch.object(orchestrator.ffmpeg_service, 'extract_streams', 
                         side_effect=Exception("FFmpeg failed")):
            
            # Process multiple videos to trigger circuit breaker
            results = []
            for i in range(5):
                video = sample_video.parent / f"test_video_{i}.mp4"
                video.touch()
                result = orchestrator.process_video_with_progress(video, Mock())
                results.append(result)
                
                if orchestrator.should_abort_batch():
                    break
            
            # Should stop after 3 failures
            assert len(results) == 3
            assert all(not r.success for r in results)
            assert orchestrator.should_abort_batch()
    
    def test_service_coordination(self, orchestrator, sample_video):
        """Test proper coordination between services"""
        call_order = []
        
        def track_service_calls(service_name):
            def wrapper(*args, **kwargs):
                call_order.append(service_name)
                return Mock()  # Return appropriate mock result
            return wrapper
        
        with patch.object(orchestrator.ffmpeg_service, 'extract_streams', 
                         track_service_calls('ffmpeg')), \
             patch.object(orchestrator.scene_service, 'analyze_video_scenes', 
                         track_service_calls('scene_detection')), \
             patch.object(orchestrator.gpu_pipeline, 'process_scene', 
                         track_service_calls('gpu_pipeline')), \
             patch.object(orchestrator.cpu_pipeline, 'process_scene', 
                         track_service_calls('cpu_pipeline')):
            
            orchestrator.process_video_with_progress(sample_video, Mock())
            
            # Verify correct service call order
            expected_order = ['ffmpeg', 'scene_detection', 'gpu_pipeline', 'cpu_pipeline']
            assert call_order[:4] == expected_order
```

## Performance Testing

### Processing Speed Tests
```python
# tests/performance/test_processing_speed.py
import pytest
import time
from pathlib import Path
from services.orchestrator import VideoProcessingOrchestrator

class TestProcessingPerformance:
    
    @pytest.fixture
    def performance_config(self):
        return {
            'performance': {
                'target_processing_time': 600,  # 10 minutes
                'max_processing_time': 1800     # 30 minutes
            }
        }
    
    @pytest.mark.performance
    def test_single_video_processing_speed(self, performance_config, sample_video):
        """Test single video processing meets performance targets"""
        orchestrator = VideoProcessingOrchestrator(performance_config)
        
        start_time = time.time()
        result = orchestrator.process_video_with_progress(sample_video, Mock())
        end_time = time.time()
        
        processing_time_seconds = end_time - start_time
        processing_time_minutes = processing_time_seconds / 60
        
        # Performance assertions
        assert processing_time_minutes < performance_config['performance']['target_processing_time'] / 60
        assert result.success, f"Processing failed in {processing_time_minutes:.2f} minutes"
        
        # Log performance metrics
        print(f"Performance: {processing_time_minutes:.2f} minutes for {sample_video.name}")
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_batch_processing_performance(self, performance_config, tmp_path):
        """Test batch processing performance and scalability"""
        # Create multiple test videos
        video_files = []
        for i in range(10):
            video_file = tmp_path / f"test_video_{i}.mp4"
            video_file.write_bytes(b"fake video content")
            video_files.append(video_file)
        
        orchestrator = VideoProcessingOrchestrator(performance_config)
        
        start_time = time.time()
        results = []
        for video in video_files:
            result = orchestrator.process_video_with_progress(video, Mock())
            results.append(result)
        end_time = time.time()
        
        total_time_minutes = (end_time - start_time) / 60
        avg_time_per_video = total_time_minutes / len(video_files)
        
        # Performance assertions
        assert avg_time_per_video < performance_config['performance']['target_processing_time'] / 60
        assert sum(r.success for r in results) >= len(video_files) * 0.95  # 95% success rate
        
        print(f"Batch Performance: {total_time_minutes:.2f} minutes total, "
              f"{avg_time_per_video:.2f} minutes per video average")
```

### Memory Usage Tests
```python
# tests/performance/test_memory_usage.py
import pytest
import psutil
import gc
from services.orchestrator import VideoProcessingOrchestrator

class TestMemoryUsage:
    
    def get_memory_usage(self):
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    @pytest.mark.performance
    def test_memory_usage_single_video(self, sample_video):
        """Test memory usage for single video processing"""
        orchestrator = VideoProcessingOrchestrator({})
        
        initial_memory = self.get_memory_usage()
        
        # Process video
        result = orchestrator.process_video_with_progress(sample_video, Mock())
        
        peak_memory = self.get_memory_usage()
        
        # Force garbage collection
        gc.collect()
        
        final_memory = self.get_memory_usage()
        memory_increase = peak_memory - initial_memory
        memory_leak = final_memory - initial_memory
        
        # Memory assertions
        assert memory_increase < 2000, f"Memory usage spike too high: {memory_increase:.2f} MB"
        assert memory_leak < 100, f"Possible memory leak: {memory_leak:.2f} MB"
        
        print(f"Memory: Initial={initial_memory:.2f}MB, Peak={peak_memory:.2f}MB, "
              f"Final={final_memory:.2f}MB, Leak={memory_leak:.2f}MB")
    
    @pytest.mark.performance
    def test_memory_usage_batch_processing(self, tmp_path):
        """Test memory doesn't accumulate during batch processing"""
        # Create multiple test videos
        video_files = []
        for i in range(5):
            video_file = tmp_path / f"test_video_{i}.mp4"
            video_file.write_bytes(b"fake video content")
            video_files.append(video_file)
        
        orchestrator = VideoProcessingOrchestrator({})
        initial_memory = self.get_memory_usage()
        memory_readings = [initial_memory]
        
        for video in video_files:
            result = orchestrator.process_video_with_progress(video, Mock())
            memory_readings.append(self.get_memory_usage())
        
        # Check memory doesn't continuously increase
        memory_growth = memory_readings[-1] - memory_readings[0]
        max_memory = max(memory_readings)
        
        assert memory_growth < 500, f"Excessive memory growth: {memory_growth:.2f} MB"
        assert max_memory < initial_memory + 3000, f"Peak memory too high: {max_memory:.2f} MB"
```

## End-to-End Testing

### Complete Pipeline Test
```python
# tests/e2e/test_complete_pipeline.py
import pytest
from pathlib import Path
from cli.video_processor import VideoProcessorCLI
from utils.config_loader import ConfigLoader

class TestCompletePipeline:
    
    @pytest.fixture
    def e2e_test_setup(self, tmp_path):
        """Set up complete test environment"""
        # Create directory structure
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        build_dir = tmp_path / "build"
        
        for directory in [input_dir, output_dir, build_dir]:
            directory.mkdir(parents=True)
        
        # Create test video
        test_video = input_dir / "test_video.mp4"
        test_video.write_bytes(b"fake video content for testing")
        
        # Create test configuration
        test_config = {
            'paths': {
                'input_dir': str(input_dir),
                'output_dir': str(output_dir),
                'build_dir': str(build_dir)
            },
            'performance': {'target_processing_time': 60},
            'error_handling': {'circuit_breaker_threshold': 1}
        }
        
        return {
            'input_dir': input_dir,
            'output_dir': output_dir,
            'config': test_config,
            'test_video': test_video
        }
    
    @pytest.mark.e2e
    def test_complete_video_processing_workflow(self, e2e_test_setup):
        """Test complete workflow from CLI to knowledge base generation"""
        setup = e2e_test_setup
        
        # Initialize CLI processor
        processor = VideoProcessorCLI(
            input_dir=str(setup['input_dir']),
            output_dir=str(setup['output_dir']),
            config=setup['config'],
            dry_run=False
        )
        
        # Mock user input for CLI interaction
        with patch('builtins.input', return_value=''):  # Simulate ENTER key
            processor.run()
        
        # Verify results
        output_files = list(setup['output_dir'].glob('*.md'))
        assert len(output_files) > 0, "No output files generated"
        
        # Verify output file content
        output_file = output_files[0]
        content = output_file.read_text()
        assert 'Video Knowledge Base' in content
        assert 'Scene-by-Scene Analysis' in content
        
        # Verify build artifacts were created and cleaned up
        build_artifacts = list(setup['build_dir'].glob('**/*'))
        # Build artifacts should exist during processing but be cleaned up after
    
    @pytest.mark.e2e
    def test_error_handling_end_to_end(self, e2e_test_setup):
        """Test end-to-end error handling and recovery"""
        setup = e2e_test_setup
        
        # Create corrupted video file
        corrupted_video = setup['input_dir'] / "corrupted_video.mp4"
        corrupted_video.write_bytes(b"not a real video file")
        
        processor = VideoProcessorCLI(
            input_dir=str(setup['input_dir']),
            output_dir=str(setup['output_dir']),
            config=setup['config'],
            dry_run=False
        )
        
        # Process should handle errors gracefully
        with patch('builtins.input', return_value=''):
            processor.run()
        
        # Should have error logs but not crash
        # Some videos might succeed, others fail
        output_files = list(setup['output_dir'].glob('*.md'))
        # At least some processing should succeed (the valid test video)
```

## Quality Gates and Release Criteria

### Pre-Commit Quality Gates
```python
# scripts/quality_gates.py - Pre-commit quality checks
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run command and return success status"""
    print(f"🔍 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} passed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(e.stdout)
        print(e.stderr)
        return False

def main():
    """Run all quality gates"""
    gates = [
        ("black --check src/", "Code formatting check"),
        ("flake8 src/", "Linting check"),
        ("mypy src/", "Type checking"),
        ("pytest tests/unit/ -v", "Unit tests"),
        ("pytest tests/integration/ -v", "Integration tests"),
        ("python scripts/security_scan.py", "Security scan")
    ]
    
    print("🚀 Running Quality Gates")
    print("=" * 50)
    
    all_passed = True
    for cmd, description in gates:
        if not run_command(cmd, description):
            all_passed = False
    
    print("=" * 50)
    if all_passed:
        print("✅ All quality gates passed")
        return 0
    else:
        print("❌ Quality gate failures detected")
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

### Release Validation Checklist
```yaml
# Release criteria configuration
release_criteria:
  code_quality:
    unit_test_coverage: 80      # Minimum 80% code coverage
    integration_tests: 100      # All integration tests must pass
    linting_score: 10          # Perfect flake8 score
    type_coverage: 90          # 90% type annotation coverage
    
  performance:
    avg_processing_time: 600    # Max 10 minutes per video
    memory_usage: 16           # Max 16GB memory usage
    success_rate: 95           # Min 95% processing success rate
    
  security:
    vulnerability_scan: pass   # No critical vulnerabilities
    dependency_audit: pass     # All dependencies up to date
    
  documentation:
    api_docs: complete         # All public APIs documented
    user_guides: complete      # Complete user documentation
    runbooks: complete         # Operational runbooks ready
```

This comprehensive quality assurance and testing documentation provides systematic procedures for ensuring the reliability, performance, and maintainability of the alexdev-video-summarizer system through all phases of development and deployment.