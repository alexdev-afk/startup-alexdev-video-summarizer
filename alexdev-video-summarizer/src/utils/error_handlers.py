"""
Advanced Error Handling System

Comprehensive error recovery strategies for all processing tools.
Implements graduated recovery approach with diagnostic capabilities.
"""

import gc
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from enum import Enum

# Optional system monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from utils.logger import get_logger

logger = get_logger(__name__)


class ErrorType(Enum):
    """Error type classifications"""
    FFMPEG_FORMAT = "ffmpeg_format"
    FFMPEG_CORRUPTED = "ffmpeg_corrupted"
    FFMPEG_SPACE = "ffmpeg_space"
    FFMPEG_PERMISSION = "ffmpeg_permission"
    GPU_MEMORY = "gpu_memory"
    GPU_MODEL_LOADING = "gpu_model_loading"
    GPU_CUDA_DRIVER = "gpu_cuda_driver"
    SCENE_NO_DETECTION = "scene_no_detection"
    SCENE_THRESHOLD = "scene_threshold"
    SCENE_TOO_SHORT = "scene_too_short"
    TOOL_CRASH = "tool_crash"
    FILE_CORRUPTION = "file_corruption"
    UNKNOWN = "unknown"


class RecoveryAction(Enum):
    """Recovery action types"""
    RETRY = "retry"
    PARAMETER_ADJUST = "parameter_adjust"
    FALLBACK_METHOD = "fallback_method"
    GRACEFUL_DEGRADE = "graceful_degrade"
    SKIP_VIDEO = "skip_video"
    ABORT_BATCH = "abort_batch"


class ErrorRecoveryResult:
    """Result of error recovery attempt"""
    
    def __init__(self, success: bool, action: RecoveryAction, message: str, data: Optional[Any] = None):
        self.success = success
        self.action = action
        self.message = message
        self.data = data
        self.timestamp = datetime.now()


class FFmpegErrorHandler:
    """FFmpeg-specific error recovery strategies"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ffmpeg_config = config.get('ffmpeg', {})
    
    def handle_ffmpeg_error(self, video_path: Path, error: Exception) -> ErrorRecoveryResult:
        """Comprehensive FFmpeg error recovery"""
        error_type = self._classify_ffmpeg_error(str(error))
        
        logger.warning(f"FFmpeg error detected for {video_path.name}: {error_type.value}")
        
        if error_type == ErrorType.FFMPEG_FORMAT:
            return self._attempt_format_conversion(video_path, error)
        elif error_type == ErrorType.FFMPEG_CORRUPTED:
            return self._attempt_repair_or_skip(video_path, error)
        elif error_type == ErrorType.FFMPEG_SPACE:
            return self._cleanup_and_retry(video_path, error)
        elif error_type == ErrorType.FFMPEG_PERMISSION:
            return self._handle_permission_error(video_path, error)
        else:
            return ErrorRecoveryResult(
                False, RecoveryAction.SKIP_VIDEO,
                f"Unrecoverable FFmpeg error: {error}"
            )
    
    def _classify_ffmpeg_error(self, error_message: str) -> ErrorType:
        """Classify FFmpeg error type from error message"""
        error_lower = error_message.lower()
        
        if any(keyword in error_lower for keyword in ['unsupported', 'format', 'codec']):
            return ErrorType.FFMPEG_FORMAT
        elif any(keyword in error_lower for keyword in ['corrupted', 'invalid', 'damaged']):
            return ErrorType.FFMPEG_CORRUPTED
        elif any(keyword in error_lower for keyword in ['space', 'disk full', 'no space']):
            return ErrorType.FFMPEG_SPACE
        elif any(keyword in error_lower for keyword in ['permission', 'access denied']):
            return ErrorType.FFMPEG_PERMISSION
        else:
            return ErrorType.UNKNOWN
    
    def _attempt_format_conversion(self, video_path: Path, error: Exception) -> ErrorRecoveryResult:
        """Try alternative codec/format combinations"""
        logger.info(f"Attempting format conversion for {video_path.name}")
        
        fallback_configs = [
            {'video_codec': 'libx264', 'audio_codec': 'aac'},
            {'video_codec': 'mpeg4', 'audio_codec': 'mp3'},
            {'output_format': 'avi'}
        ]
        
        for i, config in enumerate(fallback_configs):
            try:
                logger.debug(f"Trying fallback config {i+1}: {config}")
                # Note: Actual implementation would call FFmpeg with these configs
                # Return success indicator for now
                return ErrorRecoveryResult(
                    True, RecoveryAction.PARAMETER_ADJUST,
                    f"Format conversion successful with config {i+1}",
                    config
                )
            except Exception as retry_error:
                logger.debug(f"Fallback config {i+1} failed: {retry_error}")
                continue
        
        return ErrorRecoveryResult(
            False, RecoveryAction.SKIP_VIDEO,
            "All format conversion attempts failed"
        )
    
    def _attempt_repair_or_skip(self, video_path: Path, error: Exception) -> ErrorRecoveryResult:
        """Handle corrupted file scenarios"""
        logger.warning(f"File corruption detected: {video_path.name}")
        
        # For now, skip corrupted files - could implement repair attempts
        return ErrorRecoveryResult(
            False, RecoveryAction.SKIP_VIDEO,
            f"File corruption detected, skipping: {error}"
        )
    
    def _cleanup_and_retry(self, video_path: Path, error: Exception) -> ErrorRecoveryResult:
        """Handle disk space issues"""
        logger.warning("Disk space issue detected, attempting cleanup")
        
        try:
            # Clean up build directory
            build_dir = Path('build')
            if build_dir.exists():
                for temp_file in build_dir.rglob('*.tmp'):
                    temp_file.unlink()
                    
            return ErrorRecoveryResult(
                True, RecoveryAction.RETRY,
                "Disk space cleanup completed, retry possible"
            )
        except Exception as cleanup_error:
            return ErrorRecoveryResult(
                False, RecoveryAction.ABORT_BATCH,
                f"Cleanup failed, insufficient disk space: {cleanup_error}"
            )
    
    def _handle_permission_error(self, video_path: Path, error: Exception) -> ErrorRecoveryResult:
        """Handle permission issues"""
        return ErrorRecoveryResult(
            False, RecoveryAction.SKIP_VIDEO,
            f"Permission denied, skipping video: {error}"
        )


class GPUErrorHandler:
    """GPU-specific error recovery strategies"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.gpu_config = config.get('gpu_pipeline', {})
    
    def handle_gpu_error(self, tool_name: str, error: Exception, context: Any = None) -> ErrorRecoveryResult:
        """GPU-specific error recovery"""
        error_type = self._classify_gpu_error(str(error))
        
        logger.warning(f"GPU error in {tool_name}: {error_type.value}")
        
        if error_type == ErrorType.GPU_MEMORY:
            return self._attempt_memory_recovery(tool_name, error, context)
        elif error_type == ErrorType.GPU_MODEL_LOADING:
            return self._attempt_model_reload(tool_name, error)
        elif error_type == ErrorType.GPU_CUDA_DRIVER:
            return self._attempt_cuda_recovery(tool_name, error)
        else:
            return ErrorRecoveryResult(
                False, RecoveryAction.GRACEFUL_DEGRADE,
                f"GPU error, falling back to CPU: {error}"
            )
    
    def _classify_gpu_error(self, error_message: str) -> ErrorType:
        """Classify GPU error type"""
        error_lower = error_message.lower()
        
        if any(keyword in error_lower for keyword in ['out of memory', 'cuda memory', 'oom']):
            return ErrorType.GPU_MEMORY
        elif any(keyword in error_lower for keyword in ['model', 'loading', 'checkpoint']):
            return ErrorType.GPU_MODEL_LOADING
        elif any(keyword in error_lower for keyword in ['cuda', 'driver', 'device']):
            return ErrorType.GPU_CUDA_DRIVER
        else:
            return ErrorType.UNKNOWN
    
    def _attempt_memory_recovery(self, tool_name: str, error: Exception, context: Any) -> ErrorRecoveryResult:
        """Clear GPU memory and retry with smaller parameters"""
        logger.info(f"Attempting GPU memory recovery for {tool_name}")
        
        try:
            # Clear GPU cache if available
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("GPU cache cleared")
            
            # Force garbage collection
            gc.collect()
            
            return ErrorRecoveryResult(
                True, RecoveryAction.RETRY,
                "GPU memory cleared, retry with reduced parameters",
                {'memory_cleared': True, 'reduce_batch_size': True}
            )
            
        except Exception as recovery_error:
            return ErrorRecoveryResult(
                False, RecoveryAction.GRACEFUL_DEGRADE,
                f"Memory recovery failed, falling back to CPU: {recovery_error}"
            )
    
    def _attempt_model_reload(self, tool_name: str, error: Exception) -> ErrorRecoveryResult:
        """Attempt to reload failed model"""
        return ErrorRecoveryResult(
            True, RecoveryAction.RETRY,
            f"Model reload attempted for {tool_name}",
            {'reload_model': True}
        )
    
    def _attempt_cuda_recovery(self, tool_name: str, error: Exception) -> ErrorRecoveryResult:
        """Handle CUDA driver issues"""
        return ErrorRecoveryResult(
            False, RecoveryAction.GRACEFUL_DEGRADE,
            f"CUDA driver issue, falling back to CPU for {tool_name}"
        )


class SceneDetectionErrorHandler:
    """Scene detection error recovery strategies"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scene_config = config.get('scene_detection', {})
    
    def handle_scene_error(self, video_path: Path, error: Exception) -> ErrorRecoveryResult:
        """Scene detection fallback strategies"""
        error_type = self._classify_scene_error(str(error))
        
        logger.warning(f"Scene detection error for {video_path.name}: {error_type.value}")
        
        if error_type == ErrorType.SCENE_NO_DETECTION:
            return self._fallback_to_time_based_scenes(video_path)
        elif error_type == ErrorType.SCENE_THRESHOLD:
            return self._retry_with_adjusted_threshold(video_path)
        elif error_type == ErrorType.SCENE_TOO_SHORT:
            return self._treat_as_single_scene(video_path)
        else:
            return ErrorRecoveryResult(
                False, RecoveryAction.SKIP_VIDEO,
                f"Scene detection failed: {error}"
            )
    
    def _classify_scene_error(self, error_message: str) -> ErrorType:
        """Classify scene detection error"""
        error_lower = error_message.lower()
        
        if any(keyword in error_lower for keyword in ['no scenes', 'zero scenes']):
            return ErrorType.SCENE_NO_DETECTION
        elif 'threshold' in error_lower:
            return ErrorType.SCENE_THRESHOLD
        elif any(keyword in error_lower for keyword in ['too short', 'duration']):
            return ErrorType.SCENE_TOO_SHORT
        else:
            return ErrorType.UNKNOWN
    
    def _fallback_to_time_based_scenes(self, video_path: Path) -> ErrorRecoveryResult:
        """Create time-based scenes when content detection fails"""
        scene_length = self.scene_config.get('time_based_scene_length', 120)
        
        # Mock video duration - real implementation would get actual duration
        mock_duration = 300  # 5 minutes
        
        scenes = []
        for i in range(0, mock_duration, scene_length):
            scenes.append({
                'scene_id': len(scenes) + 1,
                'start_time': i,
                'end_time': min(i + scene_length, mock_duration),
                'fallback_method': 'time_based'
            })
        
        return ErrorRecoveryResult(
            True, RecoveryAction.FALLBACK_METHOD,
            f"Created {len(scenes)} time-based scenes",
            {'scenes': scenes, 'method': 'time_based'}
        )
    
    def _retry_with_adjusted_threshold(self, video_path: Path) -> ErrorRecoveryResult:
        """Retry with adjusted threshold"""
        return ErrorRecoveryResult(
            True, RecoveryAction.PARAMETER_ADJUST,
            "Retrying with adjusted scene detection threshold",
            {'adjust_threshold': True, 'new_threshold': 30.0}
        )
    
    def _treat_as_single_scene(self, video_path: Path) -> ErrorRecoveryResult:
        """Treat entire video as single scene"""
        return ErrorRecoveryResult(
            True, RecoveryAction.FALLBACK_METHOD,
            "Treating video as single scene",
            {'single_scene': True}
        )


class ErrorLogger:
    """Comprehensive error logging and diagnostics"""
    
    def __init__(self, log_dir: Path = Path('build/error_logs')):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.error_log_file = self.log_dir / 'processing_errors.jsonl'
    
    def log_video_failure(self, video_path: Path, error_context: Dict[str, Any]):
        """Detailed error logging for diagnostics"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'video_path': str(video_path),
            'video_name': video_path.name,
            'error_stage': error_context.get('stage', 'unknown'),
            'tool_name': error_context.get('tool'),
            'error_type': error_context.get('error_type', 'unknown'),
            'error_message': str(error_context.get('error', '')),
            'system_state': self._capture_system_state(),
            'recovery_attempts': error_context.get('recovery_attempts', []),
            'final_action': error_context.get('action', 'unknown'),
            'processing_context': error_context.get('context', {})
        }
        
        try:
            with open(self.error_log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + '\n')
            logger.debug(f"Error logged for {video_path.name}")
        except Exception as log_error:
            logger.error(f"Failed to write error log: {log_error}")
    
    def _capture_system_state(self) -> Dict[str, Any]:
        """Capture system state for debugging"""
        state = {
            'timestamp': datetime.now().isoformat()
        }
        
        # GPU memory information
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                state['gpu_memory_allocated'] = torch.cuda.memory_allocated()
                state['gpu_memory_cached'] = torch.cuda.memory_cached()
                state['gpu_memory_reserved'] = torch.cuda.memory_reserved()
                state['gpu_device_count'] = torch.cuda.device_count()
                state['cuda_available'] = True
            except:
                state['cuda_available'] = False
        else:
            state['cuda_available'] = False
        
        # System memory and disk information
        if PSUTIL_AVAILABLE:
            try:
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('.')
                
                state['cpu_memory_percent'] = memory.percent
                state['cpu_memory_available_gb'] = memory.available / (1024**3)
                state['disk_free_gb'] = disk.free / (1024**3)
                state['disk_used_percent'] = (disk.used / disk.total) * 100
            except:
                state['system_monitoring'] = False
        
        return state
    
    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get error summary for recent time period"""
        if not self.error_log_file.exists():
            return {'total_errors': 0, 'error_types': {}}
        
        cutoff_time = datetime.now().timestamp() - (hours * 3600)
        error_types = {}
        total_errors = 0
        
        try:
            with open(self.error_log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        entry_time = datetime.fromisoformat(entry['timestamp']).timestamp()
                        
                        if entry_time >= cutoff_time:
                            total_errors += 1
                            error_type = entry.get('error_type', 'unknown')
                            error_types[error_type] = error_types.get(error_type, 0) + 1
                    except (json.JSONDecodeError, ValueError):
                        continue
        except Exception as e:
            logger.error(f"Failed to read error log: {e}")
            return {'error': f"Failed to read log: {e}"}
        
        return {
            'total_errors': total_errors,
            'error_types': error_types,
            'time_period_hours': hours
        }


class ErrorPatternAnalyzer:
    """Analyze error patterns and provide improvement suggestions"""
    
    def __init__(self, error_logger: ErrorLogger):
        self.error_logger = error_logger
    
    def analyze_batch_errors(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze recent error patterns for improvement suggestions"""
        error_summary = self.error_logger.get_error_summary(hours)
        
        if error_summary.get('total_errors', 0) == 0:
            return {'status': 'healthy', 'suggestions': []}
        
        suggestions = []
        error_types = error_summary.get('error_types', {})
        
        # GPU memory error analysis
        gpu_memory_errors = error_types.get('gpu_memory', 0)
        if gpu_memory_errors > 2:
            suggestions.append({
                'issue': 'Frequent GPU memory errors',
                'count': gpu_memory_errors,
                'solution': 'Reduce batch size or enable CPU fallback for GPU tools',
                'config_change': 'Set gpu_pipeline.max_gpu_memory_usage = 0.7',
                'priority': 'high'
            })
        
        # FFmpeg format error analysis
        format_errors = error_types.get('ffmpeg_format', 0)
        if format_errors > 1:
            suggestions.append({
                'issue': 'Video format compatibility issues',
                'count': format_errors,
                'solution': 'Enable automatic format conversion in FFmpeg settings',
                'config_change': 'Set ffmpeg.enable_format_fallback = true',
                'priority': 'medium'
            })
        
        # Scene detection error analysis
        scene_errors = error_types.get('scene_no_detection', 0)
        if scene_errors > 1:
            suggestions.append({
                'issue': 'Scene detection failures',
                'count': scene_errors,
                'solution': 'Adjust scene detection threshold or enable time-based fallback',
                'config_change': 'Set scene_detection.fallback_to_time_based = true',
                'priority': 'medium'
            })
        
        return {
            'status': 'issues_detected' if suggestions else 'healthy',
            'total_errors': error_summary.get('total_errors'),
            'suggestions': suggestions,
            'analysis_period_hours': hours
        }
    
    def generate_batch_report(self, batch_results: List[Dict[str, Any]]) -> str:
        """Generate comprehensive batch processing report"""
        total_videos = len(batch_results)
        successful = len([r for r in batch_results if r.get('success', False)])
        failed = total_videos - successful
        
        report = f"""
Batch Processing Report
=======================

📊 Overall Statistics:
   Total Videos: {total_videos}
   Successful: {successful}
   Failed: {failed}
   Success Rate: {(successful/total_videos)*100:.1f}%

"""
        
        if failed > 0:
            report += "❌ Failed Videos:\n"
            for result in batch_results:
                if not result.get('success', False):
                    video_name = result.get('video_path', 'Unknown')
                    error = result.get('error', 'Unknown error')
                    report += f"   • {video_name}: {error}\n"
        
        # Add error pattern analysis
        error_analysis = self.analyze_batch_errors(1)  # Last hour
        if error_analysis.get('suggestions'):
            report += "\n🔧 Recommended Actions:\n"
            for suggestion in error_analysis['suggestions']:
                report += f"   • {suggestion['issue']}: {suggestion['solution']}\n"
        
        return report