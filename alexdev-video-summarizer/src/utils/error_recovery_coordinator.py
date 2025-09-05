"""
Error Recovery Coordinator

Orchestrates comprehensive error recovery across all processing stages.
Implements graduated recovery strategies with diagnostic reporting.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
import time

from utils.error_handlers import (
    FFmpegErrorHandler, GPUErrorHandler, SceneDetectionErrorHandler,
    ErrorLogger, ErrorPatternAnalyzer, ErrorRecoveryResult, 
    RecoveryAction, ErrorType
)
from utils.circuit_breaker import CircuitBreaker
from utils.logger import get_logger

logger = get_logger(__name__)


class ProcessingStage(Enum):
    """Processing stage identification"""
    INITIALIZATION = "initialization"
    FFMPEG_EXTRACTION = "ffmpeg_extraction"
    SCENE_DETECTION = "scene_detection"
    SCENE_SPLITTING = "scene_splitting"
    GPU_PROCESSING = "gpu_processing"
    CPU_PROCESSING = "cpu_processing"
    KNOWLEDGE_GENERATION = "knowledge_generation"
    CLEANUP = "cleanup"


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"           # Warning, continue processing
    MEDIUM = "medium"     # Recoverable error, attempt recovery
    HIGH = "high"         # Skip current item, continue batch
    CRITICAL = "critical" # Abort batch processing


class ProcessingErrorContext:
    """Context information for error processing"""
    
    def __init__(self, 
                 video_path: Path,
                 stage: ProcessingStage,
                 tool_name: Optional[str] = None,
                 scene_id: Optional[str] = None):
        self.video_path = video_path
        self.stage = stage
        self.tool_name = tool_name
        self.scene_id = scene_id
        self.error_history: List[Dict[str, Any]] = []
        self.recovery_attempts: List[ErrorRecoveryResult] = []
        self.start_time = time.time()
    
    def add_error(self, error: Exception, context: Dict[str, Any] = None):
        """Add error to context history"""
        self.error_history.append({
            'timestamp': time.time(),
            'error': str(error),
            'error_type': type(error).__name__,
            'context': context or {}
        })
    
    def add_recovery_attempt(self, result: ErrorRecoveryResult):
        """Add recovery attempt to history"""
        self.recovery_attempts.append(result)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get error context summary"""
        return {
            'video_path': str(self.video_path),
            'video_name': self.video_path.name,
            'stage': self.stage.value,
            'tool_name': self.tool_name,
            'scene_id': self.scene_id,
            'error_count': len(self.error_history),
            'recovery_attempts': len(self.recovery_attempts),
            'processing_time': time.time() - self.start_time,
            'last_error': self.error_history[-1] if self.error_history else None
        }


class ErrorRecoveryCoordinator:
    """Coordinates error recovery across all processing stages"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.error_config = config.get('error_handling', {})
        
        # Initialize error handlers
        self.ffmpeg_handler = FFmpegErrorHandler(config)
        self.gpu_handler = GPUErrorHandler(config)
        self.scene_handler = SceneDetectionErrorHandler(config)
        
        # Initialize logging and analysis
        self.error_logger = ErrorLogger()
        self.pattern_analyzer = ErrorPatternAnalyzer(self.error_logger)
        
        # Circuit breaker
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=self.error_config.get('circuit_breaker_threshold', 3)
        )
        
        # Recovery settings
        self.max_recovery_attempts = self.error_config.get('max_recovery_attempts', 3)
        self.enable_progressive_degradation = self.error_config.get('enable_progressive_degradation', True)
        
        logger.info("Error Recovery Coordinator initialized")
    
    def handle_processing_error(self, 
                              error: Exception, 
                              context: ProcessingErrorContext,
                              progress_callback: Optional[Callable] = None) -> ErrorRecoveryResult:
        """
        Main error handling entry point
        
        Args:
            error: The exception that occurred
            context: Processing context information
            progress_callback: Optional callback for progress updates
            
        Returns:
            ErrorRecoveryResult indicating recovery action
        """
        logger.error(f"Processing error in {context.stage.value} for {context.video_path.name}: {error}")
        
        # Add error to context
        context.add_error(error)
        
        # Report error to progress callback
        if progress_callback:
            progress_callback('error', {
                'stage': context.stage.value,
                'error': str(error),
                'video': context.video_path.name,
                'tool': context.tool_name,
                'scene': context.scene_id
            })
        
        # Attempt recovery based on stage and error type
        recovery_result = self._attempt_recovery(error, context)
        context.add_recovery_attempt(recovery_result)
        
        # Log the error with full context
        self._log_error(error, context, recovery_result)
        
        # Update circuit breaker based on result
        if recovery_result.success:
            self.circuit_breaker.record_success()
        else:
            self.circuit_breaker.record_failure()
            
        # Check if batch should be aborted
        if self.circuit_breaker.should_trip() and recovery_result.action != RecoveryAction.ABORT_BATCH:
            logger.critical("Circuit breaker tripped - recommending batch abort")
            recovery_result = ErrorRecoveryResult(
                False, RecoveryAction.ABORT_BATCH,
                "Circuit breaker activated due to consecutive failures"
            )
        
        return recovery_result
    
    def _attempt_recovery(self, error: Exception, context: ProcessingErrorContext) -> ErrorRecoveryResult:
        """Attempt error recovery based on stage and error type"""
        
        # Check if max recovery attempts exceeded
        if len(context.recovery_attempts) >= self.max_recovery_attempts:
            return ErrorRecoveryResult(
                False, RecoveryAction.SKIP_VIDEO,
                f"Maximum recovery attempts ({self.max_recovery_attempts}) exceeded"
            )
        
        # Stage-specific error handling
        if context.stage == ProcessingStage.FFMPEG_EXTRACTION:
            return self.ffmpeg_handler.handle_ffmpeg_error(context.video_path, error)
        
        elif context.stage == ProcessingStage.SCENE_DETECTION:
            return self.scene_handler.handle_scene_error(context.video_path, error)
        
        elif context.stage in [ProcessingStage.GPU_PROCESSING, ProcessingStage.CPU_PROCESSING]:
            return self.gpu_handler.handle_gpu_error(context.tool_name or 'unknown', error, context)
        
        elif context.stage == ProcessingStage.SCENE_SPLITTING:
            return self._handle_scene_splitting_error(error, context)
        
        elif context.stage == ProcessingStage.KNOWLEDGE_GENERATION:
            return self._handle_knowledge_generation_error(error, context)
        
        else:
            # Generic error handling
            return self._handle_generic_error(error, context)
    
    def _handle_scene_splitting_error(self, error: Exception, context: ProcessingErrorContext) -> ErrorRecoveryResult:
        """Handle scene splitting errors"""
        error_str = str(error).lower()
        
        if 'disk' in error_str or 'space' in error_str:
            return ErrorRecoveryResult(
                False, RecoveryAction.ABORT_BATCH,
                "Insufficient disk space for scene splitting"
            )
        else:
            return ErrorRecoveryResult(
                True, RecoveryAction.GRACEFUL_DEGRADE,
                "Scene splitting failed, continuing with single scene processing",
                {'skip_scene_splitting': True}
            )
    
    def _handle_knowledge_generation_error(self, error: Exception, context: ProcessingErrorContext) -> ErrorRecoveryResult:
        """Handle knowledge generation errors"""
        return ErrorRecoveryResult(
            True, RecoveryAction.GRACEFUL_DEGRADE,
            "Knowledge generation failed, saving raw analysis data",
            {'save_raw_data': True}
        )
    
    def _handle_generic_error(self, error: Exception, context: ProcessingErrorContext) -> ErrorRecoveryResult:
        """Handle generic processing errors"""
        error_str = str(error).lower()
        
        # Check for common patterns
        if any(keyword in error_str for keyword in ['memory', 'oom']):
            return ErrorRecoveryResult(
                True, RecoveryAction.RETRY,
                "Memory issue detected, attempting cleanup and retry",
                {'memory_cleanup': True}
            )
        elif 'permission' in error_str:
            return ErrorRecoveryResult(
                False, RecoveryAction.SKIP_VIDEO,
                f"Permission error, skipping video: {error}"
            )
        else:
            return ErrorRecoveryResult(
                False, RecoveryAction.SKIP_VIDEO,
                f"Unrecoverable error: {error}"
            )
    
    def _log_error(self, error: Exception, context: ProcessingErrorContext, recovery: ErrorRecoveryResult):
        """Log error with comprehensive context"""
        error_context = {
            'stage': context.stage.value,
            'tool': context.tool_name,
            'scene': context.scene_id,
            'error': error,
            'error_type': self._classify_error_type(error, context),
            'recovery_attempts': [
                {
                    'success': attempt.success,
                    'action': attempt.action.value,
                    'message': attempt.message,
                    'timestamp': attempt.timestamp.isoformat()
                }
                for attempt in context.recovery_attempts
            ],
            'action': recovery.action.value,
            'context': context.get_summary()
        }
        
        self.error_logger.log_video_failure(context.video_path, error_context)
    
    def _classify_error_type(self, error: Exception, context: ProcessingErrorContext) -> str:
        """Classify error type for analysis"""
        error_str = str(error).lower()
        
        # GPU/CUDA errors
        if any(keyword in error_str for keyword in ['cuda', 'gpu', 'memory', 'oom']):
            return 'gpu_error'
        # FFmpeg errors
        elif any(keyword in error_str for keyword in ['ffmpeg', 'codec', 'format']):
            return 'ffmpeg_error'
        # Scene detection errors
        elif 'scene' in error_str:
            return 'scene_error'
        # File system errors
        elif any(keyword in error_str for keyword in ['permission', 'access', 'file']):
            return 'filesystem_error'
        else:
            return 'unknown_error'
    
    def get_batch_error_report(self, batch_results: List[Dict[str, Any]]) -> str:
        """Generate comprehensive batch error report"""
        return self.pattern_analyzer.generate_batch_report(batch_results)
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics"""
        circuit_stats = self.circuit_breaker.get_statistics()
        error_summary = self.error_logger.get_error_summary(24)  # Last 24 hours
        error_analysis = self.pattern_analyzer.analyze_batch_errors(24)
        
        return {
            'circuit_breaker': circuit_stats,
            'recent_errors': error_summary,
            'error_analysis': error_analysis,
            'recovery_config': {
                'max_recovery_attempts': self.max_recovery_attempts,
                'progressive_degradation': self.enable_progressive_degradation,
                'circuit_breaker_threshold': self.circuit_breaker.failure_threshold
            }
        }
    
    def should_abort_batch(self) -> bool:
        """Check if batch processing should be aborted"""
        return self.circuit_breaker.should_trip()
    
    def reset_circuit_breaker(self):
        """Reset circuit breaker for new batch"""
        self.circuit_breaker.reset()
        logger.info("Circuit breaker reset for new batch")
    
    def get_failure_analysis(self) -> Dict[str, str]:
        """Get failure analysis and recommendations"""
        return self.circuit_breaker.get_failure_analysis()
    
    def create_processing_context(self, 
                                video_path: Path, 
                                stage: ProcessingStage,
                                tool_name: Optional[str] = None,
                                scene_id: Optional[str] = None) -> ProcessingErrorContext:
        """Create processing error context"""
        return ProcessingErrorContext(video_path, stage, tool_name, scene_id)