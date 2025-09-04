"""
Circuit breaker for batch processing reliability.

Implements fail-fast logic to prevent cascading failures during batch video processing.
"""

from typing import Dict, Any
import time

from utils.logger import get_logger

logger = get_logger(__name__)


class CircuitBreaker:
    """Circuit breaker for batch processing"""
    
    def __init__(self, failure_threshold: int = 3):
        """
        Initialize circuit breaker
        
        Args:
            failure_threshold: Number of consecutive failures before tripping
        """
        self.failure_threshold = failure_threshold
        self.consecutive_failures = 0
        self.total_processed = 0
        self.total_failures = 0
        self.last_failure_time = None
        self.is_tripped = False
        
        logger.info(f"Circuit breaker initialized - threshold: {failure_threshold}")
    
    def record_success(self):
        """Record successful processing"""
        self.consecutive_failures = 0
        self.total_processed += 1
        self.is_tripped = False
        
        logger.debug(f"Success recorded - consecutive failures reset to 0")
    
    def record_failure(self):
        """Record processing failure"""
        self.consecutive_failures += 1
        self.total_failures += 1
        self.total_processed += 1
        self.last_failure_time = time.time()
        
        logger.warning(
            f"Failure recorded - consecutive: {self.consecutive_failures}/{self.failure_threshold}"
        )
        
        # Check if circuit breaker should trip
        if self.consecutive_failures >= self.failure_threshold:
            self.trip()
    
    def trip(self):
        """Trip the circuit breaker"""
        self.is_tripped = True
        logger.error(
            f"🚨 CIRCUIT BREAKER TRIPPED - {self.consecutive_failures} consecutive failures"
        )
    
    def should_trip(self) -> bool:
        """Check if circuit breaker should prevent further processing"""
        return self.is_tripped
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        success_rate = 0.0
        if self.total_processed > 0:
            success_count = self.total_processed - self.total_failures
            success_rate = (success_count / self.total_processed) * 100
        
        return {
            'is_tripped': self.is_tripped,
            'consecutive_failures': self.consecutive_failures,
            'failure_threshold': self.failure_threshold,
            'total_processed': self.total_processed,
            'total_failures': self.total_failures,
            'success_rate': success_rate,
            'last_failure_time': self.last_failure_time
        }
    
    def reset(self):
        """Reset circuit breaker to initial state"""
        self.consecutive_failures = 0
        self.total_processed = 0
        self.total_failures = 0
        self.last_failure_time = None
        self.is_tripped = False
        
        logger.info("Circuit breaker reset")
    
    def get_failure_analysis(self) -> Dict[str, str]:
        """Get failure analysis and recommendations"""
        if not self.is_tripped:
            return {'status': 'healthy', 'recommendation': 'Continue processing'}
        
        failure_rate = (self.total_failures / max(self.total_processed, 1)) * 100
        
        if failure_rate > 50:
            return {
                'status': 'critical',
                'recommendation': (
                    'High failure rate detected. Check system resources, '
                    'GPU memory, and input file quality before continuing.'
                )
            }
        elif self.consecutive_failures >= self.failure_threshold:
            return {
                'status': 'pattern_failure',
                'recommendation': (
                    'Consecutive failure pattern detected. Check for systematic '
                    'issues like insufficient GPU memory or corrupted input files.'
                )
            }
        else:
            return {
                'status': 'moderate',
                'recommendation': 'Review failed videos and retry with adjusted settings.'
            }