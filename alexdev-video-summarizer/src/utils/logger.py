"""
Logging utilities for video processing pipeline.

Provides structured logging with component-specific configuration.
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Dict, Any
import sys


def setup_logging(log_level: str = 'INFO', config: Dict[str, Any] = None):
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        config: Optional logging configuration from config file
    """
    
    # Create logs directory
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler for main log
    main_log_file = logs_dir / 'processing.log'
    file_handler = logging.handlers.RotatingFileHandler(
        main_log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=3
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)
    
    # Error log file
    error_log_file = logs_dir / 'errors.log'
    error_handler = logging.handlers.RotatingFileHandler(
        error_log_file,
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=3
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(error_handler)
    
    # Configure component-specific loggers if config provided
    if config and 'logging' in config:
        _configure_component_loggers(config['logging'])
    
    logging.info(f"Logging initialized - Level: {log_level}")


def _configure_component_loggers(logging_config: Dict[str, Any]):
    """Configure component-specific logging levels"""
    
    services_config = logging_config.get('services', {})
    
    for service_name, service_level in services_config.items():
        logger = logging.getLogger(f'services.{service_name}')
        logger.setLevel(getattr(logging, service_level.upper()))


def get_logger(name: str) -> logging.Logger:
    """
    Get logger for specific component
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)