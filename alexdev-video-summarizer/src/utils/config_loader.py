"""
Configuration loading and validation utilities.

Handles YAML configuration files with validation and schema enforcement.
"""

import yaml
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class ConfigValidationError(Exception):
    """Configuration validation error"""
    message: str
    config_file: str
    missing_keys: list = None


class ConfigLoader:
    """Configuration loader with validation"""
    
    @staticmethod
    def load_config(config_file: str) -> Dict[str, Any]:
        """
        Load and validate YAML configuration file
        
        Args:
            config_file: Path to YAML configuration file
            
        Returns:
            Validated configuration dictionary
            
        Raises:
            ConfigValidationError: If configuration is invalid
        """
        config_path = Path(config_file)
        
        if not config_path.exists():
            raise ConfigValidationError(
                f"Configuration file not found: {config_file}",
                config_file
            )
            
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigValidationError(
                f"Invalid YAML syntax: {str(e)}",
                config_file
            )
            
        # Validate required configuration sections
        ConfigLoader._validate_config(config, config_file)
        
        # Load paths configuration if separate file exists
        paths_config = ConfigLoader._load_paths_config()
        if paths_config:
            config['paths'] = paths_config
            
        return config
    
    @staticmethod
    def _validate_config(config: Dict[str, Any], config_file: str):
        """Validate configuration structure"""
        required_sections = [
            'app',
            'paths', 
            'ffmpeg',
            'scene_detection',
            'gpu_pipeline',
            'cpu_pipeline',
            'error_handling'
        ]
        
        missing_sections = []
        for section in required_sections:
            if section not in config:
                missing_sections.append(section)
                
        if missing_sections:
            raise ConfigValidationError(
                f"Missing required configuration sections: {', '.join(missing_sections)}",
                config_file,
                missing_sections
            )
            
        # Validate critical settings
        ConfigLoader._validate_critical_settings(config, config_file)
        
    @staticmethod
    def _validate_critical_settings(config: Dict[str, Any], config_file: str):
        """Validate critical configuration values"""
        # Validate circuit breaker threshold
        threshold = config.get('error_handling', {}).get('circuit_breaker_threshold')
        if threshold is not None and (threshold < 1 or threshold > 10):
            raise ConfigValidationError(
                "circuit_breaker_threshold must be between 1 and 10",
                config_file
            )
            
        # Validate GPU memory usage
        gpu_memory = config.get('gpu_pipeline', {}).get('max_gpu_memory_usage')
        if gpu_memory is not None and (gpu_memory < 0.1 or gpu_memory > 1.0):
            raise ConfigValidationError(
                "max_gpu_memory_usage must be between 0.1 and 1.0",
                config_file
            )
            
    @staticmethod
    def _load_paths_config() -> Dict[str, Any]:
        """Load separate paths configuration file if it exists"""
        paths_file = Path('config/paths.yaml')
        
        if not paths_file.exists():
            return None
            
        try:
            with open(paths_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except yaml.YAMLError:
            # If paths file is invalid, continue without it
            return None