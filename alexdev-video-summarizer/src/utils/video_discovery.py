"""
Video file discovery utilities.

Finds and validates video files in input directories.
"""

import os
from pathlib import Path
from typing import List, Set
import logging

from utils.logger import get_logger

logger = get_logger(__name__)


class VideoDiscovery:
    """Video file discovery and validation"""
    
    # Supported video file extensions
    SUPPORTED_EXTENSIONS = {
        '.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v',
        '.MP4', '.AVI', '.MOV', '.MKV', '.WEBM', '.M4V'
    }
    
    def __init__(self, input_dir: Path):
        """
        Initialize video discovery
        
        Args:
            input_dir: Directory to search for video files
        """
        self.input_dir = Path(input_dir)
        
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
            
        if not self.input_dir.is_dir():
            raise NotADirectoryError(f"Input path is not a directory: {input_dir}")
    
    def find_videos(self) -> List[Path]:
        """
        Find all supported video files in input directory
        
        Returns:
            List of Path objects for discovered video files
        """
        logger.info(f"Searching for videos in: {self.input_dir}")
        
        videos = []
        
        # Search input directory (non-recursive for now)
        for file_path in self.input_dir.iterdir():
            logger.debug(f"Checking file: {file_path.name}, is_file: {file_path.is_file()}, extension: {file_path.suffix}")
            if file_path.is_file() and self._is_supported_video(file_path):
                logger.debug(f"File {file_path.name} has supported extension, validating...")
                if self._validate_video_file(file_path):
                    videos.append(file_path)
                    logger.debug(f"Added valid video: {file_path.name}")
                else:
                    logger.warning(f"Skipping invalid video file: {file_path.name}")
            else:
                logger.debug(f"Skipping {file_path.name}: not a video file or not supported extension")
        
        # Sort by name for consistent processing order
        videos.sort(key=lambda x: x.name.lower())
        
        logger.info(f"Found {len(videos)} valid video files")
        return videos
    
    def _is_supported_video(self, file_path: Path) -> bool:
        """Check if file has supported video extension"""
        return file_path.suffix in self.SUPPORTED_EXTENSIONS
    
    def _validate_video_file(self, file_path: Path) -> bool:
        """
        Basic validation of video file
        
        Args:
            file_path: Path to video file
            
        Returns:
            True if file appears to be a valid video
        """
        try:
            # In development mode, be more lenient for testing
            import sys
            config_path = Path(__file__).parent.parent.parent / 'config' / 'processing.yaml'
            if config_path.exists():
                import yaml
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                if config.get('development', {}).get('mock_ai_services', False):
                    # In mock mode, just check file exists and has extension
                    return file_path.exists() and file_path.stat().st_size > 0
            
            # Check file size (must be > 1KB)
            if file_path.stat().st_size < 1024:
                logger.warning(f"Video file too small: {file_path.name}")
                return False
            
            # Check file is readable
            with open(file_path, 'rb') as f:
                # Read first few bytes to check it's not empty/corrupted
                header = f.read(16)
                if len(header) < 16:
                    logger.warning(f"Video file header too short: {file_path.name}")
                    return False
            
            return True
            
        except (OSError, IOError) as e:
            logger.error(f"Cannot access video file {file_path.name}: {e}")
            return False
    
    def get_video_info(self, video_path: Path) -> dict:
        """
        Get basic information about video file
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video information
        """
        try:
            stat = video_path.stat()
            return {
                'name': video_path.name,
                'size_bytes': stat.st_size,
                'size_mb': stat.st_size / (1024 * 1024),
                'modified_time': stat.st_mtime,
                'extension': video_path.suffix.lower()
            }
        except (OSError, IOError) as e:
            logger.error(f"Cannot get info for video {video_path.name}: {e}")
            return {
                'name': video_path.name,
                'size_bytes': 0,
                'size_mb': 0.0,
                'modified_time': 0,
                'extension': video_path.suffix.lower(),
                'error': str(e)
            }