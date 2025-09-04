"""
FFmpeg Service for video/audio separation and scene processing.

Handles all media file preparation for downstream AI tools.
Based on feature specification: ffmpeg-foundation.md
"""

import os
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import json

from utils.logger import get_logger

logger = get_logger(__name__)


class FFmpegError(Exception):
    """FFmpeg processing error"""
    pass


class FFmpegService:
    """FFmpeg service for video/audio processing"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize FFmpeg service
        
        Args:
            config: Configuration dictionary with FFmpeg settings
        """
        self.config = config
        self.ffmpeg_config = config.get('ffmpeg', {})
        self.paths_config = config.get('paths', {})
        
        # Determine FFmpeg executable path
        self.ffmpeg_path = self._find_ffmpeg_executable()
        
        # Verify FFmpeg is available
        self.verify_ffmpeg_availability()
        
        # Create build directory
        self.build_dir = Path(self.paths_config.get('build_dir', 'build'))
        self.build_dir.mkdir(exist_ok=True)
        
    def _find_ffmpeg_executable(self) -> str:
        """Find FFmpeg executable path"""
        import platform
        
        # Try common Windows paths first
        if platform.system() == 'Windows':
            # WinGet installation path
            winget_path = Path.home() / "AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-8.0-full_build/bin/ffmpeg.exe"
            if winget_path.exists():
                return str(winget_path)
            
            # WinGet Links path
            links_path = Path.home() / "AppData/Local/Microsoft/WinGet/Links/ffmpeg.exe"
            if links_path.exists():
                return str(links_path)
        
        # Default to PATH lookup
        return 'ffmpeg'
    
    def verify_ffmpeg_availability(self):
        """Verify FFmpeg is installed and accessible"""
        # Skip FFmpeg check in development mode
        if self.config.get('development', {}).get('skip_ffmpeg_check', False):
            logger.info("FFmpeg verification skipped (development mode)")
            return
            
        try:
            result = subprocess.run(
                [self.ffmpeg_path, '-version'], 
                capture_output=True, 
                text=True,
                check=True
            )
            logger.info(f"FFmpeg verification successful: {self.ffmpeg_path}")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise FFmpegError(
                f"FFmpeg not found or not working at: {self.ffmpeg_path}. Please check installation."
            ) from e
    
    def extract_streams(self, video_path: Path) -> Tuple[Path, Path]:
        """
        Extract audio and video streams from input video
        
        Args:
            video_path: Path to input video file
            
        Returns:
            Tuple of (audio_path, video_path)
            
        Raises:
            FFmpegError: If extraction fails
        """
        logger.info(f"Extracting streams from: {video_path.name}")
        
        # Create output directory for this video
        video_name = video_path.stem
        output_dir = self.build_dir / video_name
        output_dir.mkdir(exist_ok=True)
        
        # Define output paths
        audio_path = output_dir / "audio.wav"
        processed_video_path = output_dir / "video.mp4"
        
        try:
            # Extract audio stream (optimized for Whisper)
            self._extract_audio_stream(video_path, audio_path)
            
            # Extract video stream (standardized format)
            self._extract_video_stream(video_path, processed_video_path)
            
            # Validate outputs
            self._validate_extraction_outputs(audio_path, processed_video_path)
            
            logger.info(f"Stream extraction complete: {video_name}")
            return audio_path, processed_video_path
            
        except Exception as e:
            # Clean up partial outputs on failure
            self._cleanup_partial_outputs(output_dir)
            raise FFmpegError(f"Stream extraction failed for {video_name}: {str(e)}") from e
    
    def _extract_audio_stream(self, video_path: Path, audio_path: Path):
        """Extract audio stream with Whisper-optimized settings"""
        audio_config = self.ffmpeg_config.get('audio', {})
        
        cmd = [
            self.ffmpeg_path, '-y',  # Overwrite output
            '-i', str(video_path),
            '-vn',  # No video
            '-acodec', audio_config.get('codec', 'pcm_s16le'),
            '-ar', str(audio_config.get('sample_rate', 22050)),
            '-ac', str(audio_config.get('channels', 2)),
            str(audio_path)
        ]
        
        logger.debug(f"Audio extraction command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=300  # 5 minute timeout
            )
            logger.debug(f"Audio extraction successful: {audio_path.name}")
        except subprocess.TimeoutExpired:
            raise FFmpegError("Audio extraction timed out")
        except subprocess.CalledProcessError as e:
            logger.error(f"Audio extraction failed: {e.stderr}")
            raise FFmpegError(f"Audio extraction failed: {e.stderr}")
    
    def _extract_video_stream(self, video_path: Path, video_out_path: Path):
        """Extract video stream with standardized settings"""
        video_config = self.ffmpeg_config.get('video', {})
        
        cmd = [
            self.ffmpeg_path, '-y',  # Overwrite output
            '-i', str(video_path),
            '-an',  # No audio
            '-vcodec', video_config.get('codec', 'libx264'),
        ]
        
        # Add quality settings
        if video_config.get('quality') == 'high':
            cmd.extend(['-crf', '18'])
        
        # Preserve resolution and framerate by default
        if video_config.get('preserve_resolution', True):
            cmd.extend(['-vf', 'scale=-2:-2'])  # Ensure even dimensions
            
        cmd.append(str(video_out_path))
        
        logger.debug(f"Video extraction command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=600  # 10 minute timeout
            )
            logger.debug(f"Video extraction successful: {video_out_path.name}")
        except subprocess.TimeoutExpired:
            raise FFmpegError("Video extraction timed out")
        except subprocess.CalledProcessError as e:
            logger.error(f"Video extraction failed: {e.stderr}")
            raise FFmpegError(f"Video extraction failed: {e.stderr}")
    
    def _validate_extraction_outputs(self, audio_path: Path, video_path: Path):
        """Validate that extraction outputs are valid"""
        # Check audio file
        if not audio_path.exists() or audio_path.stat().st_size < 1024:
            raise FFmpegError(f"Audio extraction failed: invalid output {audio_path}")
            
        # Check video file  
        if not video_path.exists() or video_path.stat().st_size < 1024:
            raise FFmpegError(f"Video extraction failed: invalid output {video_path}")
            
        logger.debug("Extraction outputs validated successfully")
    
    def split_by_scenes(self, video_path: Path, scene_boundaries: List[Dict]) -> List[Path]:
        """
        Split video into individual scene files
        
        Args:
            video_path: Path to video file (processed by extract_streams)
            scene_boundaries: List of scene boundary dictionaries
            
        Returns:
            List of paths to individual scene files
        """
        logger.info(f"Splitting video into {len(scene_boundaries)} scenes")
        
        # Create scenes directory
        scenes_dir = video_path.parent / "scenes"
        scenes_dir.mkdir(exist_ok=True)
        
        scene_files = []
        
        for boundary in scene_boundaries:
            scene_id = boundary['scene_id']
            start_seconds = boundary['start_seconds']
            end_seconds = boundary.get('end_seconds')
            
            # Define scene file path
            scene_file = scenes_dir / f"scene_{scene_id:03d}.mp4"
            
            try:
                self._extract_scene(video_path, scene_file, start_seconds, end_seconds)
                scene_files.append(scene_file)
                logger.debug(f"Scene {scene_id} extracted: {scene_file.name}")
                
            except Exception as e:
                logger.error(f"Failed to extract scene {scene_id}: {str(e)}")
                # Continue with other scenes rather than failing completely
                continue
        
        logger.info(f"Scene splitting complete: {len(scene_files)}/{len(scene_boundaries)} scenes")
        return scene_files
    
    def _extract_scene(self, video_path: Path, scene_file: Path, start_seconds: float, end_seconds: Optional[float]):
        """Extract a single scene from video"""
        cmd = [
            self.ffmpeg_path, '-y',
            '-i', str(video_path),
            '-ss', str(start_seconds)
        ]
        
        if end_seconds:
            duration = end_seconds - start_seconds
            cmd.extend(['-t', str(duration)])
            
        cmd.extend([
            '-c', 'copy',  # Copy without re-encoding for speed
            str(scene_file)
        ])
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=120  # 2 minute timeout per scene
            )
        except subprocess.TimeoutExpired:
            raise FFmpegError(f"Scene extraction timed out")
        except subprocess.CalledProcessError as e:
            raise FFmpegError(f"Scene extraction failed: {e.stderr}")
    
    def get_build_directory(self, video_path: Path) -> Path:
        """Get build directory for a video"""
        video_name = video_path.stem
        return self.build_dir / video_name
    
    def _cleanup_partial_outputs(self, output_dir: Path):
        """Clean up partial outputs on failure"""
        try:
            if output_dir.exists():
                shutil.rmtree(output_dir)
                logger.debug(f"Cleaned up partial outputs: {output_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up partial outputs: {e}")
    
    def get_video_info(self, video_path: Path) -> Dict[str, Any]:
        """
        Get detailed video information using ffprobe
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video information
        """
        # Use ffprobe from same directory as ffmpeg
        ffprobe_path = self.ffmpeg_path.replace('ffmpeg.exe', 'ffprobe.exe') if 'ffmpeg.exe' in self.ffmpeg_path else self.ffmpeg_path.replace('ffmpeg', 'ffprobe')
        
        cmd = [
            ffprobe_path, '-v', 'quiet',
            '-print_format', 'json',
            '-show_format', '-show_streams',
            str(video_path)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=30
            )
            
            info = json.loads(result.stdout)
            
            # Extract useful information
            format_info = info.get('format', {})
            video_stream = next(
                (s for s in info.get('streams', []) if s.get('codec_type') == 'video'),
                {}
            )
            
            return {
                'duration': float(format_info.get('duration', 0)),
                'size_bytes': int(format_info.get('size', 0)),
                'format_name': format_info.get('format_name', ''),
                'width': video_stream.get('width', 0),
                'height': video_stream.get('height', 0),
                'fps': eval(video_stream.get('avg_frame_rate', '0/1')) if '/' in str(video_stream.get('avg_frame_rate', '')) else 0,
                'codec': video_stream.get('codec_name', '')
            }
            
        except (subprocess.CalledProcessError, json.JSONDecodeError, Exception) as e:
            logger.warning(f"Could not get video info for {video_path.name}: {e}")
            return {
                'duration': 0,
                'size_bytes': video_path.stat().st_size if video_path.exists() else 0,
                'error': str(e)
            }