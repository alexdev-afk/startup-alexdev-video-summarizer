import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

from utils.logger import get_logger
from utils.processing_context import VideoProcessingContext

logger = get_logger(__name__)

class DemucsService:
    """
    A service to separate audio sources using the Demucs library.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the DemucsService.
        Args:
            config: The application configuration dictionary.
        """
        self.config = config
        self.paths_config = config.get('paths', {})
        logger.info("DemucsService initialized.")

    def separate_audio(self, context: VideoProcessingContext) -> Tuple[Path, Path]:
        """
        Separates the main audio file into 'vocals' and 'no_vocals' stems.

        Args:
            context: The processing context containing paths and video info.

        Returns:
            A tuple containing the paths to the vocals.wav and no_vocals.wav files.
        """
        audio_file = context.get_audio_file_path()
        if not audio_file.exists():
            logger.error(f"Input audio file not found for Demucs: {audio_file}")
            raise FileNotFoundError(f"Demucs input audio not found: {audio_file}")

        logger.info(f"Starting audio separation for {audio_file} using htdemucs...")

        # Construct the command to run Demucs
        # Output directly to build directory with custom filename pattern
        command = [
            sys.executable, '-m', 'demucs',
            '-n', 'htdemucs',
            '--two-stems=vocals',
            '--filename', '{stem}.{ext}',  # Output directly as vocals.wav and no_vocals.wav
            '-o', str(context.build_directory),
            str(audio_file)
        ]

        try:
            # Execute the command
            process = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                encoding='utf-8'
            )
            logger.info("Demucs process completed successfully.")
            logger.debug(f"Demucs stdout: {process.stdout}")

            # Define the expected output paths from Demucs
            # Demucs creates: build/{video_name}/htdemucs/vocals.wav and no_vocals.wav
            model_output_dir = context.build_directory / 'htdemucs'
            temp_vocals_path = model_output_dir / 'vocals.wav'
            temp_no_vocals_path = model_output_dir / 'no_vocals.wav'

            if not temp_vocals_path.exists() or not temp_no_vocals_path.exists():
                logger.error(f"Demucs did not produce the expected output files.")
                logger.error(f"Looked in: {model_output_dir}")
                logger.error(f"Expected: {temp_vocals_path} and {temp_no_vocals_path}")
                # List what's actually there for debugging
                if model_output_dir.exists():
                    files = list(model_output_dir.glob("*"))
                    logger.error(f"Found files: {files}")
                raise FileNotFoundError("Demucs output files not found after processing.")

            # Move files to build/{video_name}/ for easy access
            vocals_path = context.build_directory / 'vocals.wav'
            no_vocals_path = context.build_directory / 'no_vocals.wav'
            
            import shutil
            shutil.move(str(temp_vocals_path), str(vocals_path))
            shutil.move(str(temp_no_vocals_path), str(no_vocals_path))
            
            # Clean up htdemucs directory
            shutil.rmtree(context.build_directory / 'htdemucs', ignore_errors=True)

            logger.info(f"Vocals stem saved to: {vocals_path}")
            logger.info(f"Instrumental stem saved to: {no_vocals_path}")

            return vocals_path, no_vocals_path

        except subprocess.CalledProcessError as e:
            logger.error(f"Demucs process failed with exit code {e.returncode}.")
            logger.error(f"Demucs stderr: {e.stderr}")
            raise RuntimeError("Demucs audio separation failed.") from e
        except FileNotFoundError:
            logger.error("`python` command not found. Ensure Python is in the system's PATH.")
            raise
