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

        # Define output directory for Demucs stems
        demucs_output_dir = context.get_audio_analysis_path("demucs_output")
        demucs_output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting audio separation for {audio_file} using htdemucs...")

        # Construct the command to run Demucs
        # We use 'python -m demucs' for reliability
        command = [
            sys.executable, '-m', 'demucs',
            '-n', 'htdemucs',
            '--two-stems=vocals',
            '-o', str(demucs_output_dir),
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

            # Define the expected output paths
            # Demucs creates a subdirectory named after the model
            model_output_dir = demucs_output_dir / 'htdemucs' / context.video_name
            vocals_path = model_output_dir / 'vocals.wav'
            no_vocals_path = model_output_dir / 'no_vocals.wav'

            if not vocals_path.exists() or not no_vocals_path.exists():
                logger.error(f"Demucs did not produce the expected output files.")
                logger.error(f"Looked in: {model_output_dir}")
                raise FileNotFoundError("Demucs output files not found after processing.")

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
