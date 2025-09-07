#!/usr/bin/env python3
"""
alexdev-video-summarizer
Scene-based institutional knowledge extraction from video libraries

Main entry point for CLI application.
"""

import click
import sys
import os
import contextlib
from pathlib import Path
from io import StringIO

# Fix Windows console encoding for Unicode support
if sys.platform.startswith('win'):
    try:
        # Set console to UTF-8 mode
        os.system('chcp 65001 > nul 2>&1')
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'replace')
    except Exception:
        # Fallback if encoding fix fails
        pass

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from cli.video_processor import VideoProcessorCLI
from utils.config_loader import ConfigLoader
from utils.logger import setup_logging


@contextlib.contextmanager
def _suppress_output():
    """Context manager to suppress stdout/stderr during initialization"""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        # Redirect stdout/stderr to string buffers
        sys.stdout = StringIO()
        sys.stderr = StringIO()
        yield
    finally:
        # Restore original stdout/stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr


@click.command()
@click.option('--input', '-i', 'input_path', 
              type=click.Path(exists=True, file_okay=True, dir_okay=True),
              default='input',
              help='Input directory containing video files or single video file')
@click.option('--output', '-o', 'output_dir',
              type=click.Path(file_okay=False, dir_okay=True),
              default='output',
              help='Output directory for knowledge base files')
@click.option('--config', '-c', 'config_file',
              type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default='config/processing.yaml',
              help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True,
              help='Enable verbose logging')
@click.option('--dry-run', is_flag=True,
              help='Show what would be processed without executing')
def main(input_path, output_dir, config_file, verbose, dry_run):
    """
    Process video library for institutional knowledge extraction.
    
    Transforms videos into searchable scene-by-scene knowledge bases using
    8-tool AI pipeline: FFmpeg + PySceneDetect + 7 AI analysis tools.
    """
    
    # Setup logging - suppress unless verbose mode
    if verbose:
        log_level = 'DEBUG'
        setup_logging(log_level)
    else:
        # Suppress all logging output for clean CLI
        import logging
        logging.getLogger().setLevel(logging.CRITICAL)
    
    try:
        # Load configuration and initialize with suppressed output
        with _suppress_output():
            config = ConfigLoader.load_config(config_file)
            
            # Initialize CLI processor
            processor = VideoProcessorCLI(
                input_path=input_path,
                output_dir=output_dir,
                config=config,
                dry_run=dry_run,
                verbose=verbose
            )
        
        # Run processing pipeline
        processor.run()
        
    except KeyboardInterrupt:
        click.echo("\n[STOP] Processing interrupted by user", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"\n[ERROR] Error: {str(e)}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()