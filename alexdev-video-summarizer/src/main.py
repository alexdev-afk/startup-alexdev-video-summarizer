#!/usr/bin/env python3
"""
alexdev-video-summarizer
Scene-based institutional knowledge extraction from video libraries

Main entry point for CLI application.
"""

import click
import sys
import os
from pathlib import Path

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from cli.video_processor import VideoProcessorCLI
from utils.config_loader import ConfigLoader
from utils.logger import setup_logging


@click.command()
@click.option('--input', '-i', 'input_dir', 
              type=click.Path(exists=True, file_okay=False, dir_okay=True),
              default='input',
              help='Input directory containing video files')
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
def main(input_dir, output_dir, config_file, verbose, dry_run):
    """
    Process video library for institutional knowledge extraction.
    
    Transforms videos into searchable scene-by-scene knowledge bases using
    8-tool AI pipeline: FFmpeg + PySceneDetect + 7 AI analysis tools.
    """
    
    # Setup logging
    log_level = 'DEBUG' if verbose else 'INFO'
    setup_logging(log_level)
    
    try:
        # Load configuration
        config = ConfigLoader.load_config(config_file)
        
        # Initialize CLI processor
        processor = VideoProcessorCLI(
            input_dir=input_dir,
            output_dir=output_dir,
            config=config,
            dry_run=dry_run
        )
        
        # Run processing pipeline
        processor.run()
        
    except KeyboardInterrupt:
        click.echo("\n🛑 Processing interrupted by user", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"\n❌ Error: {str(e)}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()