# Development Setup Guide

Complete setup instructions for alexdev-video-summarizer development environment.

## Prerequisites

### Required Software
- **Python 3.8+** - Core application runtime
- **FFmpeg** - Video/audio processing foundation
- **Git** - Version control
- **GPU Drivers** - NVIDIA CUDA drivers for GPU acceleration (recommended)

### System Requirements
- **GPU**: RTX 5070 (16GB VRAM) or similar for optimal performance
- **RAM**: 32GB recommended for batch processing  
- **Storage**: 50GB+ available space (models + processing artifacts)
- **OS**: Windows 10/11, macOS, or Linux

## Quick Setup

### Automated Setup (Recommended)
```bash
# Clone and setup in one command
git clone <repository-url>
cd alexdev-video-summarizer
python scripts/setup.py
```

### Manual Setup
```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies  
pip install -r requirements.txt

# 3. Create directories
mkdir -p input output build models logs cache temp

# 4. Verify FFmpeg installation
ffmpeg -version
```

## Development Environment

### Project Structure
```
alexdev-video-summarizer/
├── src/                    # Source code
│   ├── main.py            # CLI entry point
│   ├── cli/               # CLI interface
│   ├── services/          # Processing services  
│   ├── utils/             # Utilities
│   └── models/            # Data models
├── config/                # Configuration files
├── tests/                 # Test suites
├── scripts/              # Setup and utility scripts
├── doc/                  # Documentation
├── input/                # Source videos (runtime)
├── build/                # Processing artifacts (runtime)
├── output/               # Final knowledge bases (runtime)
└── requirements.txt      # Dependencies
```

### Configuration
- **Main Config**: `config/processing.yaml` - All processing settings
- **Path Config**: `config/paths.yaml` - Directory structure and file patterns
- **Local Config**: Create `config/local_processing.yaml` for environment-specific overrides

## Development Workflow

### Basic Usage
```bash
# Activate environment
source venv/bin/activate  # Windows: venv\Scripts\activate

# Place videos in input/ directory
cp /path/to/videos/*.mp4 input/

# Run processing
python src/main.py --input input --output output

# View results
ls output/*.md
```

### Development Commands
```bash
# Run tests
pytest tests/

# Code formatting
black src/

# Linting
flake8 src/

# Type checking  
mypy src/

# Dry run (test without processing)
python src/main.py --dry-run
```

## Service Architecture

### Core Services
- **FFmpegService**: Video/audio separation and scene splitting
- **SceneDetectionService**: PySceneDetect integration
- **GPUPipelineController**: Sequential GPU tool coordination
- **CPUPipelineController**: Parallel CPU tool processing
- **KnowledgeBaseGenerator**: Final output generation

### Service Isolation
Each AI tool runs in isolated environment:
```bash
# Service venv structure
envs/
├── whisper_env/    # Whisper transcription
├── yolo_env/       # Object detection
├── easyocr_env/    # Text extraction
├── opencv_env/     # Face detection  
├── librosa_env/    # Audio features
└── audio_env/      # Audio analysis
```

## GPU Development

### CUDA Setup
```bash
# Verify CUDA installation
nvidia-smi

# Check PyTorch CUDA support
python -c "import torch; print(torch.cuda.is_available())"

# Monitor GPU usage during development
watch -n 1 nvidia-smi
```

### GPU Memory Management
- **Sequential Processing**: Only one GPU service active at a time
- **Memory Cleanup**: Automatic cleanup between services
- **Circuit Breaker**: Automatic recovery from GPU errors

## Testing

### Test Structure
```
tests/
├── unit/              # Unit tests per service
├── integration/       # End-to-end pipeline tests
└── fixtures/         # Test data and mocks
```

### Running Tests
```bash
# All tests
pytest

# Specific test category
pytest tests/unit/
pytest tests/integration/

# With coverage
pytest --cov=src tests/

# GPU tests (requires CUDA)
pytest -m gpu tests/
```

## Debugging

### Logging
- **Console**: Real-time progress display
- **Files**: Detailed logs in `logs/` directory
- **Per-Video**: Individual processing logs in `build/{video}/logs/`

### Debug Mode
```bash
# Verbose logging
python src/main.py --verbose

# Debug configuration
PYTHONPATH=src python -m pdb src/main.py
```

### Common Issues

#### FFmpeg Not Found
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# macOS  
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

#### GPU Memory Issues
- Reduce batch size in config
- Enable CPU fallback mode
- Monitor memory with `nvidia-smi`

#### Model Download Issues  
- Ensure internet connection for first run
- Check available disk space (3GB+ needed)
- Models cached in `models/` directory

## Performance Optimization

### Development Settings
```yaml
# config/local_processing.yaml
performance:
  target_processing_time: 300  # 5 min for development
development:
  fast_mode: true              # Reduced quality for speed
  debug_artifacts: true        # Keep intermediate files
```

### Production Settings
```yaml
performance:
  target_processing_time: 600  # 10 min production
development:
  fast_mode: false
  debug_artifacts: false
```

## IDE Setup

### VS Code Configuration
```json
// .vscode/settings.json
{
    "python.pythonPath": "venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black"
}
```

### Recommended Extensions
- Python
- Pylance  
- GitLens
- YAML

## Contribution Guidelines

### Code Standards
- **Formatting**: Black with 88-character line length
- **Linting**: Flake8 compliance required
- **Type Hints**: mypy type checking for all new code
- **Documentation**: Docstrings for all public functions

### Commit Messages
```
feat(service): add GPU memory cleanup
fix(cli): resolve progress display issue  
docs(setup): update installation instructions
test(integration): add scene detection tests
```

### Pull Request Process
1. Create feature branch from main
2. Implement changes with tests
3. Run full test suite
4. Update documentation
5. Submit PR with detailed description

## Troubleshooting

### Environment Issues
```bash
# Reset virtual environment
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Permission Issues
```bash
# Fix directory permissions
chmod -R 755 input output build
```

### Model Issues
```bash
# Clear model cache
rm -rf models/*
# Models will re-download on next run
```

## Support

- **Documentation**: `doc/` directory
- **Issue Tracking**: Internal tracking system
- **Development Chat**: Team communication channels

For development questions, check existing documentation or contact the development team.