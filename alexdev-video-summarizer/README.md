# alexdev-video-summarizer

Scene-based institutional knowledge extraction from video libraries using 8-tool AI pipeline.

## Quick Start

```bash
# Setup environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Process videos
python src/main.py --input input/ --output output/

# View results
ls output/*.md
```

## System Overview

Transform video libraries into searchable institutional knowledge through:
- **FFmpeg Foundation**: Video/audio separation and scene splitting
- **PySceneDetect**: Content-aware scene boundary detection (70x performance)
- **Dual Pipeline Processing**: GPU tools (sequential) + CPU tools (parallel) per scene
- **7-Tool AI Analysis**: Whisper, YOLO, EasyOCR, OpenCV, LibROSA, pyAudioAnalysis
- **Scene-by-Scene Output**: Comprehensive knowledge base with cross-references

## Processing Pipeline

```
INPUT VIDEO → FFmpeg → PySceneDetect → Per-Scene Analysis → Knowledge Base
```

**Performance**: ~10 minutes per video for comprehensive institutional knowledge extraction  
**Scalability**: 100+ video libraries with circuit breaker protection

## Directory Structure

```
alexdev-video-summarizer/
├── src/                    # Core application code
├── config/                 # Configuration files (YAML)
├── models/                 # AI model storage
├── tests/                  # Test suites
├── scripts/               # Utility and setup scripts
├── doc/                   # Documentation
│   ├── arch/             # Architecture specifications
│   ├── dev/              # Development guides
│   └── sysops/           # Operations runbooks
├── input/                # Source videos (created at runtime)
├── build/                # Processing artifacts (created at runtime)
├── output/               # Final knowledge base files (created at runtime)
└── requirements.txt      # Python dependencies
```

## Development

See [Development Guide](doc/dev/development-setup.md) for complete setup instructions.

## Architecture

See [Technical Architecture](doc/arch/technical-architecture.md) for system design details.

## License

Internal tool - All rights reserved