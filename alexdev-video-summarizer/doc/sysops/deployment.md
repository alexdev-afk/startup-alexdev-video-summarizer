# Deployment Guide

Production deployment instructions for alexdev-video-summarizer on Windows RTX 5070 workstation.

## System Requirements

### Hardware Requirements
- **GPU**: RTX 5070 (16GB VRAM) or equivalent CUDA-capable GPU
- **CPU**: 8+ cores recommended for parallel processing
- **RAM**: 32GB minimum (64GB recommended for large batches)
- **Storage**: 1TB+ SSD for optimal performance
  - 50GB+ for application and models
  - 500GB+ for video processing artifacts
  - 100GB+ for final knowledge base outputs

### Software Requirements
- **Operating System**: Windows 10/11 (64-bit)
- **Python**: 3.8+ (3.11 recommended)
- **NVIDIA Drivers**: Latest CUDA-compatible drivers
- **FFmpeg**: Latest stable version
- **Git**: For version control and updates

## Pre-Deployment Setup

### 1. NVIDIA Driver Installation
```powershell
# Download and install latest NVIDIA drivers
# https://www.nvidia.com/Download/index.aspx

# Verify installation
nvidia-smi
```

### 2. CUDA Toolkit Setup
```powershell
# Download CUDA Toolkit 12.x
# https://developer.nvidia.com/cuda-downloads

# Verify CUDA installation
nvcc --version
```

### 3. FFmpeg Installation
```powershell
# Download FFmpeg from https://ffmpeg.org/download.html
# Extract to C:\ffmpeg\
# Add C:\ffmpeg\bin to system PATH

# Verify installation
ffmpeg -version
```

### 4. Python Environment
```powershell
# Install Python 3.11 from python.org
python --version

# Verify pip
pip --version
```

## Application Deployment

### 1. Repository Setup
```powershell
# Clone repository
git clone <repository-url> C:\alexdev-video-summarizer
cd C:\alexdev-video-summarizer

# Verify structure
dir
```

### 2. Environment Configuration
```powershell
# Create virtual environment
python -m venv venv

# Activate environment
venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### 3. Directory Structure Creation
```powershell
# Run setup script
python scripts\setup.py

# Verify directories
dir input
dir output
dir build
dir models
```

### 4. Configuration Setup
```powershell
# Copy configuration templates
copy config\processing.yaml config\local_processing.yaml
copy config\paths.yaml config\local_paths.yaml

# Edit local configuration for production settings
notepad config\local_processing.yaml
```

## Production Configuration

### GPU Optimization Settings
```yaml
# config/local_processing.yaml
gpu_pipeline:
  max_gpu_memory_usage: 0.95      # Use 95% GPU memory
  sequential_processing: true      # Required for stability
  memory_cleanup: true            # Force cleanup between tools

performance:
  target_processing_time: 600     # 10 minutes per video
  max_processing_time: 1800       # 30 minute timeout
  force_gc_after_video: true      # Force garbage collection
```

### Batch Processing Settings
```yaml
error_handling:
  circuit_breaker_threshold: 3    # Stop after 3 consecutive failures
  continue_on_tool_failure: false # Fail-fast approach
  cleanup_failed_artifacts: true  # Clean up on failures

logging:
  level: "INFO"                   # Production logging level
  file_logging: true              # Enable file logging
  max_log_size: "50MB"            # Larger log files for production
  backup_count: 5                 # More log backups
```

## Model Deployment

### Automatic Model Download
```powershell
# Activate environment
venv\Scripts\activate

# Run first-time setup (downloads models)
python src\main.py --dry-run --input input
```

### Manual Model Management
```powershell
# Models are downloaded to models/ directory:
# - YOLOv8n.pt (~6MB)
# - Whisper large-v2 (~3GB) 
# - EasyOCR models (~100MB)

# Verify models
dir models
```

## Production Validation

### 1. System Health Check
```powershell
# GPU check
nvidia-smi

# Python environment check
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# FFmpeg check  
ffmpeg -version

# Application check
python src\main.py --help
```

### 2. Processing Test
```powershell
# Create test video
# Place sample video in input/test.mp4

# Run test processing
python src\main.py --input input --output output --verbose

# Verify output
dir output\*.md
```

### 3. Performance Validation
```powershell
# Monitor GPU during test
# Open new PowerShell window
nvidia-smi -l 1

# Monitor system resources
# Task Manager → Performance tab

# Check processing logs
type logs\processing.log
```

## Production Operation

### Daily Operation Workflow
```powershell
# 1. Place videos in input directory
copy "\\network\share\videos\*.mp4" input\

# 2. Start batch processing
venv\Scripts\activate
python src\main.py --input input --output output

# 3. Monitor progress (real-time CLI display)
# Processing will show:
# - Current video progress
# - Pipeline status
# - Time estimates
# - Error reporting

# 4. Review results
dir output\*.md
```

### Automated Batch Processing
```powershell
# Create batch script: process_videos.bat
@echo off
cd /d C:\alexdev-video-summarizer
call venv\Scripts\activate.bat
python src\main.py --input input --output output
pause
```

### Task Scheduler Integration
```powershell
# Create scheduled task for overnight processing
schtasks /create /tn "VideoProcessing" /tr "C:\alexdev-video-summarizer\process_videos.bat" /sc daily /st 23:00
```

## Monitoring and Maintenance

### Performance Monitoring
```powershell
# GPU utilization
nvidia-smi dmon -s pum

# Disk usage
dir build /s
dir output /s

# Log analysis
findstr "ERROR" logs\*.log
findstr "FAILED" logs\*.log
```

### Maintenance Tasks

#### Daily Maintenance
- Check processing logs for errors
- Verify output generation
- Monitor disk space usage

#### Weekly Maintenance  
```powershell
# Clean build artifacts older than 7 days
forfiles /p build /d -7 /c "cmd /c rmdir /s /q @path"

# Rotate logs
move logs\processing.log logs\processing_backup.log

# Update models if needed
# Models auto-update on first run after updates
```

#### Monthly Maintenance
```powershell
# Update dependencies
venv\Scripts\activate
pip list --outdated
pip install --upgrade -r requirements.txt

# Update NVIDIA drivers
# Check NVIDIA GeForce Experience

# System health check
python scripts\health_check.py
```

## Backup and Recovery

### Configuration Backup
```powershell
# Backup configuration
xcopy config\*.yaml backup\config\ /y

# Backup custom scripts
xcopy scripts\*.py backup\scripts\ /y
```

### Data Backup Strategy
- **Input Videos**: Maintain source backup on network storage
- **Output Knowledge Bases**: Daily backup to network storage
- **Configuration**: Version controlled and backed up
- **Models**: Can be re-downloaded if needed

### Recovery Procedures

#### Application Recovery
```powershell
# Reinstall from clean state
rmdir /s venv
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python scripts\setup.py
```

#### Data Recovery
```powershell
# Restore configuration
xcopy backup\config\*.yaml config\ /y

# Restart processing from last known good state
python src\main.py --input input --output output
```

## Troubleshooting

### Common Issues

#### GPU Memory Errors
```powershell
# Check GPU memory usage
nvidia-smi

# Reduce memory usage in config
# gpu_pipeline.max_gpu_memory_usage: 0.8

# Restart processing
```

#### FFmpeg Errors
```powershell
# Verify FFmpeg path
where ffmpeg

# Check video file integrity
ffprobe input\problematic_video.mp4

# Try alternative codecs in config
```

#### Python Environment Issues
```powershell
# Recreate virtual environment
rmdir /s venv
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Support Contacts
- **System Administrator**: [Internal contact]
- **Development Team**: [Internal contact] 
- **GPU Issues**: NVIDIA support documentation

## Security Considerations

### Data Security
- All processing occurs locally (no cloud uploads)
- Video content remains on local workstation
- Network access only for model downloads

### System Security
- Regular Windows updates
- NVIDIA driver updates
- Python security updates via pip

### Access Control
- Restrict access to processing directories
- User account permissions
- Network share security for input/output

## Performance Optimization

### Production Tuning
```yaml
# Optimal settings for RTX 5070
gpu_pipeline:
  max_gpu_memory_usage: 0.95
  
performance:
  target_processing_time: 480    # 8 minutes (optimized)
  force_gc_after_video: true
  
cpu_pipeline:
  max_workers: 4                 # Match CPU cores
```

### Capacity Planning
- **Single Video**: 8-10 minutes average
- **100 Video Library**: 14-17 hours  
- **Storage Growth**: ~1GB per video processed
- **Network Transfer**: Plan for input/output file movement

This deployment guide provides comprehensive instructions for production operation of the alexdev-video-summarizer system.