#!/usr/bin/env python3
"""
Setup script for alexdev-video-summarizer development environment
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def run_command(cmd, description):
    """Run command with error handling"""
    print(f"\n🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Error: {e}")
        if e.stdout:
            print(f"   Output: {e.stdout}")
        if e.stderr:
            print(f"   Error output: {e.stderr}")
        return False


def check_python_version():
    """Ensure Python 3.8+ is available"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        sys.exit(1)
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")


def check_ffmpeg():
    """Check if FFmpeg is available"""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        print("✅ FFmpeg detected")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  FFmpeg not found - required for video processing")
        print("   Install from: https://ffmpeg.org/download.html")
        return False


def setup_virtual_environment():
    """Create and activate virtual environment"""
    venv_path = Path("venv")
    
    if not venv_path.exists():
        if not run_command(f"{sys.executable} -m venv venv", "Creating virtual environment"):
            return False
            
    # Activation command varies by OS
    if platform.system() == "Windows":
        activate_cmd = "venv\\Scripts\\activate"
        pip_cmd = "venv\\Scripts\\pip"
    else:
        activate_cmd = "source venv/bin/activate"
        pip_cmd = "venv/bin/pip"
        
    print(f"💡 To activate: {activate_cmd}")
    
    # Upgrade pip in venv
    return run_command(f"{pip_cmd} install --upgrade pip", "Upgrading pip")


def install_dependencies():
    """Install Python dependencies"""
    if platform.system() == "Windows":
        pip_cmd = "venv\\Scripts\\pip"
    else:
        pip_cmd = "venv/bin/pip"
        
    return run_command(f"{pip_cmd} install -r requirements.txt", "Installing dependencies")


def create_directories():
    """Create necessary directories"""
    directories = [
        "input",
        "output", 
        "build",
        "models",
        "logs",
        "cache",
        "temp"
    ]
    
    print("\n📁 Creating directories...")
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"   • {directory}/")
    print("✅ Directories created")


def download_models():
    """Download required AI models"""
    print("\n🤖 AI model setup...")
    print("   Models will be downloaded on first run:")
    print("   • YOLOv8n: ~6MB (object detection)")
    print("   • Whisper Large-v2: ~3GB (transcription)")
    print("   • EasyOCR: ~100MB (text detection)")
    print("💡 Ensure sufficient disk space and internet connection for first run")


def main():
    """Main setup workflow"""
    print("🚀 alexdev-video-summarizer Setup")
    print("=" * 50)
    
    # System checks
    check_python_version()
    ffmpeg_ok = check_ffmpeg()
    
    # Environment setup
    if not setup_virtual_environment():
        print("❌ Virtual environment setup failed")
        sys.exit(1)
        
    if not install_dependencies():
        print("❌ Dependency installation failed")
        sys.exit(1)
        
    # Project setup
    create_directories()
    download_models()
    
    # Final instructions
    print("\n🎉 Setup Complete!")
    print("\nNext steps:")
    
    if platform.system() == "Windows":
        print("1. Activate environment: venv\\Scripts\\activate")
    else:
        print("1. Activate environment: source venv/bin/activate")
        
    print("2. Place videos in input/ directory")
    print("3. Run processor: python src/main.py")
    
    if not ffmpeg_ok:
        print("\n⚠️  Remember to install FFmpeg before processing videos")
        
    print("\n📚 Documentation: doc/dev/development-setup.md")


if __name__ == "__main__":
    main()