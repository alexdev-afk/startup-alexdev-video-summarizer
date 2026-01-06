@echo off
set PATH=C:\ffmpeg\bin;%PATH%
cd /d c:\dev\startup-alexdev-video-summarizer\alexdev-video-summarizer
python src/main.py --input input/ --output output/ --verbose
