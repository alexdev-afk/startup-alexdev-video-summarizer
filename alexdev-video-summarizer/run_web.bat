@echo off
set PATH=C:\ffmpeg\bin;%PATH%
cd /d "%~dp0"
python run_web.py
pause
