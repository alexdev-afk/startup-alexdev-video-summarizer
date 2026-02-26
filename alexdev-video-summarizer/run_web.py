"""Entry point for the Video Summarizer Web UI."""

import logging
import os
import sys
from pathlib import Path

# Ensure project root is the working directory
project_root = Path(__file__).resolve().parent
os.chdir(project_root)

# Add src/ to path for service imports
sys.path.insert(0, str(project_root / "src"))

# Windows UTF-8 support
os.environ["PYTHONUTF8"] = "1"

from utils.logger import setup_logging
from web.server import create_app

if __name__ == "__main__":
    # File handler gets full DEBUG; console only gets WARNING+
    # (avoids printing filenames to terminal which can cause encoding errors)
    setup_logging(log_level="DEBUG")
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.StreamHandler) and handler.stream is sys.stdout:
            handler.setLevel(logging.WARNING)

    app = create_app()
    print("Video Summarizer Web UI")
    print("http://localhost:3000")
    app.run(host="0.0.0.0", port=3000, debug=False, threaded=True)
