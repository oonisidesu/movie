"""
Main entry point for Video Face Swapping Tool

Provides command-line access to the face swapping functionality.
"""

import sys
from .ui.cli import main

if __name__ == '__main__':
    sys.exit(main())