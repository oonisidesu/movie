"""
Video Processing Module

This module provides comprehensive video processing capabilities for the
video face-swapping tool, including video loading, frame processing,
and output generation with audio preservation.

Components:
- VideoProcessor: Main video processing pipeline
- VideoWriter: Video output with audio preservation
- FrameProcessor: Frame-by-frame processing utilities
- VideoUtils: Video metadata and utility functions

Author: Face Swapping Tool
License: Personal/Educational Use Only
"""

from .video_processor import VideoProcessor
from .video_writer import VideoWriter
from .frame_processor import FrameProcessor
from .progress_tracker import (
    ProgressTracker,
    ConsoleProgressTracker,
    CallbackProgressTracker,
    SilentProgressTracker,
    ProgressInfo,
    create_progress_callback,
    progress_manager
)
from .utils.video_utils import (
    validate_video_file,
    get_video_info,
    extract_audio,
    merge_audio,
    calculate_memory_usage
)

__version__ = "1.0.0"
__all__ = [
    "VideoProcessor",
    "VideoWriter", 
    "FrameProcessor",
    "ProgressTracker",
    "ConsoleProgressTracker",
    "CallbackProgressTracker", 
    "SilentProgressTracker",
    "ProgressInfo",
    "create_progress_callback",
    "progress_manager",
    "validate_video_file",
    "get_video_info",
    "extract_audio",
    "merge_audio",
    "calculate_memory_usage"
]