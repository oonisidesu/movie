"""
Face Detection Module

This module provides comprehensive face detection, tracking, and quality assessment
capabilities for the video face-swapping tool.

Components:
- FaceDetector: Core face detection using OpenCV and MediaPipe
- FaceTracker: Face tracking across video frames with consistent IDs
- FaceQualityAssessor: Assessment of face image quality for processing
"""

from .detector import FaceDetector
from .tracker import FaceTracker, MultiPersonTracker
from .quality import FaceQualityAssessor

__version__ = "1.0.0"
__all__ = [
    "FaceDetector",
    "FaceTracker", 
    "MultiPersonTracker",
    "FaceQualityAssessor"
]