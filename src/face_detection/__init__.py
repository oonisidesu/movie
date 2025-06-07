"""
Face Detection Module

This module provides comprehensive face detection, tracking, and quality assessment
capabilities for the video face-swapping tool.

Key Components:
- FaceDetector: Multi-backend face detection (OpenCV, MediaPipe)
- FaceTracker: Face tracking across video frames  
- FaceQualityAssessor: Face quality evaluation for swapping suitability

Author: Face Swapping Tool
License: Personal/Educational Use Only
"""

from .detector import (
    FaceDetector,
    FaceDetection, 
    DetectionBackend,
    get_available_backends,
    create_face_detector
)

from .tracker import (
    FaceTracker,
    TrackedFace
)

from .quality import (
    FaceQualityAssessor,
    QualityMetrics,
    QualityLevel
)

from .landmarks import (
    LandmarkDetector,
    FaceLandmarks,
    estimate_face_pose
)

__version__ = "1.0.0"
__author__ = "Face Swapping Tool"

__all__ = [
    # Detector classes and enums
    'FaceDetector',
    'FaceDetection', 
    'DetectionBackend',
    'get_available_backends',
    'create_face_detector',
    
    # Tracker classes
    'FaceTracker',
    'TrackedFace',
    
    # Quality assessment classes
    'FaceQualityAssessor', 
    'QualityMetrics',
    'QualityLevel',
    
    # Landmark detection classes
    'LandmarkDetector',
    'FaceLandmarks',
    'estimate_face_pose'
]