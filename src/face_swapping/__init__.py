"""
Face Swapping Module

This module provides face swapping functionality for the video face-swapping tool,
including classical computer vision approaches and deep learning-based methods.

Components:
- FaceSwapper: Main face swapping interface
- ClassicalSwapper: Traditional CV-based face swapping
- DeepSwapper: Deep learning-based face swapping
- FaceAligner: Face alignment and preprocessing
- FaceBlender: Post-processing and blending utilities

Author: Face Swapping Tool
License: Personal/Educational Use Only
"""

from .swapper import FaceSwapper, SwapMethod
from .classical import ClassicalFaceSwapper
from .alignment import FaceAligner, AlignmentMethod
from .blending import FaceBlender, BlendingMethod
from .utils import (
    validate_face_images,
    calculate_face_similarity,
    extract_face_region,
    resize_face_to_target
)

__version__ = "1.0.0"
__all__ = [
    "FaceSwapper",
    "SwapMethod",
    "ClassicalFaceSwapper", 
    "FaceAligner",
    "AlignmentMethod",
    "FaceBlender",
    "BlendingMethod",
    "validate_face_images",
    "calculate_face_similarity",
    "extract_face_region",
    "resize_face_to_target"
]