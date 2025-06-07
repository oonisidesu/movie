"""
Face Swapper - Main interface for face swapping operations

Provides a unified interface for different face swapping algorithms
and manages the overall face swapping pipeline.
"""

import cv2
import numpy as np
import logging
from enum import Enum
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass

from ..face_detection import FaceDetector, FaceDetection
from .alignment import FaceAligner, AlignmentMethod
from .blending import FaceBlender, BlendingMethod
from .classical import ClassicalFaceSwapper
from .utils import validate_face_images, calculate_face_similarity

logger = logging.getLogger(__name__)


class SwapMethod(Enum):
    """Available face swapping methods."""
    CLASSICAL = "classical"
    DEEP_LEARNING = "deep_learning"
    HYBRID = "hybrid"


@dataclass
class SwapResult:
    """Result of face swapping operation."""
    success: bool
    swapped_image: Optional[np.ndarray] = None
    confidence: float = 0.0
    processing_time: float = 0.0
    method_used: Optional[SwapMethod] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SwapConfig:
    """Configuration for face swapping operations."""
    method: SwapMethod = SwapMethod.CLASSICAL
    alignment_method: AlignmentMethod = AlignmentMethod.LANDMARKS
    blending_method: BlendingMethod = BlendingMethod.POISSON
    
    # Quality settings
    target_size: Tuple[int, int] = (256, 256)
    smoothing_factor: float = 0.7
    color_correction: bool = True
    
    # Advanced settings
    face_similarity_threshold: float = 0.3
    edge_feathering: int = 5
    blend_ratio: float = 0.8
    
    # Performance settings
    enable_gpu: bool = False
    batch_processing: bool = False
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if not 0.0 <= self.smoothing_factor <= 1.0:
            raise ValueError("Smoothing factor must be between 0.0 and 1.0")
        
        if not 0.0 <= self.face_similarity_threshold <= 1.0:
            raise ValueError("Face similarity threshold must be between 0.0 and 1.0")
        
        if not 0.0 <= self.blend_ratio <= 1.0:
            raise ValueError("Blend ratio must be between 0.0 and 1.0")
        
        if self.edge_feathering < 0:
            raise ValueError("Edge feathering must be non-negative")
        
        if self.target_size[0] <= 0 or self.target_size[1] <= 0:
            raise ValueError("Target size must have positive dimensions")


class FaceSwapper:
    """
    Main face swapping class that provides a unified interface
    for different face swapping algorithms.
    """
    
    def __init__(self, config: Optional[SwapConfig] = None):
        """
        Initialize face swapper.
        
        Args:
            config: Configuration for face swapping operations
        """
        self.config = config or SwapConfig()
        self.config.validate()
        
        # Initialize components
        self.face_detector = None
        self.face_aligner = None
        self.face_blender = None
        self.classical_swapper = None
        self.deep_swapper = None
        
        # Statistics
        self.total_swaps = 0
        self.successful_swaps = 0
        self.total_processing_time = 0.0
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Initialize face swapping components."""
        try:
            # Initialize face detector
            from ..face_detection import DetectionBackend
            self.face_detector = FaceDetector(
                backend=DetectionBackend.OPENCV_HAAR,
                confidence_threshold=0.5
            )
            
            # Initialize face aligner
            self.face_aligner = FaceAligner(
                method=self.config.alignment_method
            )
            
            # Initialize face blender
            self.face_blender = FaceBlender(
                method=self.config.blending_method,
                feathering=self.config.edge_feathering
            )
            
            # Initialize classical swapper
            self.classical_swapper = ClassicalFaceSwapper(
                target_size=self.config.target_size,
                smoothing_factor=self.config.smoothing_factor
            )
            
            # Deep learning swapper will be initialized on demand
            self.deep_swapper = None
            
            logger.info("Face swapping components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize face swapping components: {e}")
            raise
    
    def swap_faces(self, source_image: np.ndarray, target_image: np.ndarray,
                   source_face: Optional[FaceDetection] = None,
                   target_face: Optional[FaceDetection] = None) -> SwapResult:
        """
        Swap faces between source and target images.
        
        Args:
            source_image: Source image containing the face to swap
            target_image: Target image where face will be placed
            source_face: Optional pre-detected source face
            target_face: Optional pre-detected target face
            
        Returns:
            SwapResult with swapped image and metadata
        """
        import time
        start_time = time.time()
        
        try:
            # Validate inputs
            if not validate_face_images(source_image, target_image):
                return SwapResult(
                    success=False,
                    error_message="Invalid input images"
                )
            
            # Detect faces if not provided
            if source_face is None:
                source_faces = self.face_detector.detect_faces(source_image)
                if not source_faces:
                    return SwapResult(
                        success=False,
                        error_message="No face detected in source image"
                    )
                source_face = source_faces[0]  # Use first detected face
            
            if target_face is None:
                target_faces = self.face_detector.detect_faces(target_image)
                if not target_faces:
                    return SwapResult(
                        success=False,
                        error_message="No face detected in target image"
                    )
                target_face = target_faces[0]  # Use first detected face
            
            # Check face similarity if threshold is set
            if self.config.face_similarity_threshold > 0:
                similarity = calculate_face_similarity(source_face, target_face)
                if similarity < self.config.face_similarity_threshold:
                    logger.warning(f"Low face similarity: {similarity:.3f}")
            
            # Perform face swapping based on method
            if self.config.method == SwapMethod.CLASSICAL:
                result = self._swap_classical(
                    source_image, target_image, source_face, target_face
                )
            elif self.config.method == SwapMethod.DEEP_LEARNING:
                result = self._swap_deep_learning(
                    source_image, target_image, source_face, target_face
                )
            elif self.config.method == SwapMethod.HYBRID:
                result = self._swap_hybrid(
                    source_image, target_image, source_face, target_face
                )
            else:
                return SwapResult(
                    success=False,
                    error_message=f"Unsupported swap method: {self.config.method}"
                )
            
            # Update statistics
            processing_time = time.time() - start_time
            self.total_swaps += 1
            self.total_processing_time += processing_time
            
            if result.success:
                self.successful_swaps += 1
            
            result.processing_time = processing_time
            result.method_used = self.config.method
            
            return result
            
        except Exception as e:
            logger.error(f"Face swapping failed: {e}")
            return SwapResult(
                success=False,
                error_message=str(e),
                processing_time=time.time() - start_time
            )
    
    def _swap_classical(self, source_image: np.ndarray, target_image: np.ndarray,
                       source_face: FaceDetection, target_face: FaceDetection) -> SwapResult:
        """
        Perform classical computer vision face swapping.
        
        Args:
            source_image: Source image
            target_image: Target image  
            source_face: Detected source face
            target_face: Detected target face
            
        Returns:
            SwapResult with classical swap result
        """
        try:
            # Use classical swapper
            swapped_image = self.classical_swapper.swap_faces(
                source_image, target_image, source_face, target_face
            )
            
            if swapped_image is not None:
                return SwapResult(
                    success=True,
                    swapped_image=swapped_image,
                    confidence=0.8,  # Classical methods have consistent quality
                    metadata={
                        "method": "classical",
                        "algorithm": "delaunay_triangulation"
                    }
                )
            else:
                return SwapResult(
                    success=False,
                    error_message="Classical face swapping failed"
                )
                
        except Exception as e:
            logger.error(f"Classical face swapping error: {e}")
            return SwapResult(
                success=False,
                error_message=f"Classical swapping failed: {e}"
            )
    
    def _swap_deep_learning(self, source_image: np.ndarray, target_image: np.ndarray,
                           source_face: FaceDetection, target_face: FaceDetection) -> SwapResult:
        """
        Perform deep learning-based face swapping.
        
        Args:
            source_image: Source image
            target_image: Target image
            source_face: Detected source face
            target_face: Detected target face
            
        Returns:
            SwapResult with deep learning swap result
        """
        # TODO: Implement deep learning face swapping
        logger.warning("Deep learning face swapping not yet implemented")
        return SwapResult(
            success=False,
            error_message="Deep learning method not implemented yet"
        )
    
    def _swap_hybrid(self, source_image: np.ndarray, target_image: np.ndarray,
                    source_face: FaceDetection, target_face: FaceDetection) -> SwapResult:
        """
        Perform hybrid face swapping (classical + deep learning enhancement).
        
        Args:
            source_image: Source image
            target_image: Target image
            source_face: Detected source face
            target_face: Detected target face
            
        Returns:
            SwapResult with hybrid swap result
        """
        # TODO: Implement hybrid face swapping
        logger.warning("Hybrid face swapping not yet implemented")
        
        # Fallback to classical method for now
        return self._swap_classical(source_image, target_image, source_face, target_face)
    
    def batch_swap_faces(self, source_image: np.ndarray, 
                        target_images: List[np.ndarray]) -> List[SwapResult]:
        """
        Perform batch face swapping on multiple target images.
        
        Args:
            source_image: Source image with face to swap
            target_images: List of target images
            
        Returns:
            List of SwapResult objects
        """
        results = []
        
        # Detect source face once for efficiency
        source_faces = self.face_detector.detect_faces(source_image)
        if not source_faces:
            error_result = SwapResult(
                success=False,
                error_message="No face detected in source image"
            )
            return [error_result] * len(target_images)
        
        source_face = source_faces[0]
        
        # Process each target image
        for target_image in target_images:
            result = self.swap_faces(
                source_image, target_image, 
                source_face=source_face
            )
            results.append(result)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get face swapping statistics.
        
        Returns:
            Dictionary with statistics
        """
        success_rate = 0.0
        avg_processing_time = 0.0
        
        if self.total_swaps > 0:
            success_rate = self.successful_swaps / self.total_swaps
            avg_processing_time = self.total_processing_time / self.total_swaps
        
        return {
            "total_swaps": self.total_swaps,
            "successful_swaps": self.successful_swaps,
            "success_rate": success_rate,
            "total_processing_time": self.total_processing_time,
            "average_processing_time": avg_processing_time,
            "config": {
                "method": self.config.method.value,
                "alignment_method": self.config.alignment_method.value,
                "blending_method": self.config.blending_method.value,
                "target_size": self.config.target_size
            }
        }
    
    def reset_statistics(self) -> None:
        """Reset face swapping statistics."""
        self.total_swaps = 0
        self.successful_swaps = 0
        self.total_processing_time = 0.0
        logger.info("Face swapping statistics reset")
    
    def update_config(self, new_config: SwapConfig) -> None:
        """
        Update face swapping configuration.
        
        Args:
            new_config: New configuration to apply
        """
        new_config.validate()
        old_config = self.config
        self.config = new_config
        
        # Reinitialize components if necessary
        if (old_config.alignment_method != new_config.alignment_method or
            old_config.blending_method != new_config.blending_method):
            
            logger.info("Reinitializing components due to config change")
            self._initialize_components()
        
        logger.info("Face swapping configuration updated")


def create_face_swapper(method: SwapMethod = SwapMethod.CLASSICAL,
                       **kwargs) -> FaceSwapper:
    """
    Factory function to create a face swapper with common configurations.
    
    Args:
        method: Face swapping method to use
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured FaceSwapper instance
    """
    config = SwapConfig(method=method, **kwargs)
    return FaceSwapper(config)