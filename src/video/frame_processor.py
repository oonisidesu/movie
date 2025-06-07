"""
Frame Processor - Frame-by-frame processing utilities

Provides efficient frame processing capabilities for the video face-swapping
pipeline with support for custom processing functions and optimizations.
"""

import numpy as np
import cv2
import logging
from typing import Callable, List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class FrameProcessingStats:
    """Statistics for frame processing operations."""
    total_frames: int = 0
    processed_frames: int = 0
    failed_frames: int = 0
    total_processing_time: float = 0.0
    average_processing_time: float = 0.0
    fps: float = 0.0


class FrameProcessor:
    """
    Frame-by-frame processing pipeline for video manipulation.
    
    Provides a flexible framework for applying various processing operations
    to video frames, with support for face swapping and other transformations.
    """
    
    def __init__(self, enable_stats: bool = True):
        """
        Initialize frame processor.
        
        Args:
            enable_stats: Whether to collect processing statistics
        """
        self.enable_stats = enable_stats
        self.stats = FrameProcessingStats()
        
        # Processing pipeline
        self.processors: List[Callable[[np.ndarray], np.ndarray]] = []
        self.preprocessing_functions: List[Callable[[np.ndarray], np.ndarray]] = []
        self.postprocessing_functions: List[Callable[[np.ndarray], np.ndarray]] = []
        
        # Processing options
        self.resize_frames = False
        self.target_size: Optional[Tuple[int, int]] = None
        self.color_space = 'BGR'  # Default OpenCV color space
        
    def add_preprocessor(self, func: Callable[[np.ndarray], np.ndarray]) -> None:
        """
        Add preprocessing function to the pipeline.
        
        Args:
            func: Function that takes and returns a frame array
        """
        self.preprocessing_functions.append(func)
        logger.debug(f"Added preprocessor: {func.__name__}")
    
    def add_processor(self, func: Callable[[np.ndarray], np.ndarray]) -> None:
        """
        Add main processing function to the pipeline.
        
        Args:
            func: Function that takes and returns a frame array
        """
        self.processors.append(func)
        logger.debug(f"Added processor: {func.__name__}")
    
    def add_postprocessor(self, func: Callable[[np.ndarray], np.ndarray]) -> None:
        """
        Add postprocessing function to the pipeline.
        
        Args:
            func: Function that takes and returns a frame array
        """
        self.postprocessing_functions.append(func)
        logger.debug(f"Added postprocessor: {func.__name__}")
    
    def set_resize_target(self, width: int, height: int) -> None:
        """
        Set target size for frame resizing.
        
        Args:
            width: Target width
            height: Target height
        """
        self.resize_frames = True
        self.target_size = (width, height)
        logger.info(f"Frame resizing enabled: {width}x{height}")
    
    def disable_resize(self) -> None:
        """Disable frame resizing."""
        self.resize_frames = False
        self.target_size = None
        logger.info("Frame resizing disabled")
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame through the complete pipeline.
        
        Args:
            frame: Input frame array
            
        Returns:
            Processed frame array
        """
        if frame is None or frame.size == 0:
            logger.warning("Empty frame provided")
            return frame
        
        start_time = time.time() if self.enable_stats else 0
        
        try:
            processed_frame = frame.copy()
            
            # Preprocessing
            for preprocess_func in self.preprocessing_functions:
                try:
                    processed_frame = preprocess_func(processed_frame)
                except Exception as e:
                    logger.error(f"Preprocessing failed: {e}")
                    processed_frame = frame.copy()  # Fallback to original
            
            # Resize if needed
            if self.resize_frames and self.target_size:
                processed_frame = cv2.resize(processed_frame, self.target_size)
            
            # Main processing
            for process_func in self.processors:
                try:
                    processed_frame = process_func(processed_frame)
                except Exception as e:
                    logger.error(f"Main processing failed: {e}")
                    # Continue with current frame state
            
            # Postprocessing
            for postprocess_func in self.postprocessing_functions:
                try:
                    processed_frame = postprocess_func(processed_frame)
                except Exception as e:
                    logger.error(f"Postprocessing failed: {e}")
                    # Continue with current frame state
            
            # Update statistics
            if self.enable_stats:
                processing_time = time.time() - start_time
                self._update_stats(processing_time, success=True)
            
            return processed_frame
            
        except Exception as e:
            logger.error(f"Frame processing failed: {e}")
            if self.enable_stats:
                processing_time = time.time() - start_time
                self._update_stats(processing_time, success=False)
            return frame  # Return original frame on failure
    
    def process_frame_batch(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Process a batch of frames.
        
        Args:
            frames: List of frame arrays
            
        Returns:
            List of processed frame arrays
        """
        processed_frames = []
        
        for i, frame in enumerate(frames):
            try:
                processed_frame = self.process_frame(frame)
                processed_frames.append(processed_frame)
            except Exception as e:
                logger.error(f"Failed to process frame {i}: {e}")
                processed_frames.append(frame)  # Fallback to original
        
        return processed_frames
    
    def _update_stats(self, processing_time: float, success: bool) -> None:
        """Update processing statistics."""
        self.stats.total_frames += 1
        
        if success:
            self.stats.processed_frames += 1
        else:
            self.stats.failed_frames += 1
        
        self.stats.total_processing_time += processing_time
        
        if self.stats.processed_frames > 0:
            self.stats.average_processing_time = (
                self.stats.total_processing_time / self.stats.processed_frames
            )
            self.stats.fps = 1.0 / self.stats.average_processing_time
    
    def get_stats(self) -> FrameProcessingStats:
        """Get current processing statistics."""
        return self.stats
    
    def reset_stats(self) -> None:
        """Reset processing statistics."""
        self.stats = FrameProcessingStats()
    
    def clear_pipeline(self) -> None:
        """Clear all processing functions from the pipeline."""
        self.processors.clear()
        self.preprocessing_functions.clear()
        self.postprocessing_functions.clear()
        logger.info("Processing pipeline cleared")


class FaceSwappingFrameProcessor(FrameProcessor):
    """
    Specialized frame processor for face swapping operations.
    
    Extends the base FrameProcessor with face swapping specific
    optimizations and utilities.
    """
    
    def __init__(self, enable_stats: bool = True):
        """Initialize face swapping frame processor."""
        super().__init__(enable_stats)
        
        # Face swapping specific settings
        self.face_detector = None
        self.face_tracker = None
        self.quality_assessor = None
        
        # Processing options
        self.quality_threshold = 0.6
        self.enable_face_tracking = True
        self.enable_quality_filtering = True
        
    def setup_face_processing(self, face_detector=None, face_tracker=None, 
                            quality_assessor=None) -> None:
        """
        Setup face processing components.
        
        Args:
            face_detector: Face detection instance
            face_tracker: Face tracking instance
            quality_assessor: Face quality assessment instance
        """
        self.face_detector = face_detector
        self.face_tracker = face_tracker
        self.quality_assessor = quality_assessor
        
        logger.info("Face processing components configured")
    
    def add_face_swap_processor(self, source_face_path: str, 
                              target_face_encoding: Optional[np.ndarray] = None) -> None:
        """
        Add face swapping processor to the pipeline.
        
        Args:
            source_face_path: Path to source face image
            target_face_encoding: Pre-computed target face encoding
        """
        def face_swap_processor(frame: np.ndarray) -> np.ndarray:
            """Face swapping processing function."""
            try:
                # Detect faces in frame
                if self.face_detector:
                    faces = self.face_detector.detect_faces(frame)
                    
                    # Track faces if enabled
                    if self.face_tracker and self.enable_face_tracking:
                        tracked_faces = self.face_tracker.process_frame(frame)
                        # Use tracked faces for consistency
                        faces = tracked_faces
                    
                    # Process each detected face
                    for face in faces:
                        # Quality check if enabled
                        if (self.quality_assessor and self.enable_quality_filtering):
                            face_region = self.face_detector.get_face_region(frame, face)
                            quality = self.quality_assessor.assess_quality(face_region)
                            
                            if quality['overall_score'] < self.quality_threshold:
                                logger.debug(f"Skipping low quality face: {quality['overall_score']:.2f}")
                                continue
                        
                        # TODO: Implement actual face swapping
                        # This would integrate with the face swapping algorithm
                        # For now, just mark the face with a rectangle
                        x, y, w, h = face['bbox']
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, "FACE DETECTED", (x, y - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                return frame
                
            except Exception as e:
                logger.error(f"Face swap processing failed: {e}")
                return frame
        
        self.add_processor(face_swap_processor)
        logger.info(f"Face swap processor added: {source_face_path}")


# Utility processing functions
def resize_frame(target_width: int, target_height: int) -> Callable[[np.ndarray], np.ndarray]:
    """
    Create a frame resizing function.
    
    Args:
        target_width: Target width
        target_height: Target height
        
    Returns:
        Frame resizing function
    """
    def resize_func(frame: np.ndarray) -> np.ndarray:
        return cv2.resize(frame, (target_width, target_height))
    
    resize_func.__name__ = f"resize_{target_width}x{target_height}"
    return resize_func


def normalize_frame() -> Callable[[np.ndarray], np.ndarray]:
    """
    Create a frame normalization function.
    
    Returns:
        Frame normalization function
    """
    def normalize_func(frame: np.ndarray) -> np.ndarray:
        return cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
    
    normalize_func.__name__ = "normalize_frame"
    return normalize_func


def denoise_frame(strength: int = 10) -> Callable[[np.ndarray], np.ndarray]:
    """
    Create a frame denoising function.
    
    Args:
        strength: Denoising strength
        
    Returns:
        Frame denoising function
    """
    def denoise_func(frame: np.ndarray) -> np.ndarray:
        return cv2.fastNlMeansDenoisingColored(frame, None, strength, strength, 7, 21)
    
    denoise_func.__name__ = f"denoise_strength_{strength}"
    return denoise_func


def enhance_contrast(alpha: float = 1.2, beta: int = 10) -> Callable[[np.ndarray], np.ndarray]:
    """
    Create a contrast enhancement function.
    
    Args:
        alpha: Contrast multiplier
        beta: Brightness offset
        
    Returns:
        Contrast enhancement function
    """
    def enhance_func(frame: np.ndarray) -> np.ndarray:
        return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    
    enhance_func.__name__ = f"enhance_contrast_a{alpha}_b{beta}"
    return enhance_func