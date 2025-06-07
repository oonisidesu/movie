#!/usr/bin/env python3
"""
Face Swapping Demo Application

Demonstrates the face swapping functionality with classical computer vision
techniques for educational and testing purposes.
"""

import cv2
import numpy as np
import sys
import os
import logging
import argparse
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.face_detection import FaceDetector, DetectionBackend
from src.face_swapping import FaceSwapper, SwapMethod, SwapConfig
from src.utils.logging_config import setup_logging

# Setup logging
setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description='Face Swapping Demo')
    parser.add_argument('--source', required=True, help='Source face image path')
    parser.add_argument('--target', required=True, help='Target image path')
    parser.add_argument('--output', required=True, help='Output image path')
    parser.add_argument('--method', choices=['classical', 'deep'], default='classical',
                       help='Face swapping method')
    parser.add_argument('--show', action='store_true', help='Show result in window')
    
    args = parser.parse_args()
    
    try:
        # Validate input files
        if not os.path.exists(args.source):
            logger.error(f"Source image not found: {args.source}")
            return 1
        
        if not os.path.exists(args.target):
            logger.error(f"Target image not found: {args.target}")
            return 1
        
        # Load images
        logger.info("Loading images...")
        source_image = cv2.imread(args.source)
        target_image = cv2.imread(args.target)
        
        if source_image is None:
            logger.error(f"Failed to load source image: {args.source}")
            return 1
        
        if target_image is None:
            logger.error(f"Failed to load target image: {args.target}")
            return 1
        
        logger.info(f"Source image shape: {source_image.shape}")
        logger.info(f"Target image shape: {target_image.shape}")
        
        # Initialize face detector
        logger.info("Initializing face detector...")
        face_detector = FaceDetector(
            backend=DetectionBackend.OPENCV_HAAR,
            min_detection_confidence=0.5
        )
        
        # Detect faces
        logger.info("Detecting faces...")
        source_faces = face_detector.detect_faces(source_image)
        target_faces = face_detector.detect_faces(target_image)
        
        if not source_faces:
            logger.error("No face detected in source image")
            return 1
        
        if not target_faces:
            logger.error("No face detected in target image")
            return 1
        
        logger.info(f"Found {len(source_faces)} face(s) in source image")
        logger.info(f"Found {len(target_faces)} face(s) in target image")
        
        # Use first detected face from each image
        source_face = source_faces[0]
        target_face = target_faces[0]
        
        # Initialize face swapper
        logger.info("Initializing face swapper...")
        swap_method = SwapMethod.CLASSICAL if args.method == 'classical' else SwapMethod.DEEP_LEARNING
        
        config = SwapConfig(
            method=swap_method,
            target_size=(256, 256),
            smoothing_factor=0.8,
            color_correction=True
        )
        
        face_swapper = FaceSwapper(config)
        
        # Perform face swap
        logger.info("Performing face swap...")
        swap_result = face_swapper.swap_faces(
            source_image, target_image,
            source_face, target_face
        )
        
        if not swap_result.success:
            logger.error(f"Face swapping failed: {swap_result.error_message}")
            return 1
        
        logger.info(f"Face swap completed in {swap_result.processing_time:.2f}s")
        logger.info(f"Confidence: {swap_result.confidence:.3f}")
        
        # Save result
        logger.info(f"Saving result to: {args.output}")
        
        # Create output directory if needed
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        success = cv2.imwrite(args.output, swap_result.swapped_image)
        if not success:
            logger.error(f"Failed to save result image: {args.output}")
            return 1
        
        logger.info("Face swap demo completed successfully!")
        
        # Show result if requested
        if args.show:
            show_results(source_image, target_image, swap_result.swapped_image)
        
        # Print statistics
        stats = face_swapper.get_statistics()
        logger.info(f"Face swapper statistics: {stats}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        return 1


def show_results(source_image: np.ndarray, target_image: np.ndarray, 
                result_image: np.ndarray) -> None:
    """
    Display results in OpenCV windows.
    
    Args:
        source_image: Source image
        target_image: Target image
        result_image: Face swap result
    """
    try:
        # Resize images for display if they're too large
        max_height = 600
        
        def resize_for_display(image):
            if image.shape[0] > max_height:
                scale = max_height / image.shape[0]
                new_width = int(image.shape[1] * scale)
                return cv2.resize(image, (new_width, max_height))
            return image
        
        source_display = resize_for_display(source_image)
        target_display = resize_for_display(target_image)
        result_display = resize_for_display(result_image)
        
        # Create combined display
        combined_width = source_display.shape[1] + target_display.shape[1] + result_display.shape[1]
        combined_height = max(source_display.shape[0], target_display.shape[0], result_display.shape[0])
        
        combined = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
        
        # Place images side by side
        x_offset = 0
        
        # Source image
        combined[0:source_display.shape[0], x_offset:x_offset+source_display.shape[1]] = source_display
        cv2.putText(combined, "Source", (x_offset + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        x_offset += source_display.shape[1]
        
        # Target image
        combined[0:target_display.shape[0], x_offset:x_offset+target_display.shape[1]] = target_display
        cv2.putText(combined, "Target", (x_offset + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        x_offset += target_display.shape[1]
        
        # Result image
        combined[0:result_display.shape[0], x_offset:x_offset+result_display.shape[1]] = result_display
        cv2.putText(combined, "Result", (x_offset + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display
        cv2.imshow("Face Swap Demo - Press any key to close", combined)
        logger.info("Press any key to close the display window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        logger.error(f"Display failed: {e}")


def create_sample_images():
    """Create sample images for testing (if needed)."""
    # This function could create simple test images with drawn faces
    # for testing purposes when real face images are not available
    pass


if __name__ == "__main__":
    sys.exit(main())