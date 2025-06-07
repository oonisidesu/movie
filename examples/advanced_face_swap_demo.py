#!/usr/bin/env python3
"""
Advanced Face Swapping Demo Application

Demonstrates high-quality face swapping using facial landmarks 
and Delaunay triangulation.
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

from src.face_detection import FaceDetector, DetectionBackend, LandmarkDetector
from src.face_swapping import AdvancedFaceSwapper, FaceSwapper, SwapMethod, SwapConfig
from src.face_swapping.hybrid_swapper import HybridFaceSwapper
from src.utils.logging_config import setup_logging

# Setup logging
setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)


def visualize_landmarks(image: np.ndarray, face_detection, landmark_detector):
    """Visualize detected landmarks on the image."""
    landmarks = landmark_detector.detect_landmarks(image, face_detection.bbox)
    if landmarks:
        result = landmark_detector.draw_landmarks(image, landmarks)
        return result, landmarks
    return image, None


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description='Advanced Face Swapping Demo')
    parser.add_argument('--source', required=True, help='Source face image path')
    parser.add_argument('--target', required=True, help='Target image path')
    parser.add_argument('--output', required=True, help='Output image path')
    parser.add_argument('--method', choices=['basic', 'advanced', 'hybrid'], default='hybrid',
                       help='Face swapping method')
    parser.add_argument('--landmark-method', choices=['basic', 'mediapipe', 'dlib'], 
                       default='basic', help='Landmark detection method')
    parser.add_argument('--show-landmarks', action='store_true', 
                       help='Show landmark detection visualization')
    parser.add_argument('--enable-pose-correction', action='store_true',
                       help='Enable 3D pose correction')
    parser.add_argument('--blend-mode', choices=['seamless', 'feather', 'poisson'],
                       default='seamless', help='Blending mode')
    parser.add_argument('--horizontal-shift', type=float, default=0.15,
                       help='Horizontal position adjustment (negative=left, positive=right)')
    parser.add_argument('--vertical-shift', type=float, default=0.0,
                       help='Vertical position adjustment (negative=up, positive=down)')
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
        
        # Show landmark detection if requested
        if args.show_landmarks:
            logger.info("Visualizing landmarks...")
            landmark_detector = LandmarkDetector(method=args.landmark_method)
            
            source_with_landmarks, source_landmarks = visualize_landmarks(
                source_image, source_face, landmark_detector
            )
            target_with_landmarks, target_landmarks = visualize_landmarks(
                target_image, target_face, landmark_detector
            )
            
            # Show landmark visualization
            cv2.imshow("Source Landmarks", source_with_landmarks)
            cv2.imshow("Target Landmarks", target_with_landmarks)
            logger.info("Press any key to continue with face swapping...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        # Perform face swap
        if args.method == 'advanced':
            logger.info("Initializing advanced face swapper...")
            face_swapper = AdvancedFaceSwapper(
                landmark_method=args.landmark_method,
                enable_pose_correction=args.enable_pose_correction,
                blend_mode=args.blend_mode
            )
            
            logger.info("Performing advanced face swap...")
            swap_result = face_swapper.swap_faces(
                source_image, target_image,
                source_face, target_face
            )
            
            if swap_result is None:
                logger.error("Advanced face swapping failed")
                return 1
            
            # Get statistics
            stats = face_swapper.get_statistics()
            logger.info(f"Advanced face swapper statistics: {stats}")
            
        elif args.method == 'hybrid':
            logger.info("Initializing hybrid face swapper...")
            face_swapper = HybridFaceSwapper(
                landmark_method=args.landmark_method,
                horizontal_shift=args.horizontal_shift,
                vertical_shift=args.vertical_shift
            )
            
            logger.info("Performing hybrid face swap...")
            swap_result = face_swapper.swap_faces(
                source_image, target_image,
                source_face, target_face
            )
            
            if swap_result is None:
                logger.error("Hybrid face swapping failed")
                return 1
            
            # Get statistics  
            stats = face_swapper.get_statistics()
            logger.info(f"Hybrid face swapper statistics: {stats}")
            
        else:
            # Use basic method
            logger.info("Initializing basic face swapper...")
            config = SwapConfig(
                method=SwapMethod.CLASSICAL,
                target_size=(256, 256),
                smoothing_factor=0.8,
                color_correction=True
            )
            
            face_swapper = FaceSwapper(config)
            
            logger.info("Performing basic face swap...")
            swap_result_obj = face_swapper.swap_faces(
                source_image, target_image,
                source_face, target_face
            )
            
            if not swap_result_obj.success:
                logger.error(f"Face swapping failed: {swap_result_obj.error_message}")
                return 1
            
            swap_result = swap_result_obj.swapped_image
            logger.info(f"Face swap completed in {swap_result_obj.processing_time:.2f}s")
        
        # Save result
        logger.info(f"Saving result to: {args.output}")
        
        # Create output directory if needed
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        success = cv2.imwrite(args.output, swap_result)
        if not success:
            logger.error(f"Failed to save result image: {args.output}")
            return 1
        
        logger.info("Face swap demo completed successfully!")
        
        # Show result if requested
        if args.show:
            show_results(source_image, target_image, swap_result, args.method)
        
        return 0
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def show_results(source_image: np.ndarray, target_image: np.ndarray, 
                result_image: np.ndarray, method: str) -> None:
    """
    Display results in OpenCV windows.
    
    Args:
        source_image: Source image
        target_image: Target image
        result_image: Face swap result
        method: Swap method used
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
        cv2.putText(combined, f"Result ({method})", (x_offset + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display
        cv2.imshow(f"Face Swap Demo - {method.capitalize()} Method - Press any key to close", combined)
        logger.info("Press any key to close the display window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        logger.error(f"Display failed: {e}")


if __name__ == "__main__":
    sys.exit(main())