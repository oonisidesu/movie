#!/usr/bin/env python3
"""
Batch Face Swapping for target_faces folder

Applies face swapping to all images in target_faces folder using a source face.
"""

import cv2
import numpy as np
import sys
import os
import logging
import argparse
from pathlib import Path
import glob

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.face_detection import FaceDetector, DetectionBackend
from src.face_swapping.hybrid_swapper import HybridFaceSwapper
from src.utils.logging_config import setup_logging

# Setup logging
setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main batch processing function."""
    parser = argparse.ArgumentParser(description='Batch Face Swapping for target_faces')
    parser.add_argument('--source', required=True, help='Source face image path')
    parser.add_argument('--target-folder', default='target_faces', help='Target faces folder')
    parser.add_argument('--output-folder', default='demo_output/batch_results', help='Output folder')
    parser.add_argument('--horizontal-shift', type=float, default=0.2, help='Horizontal position adjustment')
    parser.add_argument('--vertical-shift', type=float, default=0.0, help='Vertical position adjustment')
    parser.add_argument('--extensions', nargs='+', default=['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG'], 
                       help='Image file extensions to process')
    
    args = parser.parse_args()
    
    try:
        # Create output folder
        output_path = Path(args.output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load source image
        logger.info("Loading source image...")
        source_image = cv2.imread(args.source)
        if source_image is None:
            logger.error(f"Failed to load source image: {args.source}")
            return 1
        
        # Initialize components
        logger.info("Initializing face detector and swapper...")
        face_detector = FaceDetector(
            backend=DetectionBackend.OPENCV_HAAR,
            min_detection_confidence=0.5
        )
        
        face_swapper = HybridFaceSwapper(
            landmark_method='dlib',
            horizontal_shift=args.horizontal_shift,
            vertical_shift=args.vertical_shift
        )
        
        # Detect source face
        source_faces = face_detector.detect_faces(source_image)
        if not source_faces:
            logger.error("No face detected in source image")
            return 1
        
        source_face = source_faces[0]
        logger.info(f"Source face detected: {source_face.bbox}")
        
        # Find all target images
        target_files = []
        for ext in args.extensions:
            pattern = os.path.join(args.target_folder, f"*.{ext}")
            target_files.extend(glob.glob(pattern))
        
        if not target_files:
            logger.error(f"No target images found in {args.target_folder}")
            return 1
        
        logger.info(f"Found {len(target_files)} target images to process")
        
        # Process each target image
        successful_swaps = 0
        for i, target_file in enumerate(target_files):
            try:
                logger.info(f"Processing {i+1}/{len(target_files)}: {os.path.basename(target_file)}")
                
                # Load target image
                target_image = cv2.imread(target_file)
                if target_image is None:
                    logger.warning(f"Failed to load target image: {target_file}")
                    continue
                
                # Detect faces in target
                target_faces = face_detector.detect_faces(target_image)
                if not target_faces:
                    logger.warning(f"No face detected in {target_file}")
                    continue
                
                target_face = target_faces[0]
                
                # Perform face swap
                swapped_image = face_swapper.swap_faces(
                    source_image, target_image, source_face, target_face
                )
                
                if swapped_image is not None:
                    # Save result
                    filename = os.path.basename(target_file)
                    name, ext = os.path.splitext(filename)
                    output_file = output_path / f"{name}_swapped{ext}"
                    
                    success = cv2.imwrite(str(output_file), swapped_image)
                    if success:
                        logger.info(f"Saved: {output_file}")
                        successful_swaps += 1
                    else:
                        logger.error(f"Failed to save: {output_file}")
                else:
                    logger.warning(f"Face swap failed for {target_file}")
                    
            except Exception as e:
                logger.error(f"Error processing {target_file}: {e}")
                continue
        
        # Show final statistics
        logger.info("="*50)
        logger.info("BATCH PROCESSING COMPLETE")
        logger.info(f"Total images processed: {len(target_files)}")
        logger.info(f"Successful swaps: {successful_swaps}")
        logger.info(f"Success rate: {successful_swaps/len(target_files)*100:.1f}%")
        logger.info(f"Output folder: {output_path}")
        
        # Show face swapper statistics
        stats = face_swapper.get_statistics()
        logger.info(f"Face swapper statistics: {stats}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())