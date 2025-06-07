#!/usr/bin/env python3
"""
Optimized Video Face Swapping Demo

Processes video with frame skipping for faster preview generation.
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
from src.face_swapping.hybrid_swapper import HybridFaceSwapper
from src.video import VideoProcessor
from src.utils.logging_config import setup_logging

# Setup logging
setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description='Optimized Video Face Swapping Demo')
    parser.add_argument('--source', required=True, help='Source face image path')
    parser.add_argument('--target-video', required=True, help='Target video path')
    parser.add_argument('--output', required=True, help='Output video path')
    parser.add_argument('--duration', type=float, default=60.0, help='Duration in seconds to process')
    parser.add_argument('--frame-skip', type=int, default=3, help='Process every Nth frame (1=all frames)')
    parser.add_argument('--start-time', type=float, default=0.0, help='Start time in seconds')
    parser.add_argument('--horizontal-shift', type=float, default=0.2, help='Horizontal position adjustment')
    parser.add_argument('--vertical-shift', type=float, default=0.0, help='Vertical position adjustment')
    
    args = parser.parse_args()
    
    try:
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
            enable_expression_transfer=True,
            horizontal_shift=args.horizontal_shift,
            vertical_shift=args.vertical_shift
        )
        
        # Detect source face
        source_faces = face_detector.detect_faces(source_image)
        if not source_faces:
            logger.error("No face detected in source image")
            return 1
        
        source_face = source_faces[0]
        
        # Load video
        logger.info("Loading target video...")
        video_processor = VideoProcessor()
        video_info = video_processor.load_video(args.target_video)
        
        fps = video_info['fps']
        start_frame = int(args.start_time * fps)
        duration_frames = int(args.duration * fps)
        end_frame = start_frame + duration_frames
        
        # Get reference frame
        reference_frame = video_processor.get_frame_at(start_frame)
        if reference_frame is None:
            reference_frame = video_processor.get_frame_at(0)
        
        reference_target_faces = face_detector.detect_faces(reference_frame)
        if not reference_target_faces:
            logger.error("No face detected in reference frame")
            return 1
        
        reference_target_face = reference_target_faces[0]
        
        # Set reference faces
        success = face_swapper.set_reference_faces(
            source_image, reference_frame,
            source_face, reference_target_face
        )
        
        if not success:
            logger.error("Failed to set reference faces")
            return 1
        
        # Setup output
        output_fps = fps / args.frame_skip  # Adjust FPS for frame skipping
        video_processor.setup_output(
            args.output,
            fps=output_fps,
            resolution=(video_info['width'], video_info['height'])
        )
        
        logger.info(f"Processing {args.duration}s from {args.start_time}s (frames {start_frame}-{end_frame}, every {args.frame_skip})")
        
        # Process frames with skipping
        processed_count = 0
        success_count = 0
        
        for frame_num in range(start_frame, end_frame, args.frame_skip):
            if frame_num >= video_info['frame_count']:
                break
                
            processed_count += 1
            if processed_count % 60 == 0:  # Progress every 60 processed frames
                logger.info(f"Processed {processed_count} frames ({success_count} successful)")
            
            frame = video_processor.get_frame_at(frame_num)
            if frame is None:
                continue
            
            # Detect faces
            target_faces = face_detector.detect_faces(frame)
            if not target_faces:
                # Use original frame if no face detected
                video_processor.video_writer.write_frame(frame)
                continue
            
            target_face = target_faces[0]
            
            # Perform face swap
            swapped_frame = face_swapper.swap_faces_with_expression(
                source_image, frame, source_face, target_face
            )
            
            if swapped_frame is not None:
                video_processor.video_writer.write_frame(swapped_frame)
                success_count += 1
            else:
                video_processor.video_writer.write_frame(frame)
        
        # Finalize
        success = video_processor.finalize_output()
        
        if success:
            logger.info(f"Video processing complete!")
            logger.info(f"Processed: {processed_count} frames")
            logger.info(f"Successful swaps: {success_count}")
            logger.info(f"Success rate: {success_count/processed_count*100:.1f}%")
            logger.info(f"Output: {args.output}")
            
            # Show statistics
            stats = face_swapper.get_statistics()
            logger.info(f"Face swapper statistics: {stats}")
        else:
            logger.error("Failed to finalize video output")
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        if 'video_processor' in locals():
            video_processor.cleanup()


if __name__ == "__main__":
    sys.exit(main())