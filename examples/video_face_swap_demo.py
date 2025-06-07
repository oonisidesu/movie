#!/usr/bin/env python3
"""
Video Face Swapping Demo with Expression Transfer

Demonstrates face swapping in video with expression transfer capabilities.
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
    parser = argparse.ArgumentParser(description='Video Face Swapping Demo with Expression Transfer')
    parser.add_argument('--source', required=True, help='Source face image path')
    parser.add_argument('--target-video', required=True, help='Target video path')
    parser.add_argument('--output', required=True, help='Output video path')
    parser.add_argument('--reference-frame', type=int, default=0, 
                       help='Frame number to use as reference for expression transfer')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum number of frames to process (for testing)')
    parser.add_argument('--preview-frames', type=int, default=5,
                       help='Number of frames to process for preview')
    parser.add_argument('--landmark-method', choices=['basic', 'mediapipe', 'dlib'], 
                       default='dlib', help='Landmark detection method')
    parser.add_argument('--horizontal-shift', type=float, default=0.15,
                       help='Horizontal position adjustment (negative=left, positive=right)')
    parser.add_argument('--vertical-shift', type=float, default=0.0,
                       help='Vertical position adjustment (negative=up, positive=down)')
    
    args = parser.parse_args()
    
    try:
        # Load source image
        logger.info("Loading source image...")
        source_image = cv2.imread(args.source)
        if source_image is None:
            logger.error(f"Failed to load source image: {args.source}")
            return 1
        
        logger.info(f"Source image shape: {source_image.shape}")
        
        # Initialize components
        logger.info("Initializing face detector and swapper...")
        face_detector = FaceDetector(
            backend=DetectionBackend.OPENCV_HAAR,
            min_detection_confidence=0.5
        )
        
        face_swapper = HybridFaceSwapper(
            landmark_method=args.landmark_method,
            enable_expression_transfer=True,
            horizontal_shift=args.horizontal_shift,
            vertical_shift=args.vertical_shift
        )
        
        # Detect source face
        logger.info("Detecting source face...")
        source_faces = face_detector.detect_faces(source_image)
        if not source_faces:
            logger.error("No face detected in source image")
            return 1
        
        source_face = source_faces[0]
        logger.info(f"Source face detected at: {source_face.bbox}")
        
        # Load video
        logger.info("Loading target video...")
        video_processor = VideoProcessor()
        video_info = video_processor.load_video(args.target_video)
        
        # Get reference frame for expression transfer
        logger.info(f"Getting reference frame {args.reference_frame}...")
        reference_frame = video_processor.get_frame_at(args.reference_frame)
        if reference_frame is None:
            logger.error(f"Failed to get reference frame {args.reference_frame}")
            return 1
        
        # Detect reference target face
        reference_target_faces = face_detector.detect_faces(reference_frame)
        if not reference_target_faces:
            logger.error("No face detected in reference frame")
            return 1
        
        reference_target_face = reference_target_faces[0]
        logger.info(f"Reference target face detected at: {reference_target_face.bbox}")
        
        # Set reference faces for expression transfer
        logger.info("Setting reference faces for expression transfer...")
        success = face_swapper.set_reference_faces(
            source_image, reference_frame,
            source_face, reference_target_face
        )
        
        if not success:
            logger.error("Failed to set reference faces")
            return 1
        
        # Process frames for preview
        max_frames = args.max_frames or args.preview_frames
        logger.info(f"Processing {max_frames} frames for preview...")
        
        # Create output directory
        output_dir = Path(args.output).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process frames
        processed_frames = []
        frame_count = 0
        
        for frame_num, frame in video_processor.get_frame_iterator(0, max_frames):
            logger.info(f"Processing frame {frame_num + 1}/{max_frames}")
            
            # Detect faces in current frame
            target_faces = face_detector.detect_faces(frame)
            if not target_faces:
                logger.warning(f"No face detected in frame {frame_num}")
                processed_frames.append(frame)
                continue
            
            target_face = target_faces[0]
            
            # Perform face swap with expression transfer
            swapped_frame = face_swapper.swap_faces_with_expression(
                source_image, frame,
                source_face, target_face
            )
            
            if swapped_frame is not None:
                processed_frames.append(swapped_frame)
                
                # Save individual frame for inspection
                frame_path = output_dir / f"frame_{frame_num:04d}.jpg"
                cv2.imwrite(str(frame_path), swapped_frame)
                logger.info(f"Saved frame: {frame_path}")
            else:
                logger.warning(f"Face swap failed for frame {frame_num}")
                processed_frames.append(frame)
            
            frame_count += 1
        
        # Create comparison video if we have processed frames
        if processed_frames:
            logger.info("Creating comparison video...")
            
            # Setup video output
            video_processor.setup_output(
                args.output,
                fps=video_info['fps'],
                resolution=(video_info['width'], video_info['height'])
            )
            
            # Write frames
            for i, frame in enumerate(processed_frames):
                video_processor.video_writer.write_frame(frame)
                logger.info(f"Writing frame {i + 1}/{len(processed_frames)}")
            
            # Finalize
            success = video_processor.finalize_output()
            
            if success:
                logger.info(f"Video processing complete: {args.output}")
                
                # Show statistics
                stats = face_swapper.get_statistics()
                logger.info(f"Face swapper statistics: {stats}")
            else:
                logger.error("Failed to finalize video output")
                return 1
        
        # Show first and last frames for comparison
        if len(processed_frames) >= 2:
            logger.info("Displaying first and last frames for comparison...")
            
            first_frame = processed_frames[0]
            last_frame = processed_frames[-1]
            
            # Resize for display
            def resize_for_display(img, max_height=400):
                if img.shape[0] > max_height:
                    scale = max_height / img.shape[0]
                    new_width = int(img.shape[1] * scale)
                    return cv2.resize(img, (new_width, max_height))
                return img
            
            first_display = resize_for_display(first_frame)
            last_display = resize_for_display(last_frame)
            
            # Create side-by-side comparison
            combined = np.hstack([first_display, last_display])
            cv2.putText(combined, f"Frame 0", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(combined, f"Frame {len(processed_frames)-1}", 
                       (first_display.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("Expression Transfer Comparison - Press any key to close", combined)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        logger.info("Video face swap demo completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # Cleanup
        if 'video_processor' in locals():
            video_processor.cleanup()


if __name__ == "__main__":
    sys.exit(main())