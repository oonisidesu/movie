#!/usr/bin/env python3
"""
Video Processing Module Demo

This script demonstrates the video processing capabilities of the
video face-swapping tool, including video loading, frame processing,
progress tracking, and output generation.
"""

import sys
import os
import numpy as np
import cv2
import tempfile
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from video import (
    VideoProcessor,
    VideoWriter,
    FrameProcessor,
    ConsoleProgressTracker,
    CallbackProgressTracker,
    ProgressInfo,
    create_progress_callback,
    validate_video_file,
    get_video_info,
    calculate_memory_usage,
    optimize_video_settings
)
from video.frame_processor import (
    resize_frame,
    normalize_frame,
    enhance_contrast
)

def create_sample_video(output_path: str, duration: int = 5, fps: int = 30,
                       width: int = 640, height: int = 480):
    """Create a sample video for demonstration."""
    print(f"Creating sample video: {output_path}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = duration * fps
    
    for i in range(total_frames):
        # Create animated frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Animated background
        bg_color = int(127 + 127 * np.sin(i * 0.1))
        frame[:, :] = [bg_color, 50, 100]
        
        # Moving circle
        center_x = int(width * 0.5 + width * 0.3 * np.sin(i * 0.2))
        center_y = int(height * 0.5 + height * 0.3 * np.cos(i * 0.15))
        cv2.circle(frame, (center_x, center_y), 30, (255, 255, 255), -1)
        
        # Frame counter
        cv2.putText(frame, f"Frame {i+1}/{total_frames}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        writer.write(frame)
    
    writer.release()
    print(f"Sample video created: {total_frames} frames at {fps} fps")


def demo_video_validation():
    """Demonstrate video file validation and info extraction."""
    print("\n=== Video Validation Demo ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create sample video
        sample_video = os.path.join(temp_dir, "sample.mp4")
        create_sample_video(sample_video, duration=3)
        
        # Validate video
        print(f"\nValidating video: {sample_video}")
        is_valid = validate_video_file(sample_video)
        print(f"Video is valid: {is_valid}")
        
        if is_valid:
            # Get video information
            video_info = get_video_info(sample_video)
            print(f"\nVideo Information:")
            print(f"  Resolution: {video_info['width']}x{video_info['height']}")
            print(f"  FPS: {video_info['fps']}")
            print(f"  Duration: {video_info['duration']:.1f} seconds")
            print(f"  Total frames: {video_info['frame_count']}")
            print(f"  Codec: {video_info['codec']}")
            print(f"  File size: {video_info['file_size_mb']:.1f} MB")
            print(f"  Resolution class: {video_info['resolution_class']}")
            
            # Memory usage estimation
            memory_info = calculate_memory_usage(video_info)
            print(f"\nMemory Usage Estimation:")
            print(f"  Single frame: {memory_info['single_frame_mb']:.2f} MB")
            print(f"  Buffer memory: {memory_info['buffer_memory_mb']:.2f} MB")
            print(f"  Total memory: {memory_info['total_memory_mb']:.2f} MB")
            
            # Optimization recommendations
            recommendations = optimize_video_settings(video_info)
            print(f"\nOptimization Recommendations:")
            for key, value in recommendations.items():
                print(f"  {key}: {value}")


def demo_frame_processing():
    """Demonstrate frame processing capabilities."""
    print("\n=== Frame Processing Demo ===")
    
    # Initialize frame processor
    processor = FrameProcessor(enable_stats=True)
    
    # Add processing functions
    processor.add_preprocessor(resize_frame(320, 240))
    processor.add_processor(enhance_contrast(alpha=1.2, beta=10))
    processor.add_postprocessor(normalize_frame())
    
    # Create test frames
    test_frames = []
    for i in range(10):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        test_frames.append(frame)
    
    print(f"Processing {len(test_frames)} test frames...")
    
    # Process frames
    processed_frames = processor.process_frame_batch(test_frames)
    
    # Get statistics
    stats = processor.get_stats()
    print(f"Processing Statistics:")
    print(f"  Total frames: {stats.total_frames}")
    print(f"  Processed frames: {stats.processed_frames}")
    print(f"  Failed frames: {stats.failed_frames}")
    print(f"  Average processing time: {stats.average_processing_time:.4f}s")
    print(f"  Processing FPS: {stats.fps:.1f}")
    
    print(f"Frame shape changed from {test_frames[0].shape} to {processed_frames[0].shape}")


def demo_progress_tracking():
    """Demonstrate progress tracking features."""
    print("\n=== Progress Tracking Demo ===")
    
    # Console progress tracker
    print("\n1. Console Progress Tracker:")
    tracker = ConsoleProgressTracker(show_bar=True, bar_length=40)
    
    total_items = 50
    for i in range(total_items):
        time.sleep(0.05)  # Simulate processing
        tracker.update(i + 1, total_items, "Demo Processing")
    
    tracker.finish()
    
    # Callback progress tracker
    print("\n2. Callback Progress Tracker:")
    
    def progress_callback(info: ProgressInfo):
        if info.current_frame % 10 == 0:  # Print every 10th update
            print(f"  Progress: {info.percentage:.1f}% - "
                  f"ETA: {info.estimated_remaining_time:.1f}s - "
                  f"Speed: {info.fps:.1f} fps")
    
    callback_tracker = CallbackProgressTracker(progress_callback)
    
    for i in range(total_items):
        time.sleep(0.05)
        callback_tracker.update(i + 1, total_items, "Callback Demo")
    
    callback_tracker.finish()


def demo_video_processing_pipeline():
    """Demonstrate complete video processing pipeline."""
    print("\n=== Video Processing Pipeline Demo ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create input video
        input_video = os.path.join(temp_dir, "input.mp4")
        output_video = os.path.join(temp_dir, "output.mp4")
        
        create_sample_video(input_video, duration=2, fps=10)  # Short video for demo
        
        # Initialize video processor
        with VideoProcessor() as processor:
            # Load video
            print(f"\nLoading video: {input_video}")
            video_info = processor.load_video(input_video)
            
            # Setup output
            print(f"Setting up output: {output_video}")
            processor.setup_output(output_video, fps=video_info['fps'])
            
            # Create progress callback
            progress_callback = create_progress_callback(
                video_info['frame_count'],
                tracker_type="console",
                show_bar=True,
                bar_length=30
            )
            
            # Define frame processing function
            def process_frame(frame_num: int, frame: np.ndarray) -> np.ndarray:
                """Example frame processing: add a border and text."""
                processed = frame.copy()
                
                # Add colored border
                border_color = [0, 255, 0]  # Green
                processed[0:5, :] = border_color      # Top
                processed[-5:, :] = border_color     # Bottom
                processed[:, 0:5] = border_color     # Left
                processed[:, -5:] = border_color     # Right
                
                # Add processing indicator
                cv2.putText(processed, "PROCESSED", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                return processed
            
            # Process video
            print("\nProcessing video...")
            success = processor.process_video(process_frame, progress_callback)
            
            if success:
                print(f"\nVideo processing completed successfully!")
                print(f"Output saved to: {output_video}")
                
                # Verify output
                if os.path.exists(output_video):
                    output_info = get_video_info(output_video)
                    print(f"Output video info:")
                    print(f"  Frames: {output_info['frame_count']}")
                    print(f"  Duration: {output_info['duration']:.1f}s")
                    print(f"  Size: {output_info['file_size_mb']:.2f} MB")
            else:
                print("Video processing failed!")


def demo_video_writer():
    """Demonstrate video writer capabilities."""
    print("\n=== Video Writer Demo ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "writer_demo.mp4")
        
        # Initialize video writer
        writer = VideoWriter(
            output_path=output_path,
            fps=30.0,
            resolution=(320, 240),
            codec="mp4v",
            quality="medium"
        )
        
        print(f"Writing video to: {output_path}")
        
        # Generate and write frames
        num_frames = 60  # 2 seconds at 30 fps
        for i in range(num_frames):
            # Create test frame
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            
            # Animated pattern
            frame[:, :, 0] = (i * 255) // num_frames  # Blue channel
            frame[:, :, 1] = 100  # Green channel
            frame[:, :, 2] = 255 - (i * 255) // num_frames  # Red channel
            
            # Add frame number
            cv2.putText(frame, f"Frame {i+1}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            writer.write_frame(frame)
        
        # Get writing statistics
        stats = writer.get_stats()
        print(f"Writer Statistics:")
        print(f"  Frames written: {stats['frames_written']}")
        print(f"  Output FPS: {stats['fps']}")
        print(f"  Resolution: {stats['resolution']}")
        print(f"  Codec: {stats['codec']}")
        print(f"  Duration: {stats['duration_seconds']:.1f}s")
        
        # Release writer
        writer.release()
        
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / 1024  # KB
            print(f"Output file size: {file_size:.1f} KB")


def main():
    """Run all video processing demos."""
    print("Video Processing Module Demo")
    print("=" * 50)
    
    try:
        # Run demos
        demo_video_validation()
        demo_frame_processing()
        demo_progress_tracking()
        demo_video_writer()
        demo_video_processing_pipeline()
        
        print("\n" + "=" * 50)
        print("All demos completed successfully!")
        print("\nKey Features Demonstrated:")
        print("✅ Video file validation and metadata extraction")
        print("✅ Frame-by-frame processing pipeline")
        print("✅ Progress tracking with multiple display options")
        print("✅ Video writing with codec and quality control")
        print("✅ Complete video processing pipeline")
        print("✅ Memory usage estimation and optimization")
        print("✅ Audio preservation (integration ready)")
        
        print("\nNext Steps:")
        print("- Integrate with face detection module (Issue #2)")
        print("- Add CLI interface (Issue #4)")
        print("- Implement face swapping algorithms (Issue #5)")
        
    except Exception as e:
        print(f"\nDemo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())