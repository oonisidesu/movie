# Video Processing Module Documentation

## Overview

The video processing module provides comprehensive video manipulation capabilities for the video face-swapping tool. It handles video file loading, frame-by-frame processing, output generation with audio preservation, and progress tracking.

## Architecture

The module consists of several key components:

- **VideoProcessor**: Main processing pipeline
- **VideoWriter**: Video output with codec support  
- **FrameProcessor**: Frame-by-frame processing utilities
- **ProgressTracker**: Progress tracking and reporting
- **VideoUtils**: Utility functions for video operations

## Components

### 1. VideoProcessor (`src/video/video_processor.py`)

The main video processing pipeline that orchestrates the entire video processing workflow.

#### Key Features
- Video file loading and validation
- Frame extraction and processing
- Audio preservation during processing
- Memory-efficient processing
- Progress tracking integration

#### Usage Example

```python
from src.video import VideoProcessor, create_progress_callback

# Initialize processor
processor = VideoProcessor()

# Load input video
video_info = processor.load_video("input.mp4")
print(f"Video: {video_info['width']}x{video_info['height']} @ {video_info['fps']} fps")

# Setup output
processor.setup_output("output.mp4", fps=30, resolution=(1920, 1080))

# Define frame processing function
def process_frame(frame_num: int, frame):
    # Your frame processing logic here
    # For example, apply face swapping
    processed_frame = frame.copy()
    # ... processing logic ...
    return processed_frame

# Create progress callback
progress_callback = create_progress_callback(
    video_info['frame_count'], 
    tracker_type="console"
)

# Process video
success = processor.process_video(process_frame, progress_callback)

if success:
    print("Video processing completed successfully!")
else:
    print("Video processing failed!")

# Cleanup
processor.cleanup()
```

#### Context Manager Usage

```python
with VideoProcessor() as processor:
    processor.load_video("input.mp4")
    processor.setup_output("output.mp4")
    # Processing...
    # Cleanup happens automatically
```

### 2. VideoWriter (`src/video/video_writer.py`)

Handles video output with support for different codecs and quality settings.

#### Supported Codecs
- **MP4V**: Good compatibility, medium quality
- **H264**: High quality, modern standard
- **XVID**: AVI format, good compression
- **MJPG**: High quality, larger files

#### Usage Example

```python
from src.video import VideoWriter

# Initialize writer
writer = VideoWriter(
    output_path="output.mp4",
    fps=30.0,
    resolution=(1920, 1080),
    codec="h264",
    quality="high"
)

# Write frames
for frame in frames:
    writer.write_frame(frame)

# Get statistics
stats = writer.get_stats()
print(f"Wrote {stats['frames_written']} frames")

# Cleanup
writer.release()
```

#### Quality Settings

```python
# Different quality levels
writer_low = VideoWriter("output_low.mp4", 30, (1920, 1080), quality="low")
writer_med = VideoWriter("output_med.mp4", 30, (1920, 1080), quality="medium") 
writer_high = VideoWriter("output_high.mp4", 30, (1920, 1080), quality="high")
```

### 3. FrameProcessor (`src/video/frame_processor.py`)

Provides flexible frame-by-frame processing capabilities with a pipeline architecture.

#### Basic Usage

```python
from src.video import FrameProcessor
from src.video.frame_processor import resize_frame, enhance_contrast

# Initialize processor
processor = FrameProcessor(enable_stats=True)

# Add processing functions
processor.add_preprocessor(resize_frame(640, 480))
processor.add_processor(enhance_contrast(alpha=1.2, beta=10))

# Process single frame
processed_frame = processor.process_frame(input_frame)

# Process batch
processed_frames = processor.process_frame_batch(frame_list)

# Get processing statistics
stats = processor.get_stats()
print(f"Processed {stats.processed_frames} frames at {stats.fps:.1f} fps")
```

#### Face Swapping Integration

```python
from src.video import FaceSwappingFrameProcessor
from src.face_detection import FaceDetector, FaceTracker, FaceQualityAssessor

# Initialize specialized processor
processor = FaceSwappingFrameProcessor()

# Setup face processing components
detector = FaceDetector()
tracker = FaceTracker()
assessor = FaceQualityAssessor()

processor.setup_face_processing(
    face_detector=detector,
    face_tracker=tracker,
    quality_assessor=assessor
)

# Add face swapping processor
processor.add_face_swap_processor("source_face.jpg")

# Process frames with face swapping
processed_frame = processor.process_frame(input_frame)
```

### 4. Progress Tracking (`src/video/progress_tracker.py`)

Comprehensive progress tracking for video processing operations.

#### Console Progress Tracker

```python
from src.video import ConsoleProgressTracker

tracker = ConsoleProgressTracker(show_bar=True, bar_length=50)

for i in range(100):
    # Simulate processing
    time.sleep(0.1)
    tracker.update(i + 1, 100, "Processing video")

tracker.finish()
```

Output:
```
Processing video: |████████████████████████████████████████████████| 100/100 (100.0%) [00:10.0 < 00:00.0, 10.0 fps]
Completed in 10.0s
```

#### Callback Progress Tracker

```python
from src.video import CallbackProgressTracker, ProgressInfo

def progress_callback(info: ProgressInfo):
    print(f"Progress: {info.percentage:.1f}% - "
          f"ETA: {info.estimated_remaining_time:.1f}s")

tracker = CallbackProgressTracker(progress_callback)

# Use with processing
for i in range(total_frames):
    tracker.update(i + 1, total_frames)
```

#### Progress Manager

```python
from src.video import progress_manager, ConsoleProgressTracker

# Start tracking multiple operations
progress_manager.start_operation(
    "video_processing", 
    ConsoleProgressTracker(),
    total_items=1000
)

# Update progress
for i in range(1000):
    progress_manager.update_operation("video_processing", i + 1)

# Finish tracking
stats = progress_manager.finish_operation("video_processing")
print(f"Processed {stats['items_processed']} items in {stats['duration']:.1f}s")
```

### 5. Video Utilities (`src/video/utils/video_utils.py`)

Utility functions for video operations.

#### Video Validation

```python
from src.video import validate_video_file, get_video_info

# Validate video file
if validate_video_file("video.mp4"):
    print("Valid video file")
    
    # Get detailed information
    info = get_video_info("video.mp4")
    print(f"Resolution: {info['resolution_class']}")
    print(f"Duration: {info['duration']:.1f} seconds")
    print(f"Codec: {info['codec']}")
    print(f"Estimated bitrate: {info['estimated_bitrate_kbps']:.0f} kbps")
```

#### Audio Handling

```python
from src.video import extract_audio, merge_audio

# Extract audio from video
success = extract_audio("input.mp4", "audio.aac")

# Process video without audio...

# Merge audio back
success = merge_audio("processed_video.mp4", "audio.aac", "final_output.mp4")
```

#### Memory Usage Estimation

```python
from src.video import calculate_memory_usage, optimize_video_settings

# Estimate memory usage
video_info = get_video_info("large_video.mp4")
memory_info = calculate_memory_usage(video_info, buffer_frames=30)

print(f"Estimated memory usage: {memory_info['total_memory_mb']:.1f} MB")

# Get optimization recommendations
recommendations = optimize_video_settings(video_info)
if 'suggested_resize' in recommendations:
    print(f"Recommend resize to: {recommendations['suggested_resize']}")
```

## Installation and Dependencies

### Required Dependencies

```bash
pip install opencv-python numpy
```

### Optional Dependencies

```bash
# For enhanced video processing
pip install ffmpeg-python

# For progress bars
pip install tqdm
```

### FFmpeg Installation

FFmpeg is required for audio extraction and merging:

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**Windows:**
Download from https://ffmpeg.org/download.html

## Performance Optimization

### Memory Management

```python
# For large videos, process in chunks
def process_large_video(input_path, output_path, chunk_size=100):
    with VideoProcessor() as processor:
        processor.load_video(input_path)
        processor.setup_output(output_path)
        
        total_frames = processor.total_frames
        
        for start_frame in range(0, total_frames, chunk_size):
            end_frame = min(start_frame + chunk_size, total_frames)
            
            # Process chunk
            success = processor.process_frames(
                process_frame_function,
                start_frame=start_frame,
                end_frame=end_frame
            )
```

### Frame Processing Optimization

```python
# Resize frames for faster processing
processor = FrameProcessor()
processor.set_resize_target(640, 480)  # Process at lower resolution

# Use efficient processing functions
processor.add_processor(resize_frame(320, 240))  # Fast resize
processor.add_processor(normalize_frame())       # Simple normalization
```

### Quality vs Speed Trade-offs

```python
# Speed-optimized settings
writer = VideoWriter(
    "output.mp4", 
    fps=30, 
    resolution=(1280, 720),  # Lower resolution
    codec="mp4v",           # Fast codec
    quality="medium"        # Balanced quality
)

# Quality-optimized settings  
writer = VideoWriter(
    "output.mp4",
    fps=30,
    resolution=(1920, 1080), # Full HD
    codec="h264",           # High-quality codec
    quality="high"          # Maximum quality
)
```

## Error Handling

### Common Issues

1. **Codec Not Supported**
```python
try:
    writer = VideoWriter("output.mp4", 30, (1920, 1080), codec="unknown")
except ValueError as e:
    print(f"Codec error: {e}")
    # Fallback to default codec
    writer = VideoWriter("output.mp4", 30, (1920, 1080), codec="mp4v")
```

2. **Video File Corruption**
```python
from src.video import check_video_integrity

integrity = check_video_integrity("suspicious_video.mp4")
if not integrity['can_read_frames']:
    print("Video file may be corrupted")
    for issue in integrity['issues']:
        print(f"Issue: {issue}")
```

3. **Memory Issues**
```python
try:
    # Process large video
    processor.process_video(frame_callback)
except MemoryError:
    # Reduce memory usage
    recommendations = optimize_video_settings(video_info)
    if 'suggested_resize' in recommendations:
        # Process at lower resolution
        processor.setup_output(
            "output.mp4", 
            resolution=recommendations['suggested_resize']
        )
```

## Integration Examples

### Complete Face Swapping Pipeline

```python
from src.video import VideoProcessor, create_progress_callback
from src.face_detection import FaceDetector, FaceTracker, FaceQualityAssessor

def face_swap_video(input_path, output_path, source_face_path):
    # Initialize components
    detector = FaceDetector()
    tracker = FaceTracker()
    assessor = FaceQualityAssessor()
    
    with VideoProcessor() as processor:
        # Load and setup
        video_info = processor.load_video(input_path)
        processor.setup_output(output_path)
        
        # Create progress tracking
        progress_callback = create_progress_callback(
            video_info['frame_count'],
            tracker_type="console"
        )
        
        def process_frame(frame_num, frame):
            # Detect and track faces
            faces = tracker.process_frame(frame)
            
            # Process each face
            for face in faces:
                # Check quality
                face_region = detector.get_face_region(frame, face)
                quality = assessor.assess_quality(face_region)
                
                if quality['overall_score'] > 0.7:
                    # Apply face swapping (placeholder)
                    # frame = apply_face_swap(frame, face, source_face_path)
                    pass
            
            return frame
        
        # Process video
        success = processor.process_video(process_frame, progress_callback)
        return success

# Usage
success = face_swap_video("input.mp4", "output.mp4", "source_face.jpg")
```

### Batch Processing

```python
def batch_process_videos(video_list, output_dir):
    from src.video import progress_manager, ConsoleProgressTracker
    
    # Start batch tracking
    progress_manager.start_operation(
        "batch_processing",
        ConsoleProgressTracker(),
        total_items=len(video_list)
    )
    
    for i, video_path in enumerate(video_list):
        output_path = os.path.join(output_dir, f"processed_{i}.mp4")
        
        # Process individual video
        success = face_swap_video(video_path, output_path, "source.jpg")
        
        # Update batch progress
        progress_manager.update_operation("batch_processing", i + 1)
    
    # Finish batch tracking
    stats = progress_manager.finish_operation("batch_processing")
    print(f"Batch completed: {stats['items_processed']} videos processed")
```

## Testing

Run the test suite:

```bash
# Run all video tests
python -m pytest tests/video/

# Run specific test file
python -m pytest tests/video/test_video_processor.py

# Run with coverage
python -m pytest tests/video/ --cov=src/video
```

## Future Enhancements

- GPU acceleration support
- Real-time video processing
- Advanced codec options
- Streaming video support
- Enhanced error recovery
- Distributed processing support