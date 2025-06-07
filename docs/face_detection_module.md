# Face Detection Module Documentation

## Overview

The face detection module provides comprehensive face detection, tracking, and quality assessment capabilities for the video face-swapping tool. It is designed to work with both images and video streams, offering multiple detection algorithms and robust tracking across frames.

## Components

### 1. FaceDetector (`src/face_detection/detector.py`)

The core face detection class that supports multiple detection methods:

- **OpenCV Haar Cascade**: Fast but less accurate, good for real-time applications
- **OpenCV DNN**: More accurate, requires pre-trained models
- **MediaPipe**: Google's solution with landmarks, excellent accuracy

#### Usage Example

```python
from src.face_detection import FaceDetector
import cv2

# Initialize detector
detector = FaceDetector(detection_confidence=0.7)

# Load image
image = cv2.imread('face.jpg')

# Detect faces
faces = detector.detect_faces(image, method='auto')

# Draw results
result_image = detector.draw_faces(image, faces)
cv2.imshow('Detected Faces', result_image)
```

#### Methods

- `detect_faces(image, method='auto')`: Detect faces in an image
- `draw_faces(image, faces, draw_landmarks=True)`: Draw detection results
- `get_face_region(image, face, padding=0.2)`: Extract face region with padding

### 2. FaceTracker (`src/face_detection/tracker.py`)

Tracks faces across video frames with consistent IDs using IoU-based matching.

#### Usage Example

```python
from src.face_detection import FaceTracker
import cv2

# Initialize tracker
tracker = FaceTracker(max_missing_frames=10, iou_threshold=0.5)

# Process video frames
cap = cv2.VideoCapture('video.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Track faces in current frame
    tracked_faces = tracker.process_frame(frame)
    
    # Process each tracked face
    for face in tracked_faces:
        track_id = face['track_id']
        bbox = face['bbox']
        age = face['age']  # How long this face has been tracked
        
        print(f"Face {track_id}: age={age}, bbox={bbox}")
```

#### Features

- **Consistent IDs**: Faces maintain the same ID across frames
- **Missing frame handling**: Continues tracking even when faces are temporarily lost
- **Multiple face support**: Can track multiple faces simultaneously
- **Track history**: Maintains movement history for each face

### 3. FaceQualityAssessor (`src/face_detection/quality.py`)

Evaluates face image quality for optimal processing results.

#### Quality Metrics

- **Blur Score**: Measures image sharpness using Laplacian variance
- **Brightness Score**: Evaluates lighting conditions
- **Size Score**: Checks if face is large enough for processing
- **Pose Score**: Estimates face orientation (frontal vs. profile)
- **Occlusion Score**: Detects if face is partially hidden

#### Usage Example

```python
from src.face_detection import FaceQualityAssessor
import cv2

# Initialize quality assessor
assessor = FaceQualityAssessor(
    blur_threshold=100.0,
    brightness_range=(50, 200),
    min_face_size=80
)

# Assess face quality
face_image = cv2.imread('face.jpg')
quality_metrics = assessor.assess_quality(face_image)

print(f"Overall quality: {quality_metrics['overall_score']:.2f}")
print(f"Blur score: {quality_metrics['blur_score']:.2f}")
print(f"Brightness score: {quality_metrics['brightness_score']:.2f}")

# Filter faces based on quality
faces_with_images = [{'image': face_image}]
good_faces = assessor.filter_faces(faces_with_images, min_quality=0.6)
```

## Installation

Install required dependencies:

```bash
pip install -r requirements_face_detection.txt
```

### Optional Dependencies

For enhanced functionality, install additional packages:

```bash
# For MediaPipe support (recommended)
pip install mediapipe

# For dlib support (advanced landmarks)
pip install dlib
```

### Model Files

For OpenCV DNN detection, download required model files:

1. **deploy.prototxt**: Network architecture definition
2. **res10_300x300_ssd_iter_140000.caffemodel**: Pre-trained weights

Place these files in the `models/` directory.

## Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/face_detection/

# Run specific test file
python -m pytest tests/face_detection/test_detector.py

# Run with coverage
python -m pytest tests/face_detection/ --cov=src/face_detection
```

## Configuration

### FaceDetector Configuration

```python
detector = FaceDetector(
    detection_confidence=0.5,  # Minimum confidence threshold
    model_selection=0          # MediaPipe model (0 or 1)
)
```

### FaceTracker Configuration

```python
tracker = FaceTracker(
    max_missing_frames=10,     # Frames before track is deleted
    iou_threshold=0.5          # IoU threshold for matching
)
```

### FaceQualityAssessor Configuration

```python
assessor = FaceQualityAssessor(
    blur_threshold=100.0,           # Laplacian variance threshold
    brightness_range=(50, 200),     # Acceptable brightness range
    min_face_size=80                # Minimum face size in pixels
)
```

## Performance Considerations

### Speed vs. Accuracy Trade-offs

1. **Fastest**: Haar Cascade (`method='opencv'` with DNN disabled)
2. **Balanced**: OpenCV DNN (`method='opencv'` with DNN enabled)
3. **Most Accurate**: MediaPipe (`method='mediapipe'`)

### Memory Usage

- Face tracking stores history for each track
- Large videos may require periodic track cleanup
- Consider reducing `max_missing_frames` for memory-constrained environments

### GPU Acceleration

- OpenCV DNN supports GPU acceleration with CUDA
- MediaPipe can utilize GPU for inference
- Set appropriate backend for your hardware

## Integration with Video Processing

```python
import cv2
from src.face_detection import FaceTracker, FaceQualityAssessor

def process_video(video_path, output_path):
    tracker = FaceTracker()
    assessor = FaceQualityAssessor()
    
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (640, 480))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Track faces
        faces = tracker.process_frame(frame)
        
        # Filter by quality
        for face in faces:
            face_region = tracker.detector.get_face_region(frame, face)
            quality = assessor.assess_quality(face_region, face)
            
            if quality['overall_score'] > 0.7:
                # Process high-quality faces
                # TODO: Apply face swapping here
                pass
        
        # Draw results
        frame_with_faces = tracker.detector.draw_faces(frame, faces)
        out.write(frame_with_faces)
    
    cap.release()
    out.release()
```

## Troubleshooting

### Common Issues

1. **"No module named 'mediapipe'"**
   - Install MediaPipe: `pip install mediapipe`
   - Or disable MediaPipe usage in the code

2. **"DNN model not found"**
   - Download required model files to `models/` directory
   - Or use Haar Cascade fallback

3. **Poor detection accuracy**
   - Try different detection methods
   - Adjust detection confidence threshold
   - Ensure good lighting conditions

4. **Memory issues with long videos**
   - Reduce `max_missing_frames` parameter
   - Implement periodic track cleanup
   - Process video in chunks

### Performance Optimization

- Use GPU acceleration when available
- Resize frames for faster processing
- Implement frame skipping for real-time applications
- Cache detection models for multiple uses

## Future Enhancements

- Support for additional face detection models
- Advanced pose estimation
- Face recognition capabilities
- Integration with face alignment algorithms
- Real-time processing optimizations