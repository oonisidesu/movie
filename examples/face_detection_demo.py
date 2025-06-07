#!/usr/bin/env python3
"""
Face Detection Module Demo

This script demonstrates the basic usage of the face detection module
with sample images and video processing.
"""

import cv2
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from face_detection import FaceDetector, FaceTracker, FaceQualityAssessor


def create_sample_image():
    """Create a sample image with a simple face-like pattern."""
    image = np.ones((300, 300, 3), dtype=np.uint8) * 128
    
    # Face outline
    cv2.ellipse(image, (150, 150), (80, 100), 0, 0, 360, (200, 180, 150), -1)
    
    # Eyes
    cv2.circle(image, (120, 130), 15, (50, 50, 50), -1)
    cv2.circle(image, (180, 130), 15, (50, 50, 50), -1)
    
    # Nose
    cv2.ellipse(image, (150, 160), (8, 15), 0, 0, 360, (150, 120, 100), -1)
    
    # Mouth
    cv2.ellipse(image, (150, 190), (25, 12), 0, 0, 180, (100, 50, 50), -1)
    
    return image


def demo_face_detection():
    """Demonstrate basic face detection."""
    print("=== Face Detection Demo ===")
    
    # Create sample image
    image = create_sample_image()
    
    # Initialize detector
    detector = FaceDetector(detection_confidence=0.3)
    
    try:
        # Detect faces
        faces = detector.detect_faces(image, method='opencv')
        print(f"Detected {len(faces)} face(s)")
        
        for i, face in enumerate(faces):
            print(f"Face {i}: bbox={face['bbox']}, confidence={face['confidence']:.2f}")
        
        # Draw results
        result_image = detector.draw_faces(image, faces)
        
        # Save result
        cv2.imwrite('face_detection_result.jpg', result_image)
        print("Result saved as 'face_detection_result.jpg'")
        
    except Exception as e:
        print(f"Face detection failed: {e}")


def demo_face_tracking():
    """Demonstrate face tracking with synthetic video."""
    print("\n=== Face Tracking Demo ===")
    
    # Initialize tracker
    tracker = FaceTracker(max_missing_frames=5)
    
    # Create synthetic video frames
    frames = []
    for i in range(10):
        frame = create_sample_image()
        # Simulate movement by shifting the image
        shift_x = i * 5
        shift_y = i * 2
        
        # Create translation matrix
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
        frames.append(frame)
    
    print(f"Processing {len(frames)} frames...")
    
    try:
        all_tracks = []
        for frame_idx, frame in enumerate(frames):
            tracks = tracker.process_frame(frame)
            all_tracks.append(tracks)
            
            if tracks:
                for track in tracks:
                    print(f"Frame {frame_idx}: Track {track['track_id']}, "
                          f"bbox={track['bbox']}, age={track['age']}")
        
        print(f"Tracking completed. Total unique tracks: {tracker.next_track_id}")
        
    except Exception as e:
        print(f"Face tracking failed: {e}")


def demo_quality_assessment():
    """Demonstrate face quality assessment."""
    print("\n=== Face Quality Assessment Demo ===")
    
    # Initialize quality assessor
    assessor = FaceQualityAssessor()
    
    # Create faces with different quality characteristics
    good_face = create_sample_image()
    
    # Blurry face
    blurry_face = cv2.GaussianBlur(good_face, (15, 15), 5)
    
    # Dark face
    dark_face = (good_face * 0.3).astype(np.uint8)
    
    # Small face
    small_face = cv2.resize(good_face, (50, 50))
    
    faces = {
        'good': good_face,
        'blurry': blurry_face,
        'dark': dark_face,
        'small': small_face
    }
    
    try:
        for name, face in faces.items():
            quality = assessor.assess_quality(face)
            print(f"\n{name.upper()} face quality:")
            print(f"  Overall: {quality['overall_score']:.2f}")
            print(f"  Blur: {quality['blur_score']:.2f}")
            print(f"  Brightness: {quality['brightness_score']:.2f}")
            print(f"  Size: {quality['size_score']:.2f}")
            print(f"  Pose: {quality['pose_score']:.2f}")
            print(f"  Occlusion: {quality['occlusion_score']:.2f}")
        
        # Test face filtering
        face_list = [{'image': face} for face in faces.values()]
        filtered_faces = assessor.filter_faces(face_list, min_quality=0.6)
        print(f"\nFiltered faces (min_quality=0.6): {len(filtered_faces)}/{len(face_list)}")
        
    except Exception as e:
        print(f"Quality assessment failed: {e}")


def demo_integration():
    """Demonstrate integration of all components."""
    print("\n=== Integration Demo ===")
    
    # Initialize all components
    detector = FaceDetector()
    tracker = FaceTracker()
    assessor = FaceQualityAssessor()
    
    # Create a frame with multiple face regions
    frame = np.ones((400, 600, 3), dtype=np.uint8) * 100
    
    # Add multiple faces
    face1 = create_sample_image()
    face2 = cv2.resize(create_sample_image(), (150, 150))
    
    frame[50:350, 50:350] = face1
    frame[100:250, 400:550] = face2
    
    try:
        # Process frame
        tracks = tracker.process_frame(frame)
        print(f"Detected and tracked {len(tracks)} faces")
        
        high_quality_faces = []
        for track in tracks:
            # Extract face region
            face_region = detector.get_face_region(frame, track)
            
            # Assess quality
            quality = assessor.assess_quality(face_region, track)
            track['quality'] = quality
            
            print(f"Track {track['track_id']}: quality={quality['overall_score']:.2f}")
            
            if quality['overall_score'] > 0.5:
                high_quality_faces.append(track)
        
        print(f"High quality faces: {len(high_quality_faces)}")
        
        # Draw results
        result_frame = detector.draw_faces(frame, tracks)
        cv2.imwrite('integration_result.jpg', result_frame)
        print("Integration result saved as 'integration_result.jpg'")
        
    except Exception as e:
        print(f"Integration demo failed: {e}")


def main():
    """Run all demos."""
    print("Face Detection Module Demo")
    print("=" * 40)
    
    # Create output directory for results
    os.makedirs('demo_output', exist_ok=True)
    os.chdir('demo_output')
    
    try:
        demo_face_detection()
        demo_face_tracking()
        demo_quality_assessment()
        demo_integration()
        
        print("\n" + "=" * 40)
        print("Demo completed successfully!")
        print("Check the 'demo_output' directory for result images.")
        
    except Exception as e:
        print(f"Demo failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())