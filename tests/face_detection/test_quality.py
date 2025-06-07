import unittest
import numpy as np
import cv2
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from face_detection.quality import FaceQualityAssessor


class TestFaceQualityAssessor(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.assessor = FaceQualityAssessor()
        
        # Create test images
        self.good_face = self._create_test_face(size=128, blur=False, brightness=128)
        self.blurry_face = self._create_test_face(size=128, blur=True, brightness=128)
        self.dark_face = self._create_test_face(size=128, blur=False, brightness=30)
        self.bright_face = self._create_test_face(size=128, blur=False, brightness=200)
        self.small_face = self._create_test_face(size=32, blur=False, brightness=128)
    
    def _create_test_face(self, size: int, blur: bool, brightness: int) -> np.ndarray:
        """Create a synthetic test face image."""
        # Create a simple face-like pattern
        face = np.ones((size, size, 3), dtype=np.uint8) * brightness
        
        # Add some facial features
        # Eyes
        eye_size = size // 8
        left_eye_center = (size // 3, size // 3)
        right_eye_center = (2 * size // 3, size // 3)
        
        cv2.circle(face, left_eye_center, eye_size, (0, 0, 0), -1)
        cv2.circle(face, right_eye_center, eye_size, (0, 0, 0), -1)
        
        # Mouth
        mouth_center = (size // 2, 2 * size // 3)
        mouth_size = (size // 4, size // 8)
        cv2.ellipse(face, mouth_center, mouth_size, 0, 0, 180, (0, 0, 0), -1)
        
        # Apply blur if requested
        if blur:
            face = cv2.GaussianBlur(face, (15, 15), 5)
        
        return face
    
    def test_init(self):
        """Test assessor initialization."""
        assessor = FaceQualityAssessor(
            blur_threshold=150.0,
            brightness_range=(40, 220),
            min_face_size=100
        )
        
        self.assertEqual(assessor.blur_threshold, 150.0)
        self.assertEqual(assessor.brightness_range, (40, 220))
        self.assertEqual(assessor.min_face_size, 100)
    
    def test_assess_quality_good_face(self):
        """Test quality assessment for good face."""
        quality = self.assessor.assess_quality(self.good_face)
        
        # Should have all required metrics
        required_keys = ['overall_score', 'blur_score', 'brightness_score', 
                        'size_score', 'pose_score', 'occlusion_score']
        for key in required_keys:
            self.assertIn(key, quality)
            self.assertGreaterEqual(quality[key], 0.0)
            self.assertLessEqual(quality[key], 1.0)
        
        # Good face should have decent overall score
        self.assertGreater(quality['overall_score'], 0.5)
    
    def test_assess_quality_invalid_input(self):
        """Test quality assessment with invalid input."""
        # Test with None
        quality = self.assessor.assess_quality(None)
        self.assertEqual(quality['overall_score'], 0.0)
        
        # Test with empty array
        empty_image = np.array([])
        quality = self.assessor.assess_quality(empty_image)
        self.assertEqual(quality['overall_score'], 0.0)
    
    def test_assess_blur(self):
        """Test blur assessment."""
        good_blur = self.assessor._assess_blur(self.good_face)
        blurry_blur = self.assessor._assess_blur(self.blurry_face)
        
        # Good face should have higher blur score than blurry face
        self.assertGreater(good_blur, blurry_blur)
        self.assertGreaterEqual(good_blur, 0.0)
        self.assertLessEqual(good_blur, 1.0)
        self.assertGreaterEqual(blurry_blur, 0.0)
        self.assertLessEqual(blurry_blur, 1.0)
    
    def test_assess_brightness(self):
        """Test brightness assessment."""
        good_brightness = self.assessor._assess_brightness(self.good_face)
        dark_brightness = self.assessor._assess_brightness(self.dark_face)
        bright_brightness = self.assessor._assess_brightness(self.bright_face)
        
        # Good lighting should score higher than poor lighting
        self.assertGreater(good_brightness, dark_brightness)
        self.assertGreater(good_brightness, bright_brightness)
    
    def test_assess_size(self):
        """Test size assessment."""
        good_size = self.assessor._assess_size(self.good_face)
        small_size = self.assessor._assess_size(self.small_face)
        
        # Larger face should score higher
        self.assertGreater(good_size, small_size)
        self.assertEqual(good_size, 1.0)  # Good face meets size requirement
    
    def test_assess_pose(self):
        """Test pose assessment."""
        # Test with landmarks
        landmarks = [(42, 42), (84, 42), (64, 64), (54, 84), (74, 84)]  # Eyes, nose, mouth corners
        
        pose_score = self.assessor._assess_pose(landmarks, self.good_face.shape)
        
        self.assertGreaterEqual(pose_score, 0.0)
        self.assertLessEqual(pose_score, 1.0)
    
    def test_assess_pose_insufficient_landmarks(self):
        """Test pose assessment with insufficient landmarks."""
        landmarks = [(42, 42)]  # Only one landmark
        
        pose_score = self.assessor._assess_pose(landmarks, self.good_face.shape)
        
        # Should return default score
        self.assertEqual(pose_score, 0.7)
    
    def test_assess_occlusion(self):
        """Test occlusion assessment."""
        # Create an occluded face (add black rectangle)
        occluded_face = self.good_face.copy()
        cv2.rectangle(occluded_face, (50, 50), (100, 100), (0, 0, 0), -1)
        
        good_occlusion = self.assessor._assess_occlusion(self.good_face)
        occluded_occlusion = self.assessor._assess_occlusion(occluded_face)
        
        # Non-occluded face should score higher
        self.assertGreaterEqual(good_occlusion, occluded_occlusion)
    
    def test_assess_quality_with_landmarks(self):
        """Test quality assessment with landmark information."""
        face_info = {
            'landmarks': [(42, 42), (84, 42), (64, 64), (54, 84), (74, 84)]
        }
        
        quality = self.assessor.assess_quality(self.good_face, face_info)
        
        # Should include pose assessment
        self.assertGreater(quality['pose_score'], 0.0)
    
    def test_filter_faces(self):
        """Test face filtering based on quality."""
        faces = [
            {'image': self.good_face},
            {'image': self.blurry_face},
            {'image': self.small_face}
        ]
        
        filtered_faces = self.assessor.filter_faces(faces, min_quality=0.5)
        
        # Should filter out low quality faces
        self.assertLessEqual(len(filtered_faces), len(faces))
        
        # All filtered faces should have quality info
        for face in filtered_faces:
            self.assertIn('quality', face)
            self.assertGreaterEqual(face['quality']['overall_score'], 0.5)
    
    def test_filter_faces_empty_list(self):
        """Test filtering empty face list."""
        filtered_faces = self.assessor.filter_faces([], min_quality=0.5)
        self.assertEqual(len(filtered_faces), 0)
    
    def test_filter_faces_no_images(self):
        """Test filtering faces without image data."""
        faces = [
            {'bbox': (10, 10, 50, 50)},  # No image key
            {'image': self.good_face}
        ]
        
        filtered_faces = self.assessor.filter_faces(faces, min_quality=0.5)
        
        # Should only process faces with image data
        self.assertLessEqual(len(filtered_faces), 1)


if __name__ == '__main__':
    unittest.main()