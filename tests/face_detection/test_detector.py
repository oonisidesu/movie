import unittest
import numpy as np
import cv2
from unittest.mock import Mock, patch
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from face_detection.detector import FaceDetector


class TestFaceDetector(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = FaceDetector()
        
        # Create a simple test image (100x100 with a white square in center)
        self.test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        self.test_image[25:75, 25:75] = 255
        
    def test_init(self):
        """Test detector initialization."""
        detector = FaceDetector(detection_confidence=0.7, model_selection=1)
        self.assertEqual(detector.detection_confidence, 0.7)
        self.assertEqual(detector.model_selection, 1)
    
    def test_detect_faces_invalid_input(self):
        """Test face detection with invalid input."""
        # Test with None
        faces = self.detector.detect_faces(None)
        self.assertEqual(len(faces), 0)
        
        # Test with empty array
        empty_image = np.array([])
        faces = self.detector.detect_faces(empty_image)
        self.assertEqual(len(faces), 0)
    
    def test_detect_faces_haar_cascade(self):
        """Test face detection using Haar Cascade."""
        # Mock the cascade detector to return a face
        with patch.object(self.detector.face_cascade, 'detectMultiScale') as mock_detect:
            mock_detect.return_value = np.array([[25, 25, 50, 50]])
            
            faces = self.detector._detect_faces_haar(self.test_image)
            
            self.assertEqual(len(faces), 1)
            self.assertEqual(faces[0]['bbox'], (25, 25, 50, 50))
            self.assertEqual(faces[0]['confidence'], 1.0)
            self.assertEqual(faces[0]['id'], 0)
    
    @patch('cv2.dnn.readNetFromCaffe')
    def test_init_dnn_detector(self, mock_read_net):
        """Test DNN detector initialization."""
        mock_net = Mock()
        mock_read_net.return_value = mock_net
        
        detector = FaceDetector()
        detector._init_opencv_detector()
        
        # Should try to load DNN model
        self.assertTrue(hasattr(detector, 'dnn_net'))
    
    def test_dnn_detection(self):
        """Test DNN-based face detection."""
        if hasattr(self.detector, 'dnn_net') and self.detector.use_dnn:
            # Mock DNN network
            with patch.object(self.detector.dnn_net, 'forward') as mock_forward:
                # Mock detection result
                mock_detections = np.zeros((1, 1, 1, 7))
                mock_detections[0, 0, 0, 2] = 0.9  # confidence
                mock_detections[0, 0, 0, 3:7] = [0.25, 0.25, 0.75, 0.75]  # bbox
                mock_forward.return_value = mock_detections
                
                faces = self.detector._detect_faces_dnn(self.test_image)
                
                self.assertEqual(len(faces), 1)
                self.assertAlmostEqual(faces[0]['confidence'], 0.9)
    
    def test_mediapipe_detection(self):
        """Test MediaPipe face detection."""
        if self.detector.use_mediapipe:
            # This would require MediaPipe to be installed
            # For now, just test that the method doesn't crash
            faces = self.detector._detect_faces_mediapipe(self.test_image)
            self.assertIsInstance(faces, list)
    
    def test_draw_faces(self):
        """Test drawing faces on image."""
        faces = [
            {
                'id': 0,
                'bbox': (25, 25, 50, 50),
                'confidence': 0.9,
                'landmarks': [(30, 30), (70, 30), (50, 50)]
            }
        ]
        
        result = self.detector.draw_faces(self.test_image, faces)
        
        # Should return an image of the same size
        self.assertEqual(result.shape, self.test_image.shape)
        
        # Image should be modified (not identical to original)
        self.assertFalse(np.array_equal(result, self.test_image))
    
    def test_get_face_region(self):
        """Test face region extraction."""
        face = {
            'bbox': (25, 25, 50, 50)
        }
        
        face_region = self.detector.get_face_region(self.test_image, face, padding=0.1)
        
        # Should return a cropped region
        self.assertLess(face_region.shape[0], self.test_image.shape[0])
        self.assertLess(face_region.shape[1], self.test_image.shape[1])
    
    def test_get_face_region_with_padding(self):
        """Test face region extraction with different padding."""
        face = {'bbox': (40, 40, 20, 20)}
        
        # Test with no padding
        region_no_pad = self.detector.get_face_region(self.test_image, face, padding=0.0)
        
        # Test with padding
        region_with_pad = self.detector.get_face_region(self.test_image, face, padding=0.5)
        
        # Padded region should be larger
        self.assertGreaterEqual(region_with_pad.shape[0], region_no_pad.shape[0])
        self.assertGreaterEqual(region_with_pad.shape[1], region_no_pad.shape[1])
    
    def test_detect_faces_auto_method(self):
        """Test automatic method selection."""
        # Should not crash and return a list
        faces = self.detector.detect_faces(self.test_image, method='auto')
        self.assertIsInstance(faces, list)
    
    def test_detect_faces_specific_method(self):
        """Test specific method selection."""
        # Test OpenCV method
        faces_opencv = self.detector.detect_faces(self.test_image, method='opencv')
        self.assertIsInstance(faces_opencv, list)
        
        # Test invalid method (should fall back to OpenCV)
        faces_invalid = self.detector.detect_faces(self.test_image, method='invalid')
        self.assertIsInstance(faces_invalid, list)


if __name__ == '__main__':
    unittest.main()