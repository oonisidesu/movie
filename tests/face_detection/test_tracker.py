import unittest
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from face_detection.tracker import FaceTracker, MultiPersonTracker
from face_detection.detector import FaceDetector


class TestFaceTracker(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.tracker = FaceTracker(max_missing_frames=5, iou_threshold=0.5)
        
        # Create a simple test frame
        self.test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        self.test_frame[25:75, 25:75] = 255
    
    def test_init(self):
        """Test tracker initialization."""
        tracker = FaceTracker(max_missing_frames=10, iou_threshold=0.6)
        self.assertEqual(tracker.max_missing_frames, 10)
        self.assertEqual(tracker.iou_threshold, 0.6)
        self.assertEqual(tracker.next_track_id, 0)
        self.assertEqual(tracker.frame_count, 0)
    
    def test_calculate_iou(self):
        """Test IoU calculation."""
        bbox1 = (10, 10, 20, 20)  # x, y, w, h
        bbox2 = (15, 15, 20, 20)
        
        iou = self.tracker._calculate_iou(bbox1, bbox2)
        
        # Should be positive for overlapping boxes
        self.assertGreater(iou, 0)
        self.assertLessEqual(iou, 1.0)
    
    def test_calculate_iou_no_overlap(self):
        """Test IoU calculation for non-overlapping boxes."""
        bbox1 = (0, 0, 10, 10)
        bbox2 = (20, 20, 10, 10)
        
        iou = self.tracker._calculate_iou(bbox1, bbox2)
        
        # Should be 0 for non-overlapping boxes
        self.assertEqual(iou, 0.0)
    
    def test_calculate_iou_identical(self):
        """Test IoU calculation for identical boxes."""
        bbox = (10, 10, 20, 20)
        
        iou = self.tracker._calculate_iou(bbox, bbox)
        
        # Should be 1.0 for identical boxes
        self.assertAlmostEqual(iou, 1.0)
    
    def test_process_frame_new_detection(self):
        """Test processing frame with new face detection."""
        # Mock detector to return a face
        mock_detection = [{
            'id': 0,
            'bbox': (25, 25, 50, 50),
            'confidence': 0.9,
            'landmarks': None
        }]
        
        with unittest.mock.patch.object(self.tracker.detector, 'detect_faces', return_value=mock_detection):
            tracks = self.tracker.process_frame(self.test_frame)
        
        # Should create one new track
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0]['track_id'], 0)
        self.assertEqual(tracks[0]['bbox'], (25, 25, 50, 50))
        self.assertEqual(tracks[0]['age'], 0)
    
    def test_process_frame_track_continuation(self):
        """Test tracking face across multiple frames."""
        # First frame with detection
        mock_detection1 = [{
            'id': 0,
            'bbox': (25, 25, 50, 50),
            'confidence': 0.9,
            'landmarks': None
        }]
        
        # Second frame with similar detection
        mock_detection2 = [{
            'id': 0,
            'bbox': (26, 26, 50, 50),  # Slightly moved
            'confidence': 0.8,
            'landmarks': None
        }]
        
        with unittest.mock.patch.object(self.tracker.detector, 'detect_faces', side_effect=[mock_detection1, mock_detection2]):
            # Process first frame
            tracks1 = self.tracker.process_frame(self.test_frame)
            
            # Process second frame
            tracks2 = self.tracker.process_frame(self.test_frame)
        
        # Should maintain same track ID
        self.assertEqual(len(tracks1), 1)
        self.assertEqual(len(tracks2), 1)
        self.assertEqual(tracks1[0]['track_id'], tracks2[0]['track_id'])
        self.assertEqual(tracks2[0]['age'], 1)  # One frame older
    
    def test_process_frame_track_loss(self):
        """Test handling of lost tracks."""
        # Set up tracker with very low missing frame threshold
        tracker = FaceTracker(max_missing_frames=1)
        
        # First frame with detection
        mock_detection1 = [{
            'id': 0,
            'bbox': (25, 25, 50, 50),
            'confidence': 0.9,
            'landmarks': None
        }]
        
        # Second frame with no detection
        mock_detection2 = []
        
        with unittest.mock.patch.object(tracker.detector, 'detect_faces', side_effect=[mock_detection1, mock_detection2, mock_detection2]):
            # Process first frame
            tracks1 = tracker.process_frame(self.test_frame)
            self.assertEqual(len(tracks1), 1)
            
            # Process second frame (track should still exist but missing)
            tracks2 = tracker.process_frame(self.test_frame)
            self.assertEqual(len(tracks2), 0)  # No active tracks
            
            # Process third frame (track should be deleted)
            tracks3 = tracker.process_frame(self.test_frame)
            self.assertEqual(len(tracks3), 0)
    
    def test_get_track_history(self):
        """Test getting track history."""
        mock_detection = [{
            'id': 0,
            'bbox': (25, 25, 50, 50),
            'confidence': 0.9,
            'landmarks': None
        }]
        
        with unittest.mock.patch.object(self.tracker.detector, 'detect_faces', return_value=mock_detection):
            tracks = self.tracker.process_frame(self.test_frame)
            track_id = tracks[0]['track_id']
            
            history = self.tracker.get_track_history(track_id)
        
        self.assertIsNotNone(history)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0], (25, 25, 50, 50))
    
    def test_get_track_history_invalid_id(self):
        """Test getting history for invalid track ID."""
        history = self.tracker.get_track_history(999)
        self.assertIsNone(history)
    
    def test_reset(self):
        """Test tracker reset."""
        # Add some tracks first
        mock_detection = [{
            'id': 0,
            'bbox': (25, 25, 50, 50),
            'confidence': 0.9,
            'landmarks': None
        }]
        
        with unittest.mock.patch.object(self.tracker.detector, 'detect_faces', return_value=mock_detection):
            self.tracker.process_frame(self.test_frame)
        
        # Reset tracker
        self.tracker.reset()
        
        # Should be back to initial state
        self.assertEqual(len(self.tracker.tracks), 0)
        self.assertEqual(self.tracker.next_track_id, 0)
        self.assertEqual(self.tracker.frame_count, 0)


class TestMultiPersonTracker(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.tracker = MultiPersonTracker()
        self.test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    
    def test_init(self):
        """Test multi-person tracker initialization."""
        self.assertIsInstance(self.tracker.detector, FaceDetector)
        self.assertEqual(len(self.tracker.trackers), 0)
        self.assertEqual(len(self.tracker.person_assignments), 0)
    
    def test_assign_person_to_track(self):
        """Test manual person assignment."""
        self.tracker.assign_person_to_track(0, "person_1")
        
        self.assertEqual(self.tracker.person_assignments[0], "person_1")
    
    def test_process_frame(self):
        """Test processing frame with multiple persons."""
        mock_tracks = [
            {
                'track_id': 0,
                'bbox': (10, 10, 30, 30),
                'confidence': 0.9,
                'age': 5
            },
            {
                'track_id': 1,
                'bbox': (60, 60, 30, 30),
                'confidence': 0.8,
                'age': 3
            }
        ]
        
        # Assign persons
        self.tracker.assign_person_to_track(0, "Alice")
        self.tracker.assign_person_to_track(1, "Bob")
        
        with unittest.mock.patch.object(self.tracker.trackers['main'], 'process_frame', return_value=mock_tracks):
            result = self.tracker.process_frame(self.test_frame)
        
        # Should group tracks by person
        self.assertIn("Alice", result)
        self.assertIn("Bob", result)
        self.assertEqual(len(result["Alice"]), 1)
        self.assertEqual(len(result["Bob"]), 1)
    
    def test_process_frame_unknown_person(self):
        """Test processing frame with unassigned tracks."""
        mock_tracks = [
            {
                'track_id': 5,
                'bbox': (10, 10, 30, 30),
                'confidence': 0.9,
                'age': 5
            }
        ]
        
        with unittest.mock.patch.object(self.tracker.trackers['main'], 'process_frame', return_value=mock_tracks):
            result = self.tracker.process_frame(self.test_frame)
        
        # Should create unknown person entry
        self.assertIn("unknown_5", result)
        self.assertEqual(len(result["unknown_5"]), 1)


if __name__ == '__main__':
    unittest.main()