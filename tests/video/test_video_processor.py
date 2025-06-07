import unittest
import numpy as np
import cv2
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from video.video_processor import VideoProcessor
from video.frame_processor import FrameProcessor


class TestVideoProcessor(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.processor = VideoProcessor(temp_dir=self.temp_dir)
        
        # Create a test video file path
        self.test_video_path = os.path.join(self.temp_dir, 'test_video.mp4')
        self.test_output_path = os.path.join(self.temp_dir, 'output_video.mp4')
        
    def tearDown(self):
        """Clean up test fixtures."""
        self.processor.cleanup()
        
        # Clean up temp files
        for file in os.listdir(self.temp_dir):
            try:
                os.remove(os.path.join(self.temp_dir, file))
            except:
                pass
        
        try:
            os.rmdir(self.temp_dir)
        except:
            pass
    
    def test_init(self):
        """Test processor initialization."""
        processor = VideoProcessor("input.mp4", "output.mp4", "/tmp")
        
        self.assertEqual(processor.input_path, "input.mp4")
        self.assertEqual(processor.output_path, "output.mp4")
        self.assertEqual(processor.temp_dir, "/tmp")
        self.assertEqual(processor.current_frame, 0)
        self.assertEqual(processor.total_frames, 0)
        self.assertFalse(processor.is_processing)
        self.assertTrue(processor.preserve_audio)
    
    @patch('cv2.VideoCapture')
    @patch('video.utils.video_utils.validate_video_file')
    @patch('video.utils.video_utils.get_video_info')
    def test_load_video(self, mock_get_info, mock_validate, mock_capture):
        """Test video loading."""
        # Mock validation
        mock_validate.return_value = True
        
        # Mock video info
        mock_video_info = {
            'width': 640,
            'height': 480,
            'fps': 30.0,
            'frame_count': 900,
            'duration': 30.0
        }
        mock_get_info.return_value = mock_video_info
        
        # Mock capture
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_capture.return_value = mock_cap
        
        # Test loading
        result = self.processor.load_video(self.test_video_path)
        
        self.assertEqual(result, mock_video_info)
        self.assertEqual(self.processor.video_info, mock_video_info)
        self.assertEqual(self.processor.total_frames, 900)
        self.assertIsNotNone(self.processor.cap)
    
    @patch('cv2.VideoCapture')
    @patch('video.utils.video_utils.validate_video_file')
    def test_load_video_invalid_file(self, mock_validate, mock_capture):
        """Test loading invalid video file."""
        mock_validate.return_value = False
        
        with self.assertRaises(ValueError):
            self.processor.load_video("invalid_video.txt")
    
    @patch('cv2.VideoCapture')
    @patch('video.utils.video_utils.validate_video_file')
    @patch('video.utils.video_utils.get_video_info')
    def test_load_video_capture_failure(self, mock_get_info, mock_validate, mock_capture):
        """Test video loading when capture fails."""
        mock_validate.return_value = True
        mock_get_info.return_value = {'frame_count': 100}
        
        # Mock capture failure
        mock_cap = Mock()
        mock_cap.isOpened.return_value = False
        mock_capture.return_value = mock_cap
        
        with self.assertRaises(ValueError):
            self.processor.load_video(self.test_video_path)
    
    @patch('video.video_writer.VideoWriter')
    @patch('pathlib.Path.mkdir')
    def test_setup_output(self, mock_mkdir, mock_writer_class):
        """Test output setup."""
        # Setup processor with video info
        self.processor.video_info = {
            'width': 640,
            'height': 480,
            'fps': 30.0
        }
        
        mock_writer = Mock()
        mock_writer_class.return_value = mock_writer
        
        self.processor.setup_output(self.test_output_path)
        
        self.assertEqual(self.processor.output_path, self.test_output_path)
        self.assertIsNotNone(self.processor.video_writer)
        mock_writer_class.assert_called_once()
    
    def test_setup_output_no_video_loaded(self):
        """Test output setup without loaded video."""
        with self.assertRaises(ValueError):
            self.processor.setup_output(self.test_output_path)
    
    @patch('video.utils.video_utils.extract_audio')
    def test_extract_audio_track(self, mock_extract):
        """Test audio extraction."""
        self.processor.input_path = self.test_video_path
        mock_extract.return_value = True
        
        result = self.processor.extract_audio_track()
        
        self.assertIsNotNone(result)
        self.assertIsNotNone(self.processor.audio_path)
        mock_extract.assert_called_once()
    
    @patch('video.utils.video_utils.extract_audio')
    def test_extract_audio_track_failure(self, mock_extract):
        """Test audio extraction failure."""
        self.processor.input_path = self.test_video_path
        mock_extract.return_value = False
        
        result = self.processor.extract_audio_track()
        
        self.assertIsNone(result)
        self.assertIsNone(self.processor.audio_path)
    
    def test_extract_audio_track_no_input(self):
        """Test audio extraction with no input video."""
        result = self.processor.extract_audio_track()
        self.assertIsNone(result)
    
    def test_extract_audio_track_preserve_disabled(self):
        """Test audio extraction when preserve_audio is disabled."""
        self.processor.input_path = self.test_video_path
        self.processor.preserve_audio = False
        
        result = self.processor.extract_audio_track()
        self.assertIsNone(result)
    
    def test_get_frame_iterator(self):
        """Test frame iterator."""
        # Mock video capture
        mock_cap = Mock()
        mock_frames = [
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
            (True, np.ones((480, 640, 3), dtype=np.uint8)),
            (False, None)  # End of video
        ]
        mock_cap.read.side_effect = mock_frames
        
        self.processor.cap = mock_cap
        self.processor.total_frames = 2
        
        frames = list(self.processor.get_frame_iterator())
        
        self.assertEqual(len(frames), 2)
        self.assertEqual(frames[0][0], 0)  # Frame number
        self.assertEqual(frames[1][0], 1)  # Frame number
    
    def test_get_frame_iterator_no_video(self):
        """Test frame iterator without loaded video."""
        with self.assertRaises(ValueError):
            list(self.processor.get_frame_iterator())
    
    def test_get_frame_at(self):
        """Test getting specific frame."""
        # Mock video capture
        mock_cap = Mock()
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, test_frame)
        
        self.processor.cap = mock_cap
        self.processor.total_frames = 100
        
        frame = self.processor.get_frame_at(50)
        
        self.assertIsNotNone(frame)
        mock_cap.set.assert_called_with(cv2.CAP_PROP_POS_FRAMES, 50)
        np.testing.assert_array_equal(frame, test_frame)
    
    def test_get_frame_at_invalid_number(self):
        """Test getting frame with invalid frame number."""
        mock_cap = Mock()
        self.processor.cap = mock_cap
        self.processor.total_frames = 100
        
        # Test negative frame number
        frame = self.processor.get_frame_at(-1)
        self.assertIsNone(frame)
        
        # Test frame number beyond video length
        frame = self.processor.get_frame_at(200)
        self.assertIsNone(frame)
    
    def test_get_frame_at_no_video(self):
        """Test getting frame without loaded video."""
        frame = self.processor.get_frame_at(0)
        self.assertIsNone(frame)
    
    def test_estimate_processing_time(self):
        """Test processing time estimation."""
        self.processor.video_info = {'frame_count': 1000}
        self.processor.total_frames = 1000
        
        estimate = self.processor.estimate_processing_time(10.0)
        
        self.assertEqual(estimate['total_frames'], 1000)
        self.assertEqual(estimate['processing_fps'], 10.0)
        self.assertEqual(estimate['estimated_seconds'], 100.0)
        self.assertEqual(estimate['estimated_minutes'], 100.0 / 60)
        self.assertEqual(estimate['estimated_hours'], 100.0 / 3600)
    
    def test_estimate_processing_time_no_video(self):
        """Test processing time estimation without video info."""
        estimate = self.processor.estimate_processing_time()
        self.assertEqual(estimate, {})
    
    def test_cleanup(self):
        """Test resource cleanup."""
        # Setup mock objects
        mock_cap = Mock()
        mock_writer = Mock()
        
        self.processor.cap = mock_cap
        self.processor.video_writer = mock_writer
        self.processor.is_processing = True
        
        # Create mock audio file
        audio_file = os.path.join(self.temp_dir, 'test_audio.aac')
        with open(audio_file, 'w') as f:
            f.write('test')
        self.processor.audio_path = audio_file
        
        self.processor.cleanup()
        
        # Verify cleanup
        mock_cap.release.assert_called_once()
        mock_writer.release.assert_called_once()
        self.assertIsNone(self.processor.cap)
        self.assertIsNone(self.processor.video_writer)
        self.assertFalse(self.processor.is_processing)
        self.assertFalse(os.path.exists(audio_file))
    
    def test_context_manager(self):
        """Test using processor as context manager."""
        mock_cap = Mock()
        
        with VideoProcessor() as processor:
            processor.cap = mock_cap
        
        # Should call cleanup on exit
        mock_cap.release.assert_called_once()


class TestVideoProcessorIntegration(unittest.TestCase):
    """Integration tests for video processor."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up integration test fixtures."""
        # Clean up temp files
        for file in os.listdir(self.temp_dir):
            try:
                os.remove(os.path.join(self.temp_dir, file))
            except:
                pass
        
        try:
            os.rmdir(self.temp_dir)
        except:
            pass
    
    def create_test_video(self, path: str, frames: int = 30, width: int = 320, height: int = 240):
        """Create a test video file."""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(path, fourcc, 30.0, (width, height))
        
        for i in range(frames):
            # Create a simple test frame with frame number
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :, 0] = (i * 255) // frames  # Blue channel varies
            writer.write(frame)
        
        writer.release()
    
    @patch('video.utils.video_utils.extract_audio')
    @patch('video.utils.video_utils.merge_audio')
    def test_process_video_pipeline(self, mock_merge, mock_extract):
        """Test complete video processing pipeline."""
        # Create test video
        input_path = os.path.join(self.temp_dir, 'input.mp4')
        output_path = os.path.join(self.temp_dir, 'output.mp4')
        
        self.create_test_video(input_path, frames=10)
        
        # Mock audio functions
        mock_extract.return_value = True
        mock_merge.return_value = True
        
        # Setup processor
        processor = VideoProcessor()
        
        try:
            # Load video
            video_info = processor.load_video(input_path)
            self.assertEqual(video_info['width'], 320)
            self.assertEqual(video_info['height'], 240)
            
            # Setup output
            processor.setup_output(output_path)
            
            # Define processing function
            def simple_processor(frame_num: int, frame: np.ndarray) -> np.ndarray:
                # Add a red border
                processed = frame.copy()
                processed[0:5, :] = [0, 0, 255]  # Red top border
                return processed
            
            # Process video
            success = processor.process_video(simple_processor)
            self.assertTrue(success)
            
        finally:
            processor.cleanup()


if __name__ == '__main__':
    unittest.main()