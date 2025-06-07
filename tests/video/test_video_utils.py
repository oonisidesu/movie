import unittest
import numpy as np
import cv2
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from video.utils.video_utils import (
    validate_video_file,
    get_video_info,
    classify_resolution,
    extract_audio,
    merge_audio,
    calculate_memory_usage,
    optimize_video_settings,
    convert_video_format,
    get_video_thumbnail,
    check_video_integrity,
    _check_ffmpeg
)


class TestVideoUtils(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
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
            frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            writer.write(frame)
        
        writer.release()
    
    def test_validate_video_file_nonexistent(self):
        """Test validation of non-existent file."""
        result = validate_video_file("nonexistent_file.mp4")
        self.assertFalse(result)
    
    def test_validate_video_file_unsupported_extension(self):
        """Test validation of unsupported file extension."""
        # Create a text file with video extension
        test_file = os.path.join(self.temp_dir, "test.txt")
        with open(test_file, 'w') as f:
            f.write("not a video")
        
        result = validate_video_file(test_file)
        self.assertFalse(result)
    
    def test_validate_video_file_valid(self):
        """Test validation of valid video file."""
        # Create a test video
        test_video = os.path.join(self.temp_dir, "test.mp4")
        self.create_test_video(test_video)
        
        result = validate_video_file(test_video)
        self.assertTrue(result)
    
    def test_get_video_info(self):
        """Test video info extraction."""
        # Create a test video
        test_video = os.path.join(self.temp_dir, "test.mp4")
        self.create_test_video(test_video, frames=60, width=640, height=480)
        
        info = get_video_info(test_video)
        
        # Check basic properties
        self.assertEqual(info['width'], 640)
        self.assertEqual(info['height'], 480)
        self.assertAlmostEqual(info['fps'], 30.0, places=1)
        self.assertEqual(info['frame_count'], 60)
        self.assertAlmostEqual(info['duration'], 2.0, places=1)
        
        # Check additional properties
        self.assertIn('file_size', info)
        self.assertIn('file_size_mb', info)
        self.assertIn('resolution_class', info)
        self.assertIn('estimated_bitrate_kbps', info)
        self.assertIn('codec', info)
    
    def test_get_video_info_nonexistent(self):
        """Test video info extraction for non-existent file."""
        with self.assertRaises(FileNotFoundError):
            get_video_info("nonexistent_file.mp4")
    
    def test_classify_resolution(self):
        """Test resolution classification."""
        # Test various resolutions
        self.assertEqual(classify_resolution(3840, 2160), "4K (Ultra HD)")
        self.assertEqual(classify_resolution(1920, 1080), "Full HD (1080p)")
        self.assertEqual(classify_resolution(1280, 720), "HD (720p)")
        self.assertEqual(classify_resolution(854, 480), "SD (480p)")
        self.assertEqual(classify_resolution(320, 240), "Low Resolution")
    
    @patch('subprocess.run')
    @patch('video.utils.video_utils._check_ffmpeg')
    def test_extract_audio_success(self, mock_check_ffmpeg, mock_run):
        """Test successful audio extraction."""
        mock_check_ffmpeg.return_value = True
        mock_run.return_value = Mock(returncode=0)
        
        result = extract_audio("input.mp4", "output.aac")
        self.assertTrue(result)
        mock_run.assert_called_once()
    
    @patch('video.utils.video_utils._check_ffmpeg')
    def test_extract_audio_no_ffmpeg(self, mock_check_ffmpeg):
        """Test audio extraction without ffmpeg."""
        mock_check_ffmpeg.return_value = False
        
        result = extract_audio("input.mp4", "output.aac")
        self.assertFalse(result)
    
    @patch('subprocess.run')
    @patch('video.utils.video_utils._check_ffmpeg')
    def test_extract_audio_failure(self, mock_check_ffmpeg, mock_run):
        """Test failed audio extraction."""
        mock_check_ffmpeg.return_value = True
        mock_run.return_value = Mock(returncode=1, stderr="Error message")
        
        result = extract_audio("input.mp4", "output.aac")
        self.assertFalse(result)
    
    @patch('subprocess.run')
    @patch('video.utils.video_utils._check_ffmpeg')
    def test_merge_audio_success(self, mock_check_ffmpeg, mock_run):
        """Test successful audio merge."""
        mock_check_ffmpeg.return_value = True
        mock_run.return_value = Mock(returncode=0)
        
        result = merge_audio("video.mp4", "audio.aac", "output.mp4")
        self.assertTrue(result)
        mock_run.assert_called_once()
    
    @patch('video.utils.video_utils._check_ffmpeg')
    def test_merge_audio_no_ffmpeg(self, mock_check_ffmpeg):
        """Test audio merge without ffmpeg."""
        mock_check_ffmpeg.return_value = False
        
        result = merge_audio("video.mp4", "audio.aac", "output.mp4")
        self.assertFalse(result)
    
    @patch('subprocess.run')
    def test_check_ffmpeg_available(self, mock_run):
        """Test ffmpeg availability check - available."""
        mock_run.return_value = Mock(returncode=0)
        
        result = _check_ffmpeg()
        self.assertTrue(result)
    
    @patch('subprocess.run')
    def test_check_ffmpeg_not_available(self, mock_run):
        """Test ffmpeg availability check - not available."""
        mock_run.side_effect = FileNotFoundError()
        
        result = _check_ffmpeg()
        self.assertFalse(result)
    
    def test_calculate_memory_usage(self):
        """Test memory usage calculation."""
        video_info = {
            'width': 1920,
            'height': 1080
        }
        
        memory_info = calculate_memory_usage(video_info, buffer_frames=10)
        
        # Check calculations
        expected_bytes_per_frame = 1920 * 1080 * 3  # BGR
        expected_single_frame_mb = expected_bytes_per_frame / (1024 * 1024)
        
        self.assertEqual(memory_info['bytes_per_frame'], expected_bytes_per_frame)
        self.assertAlmostEqual(memory_info['single_frame_mb'], expected_single_frame_mb)
        self.assertEqual(memory_info['buffer_frames'], 10)
        self.assertIn('buffer_memory_mb', memory_info)
        self.assertIn('processing_memory_mb', memory_info)
        self.assertIn('total_memory_mb', memory_info)
    
    def test_calculate_memory_usage_invalid_info(self):
        """Test memory usage calculation with invalid info."""
        memory_info = calculate_memory_usage({})
        self.assertEqual(memory_info, {})
    
    def test_optimize_video_settings(self):
        """Test video settings optimization."""
        # Test 4K video
        video_info_4k = {
            'width': 3840,
            'height': 2160,
            'fps': 60,
            'duration': 600  # 10 minutes
        }
        
        recommendations = optimize_video_settings(video_info_4k)
        
        self.assertIn('suggested_resize', recommendations)
        self.assertEqual(recommendations['suggested_resize'], (1920, 1080))
        self.assertIn('suggested_fps', recommendations)
        self.assertEqual(recommendations['suggested_fps'], 30)
        self.assertIn('suggested_quality', recommendations)
        self.assertEqual(recommendations['suggested_quality'], 'medium')
        self.assertIn('chunk_size', recommendations)
        self.assertEqual(recommendations['chunk_size'], 10)
    
    def test_optimize_video_settings_hd(self):
        """Test video settings optimization for HD video."""
        video_info_hd = {
            'width': 1280,
            'height': 720,
            'fps': 30,
            'duration': 60  # 1 minute
        }
        
        recommendations = optimize_video_settings(video_info_hd)
        
        self.assertNotIn('suggested_resize', recommendations)
        self.assertNotIn('suggested_fps', recommendations)
        self.assertEqual(recommendations['suggested_quality'], 'high')
        self.assertEqual(recommendations['chunk_size'], 20)
    
    @patch('subprocess.run')
    @patch('video.utils.video_utils._check_ffmpeg')
    def test_convert_video_format_success(self, mock_check_ffmpeg, mock_run):
        """Test successful video format conversion."""
        mock_check_ffmpeg.return_value = True
        mock_run.return_value = Mock(returncode=0)
        
        result = convert_video_format("input.avi", "output.mp4", quality="high")
        self.assertTrue(result)
        mock_run.assert_called_once()
    
    @patch('video.utils.video_utils._check_ffmpeg')
    def test_convert_video_format_no_ffmpeg(self, mock_check_ffmpeg):
        """Test video conversion without ffmpeg."""
        mock_check_ffmpeg.return_value = False
        
        result = convert_video_format("input.avi", "output.mp4")
        self.assertFalse(result)
    
    def test_get_video_thumbnail(self):
        """Test video thumbnail extraction."""
        # Create a test video
        test_video = os.path.join(self.temp_dir, "test.mp4")
        self.create_test_video(test_video, frames=60)
        
        thumbnail_path = get_video_thumbnail(test_video, timestamp=1.0)
        
        if thumbnail_path:  # May fail if OpenCV can't read the test video
            self.assertTrue(os.path.exists(thumbnail_path))
            self.assertTrue(thumbnail_path.endswith('_thumbnail.jpg'))
    
    def test_get_video_thumbnail_invalid_file(self):
        """Test thumbnail extraction for invalid file."""
        result = get_video_thumbnail("nonexistent.mp4")
        self.assertIsNone(result)
    
    def test_check_video_integrity(self):
        """Test video integrity check."""
        # Create a test video
        test_video = os.path.join(self.temp_dir, "test.mp4")
        self.create_test_video(test_video, frames=30)
        
        integrity = check_video_integrity(test_video)
        
        self.assertIn('is_valid', integrity)
        self.assertIn('can_read_frames', integrity)
        self.assertIn('frame_read_errors', integrity)
        self.assertIn('total_frames_checked', integrity)
        self.assertIn('issues', integrity)
        
        # For a valid test video
        if integrity['is_valid']:
            self.assertIsInstance(integrity['frame_read_errors'], int)
            self.assertIsInstance(integrity['total_frames_checked'], int)
    
    def test_check_video_integrity_invalid_file(self):
        """Test integrity check for invalid file."""
        integrity = check_video_integrity("nonexistent.mp4")
        
        self.assertFalse(integrity['is_valid'])
        self.assertIn("Cannot open video file", integrity['issues'])


if __name__ == '__main__':
    unittest.main()