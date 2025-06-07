import unittest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import cv2

from src.video.utils.video_utils import (
    get_video_info,
    validate_video_file,
    resize_frame_maintain_aspect,
    extract_frame_at_time,
    calculate_memory_usage
)

class TestVideoUtils(unittest.TestCase):
    
    @patch('cv2.VideoCapture')
    def test_get_video_info_success(self, mock_cv2_capture):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 90,
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080,
            cv2.CAP_PROP_FOURCC: 0x5634504d
        }.get(prop, 0)
        
        mock_cv2_capture.return_value = mock_cap
        
        info = get_video_info("test.mp4")
        
        self.assertEqual(info["fps"], 30.0)
        self.assertEqual(info["frame_count"], 90)
        self.assertEqual(info["width"], 1920)
        self.assertEqual(info["height"], 1080)
        self.assertEqual(info["duration"], 3.0)  # 90 frames / 30 fps
        
    @patch('cv2.VideoCapture')
    def test_get_video_info_cannot_open(self, mock_cv2_capture):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cv2_capture.return_value = mock_cap
        
        info = get_video_info("test.mp4")
        self.assertEqual(info, {})
        
    @patch('cv2.VideoCapture')
    def test_validate_video_file_success(self, mock_cv2_capture):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_cv2_capture.return_value = mock_cap
        
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            tmp_path = tmp.name
            
        try:
            result = validate_video_file(tmp_path)
            self.assertTrue(result)
        finally:
            Path(tmp_path).unlink()
            
    def test_validate_video_file_nonexistent(self):
        result = validate_video_file("nonexistent.mp4")
        self.assertFalse(result)
        
    def test_validate_video_file_unsupported_format(self):
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            tmp_path = tmp.name
            
        try:
            result = validate_video_file(tmp_path)
            self.assertFalse(result)
        finally:
            Path(tmp_path).unlink()
            
    def test_resize_frame_maintain_aspect_wider(self):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)  # 2:1 aspect ratio
        target_size = (400, 300)  # 4:3 aspect ratio
        
        result = resize_frame_maintain_aspect(frame, target_size)
        
        self.assertEqual(result.shape, (300, 400, 3))
        
    def test_resize_frame_maintain_aspect_taller(self):
        frame = np.zeros((200, 100, 3), dtype=np.uint8)  # 1:2 aspect ratio
        target_size = (400, 300)  # 4:3 aspect ratio
        
        result = resize_frame_maintain_aspect(frame, target_size)
        
        self.assertEqual(result.shape, (300, 400, 3))
        
    @patch('cv2.VideoCapture')
    def test_extract_frame_at_time_success(self, mock_cv2_capture):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 30.0  # 30 fps
        
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, fake_frame)
        
        mock_cv2_capture.return_value = mock_cap
        
        frame = extract_frame_at_time("test.mp4", 2.0)  # 2 seconds
        
        self.assertIsNotNone(frame)
        mock_cap.set.assert_called_with(cv2.CAP_PROP_POS_FRAMES, 60)  # 2 * 30 fps
        
    @patch('cv2.VideoCapture')
    def test_extract_frame_at_time_cannot_open(self, mock_cv2_capture):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cv2_capture.return_value = mock_cap
        
        frame = extract_frame_at_time("test.mp4", 1.0)
        self.assertIsNone(frame)
        
    def test_calculate_memory_usage(self):
        video_info = {
            "width": 1920,
            "height": 1080,
            "frame_count": 100
        }
        
        memory = calculate_memory_usage(video_info)
        
        expected_single_frame = (1920 * 1080 * 3) / (1024 * 1024)
        expected_all_frames = expected_single_frame * 100
        
        self.assertAlmostEqual(memory["single_frame_mb"], expected_single_frame, places=2)
        self.assertAlmostEqual(memory["all_frames_mb"], expected_all_frames, places=2)
        self.assertAlmostEqual(memory["estimated_processing_mb"], expected_all_frames * 2, places=2)
        
    def test_calculate_memory_usage_empty_info(self):
        memory = calculate_memory_usage({})
        self.assertEqual(memory, {})

if __name__ == '__main__':
    unittest.main()