import unittest
import numpy as np
import tempfile
import cv2
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.video.video_processor import VideoProcessor

class TestVideoProcessor(unittest.TestCase):
    
    def setUp(self):
        self.processor = VideoProcessor()
        
    def tearDown(self):
        if self.processor.cap:
            self.processor.release()
            
    @patch('cv2.VideoCapture')
    def test_load_video_success(self, mock_cv2_capture):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 100,
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080,
            cv2.CAP_PROP_FOURCC: 0x5634504d  # MP4V
        }.get(prop, 0)
        
        mock_cv2_capture.return_value = mock_cap
        
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            tmp_path = tmp.name
            
        try:
            result = self.processor.load_video(tmp_path)
            self.assertTrue(result)
            self.assertEqual(self.processor.fps, 30.0)
            self.assertEqual(self.processor.frame_count, 100)
            self.assertEqual(self.processor.width, 1920)
            self.assertEqual(self.processor.height, 1080)
        finally:
            Path(tmp_path).unlink()
            
    def test_load_video_nonexistent_file(self):
        result = self.processor.load_video("nonexistent.mp4")
        self.assertFalse(result)
        
    def test_load_video_unsupported_format(self):
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            tmp_path = tmp.name
            
        try:
            result = self.processor.load_video(tmp_path)
            self.assertFalse(result)
        finally:
            Path(tmp_path).unlink()
            
    @patch('cv2.VideoCapture')
    def test_get_video_info(self, mock_cv2_capture):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 25.0,
            cv2.CAP_PROP_FRAME_COUNT: 50,
            cv2.CAP_PROP_FRAME_WIDTH: 1280,
            cv2.CAP_PROP_FRAME_HEIGHT: 720,
            cv2.CAP_PROP_FOURCC: 0x5634504d
        }.get(prop, 0)
        
        mock_cv2_capture.return_value = mock_cap
        self.processor.cap = mock_cap
        self.processor.fps = 25.0
        self.processor.frame_count = 50
        self.processor.width = 1280
        self.processor.height = 720
        self.processor.codec = "MP4V"
        
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            self.processor.video_path = Path(tmp.name)
            
        try:
            info = self.processor.get_video_info()
            self.assertEqual(info["fps"], 25.0)
            self.assertEqual(info["frame_count"], 50)
            self.assertEqual(info["width"], 1280)
            self.assertEqual(info["height"], 720)
            self.assertEqual(info["duration"], 2.0)  # 50 frames / 25 fps
        finally:
            Path(tmp.name).unlink()
            
    def test_get_video_info_no_video_loaded(self):
        info = self.processor.get_video_info()
        self.assertEqual(info, {})
        
    @patch('cv2.VideoCapture')
    def test_extract_frames(self, mock_cv2_capture):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.side_effect = [
            (True, fake_frame),
            (True, fake_frame), 
            (False, None)
        ]
        
        mock_cv2_capture.return_value = mock_cap
        self.processor.cap = mock_cap
        self.processor.frame_count = 2
        
        frames = list(self.processor.extract_frames())
        self.assertEqual(len(frames), 2)
        self.assertEqual(frames[0][0], 0)  # First frame number
        self.assertEqual(frames[1][0], 1)  # Second frame number
        self.assertTrue(np.array_equal(frames[0][1], fake_frame))
        
    def test_extract_frames_no_video_loaded(self):
        frames = list(self.processor.extract_frames())
        self.assertEqual(len(frames), 0)

if __name__ == '__main__':
    unittest.main()