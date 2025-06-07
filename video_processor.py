"""
Video processing module for face swapping CLI.

Handles video input/output operations and frame processing.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Generator
import logging

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Handles video file operations and frame processing."""
    
    def __init__(self, input_path: str, output_path: str, quality: str = 'medium'):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.quality = quality
        self.cap = None
        self.writer = None
        self._video_info = None
    
    def __enter__(self):
        """Context manager entry."""
        self._open_video()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self._close_video()
    
    def _open_video(self):
        """Open input video and prepare for processing."""
        self.cap = cv2.VideoCapture(str(self.input_path))
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video file: {self.input_path}")
        
        # Get video properties
        self._video_info = {
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'total_frames': int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration': int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) / self.cap.get(cv2.CAP_PROP_FPS)
        }
        
        logger.info(f"Video info: {self._video_info}")
    
    def _close_video(self):
        """Close video resources."""
        if self.cap:
            self.cap.release()
        if self.writer:
            self.writer.release()
    
    def get_video_info(self) -> dict:
        """Get video information."""
        if not self._video_info:
            raise RuntimeError("Video not opened. Use within context manager.")
        return self._video_info.copy()
    
    def _get_codec_and_quality(self) -> Tuple[str, float]:
        """Get video codec and quality based on settings."""
        quality_map = {
            'low': (cv2.VideoWriter_fourcc(*'XVID'), 0.5),
            'medium': (cv2.VideoWriter_fourcc(*'mp4v'), 0.7),
            'high': (cv2.VideoWriter_fourcc(*'mp4v'), 0.85),
            'best': (cv2.VideoWriter_fourcc(*'mp4v'), 1.0)
        }
        
        if self.quality in quality_map:
            return quality_map[self.quality]
        
        logger.warning(f"Unknown quality '{self.quality}', using medium")
        return quality_map['medium']
    
    def setup_writer(self, fps: Optional[float] = None):
        """Setup video writer for output."""
        if not self._video_info:
            raise RuntimeError("Video not opened")
        
        fourcc, quality_factor = self._get_codec_and_quality()
        output_fps = fps or self._video_info['fps']
        
        # Adjust dimensions based on quality
        width = int(self._video_info['width'] * quality_factor)
        height = int(self._video_info['height'] * quality_factor)
        
        # Ensure even dimensions for video encoding
        width = width - (width % 2)
        height = height - (height % 2)
        
        self.writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            output_fps,
            (width, height)
        )
        
        if not self.writer.isOpened():
            raise RuntimeError(f"Could not create output video: {self.output_path}")
        
        logger.info(f"Output video: {width}x{height} @ {output_fps}fps")
    
    def read_frames(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Generator that yields frame number and frame data."""
        if not self.cap:
            raise RuntimeError("Video not opened")
        
        frame_num = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            yield frame_num, frame
            frame_num += 1
    
    def write_frame(self, frame: np.ndarray):
        """Write a processed frame to output video."""
        if not self.writer:
            raise RuntimeError("Video writer not setup")
        
        # Resize frame if quality scaling was applied
        _, quality_factor = self._get_codec_and_quality()
        if quality_factor != 1.0:
            height, width = frame.shape[:2]
            new_width = int(width * quality_factor)
            new_height = int(height * quality_factor)
            
            # Ensure even dimensions
            new_width = new_width - (new_width % 2)
            new_height = new_height - (new_height % 2)
            
            frame = cv2.resize(frame, (new_width, new_height))
        
        self.writer.write(frame)
    
    def process_with_callback(self, frame_callback, progress_callback=None):
        """
        Process video frames with a callback function.
        
        Args:
            frame_callback: Function that takes (frame_num, frame) and returns processed frame
            progress_callback: Optional function called with (current_frame, total_frames)
        """
        if not self._video_info:
            raise RuntimeError("Video not opened")
        
        total_frames = self._video_info['total_frames']
        
        for frame_num, frame in self.read_frames():
            # Call progress callback
            if progress_callback:
                progress_callback(frame_num + 1, total_frames)
            
            # Process frame
            processed_frame = frame_callback(frame_num, frame)
            
            # Write to output
            if processed_frame is not None:
                self.write_frame(processed_frame)
    
    @staticmethod
    def extract_audio(input_path: str, output_path: str) -> bool:
        """
        Extract audio from video using ffmpeg.
        Returns True if successful, False otherwise.
        """
        import subprocess
        
        try:
            cmd = [
                'ffmpeg', '-i', input_path,
                '-vn', '-acodec', 'copy',
                '-y', output_path
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info(f"Audio extracted to: {output_path}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.warning(f"Could not extract audio: {e}")
            return False
        except FileNotFoundError:
            logger.warning("ffmpeg not found. Audio extraction skipped.")
            return False
    
    @staticmethod
    def merge_audio(video_path: str, audio_path: str, output_path: str) -> bool:
        """
        Merge video and audio using ffmpeg.
        Returns True if successful, False otherwise.
        """
        import subprocess
        
        try:
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-i', audio_path,
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-y', output_path
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info(f"Audio merged to: {output_path}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.warning(f"Could not merge audio: {e}")
            return False
        except FileNotFoundError:
            logger.warning("ffmpeg not found. Audio merge skipped.")
            return False


def get_supported_formats():
    """Get list of supported video formats."""
    return ['.mp4', '.avi', '.mov']


def validate_video_file(filepath: str) -> bool:
    """Validate that a file is a readable video."""
    try:
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            return False
        
        # Try to read one frame
        ret, _ = cap.read()
        cap.release()
        
        return ret
    except Exception:
        return False