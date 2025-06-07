import cv2
import numpy as np
import ffmpeg
from pathlib import Path
from typing import Optional, List, Tuple
import logging
import tempfile
import shutil
from tqdm import tqdm

logger = logging.getLogger(__name__)

class VideoWriter:
    """Class for writing processed frames back to video."""
    
    def __init__(self, output_path: str, fps: float, resolution: Tuple[int, int], 
                 codec: str = 'mp4v'):
        """Initialize video writer.
        
        Args:
            output_path: Path to save output video
            fps: Frames per second
            resolution: (width, height) tuple
            codec: Video codec (default: mp4v)
        """
        self.output_path = Path(output_path)
        self.fps = fps
        self.width, self.height = resolution
        self.codec = codec
        self.writer: Optional[cv2.VideoWriter] = None
        self.temp_video_path: Optional[Path] = None
        self.frame_count = 0
        
    def initialize(self) -> bool:
        """Initialize the video writer.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            
            self.temp_video_path = Path(tempfile.mktemp(suffix='.mp4'))
            
            self.writer = cv2.VideoWriter(
                str(self.temp_video_path),
                fourcc,
                self.fps,
                (self.width, self.height)
            )
            
            if not self.writer.isOpened():
                logger.error("Failed to initialize video writer")
                return False
                
            logger.info(f"Video writer initialized: {self.width}x{self.height}, {self.fps} fps")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing video writer: {e}")
            return False
            
    def write_frame(self, frame: np.ndarray) -> bool:
        """Write a single frame to the video.
        
        Args:
            frame: Frame to write (numpy array)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.writer:
            logger.error("Video writer not initialized")
            return False
            
        try:
            if frame.shape[:2] != (self.height, self.width):
                frame = cv2.resize(frame, (self.width, self.height))
                
            self.writer.write(frame)
            self.frame_count += 1
            return True
            
        except Exception as e:
            logger.error(f"Error writing frame: {e}")
            return False
            
    def write_frames(self, frames: List[np.ndarray], show_progress: bool = True) -> bool:
        """Write multiple frames to the video.
        
        Args:
            frames: List of frames to write
            show_progress: Show progress bar
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.writer:
            logger.error("Video writer not initialized")
            return False
            
        iterator = tqdm(frames, desc="Writing frames") if show_progress else frames
        
        for frame in iterator:
            if not self.write_frame(frame):
                return False
                
        return True
        
    def add_audio(self, audio_path: str) -> bool:
        """Add audio track to the video using ffmpeg.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.temp_video_path or not self.temp_video_path.exists():
            logger.error("No video to add audio to")
            return False
            
        try:
            video = ffmpeg.input(str(self.temp_video_path))
            audio = ffmpeg.input(audio_path)
            
            stream = ffmpeg.output(
                video, audio, 
                str(self.output_path), 
                vcodec='copy', 
                acodec='aac',
                strict='experimental'
            )
            
            ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
            logger.info(f"Audio added successfully")
            return True
            
        except ffmpeg.Error as e:
            logger.error(f"Error adding audio: {e.stderr.decode()}")
            return False
            
    def finalize(self, audio_path: Optional[str] = None) -> bool:
        """Finalize the video, optionally adding audio.
        
        Args:
            audio_path: Optional path to audio file
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.writer:
            self.writer.release()
            self.writer = None
            
        if not self.temp_video_path or not self.temp_video_path.exists():
            logger.error("No video to finalize")
            return False
            
        try:
            if audio_path and Path(audio_path).exists():
                success = self.add_audio(audio_path)
                if self.temp_video_path.exists():
                    self.temp_video_path.unlink()
                return success
            else:
                shutil.move(str(self.temp_video_path), str(self.output_path))
                logger.info(f"Video saved to: {self.output_path}")
                return True
                
        except Exception as e:
            logger.error(f"Error finalizing video: {e}")
            return False
            
    def cleanup(self):
        """Clean up temporary files."""
        if self.writer:
            self.writer.release()
            self.writer = None
            
        if self.temp_video_path and self.temp_video_path.exists():
            self.temp_video_path.unlink()
            
    def __del__(self):
        """Cleanup on deletion."""
        self.cleanup()