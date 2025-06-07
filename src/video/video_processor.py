import cv2
import numpy as np
import ffmpeg
from pathlib import Path
from typing import Optional, Tuple, List, Generator
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class VideoProcessor:
    """Main class for video processing operations."""
    
    SUPPORTED_FORMATS = ['.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MOV']
    
    def __init__(self):
        self.video_path: Optional[Path] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.fps: float = 0
        self.frame_count: int = 0
        self.width: int = 0
        self.height: int = 0
        self.codec: str = ""
        
    def load_video(self, video_path: str) -> bool:
        """Load video file and extract metadata.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            bool: True if successful, False otherwise
        """
        self.video_path = Path(video_path)
        
        if not self.video_path.exists():
            logger.error(f"Video file not found: {video_path}")
            return False
            
        if self.video_path.suffix not in self.SUPPORTED_FORMATS:
            logger.error(f"Unsupported video format: {self.video_path.suffix}")
            return False
            
        try:
            self.cap = cv2.VideoCapture(str(self.video_path))
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open video: {video_path}")
                return False
                
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
            self.codec = chr(fourcc & 0xff) + chr((fourcc >> 8) & 0xff) + \
                        chr((fourcc >> 16) & 0xff) + chr((fourcc >> 24) & 0xff)
            
            logger.info(f"Video loaded: {self.width}x{self.height}, {self.fps} fps, {self.frame_count} frames")
            return True
            
        except Exception as e:
            logger.error(f"Error loading video: {e}")
            return False
            
    def extract_frames(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Extract frames from the loaded video.
        
        Yields:
            Tuple[int, np.ndarray]: Frame number and frame data
        """
        if not self.cap or not self.cap.isOpened():
            logger.error("No video loaded")
            return
            
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        with tqdm(total=self.frame_count, desc="Extracting frames") as pbar:
            frame_num = 0
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                    
                yield frame_num, frame
                frame_num += 1
                pbar.update(1)
                
    def get_video_info(self) -> dict:
        """Get video metadata.
        
        Returns:
            dict: Video information
        """
        if not self.cap:
            return {}
            
        return {
            "path": str(self.video_path),
            "fps": self.fps,
            "frame_count": self.frame_count,
            "width": self.width,
            "height": self.height,
            "codec": self.codec,
            "duration": self.frame_count / self.fps if self.fps > 0 else 0
        }
        
    def extract_audio(self, output_path: str) -> bool:
        """Extract audio track from video using ffmpeg.
        
        Args:
            output_path: Path to save extracted audio
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.video_path:
            logger.error("No video loaded")
            return False
            
        try:
            stream = ffmpeg.input(str(self.video_path))
            stream = ffmpeg.output(stream, output_path, acodec='copy')
            ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)
            logger.info(f"Audio extracted to: {output_path}")
            return True
        except ffmpeg.Error as e:
            logger.error(f"Error extracting audio: {e.stderr.decode()}")
            return False
            
    def release(self):
        """Release video capture resources."""
        if self.cap:
            self.cap.release()
            self.cap = None
            
    def __del__(self):
        """Cleanup on deletion."""
        self.release()