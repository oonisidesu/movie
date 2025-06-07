"""
Video Writer - Handles video output with various codecs and quality settings

Provides video writing capabilities with support for different codecs,
quality settings, and proper resource management.
"""

import cv2
import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
import os

logger = logging.getLogger(__name__)


class VideoWriter:
    """
    Video writer with codec support and quality control.
    
    Handles video output with proper codec selection, quality settings,
    and resource management for the face swapping pipeline.
    """
    
    # Codec configurations
    CODEC_CONFIGS = {
        'mp4v': {
            'fourcc': cv2.VideoWriter_fourcc(*'mp4v'),
            'extension': '.mp4',
            'quality': 'medium'
        },
        'h264': {
            'fourcc': cv2.VideoWriter_fourcc(*'H264'),
            'extension': '.mp4',
            'quality': 'high'
        },
        'xvid': {
            'fourcc': cv2.VideoWriter_fourcc(*'XVID'),
            'extension': '.avi',
            'quality': 'medium'
        },
        'mjpg': {
            'fourcc': cv2.VideoWriter_fourcc(*'MJPG'),
            'extension': '.avi',
            'quality': 'high'
        }
    }
    
    def __init__(self, output_path: str, fps: float, 
                 resolution: Tuple[int, int], codec: str = 'mp4v',
                 quality: str = 'medium'):
        """
        Initialize video writer.
        
        Args:
            output_path: Output video file path
            fps: Frames per second
            resolution: (width, height) tuple
            codec: Video codec ('mp4v', 'h264', 'xvid', 'mjpg')
            quality: Quality setting ('low', 'medium', 'high')
        """
        self.output_path = output_path
        self.fps = fps
        self.resolution = resolution
        self.codec = codec.lower()
        self.quality = quality
        
        # Video writer object
        self.writer = None
        self.is_opened = False
        
        # Statistics
        self.frames_written = 0
        self.total_size_bytes = 0
        
        # Validate codec
        if self.codec not in self.CODEC_CONFIGS:
            logger.warning(f"Unknown codec {codec}, using mp4v")
            self.codec = 'mp4v'
        
        # Ensure output path has correct extension
        self._ensure_correct_extension()
        
        # Initialize writer
        self._initialize_writer()
    
    def _ensure_correct_extension(self):
        """Ensure output path has correct extension for codec."""
        codec_config = self.CODEC_CONFIGS[self.codec]
        expected_ext = codec_config['extension']
        current_ext = Path(self.output_path).suffix.lower()
        
        if current_ext != expected_ext:
            self.output_path = str(Path(self.output_path).with_suffix(expected_ext))
            logger.info(f"Changed output extension to {expected_ext} for codec {self.codec}")
    
    def _initialize_writer(self):
        """Initialize the OpenCV VideoWriter."""
        try:
            # Create output directory if needed
            output_dir = Path(self.output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Get codec configuration
            codec_config = self.CODEC_CONFIGS[self.codec]
            fourcc = codec_config['fourcc']
            
            # Initialize writer
            self.writer = cv2.VideoWriter(
                self.output_path,
                fourcc,
                self.fps,
                self.resolution
            )
            
            if self.writer.isOpened():
                self.is_opened = True
                logger.info(f"Video writer initialized: {self.output_path}")
                logger.info(f"Codec: {self.codec}, FPS: {self.fps}, Resolution: {self.resolution}")
            else:
                raise RuntimeError(f"Failed to initialize video writer for {self.output_path}")
                
        except Exception as e:
            logger.error(f"Video writer initialization failed: {e}")
            raise
    
    def write_frame(self, frame: np.ndarray) -> bool:
        """
        Write a single frame to the video.
        
        Args:
            frame: Frame array (BGR format)
            
        Returns:
            True if frame written successfully
        """
        if not self.is_opened or self.writer is None:
            logger.error("Video writer not initialized")
            return False
        
        try:
            # Validate frame
            if frame is None or frame.size == 0:
                logger.warning("Empty frame provided, skipping")
                return False
            
            # Ensure frame has correct dimensions
            h, w = frame.shape[:2]
            if (w, h) != self.resolution:
                frame = cv2.resize(frame, self.resolution)
            
            # Ensure frame is in correct format (BGR, uint8)
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # BGR format - write directly
                self.writer.write(frame)
            elif len(frame.shape) == 3 and frame.shape[2] == 4:
                # BGRA format - convert to BGR
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                self.writer.write(frame_bgr)
            elif len(frame.shape) == 2:
                # Grayscale - convert to BGR
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                self.writer.write(frame_bgr)
            else:
                logger.error(f"Unsupported frame format: shape={frame.shape}")
                return False
            
            self.frames_written += 1
            return True
            
        except Exception as e:
            logger.error(f"Failed to write frame {self.frames_written}: {e}")
            return False
    
    def write_frames(self, frames: list) -> int:
        """
        Write multiple frames to the video.
        
        Args:
            frames: List of frame arrays
            
        Returns:
            Number of frames written successfully
        """
        successful_writes = 0
        
        for i, frame in enumerate(frames):
            if self.write_frame(frame):
                successful_writes += 1
            else:
                logger.warning(f"Failed to write frame {i}")
        
        return successful_writes
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get writing statistics.
        
        Returns:
            Dictionary with writing statistics
        """
        stats = {
            'frames_written': self.frames_written,
            'output_path': self.output_path,
            'fps': self.fps,
            'resolution': self.resolution,
            'codec': self.codec,
            'quality': self.quality,
            'is_opened': self.is_opened,
            'duration_seconds': self.frames_written / self.fps if self.fps > 0 else 0
        }
        
        # Add file size if file exists
        if os.path.exists(self.output_path):
            stats['file_size_bytes'] = os.path.getsize(self.output_path)
            stats['file_size_mb'] = stats['file_size_bytes'] / (1024 * 1024)
        
        return stats
    
    def set_quality_params(self, quality: str) -> None:
        """
        Set quality parameters (note: limited effect with OpenCV).
        
        Args:
            quality: Quality setting ('low', 'medium', 'high')
        """
        self.quality = quality
        
        # Note: OpenCV VideoWriter has limited quality control
        # Quality is mainly determined by codec choice and bitrate
        quality_settings = {
            'low': {'bitrate_factor': 0.5},
            'medium': {'bitrate_factor': 1.0},
            'high': {'bitrate_factor': 1.5}
        }
        
        if quality in quality_settings:
            logger.info(f"Quality set to: {quality}")
        else:
            logger.warning(f"Unknown quality setting: {quality}")
    
    def flush(self) -> None:
        """Flush any pending writes."""
        if self.writer and self.is_opened:
            # OpenCV VideoWriter doesn't have explicit flush
            # but we can call release and reinitialize if needed
            pass
    
    def release(self) -> None:
        """Release video writer resources."""
        if self.writer:
            self.writer.release()
            self.writer = None
            self.is_opened = False
            
            logger.info(f"Video writer released: {self.frames_written} frames written")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class QualityVideoWriter(VideoWriter):
    """
    Enhanced video writer with advanced quality control.
    
    Provides additional quality control features and optimization
    for face swapping applications.
    """
    
    def __init__(self, output_path: str, fps: float, 
                 resolution: Tuple[int, int], codec: str = 'mp4v',
                 quality: str = 'medium', target_bitrate: Optional[int] = None):
        """
        Initialize quality video writer.
        
        Args:
            output_path: Output video file path
            fps: Frames per second
            resolution: (width, height) tuple
            codec: Video codec
            quality: Quality setting
            target_bitrate: Target bitrate in kbps (if supported)
        """
        self.target_bitrate = target_bitrate
        super().__init__(output_path, fps, resolution, codec, quality)
    
    def estimate_bitrate(self) -> int:
        """
        Estimate appropriate bitrate based on resolution and quality.
        
        Returns:
            Estimated bitrate in kbps
        """
        width, height = self.resolution
        pixel_count = width * height
        
        # Base bitrate per pixel (kbps per pixel)
        quality_multipliers = {
            'low': 0.0001,
            'medium': 0.0002,
            'high': 0.0003
        }
        
        multiplier = quality_multipliers.get(self.quality, 0.0002)
        estimated_bitrate = int(pixel_count * multiplier)
        
        # Ensure minimum bitrate
        return max(estimated_bitrate, 500)
    
    def optimize_for_face_swapping(self) -> None:
        """Optimize settings specifically for face swapping output."""
        # Face swapping benefits from:
        # - Higher quality to preserve facial details
        # - Stable frame rate
        # - Good color reproduction
        
        if self.quality == 'low':
            logger.info("Upgrading quality to medium for face swapping")
            self.quality = 'medium'
        
        # Ensure minimum resolution for face detail preservation
        width, height = self.resolution
        if width < 640 or height < 480:
            logger.warning("Low resolution may affect face swapping quality")


def create_video_writer(output_path: str, reference_video_path: str,
                       codec: Optional[str] = None,
                       quality: str = 'medium') -> VideoWriter:
    """
    Create video writer based on reference video properties.
    
    Args:
        output_path: Output video file path
        reference_video_path: Path to reference video for properties
        codec: Video codec (uses reference codec if None)
        quality: Quality setting
        
    Returns:
        Configured VideoWriter instance
    """
    from .utils.video_utils import get_video_info
    
    try:
        # Get reference video properties
        ref_info = get_video_info(reference_video_path)
        
        # Use reference properties
        fps = ref_info['fps']
        resolution = (ref_info['width'], ref_info['height'])
        ref_codec = codec or 'mp4v'  # Default fallback
        
        return VideoWriter(
            output_path=output_path,
            fps=fps,
            resolution=resolution,
            codec=ref_codec,
            quality=quality
        )
        
    except Exception as e:
        logger.error(f"Failed to create video writer from reference: {e}")
        raise