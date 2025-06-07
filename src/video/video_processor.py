"""
Video Processor - Main video processing pipeline

Handles video file loading, frame extraction, processing, and output generation.
Supports various video formats and provides memory-efficient processing.
"""

import cv2
import numpy as np
import logging
from typing import Callable, Iterator, Optional, Dict, Any, Tuple
from pathlib import Path
import tempfile
import os

from .utils.video_utils import validate_video_file, get_video_info, extract_audio, merge_audio
from .frame_processor import FrameProcessor
from .video_writer import VideoWriter

logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Main video processing pipeline for face swapping operations.
    
    Handles video loading, frame-by-frame processing, and output generation
    with audio preservation and progress tracking.
    """
    
    SUPPORTED_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    SUPPORTED_CODECS = ['H264', 'XVID', 'MJPG', 'MP4V']
    
    def __init__(self, input_path: Optional[str] = None, 
                 output_path: Optional[str] = None,
                 temp_dir: Optional[str] = None):
        """
        Initialize video processor.
        
        Args:
            input_path: Path to input video file
            output_path: Path for output video file
            temp_dir: Directory for temporary files
        """
        self.input_path = input_path
        self.output_path = output_path
        self.temp_dir = temp_dir or tempfile.gettempdir()
        
        # Video properties
        self.cap = None
        self.video_info = {}
        self.frame_processor = FrameProcessor()
        self.video_writer = None
        
        # Processing state
        self.current_frame = 0
        self.total_frames = 0
        self.is_processing = False
        
        # Audio handling
        self.audio_path = None
        self.preserve_audio = True
        
    def load_video(self, input_path: str) -> Dict[str, Any]:
        """
        Load and validate video file.
        
        Args:
            input_path: Path to video file
            
        Returns:
            Dictionary with video information
            
        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If video format is not supported
        """
        self.input_path = input_path
        
        # Validate file
        if not validate_video_file(input_path):
            raise ValueError(f"Unsupported video format: {input_path}")
        
        # Get video information
        self.video_info = get_video_info(input_path)
        self.total_frames = self.video_info['frame_count']
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(input_path)
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video: {input_path}")
        
        logger.info(f"Loaded video: {input_path}")
        logger.info(f"Resolution: {self.video_info['width']}x{self.video_info['height']}")
        logger.info(f"FPS: {self.video_info['fps']}")
        logger.info(f"Duration: {self.video_info['duration']:.2f}s")
        logger.info(f"Total frames: {self.total_frames}")
        
        return self.video_info
    
    def setup_output(self, output_path: str, 
                    fps: Optional[float] = None,
                    resolution: Optional[Tuple[int, int]] = None,
                    codec: str = 'mp4v') -> None:
        """
        Setup video output writer.
        
        Args:
            output_path: Path for output video
            fps: Output frame rate (uses input fps if None)
            resolution: Output resolution (uses input resolution if None)
            codec: Video codec to use
        """
        self.output_path = output_path
        
        if not self.video_info:
            raise ValueError("No input video loaded. Call load_video() first.")
        
        # Use input video properties if not specified
        output_fps = fps or self.video_info['fps']
        output_resolution = resolution or (self.video_info['width'], self.video_info['height'])
        
        # Create output directory if needed
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize video writer
        self.video_writer = VideoWriter(
            output_path, 
            output_fps, 
            output_resolution,
            codec=codec
        )
        
        logger.info(f"Output setup: {output_path}")
        logger.info(f"Output FPS: {output_fps}")
        logger.info(f"Output resolution: {output_resolution}")
    
    def extract_audio_track(self) -> Optional[str]:
        """
        Extract audio from input video.
        
        Returns:
            Path to extracted audio file, or None if no audio
        """
        if not self.input_path or not self.preserve_audio:
            return None
        
        # Create temporary audio file
        audio_filename = f"audio_{os.getpid()}.aac"
        self.audio_path = os.path.join(self.temp_dir, audio_filename)
        
        try:
            success = extract_audio(self.input_path, self.audio_path)
            if success:
                logger.info(f"Audio extracted: {self.audio_path}")
                return self.audio_path
            else:
                logger.info("Audio extraction failed or no audio track found")
                logger.info("Video processing will continue without audio")
                self.audio_path = None
                return None
        except Exception as e:
            logger.error(f"Audio extraction failed: {e}")
            self.audio_path = None
            return None
    
    def process_frames(self, 
                      frame_callback: Callable[[int, np.ndarray], np.ndarray],
                      progress_callback: Optional[Callable[[int, int], None]] = None,
                      start_frame: int = 0,
                      end_frame: Optional[int] = None) -> bool:
        """
        Process video frames with callback function.
        
        Args:
            frame_callback: Function to process each frame
            progress_callback: Optional progress tracking callback
            start_frame: Frame to start processing from
            end_frame: Frame to stop processing at (None for end of video)
            
        Returns:
            True if processing completed successfully
        """
        if not self.cap or not self.video_writer:
            raise ValueError("Video not loaded or output not setup")
        
        self.is_processing = True
        end_frame = end_frame or self.total_frames
        
        try:
            # Seek to start frame
            if start_frame > 0:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            self.current_frame = start_frame
            
            while self.current_frame < end_frame:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Process frame
                try:
                    processed_frame = frame_callback(self.current_frame, frame)
                    if processed_frame is not None:
                        self.video_writer.write_frame(processed_frame)
                except Exception as e:
                    logger.error(f"Error processing frame {self.current_frame}: {e}")
                    # Continue with original frame
                    self.video_writer.write_frame(frame)
                
                self.current_frame += 1
                
                # Update progress
                if progress_callback:
                    progress_callback(self.current_frame, self.total_frames)
            
            logger.info(f"Processed {self.current_frame - start_frame} frames")
            return True
            
        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            return False
        finally:
            self.is_processing = False
    
    def finalize_output(self) -> bool:
        """
        Finalize video output and merge audio if available.
        
        Returns:
            True if finalization successful
        """
        if not self.video_writer:
            return False
        
        try:
            # Close video writer
            self.video_writer.release()
            
            # Merge audio if available
            if self.audio_path and os.path.exists(self.audio_path):
                temp_video = self.output_path + ".temp.mp4"
                os.rename(self.output_path, temp_video)
                
                success = merge_audio(temp_video, self.audio_path, self.output_path)
                
                if success:
                    os.remove(temp_video)
                    logger.info("Audio merged successfully")
                else:
                    os.rename(temp_video, self.output_path)
                    logger.warning("Audio merge failed, keeping video without audio")
                
                # Clean up temporary audio
                try:
                    os.remove(self.audio_path)
                except:
                    pass
            
            logger.info(f"Video processing complete: {self.output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Output finalization failed: {e}")
            return False
    
    def process_video(self,
                     frame_callback: Callable[[int, np.ndarray], np.ndarray],
                     progress_callback: Optional[Callable[[int, int], None]] = None) -> bool:
        """
        Complete video processing pipeline.
        
        Args:
            frame_callback: Function to process each frame
            progress_callback: Optional progress tracking callback
            
        Returns:
            True if processing completed successfully
        """
        try:
            # Extract audio
            self.extract_audio_track()
            
            # Process frames
            success = self.process_frames(frame_callback, progress_callback)
            
            if success:
                # Finalize output
                return self.finalize_output()
            
            return False
            
        except Exception as e:
            logger.error(f"Video processing pipeline failed: {e}")
            return False
    
    def get_frame_iterator(self, start_frame: int = 0, 
                          end_frame: Optional[int] = None) -> Iterator[Tuple[int, np.ndarray]]:
        """
        Get iterator for video frames.
        
        Args:
            start_frame: Frame to start from
            end_frame: Frame to end at (None for end of video)
            
        Yields:
            Tuple of (frame_number, frame_array)
        """
        if not self.cap:
            raise ValueError("Video not loaded")
        
        end_frame = end_frame or self.total_frames
        
        # Seek to start frame
        if start_frame > 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frame_num = start_frame
        while frame_num < end_frame:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            yield frame_num, frame
            frame_num += 1
    
    def get_frame_at(self, frame_number: int) -> Optional[np.ndarray]:
        """
        Get specific frame by number.
        
        Args:
            frame_number: Frame number to retrieve
            
        Returns:
            Frame array or None if frame not available
        """
        if not self.cap:
            return None
        
        if frame_number < 0 or frame_number >= self.total_frames:
            return None
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        
        return frame if ret else None
    
    def estimate_processing_time(self, 
                               frames_per_second_processing: float = 10.0) -> Dict[str, float]:
        """
        Estimate processing time based on video properties.
        
        Args:
            frames_per_second_processing: Expected processing speed
            
        Returns:
            Dictionary with time estimates
        """
        if not self.video_info:
            return {}
        
        processing_fps = frames_per_second_processing
        estimated_time = self.total_frames / processing_fps
        
        return {
            'total_frames': self.total_frames,
            'processing_fps': processing_fps,
            'estimated_seconds': estimated_time,
            'estimated_minutes': estimated_time / 60,
            'estimated_hours': estimated_time / 3600
        }
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.cap:
            self.cap.release()
            self.cap = None
        
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        
        # Clean up temporary audio file
        if self.audio_path and os.path.exists(self.audio_path):
            try:
                os.remove(self.audio_path)
            except:
                pass
        
        self.is_processing = False
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()