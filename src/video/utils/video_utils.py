"""
Video Utilities - Helper functions for video processing

Provides utility functions for video validation, metadata extraction,
audio handling, and memory usage estimation.
"""

import cv2
import os
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import platform

logger = logging.getLogger(__name__)


def validate_video_file(video_path: str) -> bool:
    """
    Validate if a file is a supported video format.
    
    Args:
        video_path: Path to video file
        
    Returns:
        True if video file is valid and supported
    """
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return False
    
    # Check file extension
    supported_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
    file_ext = Path(video_path).suffix.lower()
    
    if file_ext not in supported_extensions:
        logger.error(f"Unsupported video format: {file_ext}")
        return False
    
    # Try to open with OpenCV
    try:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            
            if ret and frame is not None:
                return True
            else:
                logger.error(f"Could not read frames from: {video_path}")
                return False
        else:
            logger.error(f"Could not open video: {video_path}")
            return False
            
    except Exception as e:
        logger.error(f"Video validation failed: {e}")
        return False


def get_video_info(video_path: str) -> Dict[str, Any]:
    """
    Extract comprehensive video metadata.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with video information
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    info = {}
    
    try:
        # Basic OpenCV properties
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get basic properties
        info['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        info['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        info['fps'] = cap.get(cv2.CAP_PROP_FPS)
        info['frame_count'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate duration
        if info['fps'] > 0:
            info['duration'] = info['frame_count'] / info['fps']
        else:
            info['duration'] = 0
        
        # Get codec information
        fourcc = cap.get(cv2.CAP_PROP_FOURCC)
        info['fourcc'] = int(fourcc)
        info['codec'] = ''.join([chr((int(fourcc) >> 8 * i) & 0xFF) for i in range(4)])
        
        cap.release()
        
        # Additional file information
        info['file_size'] = os.path.getsize(video_path)
        info['file_size_mb'] = info['file_size'] / (1024 * 1024)
        
        # Resolution classification
        info['resolution_class'] = classify_resolution(info['width'], info['height'])
        
        # Bitrate estimation (rough)
        if info['duration'] > 0:
            info['estimated_bitrate_kbps'] = (info['file_size'] * 8) / (info['duration'] * 1000)
        else:
            info['estimated_bitrate_kbps'] = 0
        
        logger.debug(f"Video info extracted: {video_path}")
        return info
        
    except Exception as e:
        logger.error(f"Failed to get video info: {e}")
        raise


def classify_resolution(width: int, height: int) -> str:
    """
    Classify video resolution into standard categories.
    
    Args:
        width: Video width
        height: Video height
        
    Returns:
        Resolution classification string
    """
    total_pixels = width * height
    
    if total_pixels >= 3840 * 2160:  # 4K
        return "4K (Ultra HD)"
    elif total_pixels >= 1920 * 1080:  # Full HD
        return "Full HD (1080p)"
    elif total_pixels >= 1280 * 720:  # HD
        return "HD (720p)"
    elif total_pixels >= 854 * 480:  # SD
        return "SD (480p)"
    else:
        return "Low Resolution"


def extract_audio(video_path: str, audio_output_path: str, 
                 audio_codec: str = 'aac') -> bool:
    """
    Extract audio from video file using ffmpeg.
    
    Args:
        video_path: Path to input video
        audio_output_path: Path for extracted audio
        audio_codec: Audio codec to use
        
    Returns:
        True if extraction successful
    """
    try:
        # Check if ffmpeg is available
        if not _check_ffmpeg():
            logger.error("ffmpeg not found. Cannot extract audio.")
            return False
        
        # Build ffmpeg command
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vn',  # No video
            '-acodec', audio_codec,
            '-y',  # Overwrite output
            audio_output_path
        ]
        
        # Run ffmpeg
        result = subprocess.run(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode == 0:
            logger.info(f"Audio extracted: {audio_output_path}")
            return True
        else:
            logger.error(f"Audio extraction failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Audio extraction error: {e}")
        return False


def merge_audio(video_path: str, audio_path: str, output_path: str) -> bool:
    """
    Merge video and audio files using ffmpeg.
    
    Args:
        video_path: Path to video file
        audio_path: Path to audio file
        output_path: Path for merged output
        
    Returns:
        True if merge successful
    """
    try:
        # Check if ffmpeg is available
        if not _check_ffmpeg():
            logger.error("ffmpeg not found. Cannot merge audio.")
            return False
        
        # Build ffmpeg command
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-i', audio_path,
            '-c:v', 'copy',  # Copy video stream
            '-c:a', 'aac',   # Re-encode audio to AAC
            '-map', '0:v:0', # Map video from first input
            '-map', '1:a:0', # Map audio from second input
            '-y',  # Overwrite output
            output_path
        ]
        
        # Run ffmpeg
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode == 0:
            logger.info(f"Audio merged: {output_path}")
            return True
        else:
            logger.error(f"Audio merge failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Audio merge error: {e}")
        return False


def _check_ffmpeg() -> bool:
    """
    Check if ffmpeg is available in the system.
    
    Returns:
        True if ffmpeg is available
    """
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def calculate_memory_usage(video_info: Dict[str, Any], 
                          buffer_frames: int = 30) -> Dict[str, float]:
    """
    Calculate estimated memory usage for video processing.
    
    Args:
        video_info: Video information dictionary
        buffer_frames: Number of frames to buffer in memory
        
    Returns:
        Dictionary with memory usage estimates
    """
    try:
        width = video_info['width']
        height = video_info['height']
        
        # Calculate bytes per frame (assuming BGR format)
        bytes_per_pixel = 3  # BGR
        bytes_per_frame = width * height * bytes_per_pixel
        
        # Memory estimates
        single_frame_mb = bytes_per_frame / (1024 * 1024)
        buffer_memory_mb = single_frame_mb * buffer_frames
        
        # Processing memory (original + processed frames)
        processing_memory_mb = single_frame_mb * 2
        
        # Total estimated memory
        total_memory_mb = buffer_memory_mb + processing_memory_mb
        
        return {
            'single_frame_mb': single_frame_mb,
            'buffer_memory_mb': buffer_memory_mb,
            'processing_memory_mb': processing_memory_mb,
            'total_memory_mb': total_memory_mb,
            'buffer_frames': buffer_frames,
            'bytes_per_frame': bytes_per_frame
        }
        
    except Exception as e:
        logger.error(f"Memory calculation failed: {e}")
        return {}


def optimize_video_settings(video_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recommend optimized settings based on video properties.
    
    Args:
        video_info: Video information dictionary
        
    Returns:
        Dictionary with recommended settings
    """
    width = video_info['width']
    height = video_info['height']
    fps = video_info['fps']
    duration = video_info['duration']
    
    recommendations = {}
    
    # Resolution recommendations
    if width > 1920 or height > 1080:
        recommendations['suggested_resize'] = (1920, 1080)
        recommendations['resize_reason'] = "4K processing is very memory intensive"
    
    # FPS recommendations
    if fps > 30:
        recommendations['suggested_fps'] = 30
        recommendations['fps_reason'] = "High FPS increases processing time significantly"
    
    # Processing chunk size
    total_pixels = width * height
    if total_pixels > 1920 * 1080:
        recommendations['chunk_size'] = 10  # Process fewer frames at once
    elif total_pixels > 1280 * 720:
        recommendations['chunk_size'] = 20
    else:
        recommendations['chunk_size'] = 30
    
    # Quality recommendations
    if duration > 300:  # 5 minutes
        recommendations['suggested_quality'] = 'medium'
        recommendations['quality_reason'] = "Long videos benefit from balanced quality/speed"
    else:
        recommendations['suggested_quality'] = 'high'
    
    return recommendations


def convert_video_format(input_path: str, output_path: str, 
                        target_format: str = 'mp4',
                        quality: str = 'medium') -> bool:
    """
    Convert video to different format using ffmpeg.
    
    Args:
        input_path: Path to input video
        output_path: Path for converted output
        target_format: Target format ('mp4', 'avi', etc.)
        quality: Quality setting ('low', 'medium', 'high')
        
    Returns:
        True if conversion successful
    """
    try:
        if not _check_ffmpeg():
            logger.error("ffmpeg not found. Cannot convert video.")
            return False
        
        # Quality settings
        quality_params = {
            'low': ['-crf', '28'],
            'medium': ['-crf', '23'],
            'high': ['-crf', '18']
        }
        
        quality_args = quality_params.get(quality, quality_params['medium'])
        
        # Build ffmpeg command
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-c:v', 'libx264',
            '-c:a', 'aac',
            *quality_args,
            '-y',
            output_path
        ]
        
        # Run conversion
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode == 0:
            logger.info(f"Video converted: {output_path}")
            return True
        else:
            logger.error(f"Video conversion failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Video conversion error: {e}")
        return False


def get_video_thumbnail(video_path: str, timestamp: float = 1.0) -> Optional[str]:
    """
    Extract a thumbnail from video at specified timestamp.
    
    Args:
        video_path: Path to video file
        timestamp: Timestamp in seconds
        
    Returns:
        Path to thumbnail image or None if failed
    """
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return None
        
        # Seek to timestamp
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            # Save thumbnail
            thumbnail_path = video_path.replace(Path(video_path).suffix, '_thumbnail.jpg')
            cv2.imwrite(thumbnail_path, frame)
            return thumbnail_path
        
        return None
        
    except Exception as e:
        logger.error(f"Thumbnail extraction failed: {e}")
        return None


def check_video_integrity(video_path: str) -> Dict[str, Any]:
    """
    Check video file integrity and report any issues.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with integrity check results
    """
    results = {
        'is_valid': False,
        'can_read_frames': False,
        'frame_read_errors': 0,
        'total_frames_checked': 0,
        'issues': []
    }
    
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            results['issues'].append("Cannot open video file")
            return results
        
        results['is_valid'] = True
        
        # Check frame reading
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_frames = min(100, total_frames)  # Sample first 100 frames
        
        read_errors = 0
        for i in range(sample_frames):
            ret, frame = cap.read()
            if not ret or frame is None:
                read_errors += 1
        
        cap.release()
        
        results['total_frames_checked'] = sample_frames
        results['frame_read_errors'] = read_errors
        results['can_read_frames'] = read_errors == 0
        
        if read_errors > 0:
            error_rate = read_errors / sample_frames
            results['issues'].append(f"Frame read error rate: {error_rate:.2%}")
        
        # Additional checks can be added here
        
    except Exception as e:
        results['issues'].append(f"Integrity check failed: {e}")
    
    return results