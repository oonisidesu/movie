import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def get_video_info(video_path: str) -> dict:
    """Get basic video information without loading the full video.
    
    Args:
        video_path: Path to video file
        
    Returns:
        dict: Video information or empty dict if error
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {}
            
        info = {
            "path": video_path,
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration": 0
        }
        
        if info["fps"] > 0:
            info["duration"] = info["frame_count"] / info["fps"]
            
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        info["codec"] = chr(fourcc & 0xff) + chr((fourcc >> 8) & 0xff) + \
                      chr((fourcc >> 16) & 0xff) + chr((fourcc >> 24) & 0xff)
        
        cap.release()
        return info
        
    except Exception as e:
        logger.error(f"Error getting video info: {e}")
        return {}

def validate_video_file(video_path: str) -> bool:
    """Validate if video file is supported and readable.
    
    Args:
        video_path: Path to video file
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        path = Path(video_path)
        if not path.exists():
            logger.error(f"Video file not found: {video_path}")
            return False
            
        supported_formats = ['.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MOV']
        if path.suffix not in supported_formats:
            logger.error(f"Unsupported format: {path.suffix}")
            return False
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            cap.release()
            return False
            
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            logger.error(f"Cannot read frames from: {video_path}")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Error validating video: {e}")
        return False

def resize_frame_maintain_aspect(frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize frame while maintaining aspect ratio.
    
    Args:
        frame: Input frame
        target_size: (width, height) target size
        
    Returns:
        np.ndarray: Resized frame
    """
    h, w = frame.shape[:2]
    target_w, target_h = target_size
    
    aspect_ratio = w / h
    target_aspect = target_w / target_h
    
    if aspect_ratio > target_aspect:
        new_w = target_w
        new_h = int(target_w / aspect_ratio)
    else:
        new_h = target_h
        new_w = int(target_h * aspect_ratio)
        
    resized = cv2.resize(frame, (new_w, new_h))
    
    result = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    
    result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return result

def extract_frame_at_time(video_path: str, time_seconds: float) -> Optional[np.ndarray]:
    """Extract a single frame at specific time.
    
    Args:
        video_path: Path to video file
        time_seconds: Time in seconds
        
    Returns:
        Optional[np.ndarray]: Frame at specified time or None if error
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(time_seconds * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        
        cap.release()
        
        return frame if ret else None
        
    except Exception as e:
        logger.error(f"Error extracting frame: {e}")
        return None

def calculate_memory_usage(video_info: dict) -> dict:
    """Calculate estimated memory usage for video processing.
    
    Args:
        video_info: Video information dictionary
        
    Returns:
        dict: Memory usage estimates in MB
    """
    if not video_info:
        return {}
        
    width = video_info.get("width", 0)
    height = video_info.get("height", 0)
    frame_count = video_info.get("frame_count", 0)
    
    bytes_per_frame = width * height * 3
    
    single_frame_mb = bytes_per_frame / (1024 * 1024)
    all_frames_mb = (bytes_per_frame * frame_count) / (1024 * 1024)
    
    return {
        "single_frame_mb": round(single_frame_mb, 2),
        "all_frames_mb": round(all_frames_mb, 2),
        "estimated_processing_mb": round(all_frames_mb * 2, 2)
    }