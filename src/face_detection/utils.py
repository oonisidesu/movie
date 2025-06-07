"""
Utility functions for face detection module.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
import os


def ensure_model_directory():
    """Ensure the models directory exists."""
    models_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    return models_dir


def download_opencv_models():
    """Download required OpenCV DNN models if not present."""
    models_dir = ensure_model_directory()
    
    # URLs for OpenCV face detection models
    prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
    
    prototxt_path = os.path.join(models_dir, 'deploy.prototxt')
    model_path = os.path.join(models_dir, 'res10_300x300_ssd_iter_140000.caffemodel')
    
    # This is a placeholder - actual download implementation would go here
    # For now, users need to manually download these files
    return prototxt_path, model_path


def align_face(image: np.ndarray, landmarks: List[Tuple[int, int]], 
               output_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """
    Align face based on eye positions.
    
    Args:
        image: Input face image
        landmarks: Facial landmarks (expecting at least 2 eye points)
        output_size: Desired output size
        
    Returns:
        Aligned face image
    """
    if len(landmarks) < 2:
        # No alignment possible, just resize
        return cv2.resize(image, output_size)
    
    # Assume first two landmarks are left and right eyes
    left_eye = landmarks[0]
    right_eye = landmarks[1]
    
    # Calculate angle between eyes
    eye_angle = np.arctan2(right_eye[1] - left_eye[1], 
                          right_eye[0] - left_eye[0])
    angle_degrees = np.degrees(eye_angle)
    
    # Calculate center point between eyes
    eye_center = ((left_eye[0] + right_eye[0]) // 2,
                  (left_eye[1] + right_eye[1]) // 2)
    
    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(eye_center, angle_degrees, 1.0)
    
    # Apply rotation
    rotated = cv2.warpAffine(image, rotation_matrix, 
                           (image.shape[1], image.shape[0]))
    
    # Resize to output size
    aligned = cv2.resize(rotated, output_size)
    
    return aligned


def crop_face_with_margin(image: np.ndarray, bbox: Tuple[int, int, int, int],
                         margin: float = 0.2) -> np.ndarray:
    """
    Crop face from image with specified margin.
    
    Args:
        image: Input image
        bbox: Bounding box (x, y, width, height)
        margin: Margin ratio to add around face
        
    Returns:
        Cropped face image
    """
    x, y, w, h = bbox
    
    # Calculate margin in pixels
    margin_x = int(w * margin)
    margin_y = int(h * margin)
    
    # Calculate expanded bounding box
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(image.shape[1], x + w + margin_x)
    y2 = min(image.shape[0], y + h + margin_y)
    
    return image[y1:y2, x1:x2]


def normalize_face_size(face: np.ndarray, target_size: int = 256) -> np.ndarray:
    """
    Normalize face to standard size while maintaining aspect ratio.
    
    Args:
        face: Input face image
        target_size: Target size for the larger dimension
        
    Returns:
        Normalized face image
    """
    h, w = face.shape[:2]
    
    if h > w:
        new_h = target_size
        new_w = int(w * target_size / h)
    else:
        new_w = target_size
        new_h = int(h * target_size / w)
    
    resized = cv2.resize(face, (new_w, new_h))
    
    # Create square image with padding if needed
    if new_h != target_size or new_w != target_size:
        square = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        y_offset = (target_size - new_h) // 2
        x_offset = (target_size - new_w) // 2
        square[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        return square
    
    return resized


def validate_face_image(image: np.ndarray, min_size: int = 64) -> bool:
    """
    Validate if image is suitable for face processing.
    
    Args:
        image: Input image
        min_size: Minimum size requirement
        
    Returns:
        True if image is valid
    """
    if image is None or image.size == 0:
        return False
    
    h, w = image.shape[:2]
    if h < min_size or w < min_size:
        return False
    
    # Check if image has reasonable pixel values
    if np.mean(image) < 5 or np.mean(image) > 250:
        return False
    
    return True


def calculate_face_area(bbox: Tuple[int, int, int, int]) -> int:
    """Calculate area of face bounding box."""
    x, y, w, h = bbox
    return w * h


def get_face_center(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """Get center point of face bounding box."""
    x, y, w, h = bbox
    return (x + w // 2, y + h // 2)