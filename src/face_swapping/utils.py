"""
Face Swapping Utilities

Provides utility functions for face swapping operations including
validation, similarity calculation, and image processing helpers.
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple, Dict, Any
from scipy.spatial.distance import cosine

from ..face_detection import FaceDetection

logger = logging.getLogger(__name__)


def validate_face_images(source_image: np.ndarray, target_image: np.ndarray) -> bool:
    """
    Validate input images for face swapping.
    
    Args:
        source_image: Source image array
        target_image: Target image array
        
    Returns:
        True if images are valid for face swapping
    """
    try:
        # Check if images are numpy arrays
        if not isinstance(source_image, np.ndarray) or not isinstance(target_image, np.ndarray):
            logger.error("Images must be numpy arrays")
            return False
        
        # Check image dimensions
        if len(source_image.shape) != 3 or len(target_image.shape) != 3:
            logger.error("Images must be 3-channel (color) images")
            return False
        
        # Check color channels
        if source_image.shape[2] != 3 or target_image.shape[2] != 3:
            logger.error("Images must have 3 color channels")
            return False
        
        # Check minimum size
        min_size = 64
        if (source_image.shape[0] < min_size or source_image.shape[1] < min_size or
            target_image.shape[0] < min_size or target_image.shape[1] < min_size):
            logger.error(f"Images must be at least {min_size}x{min_size} pixels")
            return False
        
        # Check data type
        if source_image.dtype != np.uint8 or target_image.dtype != np.uint8:
            logger.warning("Images should be uint8 type")
        
        # Check value range
        if (np.max(source_image) > 255 or np.min(source_image) < 0 or
            np.max(target_image) > 255 or np.min(target_image) < 0):
            logger.error("Image values must be in range [0, 255]")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Image validation failed: {e}")
        return False


def calculate_face_similarity(source_face: FaceDetection, target_face: FaceDetection) -> float:
    """
    Calculate similarity between two faces based on landmarks.
    
    Args:
        source_face: Source face detection
        target_face: Target face detection
        
    Returns:
        Similarity score between 0.0 and 1.0
    """
    try:
        # Extract landmarks
        source_landmarks = source_face.get('landmarks')
        target_landmarks = target_face.get('landmarks')
        
        if source_landmarks is None or target_landmarks is None:
            logger.warning("No landmarks available for similarity calculation")
            return 0.5  # Neutral similarity
        
        # Convert to numpy arrays
        if isinstance(source_landmarks, list):
            source_landmarks = np.array(source_landmarks)
        if isinstance(target_landmarks, list):
            target_landmarks = np.array(target_landmarks)
        
        # Ensure same number of landmarks
        min_landmarks = min(source_landmarks.shape[0], target_landmarks.shape[0])
        source_landmarks = source_landmarks[:min_landmarks]
        target_landmarks = target_landmarks[:min_landmarks]
        
        # Normalize landmarks (center and scale)
        source_normalized = normalize_landmarks(source_landmarks)
        target_normalized = normalize_landmarks(target_landmarks)
        
        # Calculate similarity using different metrics
        geometric_similarity = calculate_geometric_similarity(source_normalized, target_normalized)
        shape_similarity = calculate_shape_similarity(source_landmarks, target_landmarks)
        
        # Combine similarities
        overall_similarity = (geometric_similarity + shape_similarity) / 2.0
        
        return np.clip(overall_similarity, 0.0, 1.0)
        
    except Exception as e:
        logger.error(f"Similarity calculation failed: {e}")
        return 0.0


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Normalize landmarks by centering and scaling.
    
    Args:
        landmarks: Facial landmarks array
        
    Returns:
        Normalized landmarks
    """
    try:
        # Center landmarks
        centroid = np.mean(landmarks, axis=0)
        centered = landmarks - centroid
        
        # Scale landmarks
        distances = np.linalg.norm(centered, axis=1)
        max_distance = np.max(distances)
        
        if max_distance > 0:
            normalized = centered / max_distance
        else:
            normalized = centered
        
        return normalized
        
    except Exception as e:
        logger.error(f"Landmark normalization failed: {e}")
        return landmarks


def calculate_geometric_similarity(landmarks1: np.ndarray, landmarks2: np.ndarray) -> float:
    """
    Calculate geometric similarity between normalized landmarks.
    
    Args:
        landmarks1: First set of normalized landmarks
        landmarks2: Second set of normalized landmarks
        
    Returns:
        Geometric similarity score
    """
    try:
        # Calculate pairwise distances
        distances = np.linalg.norm(landmarks1 - landmarks2, axis=1)
        
        # Convert to similarity (lower distance = higher similarity)
        avg_distance = np.mean(distances)
        similarity = np.exp(-avg_distance * 5.0)  # Exponential decay
        
        return similarity
        
    except Exception as e:
        logger.error(f"Geometric similarity calculation failed: {e}")
        return 0.0


def calculate_shape_similarity(landmarks1: np.ndarray, landmarks2: np.ndarray) -> float:
    """
    Calculate shape similarity using face proportions.
    
    Args:
        landmarks1: First set of landmarks
        landmarks2: Second set of landmarks
        
    Returns:
        Shape similarity score
    """
    try:
        if landmarks1.shape[0] < 68 or landmarks2.shape[0] < 68:
            logger.warning("Insufficient landmarks for shape analysis")
            return 0.5
        
        # Calculate face measurements
        measurements1 = extract_face_measurements(landmarks1)
        measurements2 = extract_face_measurements(landmarks2)
        
        if not measurements1 or not measurements2:
            return 0.5
        
        # Compare measurements
        similarities = []
        
        for key in measurements1:
            if key in measurements2:
                val1, val2 = measurements1[key], measurements2[key]
                if val1 > 0 and val2 > 0:
                    ratio = min(val1, val2) / max(val1, val2)
                    similarities.append(ratio)
        
        if similarities:
            return np.mean(similarities)
        else:
            return 0.5
            
    except Exception as e:
        logger.error(f"Shape similarity calculation failed: {e}")
        return 0.0


def extract_face_measurements(landmarks: np.ndarray) -> Dict[str, float]:
    """
    Extract facial measurements from landmarks.
    
    Args:
        landmarks: 68-point facial landmarks
        
    Returns:
        Dictionary of face measurements
    """
    try:
        measurements = {}
        
        # Eye distances
        left_eye_center = np.mean(landmarks[42:48], axis=0)
        right_eye_center = np.mean(landmarks[36:42], axis=0)
        eye_distance = np.linalg.norm(left_eye_center - right_eye_center)
        
        # Face width (jaw width)
        face_width = np.linalg.norm(landmarks[0] - landmarks[16])
        
        # Face height (forehead to chin)
        face_height = np.linalg.norm(landmarks[8] - np.mean(landmarks[19:25], axis=0))
        
        # Nose length
        nose_length = np.linalg.norm(landmarks[27] - landmarks[33])
        
        # Mouth width
        mouth_width = np.linalg.norm(landmarks[48] - landmarks[54])
        
        # Store normalized measurements
        if eye_distance > 0:
            measurements['face_width_ratio'] = face_width / eye_distance
            measurements['face_height_ratio'] = face_height / eye_distance
            measurements['nose_length_ratio'] = nose_length / eye_distance
            measurements['mouth_width_ratio'] = mouth_width / eye_distance
        
        return measurements
        
    except Exception as e:
        logger.error(f"Face measurement extraction failed: {e}")
        return {}


def extract_face_region(image: np.ndarray, face_detection: FaceDetection,
                       padding: float = 0.3) -> Optional[np.ndarray]:
    """
    Extract face region from image with padding.
    
    Args:
        image: Source image
        face_detection: Face detection with bounding box
        padding: Padding ratio (0.3 = 30% padding)
        
    Returns:
        Extracted face region or None if extraction failed
    """
    try:
        bbox = face_detection.get('bbox')
        if bbox is None:
            logger.error("No bounding box in face detection")
            return None
        
        x, y, w, h = bbox
        
        # Add padding
        pad_w = int(w * padding)
        pad_h = int(h * padding)
        
        # Calculate padded coordinates
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(image.shape[1], x + w + pad_w)
        y2 = min(image.shape[0], y + h + pad_h)
        
        # Extract region
        face_region = image[y1:y2, x1:x2]
        
        if face_region.size == 0:
            logger.error("Extracted face region is empty")
            return None
        
        return face_region
        
    except Exception as e:
        logger.error(f"Face region extraction failed: {e}")
        return None


def resize_face_to_target(face_image: np.ndarray, target_size: Tuple[int, int],
                         interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    """
    Resize face image to target size.
    
    Args:
        face_image: Face image to resize
        target_size: Target size (width, height)
        interpolation: Interpolation method
        
    Returns:
        Resized face image
    """
    try:
        if face_image.shape[:2] == target_size[::-1]:  # OpenCV uses (height, width)
            return face_image
        
        resized = cv2.resize(face_image, target_size, interpolation=interpolation)
        return resized
        
    except Exception as e:
        logger.error(f"Face resizing failed: {e}")
        return face_image


def apply_color_correction(source_image: np.ndarray, target_image: np.ndarray,
                          mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Apply color correction to match source to target colors.
    
    Args:
        source_image: Source image to correct
        target_image: Reference target image
        mask: Optional mask for region-specific correction
        
    Returns:
        Color-corrected source image
    """
    try:
        result = source_image.copy().astype(np.float32)
        
        # Use mask if provided, otherwise use entire image
        if mask is not None:
            mask_bool = mask > 127
        else:
            mask_bool = np.ones(source_image.shape[:2], dtype=bool)
        
        # Apply correction per channel
        for c in range(source_image.shape[2]):
            src_channel = source_image[:, :, c].astype(np.float32)
            tgt_channel = target_image[:, :, c].astype(np.float32)
            
            # Calculate statistics
            src_pixels = src_channel[mask_bool]
            tgt_pixels = tgt_channel[mask_bool]
            
            if len(src_pixels) > 0 and len(tgt_pixels) > 0:
                src_mean = np.mean(src_pixels)
                src_std = np.std(src_pixels)
                tgt_mean = np.mean(tgt_pixels)
                tgt_std = np.std(tgt_pixels)
                
                # Apply color transfer
                if src_std > 0:
                    corrected = (src_channel - src_mean) * (tgt_std / src_std) + tgt_mean
                    result[:, :, c] = corrected
        
        return np.clip(result, 0, 255).astype(np.uint8)
        
    except Exception as e:
        logger.error(f"Color correction failed: {e}")
        return source_image


def enhance_face_quality(face_image: np.ndarray, enhancement_level: float = 0.5) -> np.ndarray:
    """
    Enhance face image quality using various techniques.
    
    Args:
        face_image: Face image to enhance
        enhancement_level: Enhancement strength (0.0 to 1.0)
        
    Returns:
        Enhanced face image
    """
    try:
        if enhancement_level <= 0:
            return face_image
        
        enhanced = face_image.copy()
        
        # Apply sharpening
        if enhancement_level > 0.3:
            kernel = np.array([[-1, -1, -1],
                             [-1,  9, -1],
                             [-1, -1, -1]]) * enhancement_level * 0.3
            enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        # Apply contrast enhancement
        if enhancement_level > 0.5:
            enhanced = cv2.convertScaleAbs(enhanced, 
                                         alpha=1.0 + enhancement_level * 0.2,
                                         beta=0)
        
        # Apply noise reduction
        if enhancement_level > 0.7:
            enhanced = cv2.bilateralFilter(enhanced, 9, 
                                         int(enhancement_level * 30),
                                         int(enhancement_level * 30))
        
        return enhanced
        
    except Exception as e:
        logger.error(f"Face enhancement failed: {e}")
        return face_image


def calculate_face_pose(landmarks: np.ndarray) -> Dict[str, float]:
    """
    Calculate face pose angles from landmarks.
    
    Args:
        landmarks: 68-point facial landmarks
        
    Returns:
        Dictionary with pose angles (yaw, pitch, roll)
    """
    try:
        if landmarks.shape[0] < 68:
            return {'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0}
        
        # Define 3D model points (simplified)
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype=np.float32)
        
        # 2D image points from landmarks
        image_points = np.array([
            landmarks[33],    # Nose tip
            landmarks[8],     # Chin
            landmarks[45],    # Left eye left corner
            landmarks[36],    # Right eye right corner
            landmarks[54],    # Left mouth corner
            landmarks[48]     # Right mouth corner
        ], dtype=np.float32)
        
        # Camera internals (estimated)
        size = max(landmarks[:, 0]) - min(landmarks[:, 0])
        focal_length = size
        center = (size / 2, size / 2)
        camera_matrix = np.array([[focal_length, 0, center[0]],
                                 [0, focal_length, center[1]],
                                 [0, 0, 1]], dtype=np.float32)
        
        # Solve PnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, None)
        
        if success:
            # Convert rotation vector to rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            
            # Extract Euler angles
            sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
            
            singular = sy < 1e-6
            
            if not singular:
                x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
                y = np.arctan2(-rotation_matrix[2, 0], sy)
                z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
            else:
                x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
                y = np.arctan2(-rotation_matrix[2, 0], sy)
                z = 0
            
            # Convert to degrees
            pitch = np.degrees(x)
            yaw = np.degrees(y)
            roll = np.degrees(z)
            
            return {'yaw': yaw, 'pitch': pitch, 'roll': roll}
        
        return {'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0}
        
    except Exception as e:
        logger.error(f"Pose calculation failed: {e}")
        return {'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0}


def is_face_frontal(landmarks: np.ndarray, threshold: float = 25.0) -> bool:
    """
    Check if face is in frontal pose.
    
    Args:
        landmarks: Facial landmarks
        threshold: Maximum angle threshold for frontal pose
        
    Returns:
        True if face is frontal
    """
    try:
        pose = calculate_face_pose(landmarks)
        
        # Check if all angles are within threshold
        return (abs(pose['yaw']) < threshold and 
                abs(pose['pitch']) < threshold and 
                abs(pose['roll']) < threshold)
        
    except Exception as e:
        logger.error(f"Frontal face check failed: {e}")
        return False