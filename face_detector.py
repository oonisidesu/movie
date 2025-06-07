"""
Face detection utilities for the face swapping CLI.

Provides face detection and selection capabilities.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class FaceDetector:
    """Face detection using OpenCV's built-in classifiers."""
    
    def __init__(self, use_dnn: bool = False):
        self.use_dnn = use_dnn
        self.face_cascade = None
        self.net = None
        self._load_models()
    
    def _load_models(self):
        """Load face detection models."""
        try:
            # Load Haar cascade for basic detection
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            if self.face_cascade.empty():
                raise RuntimeError("Could not load Haar cascade")
            
            # Optionally load DNN model for better detection
            if self.use_dnn:
                try:
                    # Try to load OpenCV's DNN face detection model
                    # Note: This would require downloading the model files
                    # For now, we'll stick with Haar cascades
                    logger.info("DNN face detection not implemented yet, using Haar cascades")
                    self.use_dnn = False
                except Exception as e:
                    logger.warning(f"Could not load DNN model: {e}")
                    self.use_dnn = False
            
            logger.info(f"Face detector initialized (DNN: {self.use_dnn})")
            
        except Exception as e:
            raise RuntimeError(f"Could not initialize face detector: {e}")
    
    def detect_faces(self, image: np.ndarray, min_size: Tuple[int, int] = (30, 30)) -> List[Dict]:
        """
        Detect faces in an image.
        
        Args:
            image: Input image (BGR format)
            min_size: Minimum face size (width, height)
            
        Returns:
            List of face dictionaries with 'bbox', 'confidence', and 'landmarks'
        """
        if image is None or image.size == 0:
            return []
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces using Haar cascade
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=min_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Convert to list of dictionaries
        face_list = []
        for i, (x, y, w, h) in enumerate(faces):
            face_dict = {
                'id': i,
                'bbox': (x, y, w, h),
                'confidence': 1.0,  # Haar cascades don't provide confidence
                'center': (x + w // 2, y + h // 2),
                'area': w * h
            }
            face_list.append(face_dict)
        
        # Sort by area (largest first)
        face_list.sort(key=lambda f: f['area'], reverse=True)
        
        return face_list
    
    def extract_face_region(self, image: np.ndarray, bbox: Tuple[int, int, int, int], 
                          padding: float = 0.2) -> np.ndarray:
        """
        Extract face region from image with optional padding.
        
        Args:
            image: Input image
            bbox: Face bounding box (x, y, w, h)
            padding: Padding factor (0.2 = 20% padding)
            
        Returns:
            Extracted face region
        """
        x, y, w, h = bbox
        
        # Calculate padding
        pad_w = int(w * padding)
        pad_h = int(h * padding)
        
        # Expand bounding box with padding
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(image.shape[1], x + w + pad_w)
        y2 = min(image.shape[0], y + h + pad_h)
        
        return image[y1:y2, x1:x2]
    
    def draw_faces(self, image: np.ndarray, faces: List[Dict], 
                   show_ids: bool = True) -> np.ndarray:
        """
        Draw face bounding boxes on image.
        
        Args:
            image: Input image
            faces: List of face dictionaries
            show_ids: Whether to show face IDs
            
        Returns:
            Image with drawn faces
        """
        result = image.copy()
        
        for face in faces:
            x, y, w, h = face['bbox']
            face_id = face['id']
            
            # Draw bounding box
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw face ID
            if show_ids:
                cv2.putText(
                    result,
                    f"Face {face_id}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )
        
        return result


def preview_face_detection(image_path: str, output_path: Optional[str] = None) -> List[Dict]:
    """
    Preview face detection on an image.
    
    Args:
        image_path: Path to input image
        output_path: Optional path to save preview image
        
    Returns:
        List of detected faces
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    
    # Detect faces
    detector = FaceDetector()
    faces = detector.detect_faces(image)
    
    logger.info(f"Detected {len(faces)} faces in {image_path}")
    
    # Draw faces on image
    preview_image = detector.draw_faces(image, faces)
    
    # Save or display preview
    if output_path:
        cv2.imwrite(output_path, preview_image)
        logger.info(f"Preview saved to: {output_path}")
    else:
        # Display using OpenCV (if display is available)
        try:
            cv2.imshow('Face Detection Preview', preview_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error:
            logger.warning("Could not display preview (no display available)")
    
    return faces


def get_largest_face(faces: List[Dict]) -> Optional[Dict]:
    """Get the largest detected face."""
    if not faces:
        return None
    return max(faces, key=lambda f: f['area'])


def select_face_by_index(faces: List[Dict], index: int) -> Optional[Dict]:
    """Select face by index (0-based)."""
    if not faces or index < 0 or index >= len(faces):
        return None
    return faces[index]


def validate_face_image(image_path: str) -> bool:
    """Validate that an image contains at least one detectable face."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            return False
        
        detector = FaceDetector()
        faces = detector.detect_faces(image)
        
        return len(faces) > 0
        
    except Exception as e:
        logger.error(f"Error validating face image: {e}")
        return False