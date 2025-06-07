"""
Face Detection Module - Main FaceDetector Class

This module provides face detection capabilities using multiple backends:
- OpenCV Haar Cascades (fast, basic)
- OpenCV DNN (better accuracy)
- MediaPipe (Google's solution, very accurate)

Author: Face Swapping Tool
License: Personal/Educational Use Only
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
import os

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None


class DetectionBackend(Enum):
    """Supported face detection backends."""
    OPENCV_HAAR = "opencv_haar"
    OPENCV_DNN = "opencv_dnn"
    MEDIAPIPE = "mediapipe"


class FaceDetection:
    """Container for face detection results."""
    
    def __init__(self, bbox: Tuple[int, int, int, int], confidence: float = 1.0, 
                 landmarks: Optional[np.ndarray] = None):
        """
        Initialize face detection result.
        
        Args:
            bbox: Bounding box as (x, y, width, height)
            confidence: Detection confidence score (0.0 to 1.0)
            landmarks: Optional facial landmarks array
        """
        self.bbox = bbox
        self.confidence = confidence
        self.landmarks = landmarks
        
    @property
    def x(self) -> int:
        """X coordinate of top-left corner."""
        return self.bbox[0]
    
    @property
    def y(self) -> int:
        """Y coordinate of top-left corner."""
        return self.bbox[1]
    
    @property
    def width(self) -> int:
        """Width of bounding box."""
        return self.bbox[2]
    
    @property
    def height(self) -> int:
        """Height of bounding box."""
        return self.bbox[3]
    
    @property
    def area(self) -> int:
        """Area of the bounding box."""
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[int, int]:
        """Center point of the bounding box."""
        return (self.x + self.width // 2, self.y + self.height // 2)


class FaceDetector:
    """
    Main face detection class supporting multiple backends.
    
    This class provides a unified interface for face detection using different
    algorithms and frameworks. It automatically handles model loading and
    provides consistent output format across all backends.
    """
    
    def __init__(self, backend: DetectionBackend = DetectionBackend.OPENCV_DNN,
                 min_detection_confidence: float = 0.5,
                 model_selection: int = 0):
        """
        Initialize the face detector.
        
        Args:
            backend: Detection backend to use
            min_detection_confidence: Minimum confidence threshold for detections
            model_selection: Model selection for MediaPipe (0 or 1)
        """
        self.backend = backend
        self.min_confidence = min_detection_confidence
        self.model_selection = model_selection
        self.detector = None
        self.logger = logging.getLogger(__name__)
        
        # Initialize the selected backend
        self._initialize_detector()
    
    def _initialize_detector(self) -> None:
        """Initialize the selected detection backend."""
        try:
            if self.backend == DetectionBackend.OPENCV_HAAR:
                self._init_opencv_haar()
            elif self.backend == DetectionBackend.OPENCV_DNN:
                self._init_opencv_dnn()
            elif self.backend == DetectionBackend.MEDIAPIPE:
                self._init_mediapipe()
            else:
                raise ValueError(f"Unsupported backend: {self.backend}")
                
            self.logger.info(f"Initialized {self.backend.value} face detector")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize {self.backend.value}: {e}")
            raise
    
    def _init_opencv_haar(self) -> None:
        """Initialize OpenCV Haar Cascade detector."""
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if not os.path.exists(cascade_path):
            raise FileNotFoundError(f"Haar cascade file not found: {cascade_path}")
        
        self.detector = cv2.CascadeClassifier(cascade_path)
        if self.detector.empty():
            raise RuntimeError("Failed to load Haar cascade classifier")
    
    def _init_opencv_dnn(self) -> None:
        """Initialize OpenCV DNN face detector."""
        # Use OpenCV's pre-trained DNN model
        # You may need to download these files separately
        prototxt_path = "models/deploy.prototxt"
        model_path = "models/res10_300x300_ssd_iter_140000.caffemodel"
        
        # For now, create a placeholder - in production you'd download/include these files
        try:
            self.detector = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        except cv2.error:
            # Fallback to a simpler approach if model files aren't available
            self.logger.warning("DNN model files not found, falling back to Haar cascades")
            self.backend = DetectionBackend.OPENCV_HAAR
            self._init_opencv_haar()
    
    def _init_mediapipe(self) -> None:
        """Initialize MediaPipe face detector."""
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe not available. Install with: pip install mediapipe")
        
        self.mp_face_detection = mp.solutions.face_detection
        self.detector = self.mp_face_detection.FaceDetection(
            model_selection=self.model_selection,
            min_detection_confidence=self.min_confidence
        )
    
    def detect_faces(self, image: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces in an image.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of FaceDetection objects
        """
        if image is None or image.size == 0:
            return []
        
        try:
            if self.backend == DetectionBackend.OPENCV_HAAR:
                return self._detect_opencv_haar(image)
            elif self.backend == DetectionBackend.OPENCV_DNN:
                return self._detect_opencv_dnn(image)
            elif self.backend == DetectionBackend.MEDIAPIPE:
                return self._detect_mediapipe(image)
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"Face detection failed: {e}")
            return []
    
    def _detect_opencv_haar(self, image: np.ndarray) -> List[FaceDetection]:
        """Detect faces using OpenCV Haar Cascades."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        detections = []
        for (x, y, w, h) in faces:
            detection = FaceDetection(
                bbox=(x, y, w, h),
                confidence=1.0  # Haar cascades don't provide confidence scores
            )
            detections.append(detection)
        
        return detections
    
    def _detect_opencv_dnn(self, image: np.ndarray) -> List[FaceDetection]:
        """Detect faces using OpenCV DNN."""
        if self.detector is None:
            return self._detect_opencv_haar(image)  # Fallback
        
        (h, w) = image.shape[:2]
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                   (300, 300), (104.0, 177.0, 123.0))
        
        # Pass blob through network
        self.detector.setInput(blob)
        detections_dnn = self.detector.forward()
        
        detections = []
        for i in range(0, detections_dnn.shape[2]):
            confidence = detections_dnn[0, 0, i, 2]
            
            if confidence > self.min_confidence:
                box = detections_dnn[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")
                
                # Ensure valid bounding box
                x = max(0, x)
                y = max(0, y)
                x1 = min(w, x1)
                y1 = min(h, y1)
                
                if x1 > x and y1 > y:
                    detection = FaceDetection(
                        bbox=(x, y, x1 - x, y1 - y),
                        confidence=float(confidence)
                    )
                    detections.append(detection)
        
        return detections
    
    def _detect_mediapipe(self, image: np.ndarray) -> List[FaceDetection]:
        """Detect faces using MediaPipe."""
        if not MEDIAPIPE_AVAILABLE:
            return []
        
        # Convert BGR to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb_image)
        
        detections = []
        if results.detections:
            h, w = image.shape[:2]
            
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                
                # Convert relative coordinates to absolute
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Ensure valid bounding box
                x = max(0, x)
                y = max(0, y)
                width = min(width, w - x)
                height = min(height, h - y)
                
                if width > 0 and height > 0:
                    face_detection = FaceDetection(
                        bbox=(x, y, width, height),
                        confidence=detection.score[0]
                    )
                    detections.append(face_detection)
        
        return detections
    
    def detect_largest_face(self, image: np.ndarray) -> Optional[FaceDetection]:
        """
        Detect and return the largest face in the image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            FaceDetection object for the largest face, or None if no faces found
        """
        faces = self.detect_faces(image)
        if not faces:
            return None
        
        # Return face with largest area
        return max(faces, key=lambda f: f.area)
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """
        Update the minimum confidence threshold.
        
        Args:
            threshold: New confidence threshold (0.0 to 1.0)
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
        
        self.min_confidence = threshold
        
        # Reinitialize MediaPipe detector if needed
        if self.backend == DetectionBackend.MEDIAPIPE:
            self._init_mediapipe()
    
    def get_face_region(self, image: np.ndarray, detection: FaceDetection,
                       padding: float = 0.1) -> Optional[np.ndarray]:
        """
        Extract face region from image with optional padding.
        
        Args:
            image: Source image
            detection: Face detection result
            padding: Relative padding to add around face (0.1 = 10%)
            
        Returns:
            Cropped face image or None if invalid
        """
        if image is None or detection is None:
            return None
        
        h, w = image.shape[:2]
        
        # Calculate padding
        pad_w = int(detection.width * padding)
        pad_h = int(detection.height * padding)
        
        # Expand bounding box with padding
        x1 = max(0, detection.x - pad_w)
        y1 = max(0, detection.y - pad_h)
        x2 = min(w, detection.x + detection.width + pad_w)
        y2 = min(h, detection.y + detection.height + pad_h)
        
        return image[y1:y2, x1:x2]
    
    def __del__(self):
        """Cleanup detector resources."""
        if hasattr(self, 'detector') and self.detector is not None:
            if self.backend == DetectionBackend.MEDIAPIPE and MEDIAPIPE_AVAILABLE:
                self.detector.close()


def get_available_backends() -> List[DetectionBackend]:
    """
    Get list of available detection backends.
    
    Returns:
        List of available DetectionBackend enum values
    """
    available = [DetectionBackend.OPENCV_HAAR, DetectionBackend.OPENCV_DNN]
    
    if MEDIAPIPE_AVAILABLE:
        available.append(DetectionBackend.MEDIAPIPE)
    
    return available


def create_face_detector(backend: str = "opencv_dnn", **kwargs) -> FaceDetector:
    """
    Factory function to create a face detector.
    
    Args:
        backend: Backend name as string
        **kwargs: Additional arguments for FaceDetector
        
    Returns:
        Configured FaceDetector instance
    """
    backend_enum = DetectionBackend(backend)
    return FaceDetector(backend=backend_enum, **kwargs)