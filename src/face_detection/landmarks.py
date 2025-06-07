"""
Facial Landmark Detection Module

Provides facial landmark detection functionality using various methods.
Currently implements a basic estimation based on face bounding boxes,
with plans to integrate MediaPipe/dlib when available.
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FaceLandmarks:
    """Container for facial landmarks."""
    points: np.ndarray  # Shape: (n_points, 2)
    confidence: float = 1.0
    
    @property
    def num_points(self) -> int:
        """Number of landmark points."""
        return self.points.shape[0]
    
    def get_point(self, index: int) -> Tuple[int, int]:
        """Get specific landmark point."""
        if 0 <= index < self.num_points:
            return tuple(self.points[index].astype(int))
        return None
    
    def get_region(self, region_name: str) -> np.ndarray:
        """Get landmarks for specific facial region."""
        regions = {
            'jaw': list(range(0, 17)),
            'right_eyebrow': list(range(17, 22)),
            'left_eyebrow': list(range(22, 27)),
            'nose': list(range(27, 36)),
            'right_eye': list(range(36, 42)),
            'left_eye': list(range(42, 48)),
            'mouth': list(range(48, 68))
        }
        
        if region_name in regions:
            indices = regions[region_name]
            return self.points[indices]
        return np.array([])


class LandmarkDetector:
    """
    Facial landmark detector supporting multiple backends.
    
    Currently implements basic estimation from bounding boxes.
    Will support MediaPipe and dlib when available.
    """
    
    def __init__(self, method: str = 'basic'):
        """
        Initialize landmark detector.
        
        Args:
            method: Detection method ('basic', 'mediapipe', 'dlib')
        """
        self.method = method
        self.initialized = False
        
        # Try to initialize selected method
        if method == 'mediapipe':
            self._init_mediapipe()
        elif method == 'dlib':
            self._init_dlib()
        else:
            self._init_basic()
    
    def _init_basic(self):
        """Initialize basic landmark estimation."""
        self.initialized = True
        logger.info("Initialized basic landmark detector")
    
    def _init_mediapipe(self):
        """Initialize MediaPipe face mesh."""
        try:
            import mediapipe as mp
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
            self.initialized = True
            logger.info("Initialized MediaPipe landmark detector")
        except ImportError:
            logger.warning("MediaPipe not available, falling back to basic method")
            self.method = 'basic'
            self._init_basic()
    
    def _init_dlib(self):
        """Initialize dlib landmark detector."""
        try:
            import dlib
            import os
            # Load the predictor
            # Look for model in several possible locations
            model_paths = [
                "models/shape_predictor_68_face_landmarks.dat",
                "shape_predictor_68_face_landmarks.dat",
                os.path.join(os.path.dirname(__file__), "..", "..", "models", "shape_predictor_68_face_landmarks.dat")
            ]
            
            predictor_path = None
            for path in model_paths:
                if os.path.exists(path):
                    predictor_path = path
                    break
            
            if predictor_path is None:
                raise RuntimeError("dlib model file not found")
            
            self.predictor = dlib.shape_predictor(predictor_path)
            self.initialized = True
            logger.info(f"Initialized dlib landmark detector with model: {predictor_path}")
        except (ImportError, RuntimeError) as e:
            logger.warning(f"dlib not available or model not found: {e}, falling back to basic method")
            self.method = 'basic'
            self._init_basic()
    
    def detect_landmarks(self, image: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Optional[FaceLandmarks]:
        """
        Detect facial landmarks in the given face region.
        
        Args:
            image: Input image
            face_bbox: Face bounding box (x, y, w, h)
            
        Returns:
            FaceLandmarks object or None if detection failed
        """
        if not self.initialized:
            return None
        
        if self.method == 'mediapipe':
            return self._detect_mediapipe(image, face_bbox)
        elif self.method == 'dlib':
            return self._detect_dlib(image, face_bbox)
        else:
            return self._detect_basic(image, face_bbox)
    
    def _detect_basic(self, image: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Optional[FaceLandmarks]:
        """
        Basic landmark estimation based on face bounding box.
        
        Estimates 68 landmark points based on typical facial proportions.
        """
        x, y, w, h = face_bbox
        
        # Create 68 landmark points based on facial proportions
        landmarks = []
        
        # Jaw line (17 points)
        for i in range(17):
            angle = np.pi - (i * np.pi / 16)  # From left to right
            px = x + w/2 + (w/2) * np.cos(angle)
            py = y + h - (h * 0.3) + (h * 0.4) * np.sin(angle)
            landmarks.append([px, py])
        
        # Right eyebrow (5 points)
        for i in range(5):
            px = x + w * (0.25 + i * 0.1)
            py = y + h * 0.25
            landmarks.append([px, py])
        
        # Left eyebrow (5 points)
        for i in range(5):
            px = x + w * (0.55 + i * 0.1)
            py = y + h * 0.25
            landmarks.append([px, py])
        
        # Nose bridge (4 points)
        for i in range(4):
            px = x + w * 0.5
            py = y + h * (0.3 + i * 0.08)
            landmarks.append([px, py])
        
        # Nose bottom (5 points)
        for i in range(5):
            px = x + w * (0.4 + i * 0.05)
            py = y + h * 0.6
            landmarks.append([px, py])
        
        # Right eye (6 points)
        eye_center_x = x + w * 0.3
        eye_center_y = y + h * 0.35
        for i in range(6):
            angle = i * 2 * np.pi / 6
            px = eye_center_x + w * 0.08 * np.cos(angle)
            py = eye_center_y + h * 0.04 * np.sin(angle)
            landmarks.append([px, py])
        
        # Left eye (6 points)
        eye_center_x = x + w * 0.7
        eye_center_y = y + h * 0.35
        for i in range(6):
            angle = i * 2 * np.pi / 6
            px = eye_center_x + w * 0.08 * np.cos(angle)
            py = eye_center_y + h * 0.04 * np.sin(angle)
            landmarks.append([px, py])
        
        # Outer mouth (12 points)
        mouth_center_x = x + w * 0.5
        mouth_center_y = y + h * 0.75
        for i in range(12):
            angle = i * 2 * np.pi / 12
            px = mouth_center_x + w * 0.15 * np.cos(angle)
            py = mouth_center_y + h * 0.08 * np.sin(angle)
            landmarks.append([px, py])
        
        # Inner mouth (8 points)
        for i in range(8):
            angle = i * 2 * np.pi / 8
            px = mouth_center_x + w * 0.08 * np.cos(angle)
            py = mouth_center_y + h * 0.04 * np.sin(angle)
            landmarks.append([px, py])
        
        landmarks_array = np.array(landmarks, dtype=np.float32)
        return FaceLandmarks(points=landmarks_array, confidence=0.5)
    
    def _detect_mediapipe(self, image: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Optional[FaceLandmarks]:
        """Detect landmarks using MediaPipe."""
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_image)
            
            if results.multi_face_landmarks:
                # Get first face landmarks
                face_landmarks = results.multi_face_landmarks[0]
                
                # Convert to 68-point format (subset of MediaPipe's 468 points)
                # This is a simplified mapping - full implementation would be more complex
                landmarks = []
                h, w = image.shape[:2]
                
                for i in range(min(68, len(face_landmarks.landmark))):
                    landmark = face_landmarks.landmark[i]
                    px = landmark.x * w
                    py = landmark.y * h
                    landmarks.append([px, py])
                
                landmarks_array = np.array(landmarks, dtype=np.float32)
                return FaceLandmarks(points=landmarks_array, confidence=0.9)
                
        except Exception as e:
            logger.error(f"MediaPipe detection failed: {e}")
        
        return None
    
    def _detect_dlib(self, image: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Optional[FaceLandmarks]:
        """Detect landmarks using dlib."""
        try:
            import dlib
            
            # Convert bbox to dlib rectangle
            x, y, w, h = face_bbox
            rect = dlib.rectangle(x, y, x + w, y + h)
            
            # Detect landmarks
            shape = self.predictor(image, rect)
            
            # Convert to numpy array
            landmarks = []
            for i in range(68):
                pt = shape.part(i)
                landmarks.append([pt.x, pt.y])
            
            landmarks_array = np.array(landmarks, dtype=np.float32)
            return FaceLandmarks(points=landmarks_array, confidence=0.95)
            
        except Exception as e:
            logger.error(f"dlib detection failed: {e}")
        
        return None
    
    def draw_landmarks(self, image: np.ndarray, landmarks: FaceLandmarks,
                      color: Tuple[int, int, int] = (0, 255, 0),
                      thickness: int = 1) -> np.ndarray:
        """
        Draw landmarks on image.
        
        Args:
            image: Input image
            landmarks: Detected landmarks
            color: Drawing color (BGR)
            thickness: Line thickness
            
        Returns:
            Image with drawn landmarks
        """
        result = image.copy()
        
        # Draw individual points
        for point in landmarks.points:
            cv2.circle(result, tuple(point.astype(int)), 2, color, -1)
        
        # Draw connections for specific regions
        connections = {
            'jaw': list(range(0, 17)),
            'right_eyebrow': list(range(17, 22)),
            'left_eyebrow': list(range(22, 27)),
            'nose_bridge': list(range(27, 31)),
            'nose_bottom': list(range(31, 36)),
            'right_eye': list(range(36, 42)) + [36],  # Close the loop
            'left_eye': list(range(42, 48)) + [42],   # Close the loop
            'outer_mouth': list(range(48, 60)) + [48], # Close the loop
            'inner_mouth': list(range(60, 68)) + [60]  # Close the loop
        }
        
        for region, indices in connections.items():
            for i in range(len(indices) - 1):
                pt1 = tuple(landmarks.points[indices[i]].astype(int))
                pt2 = tuple(landmarks.points[indices[i + 1]].astype(int))
                cv2.line(result, pt1, pt2, color, thickness)
        
        return result


def estimate_face_pose(landmarks: FaceLandmarks) -> Dict[str, float]:
    """
    Estimate face pose (yaw, pitch, roll) from landmarks.
    
    Args:
        landmarks: Facial landmarks
        
    Returns:
        Dictionary with pose angles
    """
    if landmarks.num_points < 68:
        return {'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0}
    
    # Use key points for pose estimation
    nose_tip = landmarks.points[33]
    chin = landmarks.points[8]
    left_eye = landmarks.points[45]
    right_eye = landmarks.points[36]
    
    # Estimate yaw (left-right rotation)
    eye_center = (left_eye + right_eye) / 2
    face_center_x = (landmarks.points[0][0] + landmarks.points[16][0]) / 2
    yaw = np.arctan2(nose_tip[0] - face_center_x, 100) * 180 / np.pi
    
    # Estimate pitch (up-down rotation)
    pitch = np.arctan2(nose_tip[1] - eye_center[1], 100) * 180 / np.pi
    
    # Estimate roll (tilt)
    eye_angle = np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
    roll = eye_angle * 180 / np.pi
    
    return {'yaw': yaw, 'pitch': pitch, 'roll': roll}