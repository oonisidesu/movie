"""
Face Quality Assessment Module - FaceQualityAssessor Class

This module provides comprehensive face quality assessment for determining
the suitability of detected faces for face swapping operations.

Quality metrics include:
- Image sharpness/blur detection
- Face pose estimation
- Lighting conditions
- Face size and resolution
- Occlusion detection
- Overall quality scoring

Author: Face Swapping Tool
License: Personal/Educational Use Only
"""

import cv2
import numpy as np
import logging
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum
import math

from .detector import FaceDetection

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None


class QualityLevel(Enum):
    """Face quality levels."""
    EXCELLENT = "excellent"
    GOOD = "good" 
    FAIR = "fair"
    POOR = "poor"


@dataclass
class QualityMetrics:
    """Container for face quality assessment results."""
    
    sharpness_score: float = 0.0
    lighting_score: float = 0.0
    pose_score: float = 0.0
    size_score: float = 0.0
    occlusion_score: float = 0.0
    overall_score: float = 0.0
    quality_level: QualityLevel = QualityLevel.POOR
    
    # Detailed metrics
    blur_variance: float = 0.0
    brightness_mean: float = 0.0
    contrast_std: float = 0.0
    face_area: int = 0
    pose_angles: Optional[Tuple[float, float, float]] = None  # pitch, yaw, roll
    
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary."""
        result = {
            'sharpness_score': self.sharpness_score,
            'lighting_score': self.lighting_score,
            'pose_score': self.pose_score,
            'size_score': self.size_score,
            'occlusion_score': self.occlusion_score,
            'overall_score': self.overall_score,
            'quality_level': self.quality_level.value,
            'blur_variance': self.blur_variance,
            'brightness_mean': self.brightness_mean,
            'contrast_std': self.contrast_std,
            'face_area': self.face_area,
        }
        
        if self.pose_angles:
            result['pose_angles'] = {
                'pitch': self.pose_angles[0],
                'yaw': self.pose_angles[1], 
                'roll': self.pose_angles[2]
            }
        
        return result


class FaceQualityAssessor:
    """
    Comprehensive face quality assessment for face swapping suitability.
    
    This class evaluates multiple aspects of face quality to determine
    how suitable a detected face is for face swapping operations.
    """
    
    def __init__(self,
                 min_face_size: int = 64,
                 max_face_size: int = 1024,
                 optimal_face_size: int = 256,
                 enable_pose_estimation: bool = True):
        """
        Initialize the quality assessor.
        
        Args:
            min_face_size: Minimum acceptable face size (pixels)
            max_face_size: Maximum face size before downsampling needed
            optimal_face_size: Optimal face size for processing
            enable_pose_estimation: Whether to enable pose estimation
        """
        self.min_face_size = min_face_size
        self.max_face_size = max_face_size
        self.optimal_face_size = optimal_face_size
        self.enable_pose_estimation = enable_pose_estimation
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize MediaPipe for pose estimation if available
        self.mp_face_mesh = None
        if enable_pose_estimation and MEDIAPIPE_AVAILABLE:
            self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
        
        # Quality thresholds
        self.thresholds = {
            'excellent': 0.8,
            'good': 0.6,
            'fair': 0.4,
            'poor': 0.0
        }
        
        # Weights for combining different quality metrics
        self.weights = {
            'sharpness': 0.25,
            'lighting': 0.20,
            'pose': 0.25,
            'size': 0.15,
            'occlusion': 0.15
        }
    
    def assess_quality(self, image: np.ndarray, 
                      detection: FaceDetection) -> QualityMetrics:
        """
        Assess overall quality of a detected face.
        
        Args:
            image: Source image containing the face
            detection: Face detection result
            
        Returns:
            QualityMetrics object with assessment results
        """
        if image is None or detection is None:
            return QualityMetrics()
        
        # Extract face region
        face_region = self._extract_face_region(image, detection)
        if face_region is None or face_region.size == 0:
            return QualityMetrics()
        
        # Calculate individual quality metrics
        metrics = QualityMetrics()
        
        try:
            # Sharpness assessment
            metrics.sharpness_score, metrics.blur_variance = self._assess_sharpness(face_region)
            
            # Lighting assessment  
            metrics.lighting_score, metrics.brightness_mean, metrics.contrast_std = self._assess_lighting(face_region)
            
            # Size assessment
            metrics.size_score, metrics.face_area = self._assess_size(detection)
            
            # Pose assessment
            if self.enable_pose_estimation:
                metrics.pose_score, metrics.pose_angles = self._assess_pose(face_region)
            else:
                metrics.pose_score = 1.0  # Assume good pose if not assessing
            
            # Occlusion assessment
            metrics.occlusion_score = self._assess_occlusion(face_region)
            
            # Calculate overall score
            metrics.overall_score = self._calculate_overall_score(metrics)
            
            # Determine quality level
            metrics.quality_level = self._determine_quality_level(metrics.overall_score)
            
        except Exception as e:
            self.logger.error(f"Quality assessment failed: {e}")
            
        return metrics
    
    def _extract_face_region(self, image: np.ndarray, 
                           detection: FaceDetection,
                           padding: float = 0.1) -> Optional[np.ndarray]:
        """Extract face region with padding."""
        h, w = image.shape[:2]
        
        # Calculate padding
        pad_w = int(detection.width * padding)
        pad_h = int(detection.height * padding)
        
        # Expand bounding box
        x1 = max(0, detection.x - pad_w)
        y1 = max(0, detection.y - pad_h)
        x2 = min(w, detection.x + detection.width + pad_w)
        y2 = min(h, detection.y + detection.height + pad_h)
        
        if x2 <= x1 or y2 <= y1:
            return None
            
        return image[y1:y2, x1:x2]
    
    def _assess_sharpness(self, face_region: np.ndarray) -> Tuple[float, float]:
        """
        Assess face sharpness using Laplacian variance.
        
        Args:
            face_region: Face image region
            
        Returns:
            (sharpness_score, blur_variance)
        """
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        
        # Calculate Laplacian variance (measure of focus)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_variance = laplacian.var()
        
        # Normalize score (higher variance = sharper image)
        # Typical range: 0-2000+, good images usually > 100
        max_variance = 1000.0
        sharpness_score = min(1.0, blur_variance / max_variance)
        
        return sharpness_score, blur_variance
    
    def _assess_lighting(self, face_region: np.ndarray) -> Tuple[float, float, float]:
        """
        Assess lighting conditions.
        
        Args:
            face_region: Face image region
            
        Returns:
            (lighting_score, brightness_mean, contrast_std)
        """
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        
        # Calculate brightness (mean intensity)
        brightness_mean = np.mean(gray)
        
        # Calculate contrast (standard deviation)
        contrast_std = np.std(gray)
        
        # Assess brightness (optimal range: 80-180)
        optimal_brightness = 128
        brightness_deviation = abs(brightness_mean - optimal_brightness)
        brightness_score = max(0.0, 1.0 - brightness_deviation / 128.0)
        
        # Assess contrast (higher is generally better, but not too high)
        # Good contrast typically > 30
        contrast_score = min(1.0, contrast_std / 50.0)
        
        # Detect overexposure and underexposure
        overexposed = np.sum(gray > 240) / gray.size
        underexposed = np.sum(gray < 15) / gray.size
        exposure_penalty = (overexposed + underexposed) * 2
        
        # Combine scores
        lighting_score = max(0.0, (brightness_score + contrast_score) / 2 - exposure_penalty)
        
        return lighting_score, brightness_mean, contrast_std
    
    def _assess_size(self, detection: FaceDetection) -> Tuple[float, int]:
        """
        Assess face size suitability.
        
        Args:
            detection: Face detection result
            
        Returns:
            (size_score, face_area)
        """
        face_area = detection.area
        face_size = min(detection.width, detection.height)
        
        # Penalty for faces that are too small
        if face_size < self.min_face_size:
            size_score = face_size / self.min_face_size
        # Penalty for faces that are too large
        elif face_size > self.max_face_size:
            size_score = max(0.5, 1.0 - (face_size - self.max_face_size) / self.max_face_size)
        # Optimal range
        else:
            # Score based on how close to optimal size
            distance_from_optimal = abs(face_size - self.optimal_face_size)
            max_distance = max(self.optimal_face_size - self.min_face_size,
                             self.max_face_size - self.optimal_face_size)
            size_score = max(0.7, 1.0 - distance_from_optimal / max_distance)
        
        return size_score, face_area
    
    def _assess_pose(self, face_region: np.ndarray) -> Tuple[float, Optional[Tuple[float, float, float]]]:
        """
        Assess face pose using MediaPipe if available.
        
        Args:
            face_region: Face image region
            
        Returns:
            (pose_score, pose_angles) where pose_angles is (pitch, yaw, roll)
        """
        if not self.enable_pose_estimation or not MEDIAPIPE_AVAILABLE or self.mp_face_mesh is None:
            return 1.0, None
        
        try:
            # Convert BGR to RGB for MediaPipe
            rgb_image = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
            results = self.mp_face_mesh.process(rgb_image)
            
            if not results.multi_face_landmarks:
                return 0.5, None  # No landmarks detected
            
            landmarks = results.multi_face_landmarks[0]
            
            # Calculate pose angles from landmarks
            pose_angles = self._calculate_pose_angles(landmarks, face_region.shape)
            
            if pose_angles is None:
                return 0.5, None
            
            pitch, yaw, roll = pose_angles
            
            # Score based on how frontal the face is
            # Ideal: pitch ~0, yaw ~0, roll ~0
            max_angle = 30.0  # degrees
            
            pitch_score = max(0.0, 1.0 - abs(pitch) / max_angle)
            yaw_score = max(0.0, 1.0 - abs(yaw) / max_angle)
            roll_score = max(0.0, 1.0 - abs(roll) / max_angle)
            
            # Yaw is most important for face swapping
            pose_score = (yaw_score * 0.5 + pitch_score * 0.3 + roll_score * 0.2)
            
            return pose_score, pose_angles
            
        except Exception as e:
            self.logger.warning(f"Pose assessment failed: {e}")
            return 0.5, None
    
    def _calculate_pose_angles(self, landmarks, image_shape) -> Optional[Tuple[float, float, float]]:
        """Calculate pose angles from facial landmarks."""
        try:
            h, w = image_shape[:2]
            
            # Key landmark points for pose estimation
            # These correspond to MediaPipe face mesh landmark indices
            nose_tip = landmarks.landmark[1]  # Nose tip
            nose_bridge = landmarks.landmark[168]  # Nose bridge
            left_eye = landmarks.landmark[33]  # Left eye corner
            right_eye = landmarks.landmark[263]  # Right eye corner
            mouth_left = landmarks.landmark[61]  # Mouth left
            mouth_right = landmarks.landmark[291]  # Mouth right
            
            # Convert normalized coordinates to pixel coordinates
            def to_pixel(landmark):
                return (int(landmark.x * w), int(landmark.y * h))
            
            nose_tip_px = to_pixel(nose_tip)
            nose_bridge_px = to_pixel(nose_bridge)
            left_eye_px = to_pixel(left_eye)
            right_eye_px = to_pixel(right_eye)
            mouth_left_px = to_pixel(mouth_left)
            mouth_right_px = to_pixel(mouth_right)
            
            # Calculate roll (rotation around z-axis)
            eye_center_x = (left_eye_px[0] + right_eye_px[0]) / 2
            eye_center_y = (left_eye_px[1] + right_eye_px[1]) / 2
            eye_angle = math.atan2(right_eye_px[1] - left_eye_px[1], 
                                 right_eye_px[0] - left_eye_px[0])
            roll = math.degrees(eye_angle)
            
            # Calculate yaw (rotation around y-axis)
            # Based on eye-to-eye distance vs eye-to-nose distance ratio
            eye_distance = abs(right_eye_px[0] - left_eye_px[0])
            nose_to_left_eye = abs(nose_tip_px[0] - left_eye_px[0])
            nose_to_right_eye = abs(nose_tip_px[0] - right_eye_px[0])
            
            if eye_distance > 0:
                if nose_to_left_eye > nose_to_right_eye:
                    yaw = -30 * (nose_to_left_eye - nose_to_right_eye) / eye_distance
                else:
                    yaw = 30 * (nose_to_right_eye - nose_to_left_eye) / eye_distance
            else:
                yaw = 0
            
            # Calculate pitch (rotation around x-axis)  
            # Based on nose bridge to nose tip vertical distance
            nose_vertical_distance = abs(nose_tip_px[1] - nose_bridge_px[1])
            mouth_center_y = (mouth_left_px[1] + mouth_right_px[1]) / 2
            nose_to_mouth_distance = abs(nose_tip_px[1] - mouth_center_y)
            
            if nose_to_mouth_distance > 0:
                pitch_ratio = nose_vertical_distance / nose_to_mouth_distance
                pitch = (pitch_ratio - 0.3) * 60  # Empirical scaling
            else:
                pitch = 0
            
            # Clamp angles to reasonable ranges
            pitch = max(-60, min(60, pitch))
            yaw = max(-60, min(60, yaw))
            roll = max(-45, min(45, roll))
            
            return (pitch, yaw, roll)
            
        except Exception as e:
            self.logger.warning(f"Pose angle calculation failed: {e}")
            return None
    
    def _assess_occlusion(self, face_region: np.ndarray) -> float:
        """
        Assess face occlusion (simple version).
        
        Args:
            face_region: Face image region
            
        Returns:
            Occlusion score (higher = less occluded)
        """
        # Simple occlusion detection based on edge density
        # More edges typically indicate less occlusion
        
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Detect edges using Canny
        edges = cv2.Canny(blurred, 50, 150)
        
        # Calculate edge density
        edge_density = np.sum(edges > 0) / edges.size
        
        # Normalize to 0-1 range (typical good faces have 0.1-0.3 edge density)
        occlusion_score = min(1.0, edge_density * 5)
        
        return occlusion_score
    
    def _calculate_overall_score(self, metrics: QualityMetrics) -> float:
        """Calculate weighted overall quality score."""
        return (
            metrics.sharpness_score * self.weights['sharpness'] +
            metrics.lighting_score * self.weights['lighting'] +
            metrics.pose_score * self.weights['pose'] +
            metrics.size_score * self.weights['size'] +
            metrics.occlusion_score * self.weights['occlusion']
        )
    
    def _determine_quality_level(self, overall_score: float) -> QualityLevel:
        """Determine quality level from overall score."""
        if overall_score >= self.thresholds['excellent']:
            return QualityLevel.EXCELLENT
        elif overall_score >= self.thresholds['good']:
            return QualityLevel.GOOD
        elif overall_score >= self.thresholds['fair']:
            return QualityLevel.FAIR
        else:
            return QualityLevel.POOR
    
    def filter_by_quality(self, 
                         detections_with_metrics: List[Tuple[FaceDetection, QualityMetrics]],
                         min_quality: QualityLevel = QualityLevel.FAIR) -> List[Tuple[FaceDetection, QualityMetrics]]:
        """
        Filter detections by minimum quality level.
        
        Args:
            detections_with_metrics: List of (detection, metrics) tuples
            min_quality: Minimum quality level to keep
            
        Returns:
            Filtered list of detections meeting quality threshold
        """
        quality_order = {
            QualityLevel.POOR: 0,
            QualityLevel.FAIR: 1, 
            QualityLevel.GOOD: 2,
            QualityLevel.EXCELLENT: 3
        }
        
        min_level = quality_order[min_quality]
        
        return [
            (detection, metrics) for detection, metrics in detections_with_metrics
            if quality_order[metrics.quality_level] >= min_level
        ]
    
    def get_best_quality_face(self, 
                            detections_with_metrics: List[Tuple[FaceDetection, QualityMetrics]]) -> Optional[Tuple[FaceDetection, QualityMetrics]]:
        """
        Get the highest quality face from a list.
        
        Args:
            detections_with_metrics: List of (detection, metrics) tuples
            
        Returns:
            Best quality face or None if list is empty
        """
        if not detections_with_metrics:
            return None
        
        return max(detections_with_metrics, key=lambda x: x[1].overall_score)
    
    def batch_assess_quality(self,
                           image: np.ndarray,
                           detections: List[FaceDetection]) -> List[Tuple[FaceDetection, QualityMetrics]]:
        """
        Assess quality for multiple face detections.
        
        Args:
            image: Source image
            detections: List of face detections
            
        Returns:
            List of (detection, metrics) tuples
        """
        results = []
        
        for detection in detections:
            metrics = self.assess_quality(image, detection)
            results.append((detection, metrics))
        
        return results
    
    def set_quality_weights(self, **weights) -> None:
        """
        Update quality assessment weights.
        
        Args:
            **weights: New weights (sharpness, lighting, pose, size, occlusion)
        """
        for metric, weight in weights.items():
            if metric in self.weights:
                self.weights[metric] = weight
        
        # Normalize weights to sum to 1.0
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            for metric in self.weights:
                self.weights[metric] /= total_weight
    
    def get_quality_report(self, metrics: QualityMetrics) -> str:
        """
        Generate human-readable quality report.
        
        Args:
            metrics: Quality metrics to report
            
        Returns:
            Formatted quality report string
        """
        report = f"Face Quality Report\n"
        report += f"==================\n"
        report += f"Overall Score: {metrics.overall_score:.3f}\n"
        report += f"Quality Level: {metrics.quality_level.value.upper()}\n\n"
        
        report += f"Individual Scores:\n"
        report += f"- Sharpness: {metrics.sharpness_score:.3f} (blur variance: {metrics.blur_variance:.1f})\n"
        report += f"- Lighting: {metrics.lighting_score:.3f} (brightness: {metrics.brightness_mean:.1f})\n"
        report += f"- Pose: {metrics.pose_score:.3f}"
        
        if metrics.pose_angles:
            pitch, yaw, roll = metrics.pose_angles
            report += f" (pitch: {pitch:.1f}°, yaw: {yaw:.1f}°, roll: {roll:.1f}°)"
        
        report += f"\n- Size: {metrics.size_score:.3f} (area: {metrics.face_area} pixels)\n"
        report += f"- Occlusion: {metrics.occlusion_score:.3f}\n"
        
        return report