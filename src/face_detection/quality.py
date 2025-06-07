import cv2
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class FaceQualityAssessor:
    def __init__(self,
                 blur_threshold: float = 100.0,
                 brightness_range: Tuple[float, float] = (50, 200),
                 min_face_size: int = 80):
        self.blur_threshold = blur_threshold
        self.brightness_range = brightness_range
        self.min_face_size = min_face_size
        
    def assess_quality(self, face_image: np.ndarray, 
                      face_info: Optional[Dict] = None) -> Dict[str, float]:
        """
        Assess the quality of a face image.
        
        Args:
            face_image: Cropped face image
            face_info: Optional face detection info (bbox, landmarks, etc.)
            
        Returns:
            Dictionary with quality metrics
        """
        results = {
            'overall_score': 0.0,
            'blur_score': 0.0,
            'brightness_score': 0.0,
            'size_score': 0.0,
            'pose_score': 0.0,
            'occlusion_score': 0.0,
        }
        
        # Check image validity
        if face_image is None or face_image.size == 0:
            return results
        
        # Assess blur
        blur_score = self._assess_blur(face_image)
        results['blur_score'] = blur_score
        
        # Assess brightness
        brightness_score = self._assess_brightness(face_image)
        results['brightness_score'] = brightness_score
        
        # Assess size
        size_score = self._assess_size(face_image)
        results['size_score'] = size_score
        
        # Assess pose (if landmarks available)
        if face_info and face_info.get('landmarks'):
            pose_score = self._assess_pose(face_info['landmarks'], face_image.shape)
            results['pose_score'] = pose_score
        else:
            results['pose_score'] = 0.7  # Default neutral score
        
        # Assess occlusion
        occlusion_score = self._assess_occlusion(face_image)
        results['occlusion_score'] = occlusion_score
        
        # Calculate overall score
        weights = {
            'blur_score': 0.3,
            'brightness_score': 0.2,
            'size_score': 0.2,
            'pose_score': 0.2,
            'occlusion_score': 0.1
        }
        
        overall_score = sum(results[key] * weight 
                          for key, weight in weights.items())
        results['overall_score'] = overall_score
        
        return results
    
    def _assess_blur(self, image: np.ndarray) -> float:
        """
        Assess image blur using Laplacian variance.
        
        Returns:
            Score between 0 (very blurry) and 1 (sharp)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Normalize to 0-1 range
        score = min(laplacian_var / self.blur_threshold, 1.0)
        return score
    
    def _assess_brightness(self, image: np.ndarray) -> float:
        """
        Assess image brightness and contrast.
        
        Returns:
            Score between 0 (poor lighting) and 1 (good lighting)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        mean_brightness = np.mean(gray)
        
        min_bright, max_bright = self.brightness_range
        
        if mean_brightness < min_bright:
            score = mean_brightness / min_bright
        elif mean_brightness > max_bright:
            score = 1.0 - (mean_brightness - max_bright) / (255 - max_bright)
        else:
            score = 1.0
        
        # Also check contrast
        std_brightness = np.std(gray)
        contrast_score = min(std_brightness / 50.0, 1.0)
        
        return (score + contrast_score) / 2.0
    
    def _assess_size(self, image: np.ndarray) -> float:
        """
        Assess face size.
        
        Returns:
            Score between 0 (too small) and 1 (good size)
        """
        height, width = image.shape[:2]
        size = min(height, width)
        
        if size < self.min_face_size:
            score = size / self.min_face_size
        else:
            score = 1.0
        
        return score
    
    def _assess_pose(self, landmarks: list, image_shape: tuple) -> float:
        """
        Assess face pose based on landmarks.
        
        Returns:
            Score between 0 (extreme pose) and 1 (frontal)
        """
        if len(landmarks) < 5:
            return 0.7  # Default score if not enough landmarks
        
        # Simple pose estimation based on facial symmetry
        # This is a simplified version - more sophisticated methods exist
        height, width = image_shape[:2]
        
        # Assuming landmarks order: left_eye, right_eye, nose, left_mouth, right_mouth
        if len(landmarks) >= 5:
            left_eye = landmarks[0]
            right_eye = landmarks[1]
            nose = landmarks[2]
            
            # Check horizontal symmetry
            eye_center_x = (left_eye[0] + right_eye[0]) / 2
            nose_offset = abs(nose[0] - eye_center_x) / width
            
            # Check if face is tilted
            eye_angle = np.arctan2(right_eye[1] - left_eye[1], 
                                  right_eye[0] - left_eye[0])
            tilt_score = 1.0 - min(abs(eye_angle) / (np.pi / 6), 1.0)
            
            symmetry_score = 1.0 - min(nose_offset * 4, 1.0)
            
            return (symmetry_score + tilt_score) / 2.0
        
        return 0.7
    
    def _assess_occlusion(self, image: np.ndarray) -> float:
        """
        Assess face occlusion using edge detection.
        
        Returns:
            Score between 0 (heavily occluded) and 1 (no occlusion)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Use edge detection to find face features
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Expected edge density range for unoccluded faces
        expected_density = 0.05
        if edge_density < expected_density * 0.5:
            score = edge_density / (expected_density * 0.5)
        elif edge_density > expected_density * 2:
            score = 1.0 - (edge_density - expected_density * 2) / expected_density
        else:
            score = 1.0
        
        return max(0, min(1, score))
    
    def filter_faces(self, faces: list, min_quality: float = 0.6) -> list:
        """
        Filter faces based on quality threshold.
        
        Args:
            faces: List of face dictionaries with 'image' key
            min_quality: Minimum quality score
            
        Returns:
            Filtered list of faces
        """
        filtered_faces = []
        
        for face in faces:
            if 'image' in face:
                quality = self.assess_quality(face['image'], face)
                face['quality'] = quality
                
                if quality['overall_score'] >= min_quality:
                    filtered_faces.append(face)
        
        return filtered_faces