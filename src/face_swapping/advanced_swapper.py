"""
Advanced Face Swapper with Landmark Detection and Delaunay Triangulation

Implements high-quality face swapping using facial landmarks and triangular warping.
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple, Dict, Any

from ..face_detection import FaceDetection
from ..face_detection.landmarks import LandmarkDetector, FaceLandmarks, estimate_face_pose
from .triangulation import FaceTriangulation, TriangularWarper, create_face_mesh
from .utils import apply_color_correction

logger = logging.getLogger(__name__)


class AdvancedFaceSwapper:
    """
    Advanced face swapper using landmarks and Delaunay triangulation.
    
    This implementation provides high-quality face swapping with:
    - 68-point facial landmark detection
    - Delaunay triangulation for precise warping
    - 3D pose estimation and correction
    - Advanced blending techniques
    """
    
    def __init__(self, landmark_method: str = 'basic',
                 enable_pose_correction: bool = True,
                 blend_mode: str = 'seamless'):
        """
        Initialize advanced face swapper.
        
        Args:
            landmark_method: Landmark detection method ('basic', 'mediapipe', 'dlib')
            enable_pose_correction: Whether to apply 3D pose correction
            blend_mode: Blending mode ('seamless', 'feather', 'poisson')
        """
        self.landmark_detector = LandmarkDetector(method=landmark_method)
        self.enable_pose_correction = enable_pose_correction
        self.blend_mode = blend_mode
        
        # Statistics
        self.swap_count = 0
        self.success_count = 0
    
    def swap_faces(self, source_image: np.ndarray, target_image: np.ndarray,
                   source_face: FaceDetection, target_face: FaceDetection) -> Optional[np.ndarray]:
        """
        Perform advanced face swapping with landmarks and triangulation.
        
        Args:
            source_image: Source image
            target_image: Target image
            source_face: Source face detection
            target_face: Target face detection
            
        Returns:
            Swapped image or None if failed
        """
        self.swap_count += 1
        
        try:
            # Detect landmarks
            source_landmarks = self.landmark_detector.detect_landmarks(
                source_image, source_face.bbox
            )
            target_landmarks = self.landmark_detector.detect_landmarks(
                target_image, target_face.bbox
            )
            
            if source_landmarks is None or target_landmarks is None:
                logger.error("Failed to detect landmarks")
                return None
            
            # Estimate face poses
            source_pose = estimate_face_pose(source_landmarks)
            target_pose = estimate_face_pose(target_landmarks)
            
            logger.info(f"Source pose: yaw={source_pose['yaw']:.1f}, "
                       f"pitch={source_pose['pitch']:.1f}, roll={source_pose['roll']:.1f}")
            logger.info(f"Target pose: yaw={target_pose['yaw']:.1f}, "
                       f"pitch={target_pose['pitch']:.1f}, roll={target_pose['roll']:.1f}")
            
            # Apply pose correction if enabled
            if self.enable_pose_correction:
                source_image_corrected, source_landmarks_corrected = self._correct_pose(
                    source_image, source_landmarks, source_pose, target_pose
                )
            else:
                source_image_corrected = source_image
                source_landmarks_corrected = source_landmarks
            
            # Use original landmarks for stable triangulation
            # Create enhanced landmarks by adding boundary points
            source_landmarks_enhanced = self._create_stable_landmarks(
                source_landmarks_corrected, source_image.shape
            )
            target_landmarks_enhanced = self._create_stable_landmarks(
                target_landmarks, target_image.shape
            )
            
            # Perform triangular warping
            warped_face = TriangularWarper.warp_face(
                source_image_corrected, target_image,
                source_landmarks_enhanced, target_landmarks_enhanced
            )
            
            # Apply color correction
            warped_face_corrected = self._apply_color_matching(
                warped_face, target_image, target_landmarks
            )
            
            # Blend the result
            result = self._blend_faces(
                warped_face_corrected, target_image, target_landmarks
            )
            
            self.success_count += 1
            logger.info("Advanced face swap completed successfully")
            
            return result
            
        except Exception as e:
            logger.error(f"Advanced face swapping failed: {e}")
            return None
    
    def _correct_pose(self, image: np.ndarray, landmarks: FaceLandmarks,
                     source_pose: Dict[str, float], target_pose: Dict[str, float]) -> Tuple[np.ndarray, FaceLandmarks]:
        """
        Correct face pose to match target.
        
        Args:
            image: Source image
            landmarks: Source landmarks
            source_pose: Source face pose
            target_pose: Target face pose
            
        Returns:
            Corrected image and landmarks
        """
        # Calculate pose difference
        yaw_diff = target_pose['yaw'] - source_pose['yaw']
        pitch_diff = target_pose['pitch'] - source_pose['pitch']
        roll_diff = target_pose['roll'] - source_pose['roll']
        
        # Apply rotation to correct roll
        if abs(roll_diff) > 5:  # Only correct significant differences
            center = np.mean(landmarks.points, axis=0)
            center_tuple = (int(center[0]), int(center[1]))
            rotation_matrix = cv2.getRotationMatrix2D(
                center_tuple, roll_diff, 1.0
            )
            
            # Rotate image
            corrected_image = cv2.warpAffine(
                image, rotation_matrix, (image.shape[1], image.shape[0])
            )
            
            # Rotate landmarks
            ones = np.ones((landmarks.points.shape[0], 1))
            landmarks_homogeneous = np.hstack([landmarks.points, ones])
            rotated_landmarks = rotation_matrix.dot(landmarks_homogeneous.T).T
            
            corrected_landmarks = FaceLandmarks(
                points=rotated_landmarks,
                confidence=landmarks.confidence
            )
        else:
            corrected_image = image
            corrected_landmarks = landmarks
        
        # TODO: Implement yaw and pitch correction using 3D face model
        
        return corrected_image, corrected_landmarks
    
    def _apply_color_matching(self, source_face: np.ndarray, target_image: np.ndarray,
                             target_landmarks: FaceLandmarks) -> np.ndarray:
        """
        Apply sophisticated color matching.
        
        Args:
            source_face: Source face region
            target_image: Target image
            target_landmarks: Target facial landmarks
            
        Returns:
            Color-corrected face
        """
        # Create mask from landmarks
        mask = self._create_face_mask(target_landmarks, target_image.shape)
        
        # Apply color correction
        corrected = apply_color_correction(source_face, target_image, mask)
        
        # Additional skin tone matching in LAB space
        source_lab = cv2.cvtColor(corrected, cv2.COLOR_BGR2LAB)
        target_lab = cv2.cvtColor(target_image, cv2.COLOR_BGR2LAB)
        
        # Match L channel (luminance) statistics
        source_l = source_lab[:, :, 0].astype(np.float32)
        target_l = target_lab[:, :, 0].astype(np.float32)
        
        # Calculate statistics in face region
        face_pixels = mask > 127
        if np.any(face_pixels):
            source_mean = np.mean(source_l[face_pixels])
            source_std = np.std(source_l[face_pixels])
            target_mean = np.mean(target_l[face_pixels])
            target_std = np.std(target_l[face_pixels])
            
            if source_std > 0:
                # Normalize and scale
                source_l = (source_l - source_mean) * (target_std / source_std) + target_mean
                source_l = np.clip(source_l, 0, 255)
                source_lab[:, :, 0] = source_l.astype(np.uint8)
        
        # Convert back to BGR
        result = cv2.cvtColor(source_lab, cv2.COLOR_LAB2BGR)
        
        return result
    
    def _create_face_mask(self, landmarks: FaceLandmarks, image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Create detailed face mask from landmarks.
        
        Args:
            landmarks: Facial landmarks
            image_shape: Image dimensions
            
        Returns:
            Face mask
        """
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        
        # Get face contour points (jaw + forehead estimation)
        jaw_points = landmarks.get_region('jaw')
        
        # Estimate forehead points
        left_eyebrow = landmarks.get_region('left_eyebrow')
        right_eyebrow = landmarks.get_region('right_eyebrow')
        
        # Create forehead arc
        forehead_points = []
        eyebrow_top = np.min(np.vstack([left_eyebrow, right_eyebrow])[:, 1])
        forehead_height = int((landmarks.get_point(8)[1] - eyebrow_top) * 0.4)
        
        # Create arc points
        for i in range(5):
            t = i / 4.0
            x = int(jaw_points[-1][0] * (1 - t) + jaw_points[0][0] * t)
            y = eyebrow_top - forehead_height
            forehead_points.append([x, y])
        
        # Combine jaw and forehead
        face_contour = np.vstack([
            jaw_points,
            forehead_points[::-1]
        ]).astype(np.int32)
        
        # Fill face region
        cv2.fillPoly(mask, [face_contour], 255)
        
        # Apply Gaussian blur for smooth edges
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        
        return mask
    
    def _blend_faces(self, warped_face: np.ndarray, target_image: np.ndarray,
                    target_landmarks: FaceLandmarks) -> np.ndarray:
        """
        Blend warped face with target using advanced techniques.
        
        Args:
            warped_face: Warped source face
            target_image: Target image
            target_landmarks: Target landmarks
            
        Returns:
            Blended result
        """
        if self.blend_mode == 'seamless':
            return self._seamless_blend(warped_face, target_image, target_landmarks)
        elif self.blend_mode == 'poisson':
            return self._poisson_blend(warped_face, target_image, target_landmarks)
        else:
            return self._feather_blend(warped_face, target_image, target_landmarks)
    
    def _seamless_blend(self, warped_face: np.ndarray, target_image: np.ndarray,
                       target_landmarks: FaceLandmarks) -> np.ndarray:
        """Seamless blending using cv2.seamlessClone."""
        # Create mask
        mask = self._create_face_mask(target_landmarks, target_image.shape)
        
        # Find center point
        center = np.mean(target_landmarks.points[:17], axis=0).astype(int)  # Jaw center
        center = (int(center[0]), int(center[1]))
        
        try:
            # Apply seamless cloning
            result = cv2.seamlessClone(
                warped_face, target_image, mask, center, cv2.NORMAL_CLONE
            )
            return result
        except:
            # Fallback to feather blend
            return self._feather_blend(warped_face, target_image, target_landmarks)
    
    def _poisson_blend(self, warped_face: np.ndarray, target_image: np.ndarray,
                      target_landmarks: FaceLandmarks) -> np.ndarray:
        """Poisson blending (similar to seamless but with mixed mode)."""
        mask = self._create_face_mask(target_landmarks, target_image.shape)
        center = np.mean(target_landmarks.points[:17], axis=0).astype(int)
        center = (int(center[0]), int(center[1]))
        
        try:
            result = cv2.seamlessClone(
                warped_face, target_image, mask, center, cv2.MIXED_CLONE
            )
            return result
        except:
            return self._feather_blend(warped_face, target_image, target_landmarks)
    
    def _feather_blend(self, warped_face: np.ndarray, target_image: np.ndarray,
                      target_landmarks: FaceLandmarks) -> np.ndarray:
        """Feathered alpha blending."""
        # Create feathered mask
        mask = self._create_face_mask(target_landmarks, target_image.shape)
        mask_3d = np.stack([mask] * 3, axis=2) / 255.0
        
        # Blend images
        result = warped_face * mask_3d + target_image * (1 - mask_3d)
        
        return result.astype(np.uint8)
    
    def _create_stable_landmarks(self, landmarks: FaceLandmarks, image_shape: Tuple[int, int]) -> FaceLandmarks:
        """
        Create stable landmarks by adding consistent boundary points.
        
        Args:
            landmarks: Original facial landmarks
            image_shape: Image dimensions
            
        Returns:
            Enhanced landmarks with boundary points
        """
        h, w = image_shape[:2]
        
        # Start with original 68 landmark points
        points = landmarks.points.copy()
        
        # Add consistent boundary points (8 points around image border)
        boundary_points = np.array([
            [0, 0],           # Top-left
            [w//2, 0],        # Top-center  
            [w-1, 0],         # Top-right
            [w-1, h//2],      # Right-center
            [w-1, h-1],       # Bottom-right
            [w//2, h-1],      # Bottom-center
            [0, h-1],         # Bottom-left
            [0, h//2]         # Left-center
        ], dtype=np.float32)
        
        # Combine landmarks with boundary points
        enhanced_points = np.vstack([points, boundary_points])
        
        return FaceLandmarks(points=enhanced_points, confidence=landmarks.confidence)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get swapping statistics."""
        success_rate = self.success_count / self.swap_count if self.swap_count > 0 else 0
        
        return {
            'total_swaps': self.swap_count,
            'successful_swaps': self.success_count,
            'success_rate': success_rate,
            'landmark_method': self.landmark_detector.method,
            'pose_correction_enabled': self.enable_pose_correction,
            'blend_mode': self.blend_mode
        }