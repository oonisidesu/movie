"""
Face Alignment Module

Provides face alignment functionality for preprocessing faces before swapping.
Includes landmark-based alignment and geometric transformations.
"""

import cv2
import numpy as np
import logging
from enum import Enum
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

from ..face_detection import FaceDetection

logger = logging.getLogger(__name__)


class AlignmentMethod(Enum):
    """Available face alignment methods."""
    LANDMARKS = "landmarks"
    SIMILARITY_TRANSFORM = "similarity"
    AFFINE_TRANSFORM = "affine"
    PROCRUSTES = "procrustes"


@dataclass
class AlignmentResult:
    """Result of face alignment operation."""
    success: bool
    aligned_image: Optional[np.ndarray] = None
    transform_matrix: Optional[np.ndarray] = None
    aligned_landmarks: Optional[np.ndarray] = None
    confidence: float = 0.0
    error_message: Optional[str] = None


class FaceAligner:
    """
    Face alignment class for preprocessing faces before swapping.
    
    Provides various alignment methods to normalize face pose,
    scale, and rotation for better swapping results.
    """
    
    # Standard face template (normalized coordinates 0-1)
    TEMPLATE_LANDMARKS = np.array([
        # Right eye center
        [0.3, 0.4],
        # Left eye center  
        [0.7, 0.4],
        # Nose tip
        [0.5, 0.6],
        # Right mouth corner
        [0.4, 0.8],
        # Left mouth corner
        [0.6, 0.8]
    ], dtype=np.float32)
    
    def __init__(self, method: AlignmentMethod = AlignmentMethod.LANDMARKS,
                 target_size: Tuple[int, int] = (256, 256)):
        """
        Initialize face aligner.
        
        Args:
            method: Alignment method to use
            target_size: Target size for aligned face
        """
        self.method = method
        self.target_size = target_size
        
        # Scale template landmarks to target size
        self.scaled_template = self.TEMPLATE_LANDMARKS * np.array([target_size[0], target_size[1]])
    
    def align_face(self, image: np.ndarray, face_detection: FaceDetection) -> AlignmentResult:
        """
        Align face based on detection landmarks.
        
        Args:
            image: Input image containing the face
            face_detection: Face detection with landmarks
            
        Returns:
            AlignmentResult with aligned face and transformation info
        """
        try:
            # Extract landmarks
            landmarks = self._extract_key_landmarks(face_detection)
            if landmarks is None:
                return AlignmentResult(
                    success=False,
                    error_message="Failed to extract landmarks"
                )
            
            # Perform alignment based on method
            if self.method == AlignmentMethod.LANDMARKS:
                return self._align_by_landmarks(image, landmarks)
            elif self.method == AlignmentMethod.SIMILARITY_TRANSFORM:
                return self._align_by_similarity(image, landmarks)
            elif self.method == AlignmentMethod.AFFINE_TRANSFORM:
                return self._align_by_affine(image, landmarks)
            elif self.method == AlignmentMethod.PROCRUSTES:
                return self._align_by_procrustes(image, landmarks)
            else:
                return AlignmentResult(
                    success=False,
                    error_message=f"Unsupported alignment method: {self.method}"
                )
                
        except Exception as e:
            logger.error(f"Face alignment failed: {e}")
            return AlignmentResult(
                success=False,
                error_message=str(e)
            )
    
    def _extract_key_landmarks(self, face_detection: FaceDetection) -> Optional[np.ndarray]:
        """
        Extract key landmarks for alignment.
        
        Args:
            face_detection: Face detection with landmarks
            
        Returns:
            Key landmarks array or None if extraction failed
        """
        if 'landmarks' not in face_detection or face_detection['landmarks'] is None:
            logger.warning("No landmarks available for alignment")
            return None
        
        landmarks = face_detection['landmarks']
        
        # Convert to numpy array if needed
        if isinstance(landmarks, list):
            landmarks = np.array(landmarks, dtype=np.float32)
        
        # Extract key points (68-point landmarks)
        if landmarks.shape[0] >= 68:
            # Extract 5 key points for alignment
            key_points = np.array([
                # Right eye center (average of eye landmarks)
                np.mean(landmarks[36:42], axis=0),
                # Left eye center
                np.mean(landmarks[42:48], axis=0),
                # Nose tip
                landmarks[33],
                # Right mouth corner
                landmarks[48],
                # Left mouth corner
                landmarks[54]
            ], dtype=np.float32)
            
            return key_points
        
        # Fallback: use available landmarks
        logger.warning(f"Insufficient landmarks for optimal alignment: {landmarks.shape[0]}")
        return landmarks
    
    def _align_by_landmarks(self, image: np.ndarray, landmarks: np.ndarray) -> AlignmentResult:
        """
        Align face using landmark-based transformation.
        
        Args:
            image: Input image
            landmarks: Facial landmarks
            
        Returns:
            AlignmentResult with aligned face
        """
        try:
            # Use similarity transform for basic alignment
            if landmarks.shape[0] >= 2:
                # Calculate angle from eye landmarks
                if landmarks.shape[0] >= 5:
                    # Use eye points
                    right_eye = landmarks[0]
                    left_eye = landmarks[1]
                else:
                    # Use first two landmarks
                    right_eye = landmarks[0]
                    left_eye = landmarks[1]
                
                # Calculate rotation angle
                delta = left_eye - right_eye
                angle = np.degrees(np.arctan2(delta[1], delta[0]))
                
                # Calculate scale based on eye distance
                eye_distance = np.linalg.norm(delta)
                template_eye_distance = np.linalg.norm(
                    self.scaled_template[1] - self.scaled_template[0]
                )
                scale = template_eye_distance / eye_distance if eye_distance > 0 else 1.0
                
                # Calculate center point
                center = np.mean(landmarks[:2], axis=0)
                
                # Create transformation matrix
                M = cv2.getRotationMatrix2D((center[0], center[1]), angle, scale)
                
                # Adjust translation to center face
                template_center = np.mean(self.scaled_template[:2], axis=0)
                M[0, 2] += template_center[0] - center[0]
                M[1, 2] += template_center[1] - center[1]
                
                # Apply transformation
                aligned_image = cv2.warpAffine(
                    image, M, self.target_size,
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT
                )
                
                # Transform landmarks
                aligned_landmarks = self._transform_landmarks(landmarks, M)
                
                return AlignmentResult(
                    success=True,
                    aligned_image=aligned_image,
                    transform_matrix=M,
                    aligned_landmarks=aligned_landmarks,
                    confidence=0.8
                )
            
            else:
                return AlignmentResult(
                    success=False,
                    error_message="Insufficient landmarks for alignment"
                )
                
        except Exception as e:
            logger.error(f"Landmark alignment failed: {e}")
            return AlignmentResult(
                success=False,
                error_message=str(e)
            )
    
    def _align_by_similarity(self, image: np.ndarray, landmarks: np.ndarray) -> AlignmentResult:
        """
        Align face using similarity transformation.
        
        Args:
            image: Input image
            landmarks: Facial landmarks
            
        Returns:
            AlignmentResult with aligned face
        """
        try:
            if landmarks.shape[0] < 2:
                return AlignmentResult(
                    success=False,
                    error_message="Need at least 2 landmarks for similarity transform"
                )
            
            # Use first N landmarks (up to template size)
            n_points = min(landmarks.shape[0], self.scaled_template.shape[0])
            src_points = landmarks[:n_points]
            dst_points = self.scaled_template[:n_points]
            
            # Estimate similarity transform
            M = cv2.estimateAffinePartial2D(src_points, dst_points)[0]
            
            if M is None:
                return AlignmentResult(
                    success=False,
                    error_message="Failed to estimate similarity transform"
                )
            
            # Apply transformation
            aligned_image = cv2.warpAffine(
                image, M, self.target_size,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT
            )
            
            # Transform landmarks
            aligned_landmarks = self._transform_landmarks(landmarks, M)
            
            return AlignmentResult(
                success=True,
                aligned_image=aligned_image,
                transform_matrix=M,
                aligned_landmarks=aligned_landmarks,
                confidence=0.9
            )
            
        except Exception as e:
            logger.error(f"Similarity alignment failed: {e}")
            return AlignmentResult(
                success=False,
                error_message=str(e)
            )
    
    def _align_by_affine(self, image: np.ndarray, landmarks: np.ndarray) -> AlignmentResult:
        """
        Align face using affine transformation.
        
        Args:
            image: Input image
            landmarks: Facial landmarks
            
        Returns:
            AlignmentResult with aligned face
        """
        try:
            if landmarks.shape[0] < 3:
                return AlignmentResult(
                    success=False,
                    error_message="Need at least 3 landmarks for affine transform"
                )
            
            # Use first 3 landmarks for affine transform
            src_points = landmarks[:3].astype(np.float32)
            dst_points = self.scaled_template[:3].astype(np.float32)
            
            # Calculate affine transform
            M = cv2.getAffineTransform(src_points, dst_points)
            
            # Apply transformation
            aligned_image = cv2.warpAffine(
                image, M, self.target_size,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT
            )
            
            # Transform landmarks
            aligned_landmarks = self._transform_landmarks(landmarks, M)
            
            return AlignmentResult(
                success=True,
                aligned_image=aligned_image,
                transform_matrix=M,
                aligned_landmarks=aligned_landmarks,
                confidence=0.85
            )
            
        except Exception as e:
            logger.error(f"Affine alignment failed: {e}")
            return AlignmentResult(
                success=False,
                error_message=str(e)
            )
    
    def _align_by_procrustes(self, image: np.ndarray, landmarks: np.ndarray) -> AlignmentResult:
        """
        Align face using Procrustes analysis.
        
        Args:
            image: Input image
            landmarks: Facial landmarks
            
        Returns:
            AlignmentResult with aligned face
        """
        try:
            n_points = min(landmarks.shape[0], self.scaled_template.shape[0])
            src_points = landmarks[:n_points]
            dst_points = self.scaled_template[:n_points]
            
            # Perform Procrustes analysis
            transform_matrix, scale, rotation, translation = self._procrustes_analysis(
                src_points, dst_points
            )
            
            if transform_matrix is None:
                return AlignmentResult(
                    success=False,
                    error_message="Procrustes analysis failed"
                )
            
            # Apply transformation
            aligned_image = cv2.warpAffine(
                image, transform_matrix, self.target_size,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT
            )
            
            # Transform landmarks
            aligned_landmarks = self._transform_landmarks(landmarks, transform_matrix)
            
            return AlignmentResult(
                success=True,
                aligned_image=aligned_image,
                transform_matrix=transform_matrix,
                aligned_landmarks=aligned_landmarks,
                confidence=0.95
            )
            
        except Exception as e:
            logger.error(f"Procrustes alignment failed: {e}")
            return AlignmentResult(
                success=False,
                error_message=str(e)
            )
    
    def _procrustes_analysis(self, src_points: np.ndarray, 
                           dst_points: np.ndarray) -> Tuple[Optional[np.ndarray], float, float, np.ndarray]:
        """
        Perform Procrustes analysis to find optimal alignment.
        
        Args:
            src_points: Source landmarks
            dst_points: Target landmarks
            
        Returns:
            Tuple of (transform_matrix, scale, rotation, translation)
        """
        try:
            # Center the points
            src_centered = src_points - np.mean(src_points, axis=0)
            dst_centered = dst_points - np.mean(dst_points, axis=0)
            
            # Calculate scale
            src_scale = np.sqrt(np.sum(src_centered ** 2))
            dst_scale = np.sqrt(np.sum(dst_centered ** 2))
            
            if src_scale == 0:
                return None, 0, 0, np.zeros(2)
            
            scale = dst_scale / src_scale
            
            # Normalize
            src_normalized = src_centered / src_scale
            dst_normalized = dst_centered / dst_scale
            
            # Calculate rotation using SVD
            H = src_normalized.T @ dst_normalized
            U, S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            
            # Ensure proper rotation (det(R) = 1)
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
            
            # Calculate angle
            angle = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
            
            # Calculate translation
            src_mean = np.mean(src_points, axis=0)
            dst_mean = np.mean(dst_points, axis=0)
            
            # Create transformation matrix
            M = cv2.getRotationMatrix2D((src_mean[0], src_mean[1]), angle, scale)
            M[0, 2] += dst_mean[0] - src_mean[0]
            M[1, 2] += dst_mean[1] - src_mean[1]
            
            return M, scale, angle, dst_mean - src_mean
            
        except Exception as e:
            logger.error(f"Procrustes analysis error: {e}")
            return None, 0, 0, np.zeros(2)
    
    def _transform_landmarks(self, landmarks: np.ndarray, 
                           transform_matrix: np.ndarray) -> np.ndarray:
        """
        Apply transformation matrix to landmarks.
        
        Args:
            landmarks: Original landmarks
            transform_matrix: 2x3 transformation matrix
            
        Returns:
            Transformed landmarks
        """
        try:
            # Add homogeneous coordinate
            landmarks_homo = np.hstack([landmarks, np.ones((landmarks.shape[0], 1))])
            
            # Apply transformation
            transformed = (transform_matrix @ landmarks_homo.T).T
            
            return transformed
            
        except Exception as e:
            logger.error(f"Landmark transformation failed: {e}")
            return landmarks
    
    def align_face_to_template(self, image: np.ndarray, 
                              face_detection: FaceDetection,
                              template_landmarks: Optional[np.ndarray] = None) -> AlignmentResult:
        """
        Align face to a specific template.
        
        Args:
            image: Input image
            face_detection: Face detection with landmarks
            template_landmarks: Custom template landmarks (optional)
            
        Returns:
            AlignmentResult with aligned face
        """
        # Use custom template if provided
        if template_landmarks is not None:
            original_template = self.scaled_template
            self.scaled_template = template_landmarks
        
        try:
            result = self.align_face(image, face_detection)
            return result
        finally:
            # Restore original template
            if template_landmarks is not None:
                self.scaled_template = original_template
    
    def get_alignment_quality(self, aligned_landmarks: np.ndarray) -> float:
        """
        Calculate alignment quality score.
        
        Args:
            aligned_landmarks: Aligned facial landmarks
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        try:
            if aligned_landmarks is None or len(aligned_landmarks) == 0:
                return 0.0
            
            # Calculate distance from template
            n_points = min(aligned_landmarks.shape[0], self.scaled_template.shape[0])
            distances = np.linalg.norm(
                aligned_landmarks[:n_points] - self.scaled_template[:n_points],
                axis=1
            )
            
            # Convert to quality score (lower distance = higher quality)
            avg_distance = np.mean(distances)
            max_distance = np.max(self.target_size) * 0.1  # 10% of max dimension
            
            quality = max(0.0, 1.0 - (avg_distance / max_distance))
            return min(1.0, quality)
            
        except Exception as e:
            logger.error(f"Quality calculation failed: {e}")
            return 0.0