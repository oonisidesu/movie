"""
Classical Face Swapping Implementation

Implements traditional computer vision techniques for face swapping using
Delaunay triangulation, affine transformations, and seamless blending.
"""

import cv2
import numpy as np
import logging
from typing import Optional, List, Tuple, Dict
from scipy.spatial import Delaunay

from ..face_detection import FaceDetection
from .utils import extract_face_region, resize_face_to_target

logger = logging.getLogger(__name__)


class ClassicalFaceSwapper:
    """
    Classical face swapping using computer vision techniques.
    
    This implementation uses facial landmarks to create a triangular mesh,
    applies affine transformations for warping, and uses blending techniques
    for seamless integration.
    """
    
    def __init__(self, target_size: Tuple[int, int] = (256, 256),
                 smoothing_factor: float = 0.7):
        """
        Initialize classical face swapper.
        
        Args:
            target_size: Target size for face processing
            smoothing_factor: Smoothing factor for blending (0.0 to 1.0)
        """
        self.target_size = target_size
        self.smoothing_factor = smoothing_factor
        
        # 68-point facial landmark indices for different face regions
        self.face_landmarks = {
            'jaw': list(range(0, 17)),
            'right_eyebrow': list(range(17, 22)),
            'left_eyebrow': list(range(22, 27)),
            'nose': list(range(27, 36)),
            'right_eye': list(range(36, 42)),
            'left_eye': list(range(42, 48)),
            'mouth': list(range(48, 68))
        }
        
        # Core face region (excluding jaw for better blending)
        self.core_face_indices = (
            self.face_landmarks['right_eyebrow'] +
            self.face_landmarks['left_eyebrow'] +
            self.face_landmarks['nose'] +
            self.face_landmarks['right_eye'] +
            self.face_landmarks['left_eye'] +
            self.face_landmarks['mouth']
        )
    
    def swap_faces(self, source_image: np.ndarray, target_image: np.ndarray,
                   source_face: FaceDetection, target_face: FaceDetection) -> Optional[np.ndarray]:
        """
        Swap faces using classical computer vision techniques.
        
        Args:
            source_image: Source image containing the face to extract
            target_image: Target image where face will be placed
            source_face: Detected source face information
            target_face: Detected target face information
            
        Returns:
            Image with swapped face, or None if swapping failed
        """
        try:
            # Extract landmarks
            source_landmarks = self._extract_landmarks(source_face)
            target_landmarks = self._extract_landmarks(target_face)
            
            if source_landmarks is None or target_landmarks is None:
                logger.error("Failed to extract landmarks")
                return None
            
            # Create result image (copy of target)
            result_image = target_image.copy()
            
            # Perform triangular warping
            warped_face = self._warp_face_triangular(
                source_image, result_image,
                source_landmarks, target_landmarks
            )
            
            if warped_face is None:
                logger.error("Face warping failed")
                return None
            
            # Apply seamless blending
            blended_image = self._seamless_blend(
                warped_face, target_image, target_landmarks
            )
            
            # Color correction
            if blended_image is not None:
                blended_image = self._color_correct(
                    blended_image, target_image, target_landmarks
                )
            
            return blended_image
            
        except Exception as e:
            logger.error(f"Classical face swapping failed: {e}")
            return None
    
    def _extract_landmarks(self, face_detection: FaceDetection) -> Optional[np.ndarray]:
        """
        Extract facial landmarks from face detection.
        
        Args:
            face_detection: Face detection with landmarks
            
        Returns:
            Array of landmark points or None if not available
        """
        if 'landmarks' not in face_detection or face_detection['landmarks'] is None:
            logger.warning("No landmarks available in face detection")
            return None
        
        landmarks = face_detection['landmarks']
        
        # Convert to numpy array if needed
        if isinstance(landmarks, list):
            landmarks = np.array(landmarks, dtype=np.float32)
        
        # Ensure we have the expected number of landmarks
        if landmarks.shape[0] < 68:
            logger.warning(f"Insufficient landmarks: {landmarks.shape[0]} (expected 68)")
            return None
        
        return landmarks[:68]  # Use first 68 landmarks
    
    def _create_delaunay_triangulation(self, landmarks: np.ndarray, 
                                     image_shape: Tuple[int, int]) -> Optional[Delaunay]:
        """
        Create Delaunay triangulation from facial landmarks.
        
        Args:
            landmarks: Facial landmark points
            image_shape: Image dimensions (height, width)
            
        Returns:
            Delaunay triangulation object or None if failed
        """
        try:
            # Add image corners to landmarks for complete triangulation
            h, w = image_shape[:2]
            corners = np.array([
                [0, 0], [w-1, 0], [w-1, h-1], [0, h-1]
            ], dtype=np.float32)
            
            # Combine landmarks with corners
            all_points = np.vstack([landmarks, corners])
            
            # Create Delaunay triangulation
            triangulation = Delaunay(all_points)
            
            return triangulation
            
        except Exception as e:
            logger.error(f"Delaunay triangulation failed: {e}")
            return None
    
    def _warp_face_triangular(self, source_image: np.ndarray, target_image: np.ndarray,
                             source_landmarks: np.ndarray, target_landmarks: np.ndarray) -> Optional[np.ndarray]:
        """
        Warp source face to target landmarks using triangular warping.
        
        Args:
            source_image: Source image
            target_image: Target image
            source_landmarks: Source facial landmarks
            target_landmarks: Target facial landmarks
            
        Returns:
            Warped face image or None if failed
        """
        try:
            # Create triangulation based on target landmarks
            triangulation = self._create_delaunay_triangulation(
                target_landmarks, target_image.shape
            )
            
            if triangulation is None:
                return None
            
            # Create result image
            result = target_image.copy()
            
            # Add corners to both landmark sets
            h, w = target_image.shape[:2]
            hs, ws = source_image.shape[:2]
            
            target_corners = np.array([
                [0, 0], [w-1, 0], [w-1, h-1], [0, h-1]
            ], dtype=np.float32)
            
            source_corners = np.array([
                [0, 0], [ws-1, 0], [ws-1, hs-1], [0, hs-1]
            ], dtype=np.float32)
            
            target_points = np.vstack([target_landmarks, target_corners])
            source_points = np.vstack([source_landmarks, source_corners])
            
            # Warp each triangle
            for triangle in triangulation.simplices:
                # Get triangle vertices
                target_triangle = target_points[triangle]
                source_triangle = source_points[triangle]
                
                # Warp triangle
                self._warp_triangle(
                    source_image, result,
                    source_triangle, target_triangle
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Triangular warping failed: {e}")
            return None
    
    def _warp_triangle(self, source_image: np.ndarray, target_image: np.ndarray,
                      source_triangle: np.ndarray, target_triangle: np.ndarray) -> None:
        """
        Warp a single triangle from source to target.
        
        Args:
            source_image: Source image
            target_image: Target image (modified in-place)
            source_triangle: Source triangle vertices
            target_triangle: Target triangle vertices
        """
        try:
            # Get bounding rectangles
            target_rect = cv2.boundingRect(target_triangle.astype(np.int32))
            source_rect = cv2.boundingRect(source_triangle.astype(np.int32))
            
            # Offset triangles by their bounding rects
            target_triangle_offset = target_triangle - [target_rect[0], target_rect[1]]
            source_triangle_offset = source_triangle - [source_rect[0], source_rect[1]]
            
            # Get affine transform
            transform_matrix = cv2.getAffineTransform(
                source_triangle_offset.astype(np.float32),
                target_triangle_offset.astype(np.float32)
            )
            
            # Extract source region
            source_patch = source_image[
                source_rect[1]:source_rect[1] + source_rect[3],
                source_rect[0]:source_rect[0] + source_rect[2]
            ]
            
            if source_patch.size == 0:
                return
            
            # Warp source patch
            warped_patch = cv2.warpAffine(
                source_patch, transform_matrix,
                (target_rect[2], target_rect[3])
            )
            
            # Create mask for triangle
            mask = np.zeros((target_rect[3], target_rect[2]), dtype=np.uint8)
            cv2.fillPoly(mask, [target_triangle_offset.astype(np.int32)], 255)
            
            # Apply mask and copy to target image
            target_region = target_image[
                target_rect[1]:target_rect[1] + target_rect[3],
                target_rect[0]:target_rect[0] + target_rect[2]
            ]
            
            # Blend warped patch with target
            for c in range(target_image.shape[2]):
                target_region[:, :, c] = np.where(
                    mask > 0,
                    warped_patch[:, :, c],
                    target_region[:, :, c]
                )
                
        except Exception as e:
            logger.debug(f"Triangle warping error (non-critical): {e}")
    
    def _seamless_blend(self, warped_image: np.ndarray, target_image: np.ndarray,
                       target_landmarks: np.ndarray) -> Optional[np.ndarray]:
        """
        Apply seamless blending using cv2.seamlessClone.
        
        Args:
            warped_image: Warped face image
            target_image: Original target image
            target_landmarks: Target facial landmarks
            
        Returns:
            Blended image or None if failed
        """
        try:
            # Create face mask from landmarks
            face_mask = self._create_face_mask(target_landmarks, target_image.shape)
            
            if face_mask is None:
                return warped_image
            
            # Calculate center point for seamless cloning
            center = np.mean(target_landmarks, axis=0).astype(np.int32)
            center = (int(center[0]), int(center[1]))
            
            # Ensure center is within image bounds
            h, w = target_image.shape[:2]
            center = (
                max(0, min(w-1, center[0])),
                max(0, min(h-1, center[1]))
            )
            
            # Apply seamless cloning
            blended = cv2.seamlessClone(
                warped_image, target_image, face_mask,
                center, cv2.NORMAL_CLONE
            )
            
            return blended
            
        except Exception as e:
            logger.warning(f"Seamless blending failed, using alpha blending: {e}")
            return self._alpha_blend(warped_image, target_image, target_landmarks)
    
    def _alpha_blend(self, warped_image: np.ndarray, target_image: np.ndarray,
                    target_landmarks: np.ndarray) -> Optional[np.ndarray]:
        """
        Apply alpha blending as fallback.
        
        Args:
            warped_image: Warped face image
            target_image: Original target image
            target_landmarks: Target facial landmarks
            
        Returns:
            Blended image or None if failed
        """
        try:
            # Create smooth face mask
            face_mask = self._create_face_mask(target_landmarks, target_image.shape, smooth=True)
            
            if face_mask is None:
                return warped_image
            
            # Convert mask to 3-channel
            mask_3d = np.stack([face_mask] * 3, axis=2) / 255.0
            
            # Apply smoothing factor
            mask_3d *= self.smoothing_factor
            
            # Blend images
            blended = (warped_image * mask_3d + target_image * (1 - mask_3d)).astype(np.uint8)
            
            return blended
            
        except Exception as e:
            logger.error(f"Alpha blending failed: {e}")
            return warped_image
    
    def _create_face_mask(self, landmarks: np.ndarray, image_shape: Tuple[int, int],
                         smooth: bool = False) -> Optional[np.ndarray]:
        """
        Create face mask from landmarks.
        
        Args:
            landmarks: Facial landmarks
            image_shape: Image dimensions
            smooth: Whether to apply smoothing to mask
            
        Returns:
            Face mask or None if failed
        """
        try:
            # Create mask
            mask = np.zeros(image_shape[:2], dtype=np.uint8)
            
            # Define face contour (jaw + forehead)
            jaw_points = landmarks[self.face_landmarks['jaw']]
            
            # Add forehead points (estimated from eyebrows)
            left_eyebrow = landmarks[self.face_landmarks['left_eyebrow']]
            right_eyebrow = landmarks[self.face_landmarks['right_eyebrow']]
            
            # Estimate forehead points
            forehead_height = np.linalg.norm(left_eyebrow[0] - landmarks[33]) * 0.6
            forehead_points = []
            
            for i in range(len(right_eyebrow)):
                point = right_eyebrow[i] + [0, -forehead_height]
                forehead_points.append(point)
            
            for i in range(len(left_eyebrow)-1, -1, -1):
                point = left_eyebrow[i] + [0, -forehead_height]
                forehead_points.append(point)
            
            # Combine all contour points
            contour_points = np.vstack([
                jaw_points,
                np.array(forehead_points)
            ]).astype(np.int32)
            
            # Fill mask
            cv2.fillPoly(mask, [contour_points], 255)
            
            # Apply smoothing if requested
            if smooth:
                mask = cv2.GaussianBlur(mask, (15, 15), 0)
            
            return mask
            
        except Exception as e:
            logger.error(f"Face mask creation failed: {e}")
            return None
    
    def _color_correct(self, blended_image: np.ndarray, target_image: np.ndarray,
                      target_landmarks: np.ndarray) -> np.ndarray:
        """
        Apply color correction to match target image.
        
        Args:
            blended_image: Blended face image
            target_image: Original target image
            target_landmarks: Target facial landmarks
            
        Returns:
            Color-corrected image
        """
        try:
            # Create face region mask
            face_mask = self._create_face_mask(target_landmarks, target_image.shape)
            
            if face_mask is None:
                return blended_image
            
            # Calculate color statistics for face regions
            target_face = cv2.bitwise_and(target_image, target_image, mask=face_mask)
            blended_face = cv2.bitwise_and(blended_image, blended_image, mask=face_mask)
            
            # Calculate mean and std for each channel
            result = blended_image.copy()
            
            for c in range(3):
                target_pixels = target_face[:, :, c][face_mask > 0]
                blended_pixels = blended_face[:, :, c][face_mask > 0]
                
                if len(target_pixels) > 0 and len(blended_pixels) > 0:
                    target_mean = np.mean(target_pixels)
                    target_std = np.std(target_pixels)
                    blended_mean = np.mean(blended_pixels)
                    blended_std = np.std(blended_pixels)
                    
                    if blended_std > 0:
                        # Apply color transfer
                        face_region = result[:, :, c]
                        face_pixels = face_region[face_mask > 0]
                        
                        # Normalize and scale
                        corrected_pixels = (face_pixels - blended_mean) * (target_std / blended_std) + target_mean
                        corrected_pixels = np.clip(corrected_pixels, 0, 255)
                        
                        face_region[face_mask > 0] = corrected_pixels
                        result[:, :, c] = face_region
            
            return result
            
        except Exception as e:
            logger.warning(f"Color correction failed: {e}")
            return blended_image