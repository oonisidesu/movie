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
            
            # If landmarks are not available, use simple bbox-based swapping
            if source_landmarks is None or target_landmarks is None:
                logger.info("Landmarks not available, using simple bbox-based swapping")
                return self._simple_bbox_swap(source_image, target_image, source_face, target_face)
            
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
    
    def _simple_bbox_swap(self, source_image: np.ndarray, target_image: np.ndarray,
                         source_face: FaceDetection, target_face: FaceDetection) -> Optional[np.ndarray]:
        """
        Enhanced face swapping using bounding boxes with color correction and better blending.
        
        Args:
            source_image: Source image
            target_image: Target image
            source_face: Source face detection
            target_face: Target face detection
            
        Returns:
            Image with swapped face or None if failed
        """
        try:
            # Get bounding boxes
            src_x, src_y, src_w, src_h = source_face.bbox
            tgt_x, tgt_y, tgt_w, tgt_h = target_face.bbox
            
            # Add padding for better blending
            padding = 0.3
            src_pad_w = int(src_w * padding)
            src_pad_h = int(src_h * padding)
            tgt_pad_w = int(tgt_w * padding)
            tgt_pad_h = int(tgt_h * padding)
            
            # Expand source region with padding
            src_x_padded = max(0, src_x - src_pad_w)
            src_y_padded = max(0, src_y - src_pad_h)
            src_x2_padded = min(source_image.shape[1], src_x + src_w + src_pad_w)
            src_y2_padded = min(source_image.shape[0], src_y + src_h + src_pad_h)
            
            # Expand target region with padding
            tgt_x_padded = max(0, tgt_x - tgt_pad_w)
            tgt_y_padded = max(0, tgt_y - tgt_pad_h)
            tgt_x2_padded = min(target_image.shape[1], tgt_x + tgt_w + tgt_pad_w)
            tgt_y2_padded = min(target_image.shape[0], tgt_y + tgt_h + tgt_pad_h)
            
            # Extract padded regions
            source_face_region = source_image[src_y_padded:src_y2_padded, src_x_padded:src_x2_padded]
            target_face_region = target_image[tgt_y_padded:tgt_y2_padded, tgt_x_padded:tgt_x2_padded]
            
            if source_face_region.size == 0:
                logger.error("Source face region is empty")
                return None
            
            # Resize source face to match target face size
            target_height = tgt_y2_padded - tgt_y_padded
            target_width = tgt_x2_padded - tgt_x_padded
            # Use INTER_CUBIC for better quality when upscaling, INTER_AREA for downscaling
            interpolation = cv2.INTER_CUBIC if (target_width > source_face_region.shape[1] or 
                                               target_height > source_face_region.shape[0]) else cv2.INTER_AREA
            resized_source = cv2.resize(source_face_region, (target_width, target_height), 
                                      interpolation=interpolation)
            
            # Color correction - match skin tone
            resized_source_corrected = self._match_skin_tone(resized_source, target_face_region)
            
            # Apply sharpening to improve clarity
            resized_source_corrected = self._sharpen_image(resized_source_corrected)
            
            # Create result image
            result = target_image.copy()
            
            # Create advanced elliptical mask with feathering
            mask = self._create_advanced_mask(target_height, target_width, tgt_w, tgt_h, padding)
            
            # Apply seamless cloning if possible
            try:
                # Calculate center for seamless cloning
                center_x = tgt_x + tgt_w // 2
                center_y = tgt_y + tgt_h // 2
                
                # Use Poisson blending for more natural results
                blended = cv2.seamlessClone(
                    resized_source_corrected, 
                    result, 
                    mask,
                    (center_x, center_y), 
                    cv2.NORMAL_CLONE
                )
                result = blended
            except:
                # Fallback to alpha blending
                mask_3d = np.stack([mask] * 3, axis=2) / 255.0
                
                # Extract target region for blending
                target_region = result[tgt_y_padded:tgt_y2_padded, tgt_x_padded:tgt_x2_padded]
                
                # Blend with improved alpha blending
                blended_region = (resized_source_corrected * mask_3d + target_region * (1 - mask_3d)).astype(np.uint8)
                
                # Place blended region back
                result[tgt_y_padded:tgt_y2_padded, tgt_x_padded:tgt_x2_padded] = blended_region
            
            logger.info("Enhanced face swap completed")
            return result
            
        except Exception as e:
            logger.error(f"Enhanced face swapping failed: {e}")
            return None
    
    def _match_skin_tone(self, source_face: np.ndarray, target_face: np.ndarray) -> np.ndarray:
        """
        Match the skin tone of source face to target face.
        
        Args:
            source_face: Source face region
            target_face: Target face region
            
        Returns:
            Color-corrected source face
        """
        try:
            # Convert to LAB color space for better color matching
            source_lab = cv2.cvtColor(source_face, cv2.COLOR_BGR2LAB)
            target_lab = cv2.cvtColor(target_face, cv2.COLOR_BGR2LAB)
            
            # Split channels
            source_l, source_a, source_b = cv2.split(source_lab)
            target_l, target_a, target_b = cv2.split(target_lab)
            
            # Calculate statistics for color channels
            source_mean_a = np.mean(source_a)
            source_mean_b = np.mean(source_b)
            source_std_a = np.std(source_a)
            source_std_b = np.std(source_b)
            
            target_mean_a = np.mean(target_a)
            target_mean_b = np.mean(target_b)
            target_std_a = np.std(target_a)
            target_std_b = np.std(target_b)
            
            # Transfer color
            if source_std_a > 0:
                source_a = (source_a - source_mean_a) * (target_std_a / source_std_a) + target_mean_a
            if source_std_b > 0:
                source_b = (source_b - source_mean_b) * (target_std_b / source_std_b) + target_mean_b
            
            # Clip values
            source_a = np.clip(source_a, 0, 255).astype(np.uint8)
            source_b = np.clip(source_b, 0, 255).astype(np.uint8)
            
            # Merge channels
            corrected_lab = cv2.merge([source_l, source_a, source_b])
            
            # Convert back to BGR
            corrected_bgr = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)
            
            return corrected_bgr
            
        except Exception as e:
            logger.warning(f"Color matching failed: {e}")
            return source_face
    
    def _create_advanced_mask(self, height: int, width: int, 
                             face_w: int, face_h: int, padding: float) -> np.ndarray:
        """
        Create an advanced mask with smooth feathering.
        
        Args:
            height: Mask height
            width: Mask width
            face_w: Face width
            face_h: Face height
            padding: Padding ratio
            
        Returns:
            Advanced blending mask
        """
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Calculate center and axes
        center_x = width // 2
        center_y = height // 2
        
        # Calculate face region within padded area
        face_center_x = int(width * (0.5))
        face_center_y = int(height * (0.5))
        
        # Create multiple ellipses for smooth gradient
        max_axes_x = int(face_w * 0.5)
        max_axes_y = int(face_h * 0.6)  # Slightly elongated for face shape
        
        # Create gradient mask
        for i in range(10):
            factor = 1.0 - (i * 0.1)
            axes_x = int(max_axes_x * factor)
            axes_y = int(max_axes_y * factor)
            value = int(255 * (1.0 - i * 0.1))
            cv2.ellipse(mask, (face_center_x, face_center_y), 
                       (axes_x, axes_y), 0, 0, 360, value, -1)
        
        # Apply moderate Gaussian blur for smooth feathering without too much blur
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        
        return mask
    
    def _sharpen_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply sharpening to improve image clarity.
        
        Args:
            image: Input image
            
        Returns:
            Sharpened image
        """
        try:
            # Create sharpening kernel
            kernel = np.array([[-1, -1, -1],
                             [-1,  9, -1],
                             [-1, -1, -1]])
            
            # Apply the sharpening kernel
            sharpened = cv2.filter2D(image, -1, kernel)
            
            # Blend with original to avoid over-sharpening
            result = cv2.addWeighted(image, 0.7, sharpened, 0.3, 0)
            
            return result
        except Exception as e:
            logger.warning(f"Sharpening failed: {e}")
            return image
    
    def _extract_landmarks(self, face_detection: FaceDetection) -> Optional[np.ndarray]:
        """
        Extract facial landmarks from face detection.
        
        Args:
            face_detection: Face detection with landmarks
            
        Returns:
            Array of landmark points or None if not available
        """
        if face_detection.landmarks is None:
            logger.warning("No landmarks available in face detection")
            return None
        
        landmarks = face_detection.landmarks
        
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