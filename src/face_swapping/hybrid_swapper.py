"""
Hybrid Face Swapper

Combines dlib landmarks for precise feature detection with stable transformation methods.
Avoids complex triangulation issues while maintaining high quality.
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple, Dict, Any

from ..face_detection import FaceDetection
from ..face_detection.landmarks import LandmarkDetector, FaceLandmarks
from .utils import apply_color_correction

logger = logging.getLogger(__name__)


class HybridFaceSwapper:
    """
    Hybrid face swapper using landmarks for feature detection but stable transformations.
    
    This approach uses dlib landmarks to:
    1. Detect precise face boundaries
    2. Calculate optimal face alignment
    3. Apply stable transformations
    4. Perform advanced blending
    5. Handle expression variations in video sequences
    """
    
    def __init__(self, landmark_method: str = 'dlib', enable_expression_transfer: bool = True,
                 horizontal_shift: float = 0.15, vertical_shift: float = 0.0):
        """
        Initialize hybrid face swapper.
        
        Args:
            landmark_method: Landmark detection method
            enable_expression_transfer: Whether to transfer facial expressions (for video)
            horizontal_shift: Horizontal position adjustment factor (negative=left, positive=right)
            vertical_shift: Vertical position adjustment factor (negative=up, positive=down)
        """
        self.landmark_detector = LandmarkDetector(method=landmark_method)
        self.enable_expression_transfer = enable_expression_transfer
        self.horizontal_shift = horizontal_shift
        self.vertical_shift = vertical_shift
        self.swap_count = 0
        self.success_count = 0
        
        # For video processing - store reference face
        self.reference_source_landmarks = None
        self.reference_target_landmarks = None
    
    def swap_faces(self, source_image: np.ndarray, target_image: np.ndarray,
                   source_face: FaceDetection, target_face: FaceDetection) -> Optional[np.ndarray]:
        """
        Perform hybrid face swapping.
        
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
            
            # Use landmark-guided alignment instead of simple bbox
            aligned_source, alignment_mask = self._align_face_with_landmarks(
                source_image, target_image, source_landmarks, target_landmarks
            )
            
            if aligned_source is None:
                logger.error("Failed to align faces with landmarks")
                return None
            
            # Enhanced color matching with alignment mask
            color_matched = self._enhanced_color_matching(
                aligned_source, target_image, alignment_mask
            )
            
            # Blend the aligned and color-matched face
            result = self._blend_aligned_faces(
                color_matched, target_image, alignment_mask
            )
            
            self.success_count += 1
            logger.info("Hybrid face swap completed successfully")
            
            return result
            
        except Exception as e:
            logger.error(f"Hybrid face swapping failed: {e}")
            return None
    
    def set_reference_faces(self, source_image: np.ndarray, target_image: np.ndarray,
                           source_face: FaceDetection, target_face: FaceDetection) -> bool:
        """
        Set reference faces for video processing to enable expression transfer.
        
        Args:
            source_image: Reference source image
            target_image: Reference target image
            source_face: Reference source face detection
            target_face: Reference target face detection
            
        Returns:
            True if reference faces were set successfully
        """
        try:
            # Detect landmarks for reference faces
            self.reference_source_landmarks = self.landmark_detector.detect_landmarks(
                source_image, source_face.bbox
            )
            self.reference_target_landmarks = self.landmark_detector.detect_landmarks(
                target_image, target_face.bbox
            )
            
            if self.reference_source_landmarks is None or self.reference_target_landmarks is None:
                logger.error("Failed to detect reference landmarks")
                return False
            
            logger.info("Reference faces set for expression transfer")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set reference faces: {e}")
            return False
    
    def swap_faces_with_expression(self, source_image: np.ndarray, target_image: np.ndarray,
                                  source_face: FaceDetection, target_face: FaceDetection,
                                  target_expression_landmarks: Optional[FaceLandmarks] = None) -> Optional[np.ndarray]:
        """
        Perform face swapping with expression transfer for video processing.
        
        Args:
            source_image: Source image (reference)
            target_image: Current target frame
            source_face: Source face detection
            target_face: Target face detection  
            target_expression_landmarks: Current target expression landmarks
            
        Returns:
            Swapped image with expression transfer or None if failed
        """
        self.swap_count += 1
        
        try:
            # Get current target landmarks for expression
            if target_expression_landmarks is None:
                target_expression_landmarks = self.landmark_detector.detect_landmarks(
                    target_image, target_face.bbox
                )
            
            if target_expression_landmarks is None:
                logger.error("Failed to detect current target landmarks")
                return None
            
            # If no reference faces set, use standard swap
            if self.reference_source_landmarks is None or self.reference_target_landmarks is None:
                return self.swap_faces(source_image, target_image, source_face, target_face)
            
            # Calculate expression difference
            expression_delta = self._calculate_expression_delta(
                self.reference_target_landmarks, target_expression_landmarks
            )
            
            # Apply expression to source face
            modified_source_landmarks = self._apply_expression_to_source(
                self.reference_source_landmarks, expression_delta
            )
            
            # Extract face regions
            source_face_region, source_center = self._extract_face_region_bbox_landmarks(
                source_image, modified_source_landmarks
            )
            target_face_region, target_center = self._extract_face_region_bbox(
                target_image, target_expression_landmarks
            )
            
            if source_face_region is None or target_face_region is None:
                logger.error("Failed to extract face regions")
                return None
            
            # Continue with standard processing
            target_h, target_w = target_face_region.shape[:2]
            if target_w <= 0 or target_h <= 0:
                return None
            source_resized = cv2.resize(source_face_region, (target_w, target_h), 
                                      interpolation=cv2.INTER_LINEAR)
            
            source_mask = self._create_face_mask_from_landmarks(
                modified_source_landmarks, source_resized.shape, source_center
            )
            color_matched = self._enhanced_color_matching(
                source_resized, target_face_region, source_mask
            )
            
            # Place the face back
            result = target_image.copy()
            x, y, w, h = target_center
            
            # Apply position adjustments for expression transfer
            face_width = abs(target_expression_landmarks.points[42, 0] - target_expression_landmarks.points[36, 0])
            face_height = abs(target_expression_landmarks.points[33, 1] - np.mean(target_expression_landmarks.points[36:48, 1]))
            
            horizontal_offset = int(face_width * self.horizontal_shift)
            vertical_offset = int(face_height * self.vertical_shift)
            
            x -= horizontal_offset
            y += vertical_offset
            
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(result.shape[1], x + w), min(result.shape[0], y + h)
            
            if x2 > x1 and y2 > y1:
                new_w, new_h = x2 - x1, y2 - y1
                if new_w > 0 and new_h > 0:
                    fitted_face = cv2.resize(color_matched, (new_w, new_h))
                    fitted_mask = cv2.resize(source_mask, (new_w, new_h))
                else:
                    return None
                
                mask_3d = np.stack([fitted_mask] * 3, axis=2) / 255.0
                if mask_3d.shape[0] > 7 and mask_3d.shape[1] > 7:
                    mask_3d = cv2.GaussianBlur(mask_3d, (7, 7), 0)
                
                region = result[y1:y2, x1:x2]
                
                try:
                    # Use improved seamless cloning with better mask preparation
                    improved_mask = self._create_improved_mask(fitted_mask)
                    center_x, center_y = (x2 - x1) // 2, (y2 - y1) // 2
                    center = (center_x, center_y)
                    
                    # Ensure center is within bounds
                    if (center_x > 0 and center_y > 0 and 
                        center_x < region.shape[1] and center_y < region.shape[0] and
                        fitted_face.shape == region.shape and
                        improved_mask.shape[:2] == region.shape[:2]):
                        
                        # Try mixed clone for more natural blending
                        blended_region = cv2.seamlessClone(
                            fitted_face, region, improved_mask, center, cv2.MIXED_CLONE
                        )
                        result[y1:y2, x1:x2] = blended_region
                    else:
                        raise ValueError("ROI bounds check failed")
                    
                except Exception:
                    # Use advanced multi-layer blending
                    advanced_blend = self._advanced_alpha_blending(fitted_face, region, mask_3d)
                    result[y1:y2, x1:x2] = advanced_blend
            
            self.success_count += 1
            logger.debug("Expression-aware face swap completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Expression-aware face swapping failed: {e}")
            return None
    
    def _create_improved_mask(self, mask: np.ndarray) -> np.ndarray:
        """Create an improved mask for better seamless cloning."""
        # Erode the mask slightly to avoid border artifacts
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        improved_mask = cv2.erode(mask, kernel, iterations=1)
        
        # Apply Gaussian blur for smoother edges
        improved_mask = cv2.GaussianBlur(improved_mask, (5, 5), 0)
        
        return improved_mask
    
    def _advanced_alpha_blending(self, source: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Advanced alpha blending with feathering and multi-scale mixing."""
        # Create feathered mask with distance transform
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) if len(mask.shape) == 3 else mask
        
        # Distance transform for smooth transitions
        dist_transform = cv2.distanceTransform(mask_gray.astype(np.uint8), cv2.DIST_L2, 5)
        dist_transform = cv2.normalize(dist_transform, None, 0, 1, cv2.NORM_MINMAX)
        
        # Create multi-scale blend
        feather_mask = np.stack([dist_transform] * 3, axis=2)
        
        # Apply different blending for different frequency components
        # Low frequency (overall color/lighting)
        source_low = cv2.GaussianBlur(source, (15, 15), 0)
        target_low = cv2.GaussianBlur(target, (15, 15), 0)
        low_blend = source_low * feather_mask + target_low * (1 - feather_mask)
        
        # High frequency (details/texture)
        source_high = source.astype(np.float32) - source_low.astype(np.float32)
        target_high = target.astype(np.float32) - target_low.astype(np.float32)
        high_blend = source_high * feather_mask + target_high * (1 - feather_mask)
        
        # Combine
        result = low_blend + high_blend
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _calculate_expression_delta(self, reference_landmarks: FaceLandmarks, 
                                   current_landmarks: FaceLandmarks) -> np.ndarray:
        """
        Calculate the difference in facial expression between reference and current landmarks.
        
        Args:
            reference_landmarks: Reference facial landmarks (neutral expression)
            current_landmarks: Current facial landmarks (with expression)
            
        Returns:
            Expression delta vector
        """
        # Focus on key expression points (mouth, eyebrows, eyes)
        expression_indices = list(range(17, 68))  # Exclude jaw outline for stability
        
        ref_points = reference_landmarks.points[expression_indices]
        cur_points = current_landmarks.points[expression_indices]
        
        # Calculate normalized delta
        ref_center = np.mean(ref_points, axis=0)
        cur_center = np.mean(cur_points, axis=0)
        
        # Normalize by face size
        ref_size = np.std(ref_points)
        cur_size = np.std(cur_points)
        
        if ref_size > 0:
            scale_factor = cur_size / ref_size
            delta = (cur_points - cur_center) * scale_factor - (ref_points - ref_center)
        else:
            delta = cur_points - ref_points
        
        return delta
    
    def _apply_expression_to_source(self, source_landmarks: FaceLandmarks,
                                   expression_delta: np.ndarray) -> FaceLandmarks:
        """
        Apply expression changes to source face landmarks.
        
        Args:
            source_landmarks: Original source landmarks
            expression_delta: Expression change vector
            
        Returns:
            Modified source landmarks with expression
        """
        modified_points = source_landmarks.points.copy()
        
        # Apply delta to expression-sensitive points
        expression_indices = list(range(17, 68))
        
        # Scale delta based on source face size
        source_expression_points = source_landmarks.points[expression_indices]
        source_center = np.mean(source_expression_points, axis=0)
        source_size = np.std(source_expression_points)
        
        if source_size > 0:
            # Scale the expression delta to match source face size
            std_delta = np.std(expression_delta)
            if std_delta > 0 and not np.isnan(std_delta):
                scaled_delta = expression_delta * (source_size / std_delta) * 0.5  # Reduce intensity
                modified_points[expression_indices] += scaled_delta
        
        return FaceLandmarks(points=modified_points, confidence=source_landmarks.confidence)
    
    def _extract_face_region_bbox_landmarks(self, image: np.ndarray, landmarks: FaceLandmarks) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        """
        Extract face region using modified landmarks.
        
        Args:
            image: Input image
            landmarks: Facial landmarks (possibly modified for expression)
            
        Returns:
            Tuple of (face_region, bbox_coordinates)
        """
        return self._extract_face_region_bbox(image, landmarks)
    
    def _align_face_with_landmarks(self, source_image: np.ndarray, target_image: np.ndarray,
                                  source_landmarks: FaceLandmarks, target_landmarks: FaceLandmarks) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Align source face to target using precise landmark matching.
        
        Args:
            source_image: Source image
            target_image: Target image
            source_landmarks: Source landmarks
            target_landmarks: Target landmarks
            
        Returns:
            Tuple of (aligned_source, alignment_mask)
        """
        try:
            # Key landmark points for alignment
            source_left_eye = np.mean(source_landmarks.points[42:48], axis=0)
            source_right_eye = np.mean(source_landmarks.points[36:42], axis=0)
            source_nose = source_landmarks.points[33]
            source_mouth = np.mean(source_landmarks.points[48:68], axis=0)
            
            target_left_eye = np.mean(target_landmarks.points[42:48], axis=0)
            target_right_eye = np.mean(target_landmarks.points[36:42], axis=0)
            target_nose = target_landmarks.points[33]
            target_mouth = np.mean(target_landmarks.points[48:68], axis=0)
            
            # Apply position adjustments to target points
            face_width = abs(target_left_eye[0] - target_right_eye[0])
            face_height = abs(target_nose[1] - np.mean([target_left_eye[1], target_right_eye[1]]))
            
            horizontal_offset = face_width * self.horizontal_shift
            vertical_offset = face_height * self.vertical_shift
            
            # Adjust target landmarks
            target_left_eye = target_left_eye.copy()
            target_right_eye = target_right_eye.copy()
            target_nose = target_nose.copy()
            target_mouth = target_mouth.copy()
            
            target_left_eye[0] -= horizontal_offset
            target_right_eye[0] -= horizontal_offset
            target_nose[0] -= horizontal_offset
            target_mouth[0] -= horizontal_offset
            
            target_left_eye[1] += vertical_offset
            target_right_eye[1] += vertical_offset
            target_nose[1] += vertical_offset
            target_mouth[1] += vertical_offset
            
            # Use 3 key points for stable affine transformation
            source_pts = np.array([
                source_left_eye,
                source_right_eye,
                source_nose
            ], dtype=np.float32)
            
            target_pts = np.array([
                target_left_eye,
                target_right_eye,
                target_nose
            ], dtype=np.float32)
            
            logger.info(f"Landmark alignment - Offset: H={horizontal_offset:.1f}, V={vertical_offset:.1f}")
            
            # Calculate affine transformation for stable alignment
            transform_matrix = cv2.getAffineTransform(source_pts, target_pts)
            
            # Apply transformation
            aligned_source = cv2.warpAffine(
                source_image, transform_matrix,
                (target_image.shape[1], target_image.shape[0]),
                flags=cv2.INTER_LINEAR
            )
            
            # Create alignment mask
            face_mask = self._create_precise_face_mask(target_landmarks, target_image.shape)
            
            return aligned_source, face_mask
            
        except Exception as e:
            logger.error(f"Landmark alignment failed: {e}")
            return None, None
    
    def _blend_aligned_faces(self, aligned_source: np.ndarray, target_image: np.ndarray, 
                            mask: np.ndarray) -> np.ndarray:
        """
        Blend aligned source face with target image using mask.
        
        Args:
            aligned_source: Aligned source image
            target_image: Target image
            mask: Face mask
            
        Returns:
            Blended result
        """
        try:
            # Find face center for seamless cloning
            mask_points = np.where(mask > 127)
            if len(mask_points[0]) == 0:
                return target_image
            
            center_y = int(np.mean(mask_points[0]))
            center_x = int(np.mean(mask_points[1]))
            center = (center_x, center_y)
            
            # Use seamless cloning
            result = cv2.seamlessClone(
                aligned_source, target_image, mask, center, cv2.NORMAL_CLONE
            )
            
            return result
            
        except Exception as e:
            logger.warning(f"Seamless cloning failed: {e}, using alpha blending")
            
            # Fallback to alpha blending
            mask_3d = np.stack([mask] * 3, axis=2) / 255.0
            mask_3d = cv2.GaussianBlur(mask_3d, (15, 15), 0)
            
            # Apply feathering
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
            eroded_mask = cv2.erode(mask, kernel, iterations=1)
            feather_mask = cv2.GaussianBlur(eroded_mask, (15, 15), 0) / 255.0
            feather_mask_3d = np.stack([feather_mask] * 3, axis=2)
            
            result = aligned_source * feather_mask_3d + target_image * (1 - feather_mask_3d)
            return result.astype(np.uint8)
    
    def _extract_face_region_bbox(self, image: np.ndarray, landmarks: FaceLandmarks) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        """
        Extract face region using bounding box from landmarks.
        
        Args:
            image: Input image
            landmarks: Facial landmarks
            
        Returns:
            Tuple of (face_region, bbox_coordinates)
        """
        try:
            # Get bounding box from landmarks
            points = landmarks.points
            if points is None or len(points) == 0:
                return None, None
            
            # Check for valid points
            valid_points = points[~np.isnan(points).any(axis=1)]
            if len(valid_points) == 0:
                return None, None
            
            x_min, y_min = np.min(valid_points, axis=0).astype(int)
            x_max, y_max = np.max(valid_points, axis=0).astype(int)
            
            # Add padding
            padding = 0.2
            w, h = x_max - x_min, y_max - y_min
            pad_x, pad_y = int(w * padding), int(h * padding)
            
            x1 = max(0, x_min - pad_x)
            y1 = max(0, y_min - pad_y)
            x2 = min(image.shape[1], x_max + pad_x)
            y2 = min(image.shape[0], y_max + pad_y)
            
            # Extract region
            face_region = image[y1:y2, x1:x2]
            bbox = (x1, y1, x2 - x1, y2 - y1)
            
            return face_region, bbox
            
        except Exception as e:
            logger.error(f"Face region extraction failed: {e}")
            return None, None
    
    def _create_face_mask_from_landmarks(self, landmarks: FaceLandmarks, shape: Tuple[int, int, int], 
                                        bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Create a precise face mask using landmark contours.
        
        Args:
            landmarks: Facial landmarks
            shape: Shape of the region (height, width, channels)
            bbox: Bounding box coordinates (x, y, w, h)
            
        Returns:
            Face mask
        """
        h, w = shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        x, y, bw, bh = bbox
        
        try:
            # Adjust landmarks to the cropped region coordinates
            adjusted_points = landmarks.points.copy()
            adjusted_points[:, 0] -= x  # Adjust x coordinates
            adjusted_points[:, 1] -= y  # Adjust y coordinates
            
            # Ensure points are within bounds
            adjusted_points[:, 0] = np.clip(adjusted_points[:, 0], 0, w - 1)
            adjusted_points[:, 1] = np.clip(adjusted_points[:, 1], 0, h - 1)
            
            # Create smooth elliptical mask based on landmark bounds instead of sharp contours
            # Get face bounds from landmarks
            x_min, y_min = np.min(adjusted_points, axis=0).astype(int)
            x_max, y_max = np.max(adjusted_points, axis=0).astype(int)
            
            # Calculate center and dimensions with padding
            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2
            
            # Use landmark spread to determine ellipse size
            width = int((x_max - x_min) * 0.6)  # Slightly larger than face
            height = int((y_max - y_min) * 0.7)
            
            # Create smooth elliptical mask
            cv2.ellipse(mask, (center_x, center_y), (width, height), 0, 0, 360, 255, -1)
            
            # Add some refinement based on key facial features
            try:
                # Create inner detailed region using key landmarks
                nose_tip = adjusted_points[30].astype(int)
                left_mouth = adjusted_points[48].astype(int)
                right_mouth = adjusted_points[54].astype(int)
                chin = adjusted_points[8].astype(int)
                
                # Create refined mouth-chin region
                chin_points = np.array([
                    left_mouth, right_mouth, chin
                ], dtype=np.int32)
                
                # Add subtle refinement without sharp edges
                cv2.fillPoly(mask, [chin_points], 255)
                
            except Exception:
                pass  # Use base ellipse if refinement fails
            
        except Exception as e:
            logger.warning(f"Landmark-based mask creation failed: {e}, using fallback")
            # Fallback to elliptical mask
            return self._create_face_mask_from_region(shape)
        
        # Advanced multi-stage smoothing for ultra-natural edges
        # First pass: remove any sharp corners
        mask = cv2.GaussianBlur(mask, (11, 11), 0)
        
        # Second pass: create distance-based feathering
        mask_binary = (mask > 127).astype(np.uint8) * 255
        distance_transform = cv2.distanceTransform(mask_binary, cv2.DIST_L2, 5)
        
        # Create smooth falloff
        max_dist = np.max(distance_transform)
        if max_dist > 0:
            # Normalize distance transform
            normalized_dist = distance_transform / max_dist
            # Apply smooth falloff function
            smooth_mask = np.power(normalized_dist, 0.7) * 255
            mask = smooth_mask.astype(np.uint8)
        
        # Final smoothing pass for ultra-soft edges
        mask = cv2.GaussianBlur(mask, (19, 19), 0)
        
        return mask
    
    def _create_face_mask_from_region(self, shape: Tuple[int, int, int]) -> np.ndarray:
        """
        Create a more natural face mask for region (fallback method).
        
        Args:
            shape: Shape of the region (height, width, channels)
            
        Returns:
            Face mask
        """
        h, w = shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Create more natural face-like contour
        center = (w // 2, h // 2)
        
        # Main face ellipse - slightly larger
        axes = (int(w * 0.38), int(h * 0.48))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        
        # Add chin area with better proportions
        chin_center = (w // 2, int(h * 0.75))
        chin_axes = (int(w * 0.28), int(h * 0.18))
        cv2.ellipse(mask, chin_center, chin_axes, 0, 0, 360, 255, -1)
        
        # Multi-stage smoothing for very natural blending
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        
        # Create gradual falloff
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.GaussianBlur(mask, (19, 19), 0)
        
        # Additional feathering
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.GaussianBlur(mask, (23, 23), 0)
        
        return mask
    
    def _enhanced_color_matching(self, source_face: np.ndarray, target_face: np.ndarray, 
                               mask: np.ndarray) -> np.ndarray:
        """
        Enhanced color matching with multiple techniques for natural skin tones.
        
        Args:
            source_face: Source face region
            target_face: Target face region
            mask: Face mask for precise color matching
            
        Returns:
            Color-corrected source face
        """
        try:
            # Step 1: Fast skin-aware color matching
            color_matched = self._advanced_skin_color_matching(source_face, target_face, mask)
            
            # Step 2: HSV-based natural color adjustment (lightweight)
            color_matched = self._hsv_color_adjustment(color_matched, target_face, mask)
            
            # Step 3: Skip histogram matching for video (too slow)
            # if mask is not None:
            #     color_matched = self._skin_histogram_matching(
            #         color_matched, target_face, mask
            #     )
            
            # Step 4: Simplified luminance balancing
            color_matched = self._fast_luminance_balance(color_matched, target_face, mask)
            
            # Step 5: Skip fine color temperature for speed
            # color_matched = self._natural_color_temperature_adjustment(color_matched, target_face, mask)
            
            return color_matched
            
        except Exception as e:
            logger.warning(f"Enhanced color matching failed: {e}")
            return source_face
    
    def _histogram_matching_masked(self, source: np.ndarray, target: np.ndarray, 
                                  mask: np.ndarray) -> np.ndarray:
        """Apply histogram matching only in masked region."""
        result = source.copy()
        
        try:
            # Convert mask to boolean
            face_region = mask > 127
            
            # Apply histogram matching per channel
            for i in range(3):
                if np.any(face_region):
                    source_channel = source[:, :, i]
                    target_channel = target[:, :, i]
                    
                    # Get histograms for face region only
                    source_hist, _ = np.histogram(source_channel[face_region], bins=256, range=(0, 256))
                    target_hist, _ = np.histogram(target_channel[face_region], bins=256, range=(0, 256))
                    
                    # Calculate cumulative distribution functions
                    source_cdf = np.cumsum(source_hist).astype(np.float64)
                    target_cdf = np.cumsum(target_hist).astype(np.float64)
                    
                    # Normalize CDFs
                    source_cdf /= source_cdf[-1]
                    target_cdf /= target_cdf[-1]
                    
                    # Create lookup table
                    lookup_table = np.interp(source_cdf, target_cdf, range(256))
                    
                    # Apply only to face region
                    matched_channel = source_channel.copy().astype(np.float32)
                    matched_channel[face_region] = lookup_table[source_channel[face_region]]
                    
                    result[:, :, i] = np.clip(matched_channel, 0, 255).astype(np.uint8)
                    
        except Exception as e:
            logger.warning(f"Histogram matching failed: {e}")
            
        return result
    
    def _advanced_skin_color_matching(self, source: np.ndarray, target: np.ndarray, 
                                    mask: np.ndarray) -> np.ndarray:
        """Advanced skin color matching using multiple color spaces."""
        result = source.copy().astype(np.float32)
        
        try:
            # Detect skin regions in both images
            source_skin_mask = self._detect_skin_region(source)
            target_skin_mask = self._detect_skin_region(target)
            
            if mask is not None:
                # Combine with face mask
                face_region = (mask > 127).astype(np.float32)
                source_skin_mask = source_skin_mask * face_region
                target_skin_mask = target_skin_mask * face_region
            
            # Calculate average skin color in LAB space
            source_lab = cv2.cvtColor(source.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
            target_lab = cv2.cvtColor(target.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
            
            if np.sum(source_skin_mask) > 100 and np.sum(target_skin_mask) > 100:
                # Calculate skin tone statistics
                source_skin_mean = np.mean(source_lab[source_skin_mask > 0.5], axis=0)
                target_skin_mean = np.mean(target_lab[target_skin_mask > 0.5], axis=0)
                
                source_skin_std = np.std(source_lab[source_skin_mask > 0.5], axis=0)
                target_skin_std = np.std(target_lab[target_skin_mask > 0.5], axis=0)
                
                # Apply gradual color transfer
                result_lab = source_lab.copy()
                for i in range(3):
                    if target_skin_std[i] > 0 and source_skin_std[i] > 0:
                        # Normalize and scale
                        normalized = (source_lab[:, :, i] - source_skin_mean[i]) / source_skin_std[i]
                        result_lab[:, :, i] = normalized * target_skin_std[i] + target_skin_mean[i]
                
                # Apply skin mask for gradual blending
                alpha = source_skin_mask[:, :, np.newaxis] * 0.7  # Reduce intensity for naturalness
                result_lab = source_lab * (1 - alpha) + result_lab * alpha
                
                # Convert back to BGR
                result = cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
                
        except Exception as e:
            logger.warning(f"Advanced skin color matching failed: {e}")
            
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _detect_skin_region(self, image: np.ndarray) -> np.ndarray:
        """Detect skin regions using HSV and YCrCb color spaces."""
        try:
            # Convert to HSV and YCrCb
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            
            # HSV skin detection
            lower_hsv = np.array([0, 20, 70], dtype=np.uint8)
            upper_hsv = np.array([20, 255, 255], dtype=np.uint8)
            mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
            
            # YCrCb skin detection
            lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
            upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)
            mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
            
            # Combine masks
            skin_mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)
            
            # Morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
            
            # Gaussian blur for smooth transitions
            skin_mask = cv2.GaussianBlur(skin_mask, (5, 5), 0)
            
            return skin_mask.astype(np.float32) / 255.0
            
        except Exception:
            return np.ones(image.shape[:2], dtype=np.float32)
    
    def _hsv_color_adjustment(self, source: np.ndarray, target: np.ndarray, 
                            mask: np.ndarray) -> np.ndarray:
        """Natural color adjustment in HSV space."""
        result = source.copy()
        
        try:
            source_hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV).astype(np.float32)
            target_hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV).astype(np.float32)
            
            if mask is not None:
                face_region = mask > 127
                
                if np.sum(face_region) > 100:
                    # Calculate target face HSV statistics
                    target_h_mean = np.mean(target_hsv[face_region, 0])
                    target_s_mean = np.mean(target_hsv[face_region, 1])
                    target_v_mean = np.mean(target_hsv[face_region, 2])
                    
                    source_h_mean = np.mean(source_hsv[face_region, 0])
                    source_s_mean = np.mean(source_hsv[face_region, 1])
                    source_v_mean = np.mean(source_hsv[face_region, 2])
                    
                    # Gentle adjustments
                    h_diff = (target_h_mean - source_h_mean) * 0.3  # Reduce hue shift
                    s_ratio = target_s_mean / max(source_s_mean, 1) * 0.5 + 0.5  # Gentle saturation
                    v_ratio = target_v_mean / max(source_v_mean, 1) * 0.7 + 0.3  # Gentle value
                    
                    # Apply adjustments with face mask
                    mask_3d = mask.astype(np.float32) / 255.0
                    
                    # Hue adjustment (with wrapping)
                    new_h = source_hsv[:, :, 0] + h_diff * mask_3d
                    new_h = np.where(new_h > 179, new_h - 180, new_h)
                    new_h = np.where(new_h < 0, new_h + 180, new_h)
                    source_hsv[:, :, 0] = new_h
                    
                    # Saturation and Value adjustments
                    source_hsv[:, :, 1] = source_hsv[:, :, 1] * (s_ratio * mask_3d + (1 - mask_3d))
                    source_hsv[:, :, 2] = source_hsv[:, :, 2] * (v_ratio * mask_3d + (1 - mask_3d))
                    
                    # Clamp values
                    source_hsv = np.clip(source_hsv, 0, 255)
                    
                    # Convert back
                    result = cv2.cvtColor(source_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
                    
        except Exception as e:
            logger.warning(f"HSV color adjustment failed: {e}")
            
        return result
    
    def _skin_histogram_matching(self, source: np.ndarray, target: np.ndarray, 
                               mask: np.ndarray) -> np.ndarray:
        """Histogram matching focused on skin regions."""
        return self._histogram_matching_masked(source, target, mask)
    
    def _balance_luminance_contrast(self, source: np.ndarray, target: np.ndarray, 
                                  mask: np.ndarray) -> np.ndarray:
        """Balance luminance and contrast for natural appearance."""
        result = source.copy().astype(np.float32)
        
        try:
            if mask is not None:
                face_region = mask > 127
                
                if np.sum(face_region) > 100:
                    # Calculate luminance statistics
                    source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY).astype(np.float32)
                    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY).astype(np.float32)
                    
                    source_mean = np.mean(source_gray[face_region])
                    target_mean = np.mean(target_gray[face_region])
                    
                    source_std = np.std(source_gray[face_region])
                    target_std = np.std(target_gray[face_region])
                    
                    # Gentle contrast and brightness adjustment
                    if source_std > 0:
                        contrast_ratio = (target_std / source_std) * 0.5 + 0.5  # Gentle contrast
                        brightness_diff = (target_mean - source_mean) * 0.5  # Gentle brightness
                        
                        mask_3d = mask.astype(np.float32) / 255.0
                        mask_3d = np.stack([mask_3d] * 3, axis=2)
                        
                        # Apply adjustments
                        adjusted = (result - source_mean) * contrast_ratio + source_mean + brightness_diff
                        result = result * (1 - mask_3d) + adjusted * mask_3d
                        
                        result = np.clip(result, 0, 255)
                        
        except Exception as e:
            logger.warning(f"Luminance contrast balancing failed: {e}")
            
        return result.astype(np.uint8)
    
    def _fast_luminance_balance(self, source: np.ndarray, target: np.ndarray, 
                              mask: np.ndarray) -> np.ndarray:
        """Fast luminance balancing for video processing."""
        result = source.copy().astype(np.float32)
        
        try:
            if mask is not None:
                face_region = mask > 127
                
                if np.sum(face_region) > 50:  # Lower threshold for speed
                    # Simple brightness matching
                    source_mean = np.mean(source[face_region])
                    target_mean = np.mean(target[face_region])
                    
                    brightness_diff = (target_mean - source_mean) * 0.3  # Gentle adjustment
                    
                    mask_3d = mask.astype(np.float32) / 255.0
                    mask_3d = np.stack([mask_3d] * 3, axis=2)
                    
                    # Apply brightness adjustment
                    result = result + brightness_diff * mask_3d
                    result = np.clip(result, 0, 255)
                    
        except Exception as e:
            logger.warning(f"Fast luminance balance failed: {e}")
            
        return result.astype(np.uint8)
    
    def _natural_color_temperature_adjustment(self, source: np.ndarray, target: np.ndarray, 
                                            mask: np.ndarray) -> np.ndarray:
        """Fine-tune color temperature for natural appearance."""
        result = source.copy().astype(np.float32)
        
        try:
            if mask is not None:
                face_region = mask > 127
                
                if np.sum(face_region) > 100:
                    # Calculate average color in face region
                    source_avg = np.mean(result[face_region], axis=0)
                    target_avg = np.mean(target.astype(np.float32)[face_region], axis=0)
                    
                    # Calculate color temperature difference (Blue vs Red balance)
                    source_temp = source_avg[0] / max(source_avg[2], 1)  # B/R ratio
                    target_temp = target_avg[0] / max(target_avg[2], 1)  # B/R ratio
                    
                    temp_diff = (target_temp - source_temp) * 0.3  # Gentle adjustment
                    
                    mask_3d = mask.astype(np.float32) / 255.0
                    
                    # Apply temperature adjustment to blue channel
                    if temp_diff != 0:
                        result[:, :, 0] = result[:, :, 0] * (1 + temp_diff * mask_3d)
                        result = np.clip(result, 0, 255)
                        
        except Exception as e:
            logger.warning(f"Color temperature adjustment failed: {e}")
            
        return result.astype(np.uint8)
    
    def _adjust_luminance(self, source: np.ndarray, target: np.ndarray, 
                         mask: np.ndarray) -> np.ndarray:
        """Adjust luminance to match target."""
        try:
            # Convert to LAB
            source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
            target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)
            
            if mask is not None:
                face_region = mask > 127
                if np.any(face_region):
                    # Adjust L channel (luminance)
                    source_l = source_lab[:, :, 0].astype(np.float32)
                    target_l = target_lab[:, :, 0].astype(np.float32)
                    
                    source_mean = np.mean(source_l[face_region])
                    target_mean = np.mean(target_l[face_region])
                    
                    # Gentle adjustment
                    adjustment = (target_mean - source_mean) * 0.6
                    source_l[face_region] += adjustment
                    
                    source_lab[:, :, 0] = np.clip(source_l, 0, 255).astype(np.uint8)
            
            return cv2.cvtColor(source_lab, cv2.COLOR_LAB2BGR)
            
        except Exception as e:
            logger.warning(f"Luminance adjustment failed: {e}")
            return source
    
    def _adjust_color_temperature(self, source: np.ndarray, target: np.ndarray, 
                                 mask: np.ndarray) -> np.ndarray:
        """Subtle color temperature adjustment."""
        try:
            if mask is not None:
                face_region = mask > 127
                if np.any(face_region):
                    # Calculate color temperature difference
                    source_blue = np.mean(source[face_region, 0].astype(np.float32))
                    source_red = np.mean(source[face_region, 2].astype(np.float32))
                    target_blue = np.mean(target[face_region, 0].astype(np.float32))
                    target_red = np.mean(target[face_region, 2].astype(np.float32))
                    
                    # Calculate temperature shift
                    blue_shift = (target_blue - source_blue) * 0.3
                    red_shift = (target_red - source_red) * 0.3
                    
                    # Apply gentle shift
                    result = source.copy().astype(np.float32)
                    result[face_region, 0] += blue_shift
                    result[face_region, 2] += red_shift
                    
                    return np.clip(result, 0, 255).astype(np.uint8)
            
            return source
            
        except Exception as e:
            logger.warning(f"Color temperature adjustment failed: {e}")
            return source
    
    def _match_skin_tone_simple(self, source_face: np.ndarray, target_face: np.ndarray) -> np.ndarray:
        """
        Simple skin tone matching.
        
        Args:
            source_face: Source face region
            target_face: Target face region
            
        Returns:
            Color-corrected source face
        """
        try:
            # Convert to LAB color space
            source_lab = cv2.cvtColor(source_face, cv2.COLOR_BGR2LAB)
            target_lab = cv2.cvtColor(target_face, cv2.COLOR_BGR2LAB)
            
            # Match each channel
            for i in range(3):
                source_channel = source_lab[:, :, i].astype(np.float32)
                target_channel = target_lab[:, :, i].astype(np.float32)
                
                source_mean, source_std = cv2.meanStdDev(source_channel)
                target_mean, target_std = cv2.meanStdDev(target_channel)
                
                if source_std > 0:
                    # Normalize and scale
                    normalized = (source_channel - source_mean) * (target_std / source_std) + target_mean
                    source_lab[:, :, i] = np.clip(normalized, 0, 255).astype(np.uint8)
            
            # Convert back to BGR
            result = cv2.cvtColor(source_lab, cv2.COLOR_LAB2BGR)
            return result
            
        except Exception as e:
            logger.warning(f"Color matching failed: {e}")
            return source_face
    
    def _extract_face_region(self, image: np.ndarray, landmarks: FaceLandmarks) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extract face region using landmarks to define precise boundaries.
        
        Args:
            image: Input image
            landmarks: Facial landmarks
            
        Returns:
            Tuple of (face_region, face_mask)
        """
        try:
            # Create face mask from landmarks
            mask = self._create_precise_face_mask(landmarks, image.shape)
            
            # Find bounding box of face region
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None, None
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contours[0])
            
            # Add padding
            padding = 0.1
            pad_x = int(w * padding)
            pad_y = int(h * padding)
            
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(image.shape[1], x + w + pad_x)
            y2 = min(image.shape[0], y + h + pad_y)
            
            # Extract region
            face_region = image[y1:y2, x1:x2]
            face_mask = mask[y1:y2, x1:x2]
            
            return face_region, face_mask
            
        except Exception as e:
            logger.error(f"Face region extraction failed: {e}")
            return None, None
    
    def _create_precise_face_mask(self, landmarks: FaceLandmarks, image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Create precise face mask using landmark contours.
        
        Args:
            landmarks: Facial landmarks
            image_shape: Image dimensions
            
        Returns:
            Face mask
        """
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        
        # Get face contour points
        # Use jaw points (0-16) and estimated forehead
        jaw_points = landmarks.points[0:17].astype(np.int32)
        
        # Estimate forehead points based on eyebrows
        left_eyebrow = landmarks.points[22:27]
        right_eyebrow = landmarks.points[17:22]
        
        # Create forehead arc
        forehead_points = []
        eyebrow_center = np.mean(np.vstack([left_eyebrow, right_eyebrow]), axis=0)
        forehead_height = int(abs(jaw_points[8][1] - eyebrow_center[1]) * 0.3)
        
        for i in range(5):
            t = i / 4.0
            x = int(jaw_points[-1][0] * (1 - t) + jaw_points[0][0] * t)
            y = int(eyebrow_center[1] - forehead_height)
            forehead_points.append([x, y])
        
        # Combine contour points
        contour_points = np.vstack([
            jaw_points,
            np.array(forehead_points[::-1])  # Reverse for proper contour
        ])
        
        # Fill contour
        cv2.fillPoly(mask, [contour_points], 255)
        
        # Smooth the mask
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        
        return mask
    
    def _calculate_alignment(self, source_landmarks: FaceLandmarks, 
                           target_landmarks: FaceLandmarks) -> np.ndarray:
        """
        Calculate alignment transformation between landmarks with position adjustment.
        
        Args:
            source_landmarks: Source facial landmarks
            target_landmarks: Target facial landmarks
            
        Returns:
            Transformation matrix
        """
        # Use key points for alignment (eyes and nose)
        source_right_eye = np.mean(source_landmarks.points[36:42], axis=0)  # Right eye center
        source_left_eye = np.mean(source_landmarks.points[42:48], axis=0)   # Left eye center
        source_nose = source_landmarks.points[33]  # Nose tip
        
        target_right_eye = np.mean(target_landmarks.points[36:42], axis=0)  # Right eye center
        target_left_eye = np.mean(target_landmarks.points[42:48], axis=0)   # Left eye center
        target_nose = target_landmarks.points[33]  # Nose tip
        
        # Calculate face center and adjust target positions
        target_face_center = np.mean([target_right_eye, target_left_eye, target_nose], axis=0)
        
        # Apply position adjustments
        face_width = abs(target_left_eye[0] - target_right_eye[0])
        face_height = abs(target_nose[1] - np.mean([target_right_eye[1], target_left_eye[1]]))
        
        horizontal_shift = face_width * self.horizontal_shift
        vertical_shift = face_height * self.vertical_shift
        
        # Adjust target points
        adjusted_target_right_eye = target_right_eye.copy()
        adjusted_target_left_eye = target_left_eye.copy()
        adjusted_target_nose = target_nose.copy()
        
        # Apply horizontal shift (negative=left, positive=right)
        adjusted_target_right_eye[0] -= horizontal_shift
        adjusted_target_left_eye[0] -= horizontal_shift
        adjusted_target_nose[0] -= horizontal_shift
        
        # Apply vertical shift (negative=up, positive=down)
        adjusted_target_right_eye[1] += vertical_shift
        adjusted_target_left_eye[1] += vertical_shift
        adjusted_target_nose[1] += vertical_shift
        
        # Calculate transformation using key points
        source_pts = np.array([
            source_right_eye,
            source_left_eye,
            source_nose
        ], dtype=np.float32)
        
        target_pts = np.array([
            adjusted_target_right_eye,
            adjusted_target_left_eye,
            adjusted_target_nose
        ], dtype=np.float32)
        
        logger.info(f"Original target key points: {np.array([target_right_eye, target_left_eye, target_nose])}")
        logger.info(f"Adjusted target key points: {target_pts}")
        logger.info(f"Position adjustments - Horizontal: {horizontal_shift:.2f}, Vertical: {vertical_shift:.2f} pixels")
        
        # Calculate affine transformation
        transform_matrix = cv2.getAffineTransform(source_pts, target_pts)
        
        return transform_matrix
    
    def _apply_alignment(self, source_region: np.ndarray, source_mask: np.ndarray,
                        transform_matrix: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Apply alignment transformation to source face.
        
        Args:
            source_region: Source face region
            source_mask: Source face mask
            transform_matrix: Transformation matrix
            target_size: Target size (height, width)
            
        Returns:
            Aligned source face
        """
        # Apply transformation to both face and mask
        aligned_face = cv2.warpAffine(
            source_region, transform_matrix, 
            (target_size[1], target_size[0]),
            flags=cv2.INTER_LINEAR
        )
        
        aligned_mask = cv2.warpAffine(
            source_mask, transform_matrix,
            (target_size[1], target_size[0]),
            flags=cv2.INTER_LINEAR
        )
        
        return aligned_face
    
    def _match_skin_tone(self, source_face: np.ndarray, target_face: np.ndarray,
                        mask: np.ndarray) -> np.ndarray:
        """
        Match skin tone between source and target faces.
        
        Args:
            source_face: Source face region
            target_face: Target face region
            mask: Face mask
            
        Returns:
            Color-corrected source face
        """
        try:
            # Convert to LAB color space
            source_lab = cv2.cvtColor(source_face, cv2.COLOR_BGR2LAB)
            target_lab = cv2.cvtColor(target_face, cv2.COLOR_BGR2LAB)
            
            # Apply color transfer in face region
            if mask is not None:
                face_pixels = mask > 127
                
                for i in range(3):  # L, A, B channels
                    if np.any(face_pixels):
                        source_channel = source_lab[:, :, i].astype(np.float32)
                        target_channel = target_lab[:, :, i].astype(np.float32)
                        
                        source_mean = np.mean(source_channel[face_pixels])
                        source_std = np.std(source_channel[face_pixels])
                        target_mean = np.mean(target_channel[face_pixels])
                        target_std = np.std(target_channel[face_pixels])
                        
                        if source_std > 0:
                            # Normalize and scale
                            normalized = (source_channel - source_mean) * (target_std / source_std) + target_mean
                            source_lab[:, :, i] = np.clip(normalized, 0, 255).astype(np.uint8)
            
            # Convert back to BGR
            corrected = cv2.cvtColor(source_lab, cv2.COLOR_LAB2BGR)
            return corrected
            
        except Exception as e:
            logger.warning(f"Color matching failed: {e}")
            return source_face
    
    def _blend_faces(self, aligned_source: np.ndarray, target_image: np.ndarray,
                    target_landmarks: FaceLandmarks, source_mask: np.ndarray) -> np.ndarray:
        """
        Blend aligned source face with target image.
        
        Args:
            aligned_source: Aligned source image
            target_image: Target image
            target_landmarks: Target landmarks
            source_mask: Source face mask (already transformed)
            
        Returns:
            Blended result
        """
        result = target_image.copy()
        
        try:
            # Find face center for seamless cloning
            center = np.mean(target_landmarks.points[0:17], axis=0).astype(int)
            center = (int(center[0]), int(center[1]))
            
            # Use seamless cloning with the source mask
            blended = cv2.seamlessClone(
                aligned_source, target_image, source_mask, center, cv2.NORMAL_CLONE
            )
            
            return blended
            
        except Exception as e:
            logger.warning(f"Seamless cloning failed: {e}, using alpha blending")
            
            # Fallback to alpha blending using the source mask
            # Create 3D mask with smoothing
            mask_3d = np.stack([source_mask] * 3, axis=2) / 255.0
            if mask_3d.shape[0] > 15 and mask_3d.shape[1] > 15:
                mask_3d = cv2.GaussianBlur(mask_3d, (15, 15), 0)
            
            # Apply feathering at the edges
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            eroded_mask = cv2.erode(source_mask, kernel, iterations=1)
            if eroded_mask.shape[0] > 5 and eroded_mask.shape[1] > 5:
                feather_mask = cv2.GaussianBlur(eroded_mask, (5, 5), 0) / 255.0
            else:
                feather_mask = eroded_mask / 255.0
            feather_mask_3d = np.stack([feather_mask] * 3, axis=2)
            
            # Blend with feathered edges
            blended = aligned_source * feather_mask_3d + target_image * (1 - feather_mask_3d)
            return blended.astype(np.uint8)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get swapping statistics."""
        success_rate = self.success_count / self.swap_count if self.swap_count > 0 else 0
        
        return {
            'total_swaps': self.swap_count,
            'successful_swaps': self.success_count,
            'success_rate': success_rate,
            'method': 'hybrid',
            'landmark_method': self.landmark_detector.method
        }