"""
Face Blending Module

Provides advanced blending techniques for seamless face integration
including Poisson blending, multi-band blending, and gradient domain methods.
"""

import cv2
import numpy as np
import logging
from enum import Enum
from typing import Optional, Tuple, Dict, Any
from scipy import ndimage

logger = logging.getLogger(__name__)


class BlendingMethod(Enum):
    """Available blending methods."""
    ALPHA = "alpha"
    POISSON = "poisson"
    MULTIBAND = "multiband"
    GRADIENT = "gradient"
    SEAMLESS = "seamless"


class FaceBlender:
    """
    Advanced face blending for seamless integration.
    
    Provides multiple blending techniques to achieve natural-looking
    face swaps with smooth transitions and color matching.
    """
    
    def __init__(self, method: BlendingMethod = BlendingMethod.POISSON,
                 feathering: int = 5):
        """
        Initialize face blender.
        
        Args:
            method: Blending method to use
            feathering: Edge feathering amount in pixels
        """
        self.method = method
        self.feathering = feathering
    
    def blend_faces(self, source_image: np.ndarray, target_image: np.ndarray,
                   mask: np.ndarray, center: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
        """
        Blend source face into target image using specified method.
        
        Args:
            source_image: Source image with face to blend
            target_image: Target image to blend into
            mask: Binary mask defining blend region
            center: Center point for blending (auto-calculated if None)
            
        Returns:
            Blended image or None if blending failed
        """
        try:
            # Validate inputs
            if source_image.shape != target_image.shape:
                logger.error("Source and target images must have same dimensions")
                return None
            
            if mask.shape[:2] != source_image.shape[:2]:
                logger.error("Mask dimensions must match image dimensions")
                return None
            
            # Calculate center if not provided
            if center is None:
                center = self._calculate_mask_center(mask)
            
            # Apply selected blending method
            if self.method == BlendingMethod.ALPHA:
                return self._alpha_blend(source_image, target_image, mask)
            elif self.method == BlendingMethod.POISSON:
                return self._poisson_blend(source_image, target_image, mask, center)
            elif self.method == BlendingMethod.MULTIBAND:
                return self._multiband_blend(source_image, target_image, mask)
            elif self.method == BlendingMethod.GRADIENT:
                return self._gradient_blend(source_image, target_image, mask)
            elif self.method == BlendingMethod.SEAMLESS:
                return self._seamless_blend(source_image, target_image, mask, center)
            else:
                logger.error(f"Unsupported blending method: {self.method}")
                return None
                
        except Exception as e:
            logger.error(f"Face blending failed: {e}")
            return None
    
    def _calculate_mask_center(self, mask: np.ndarray) -> Tuple[int, int]:
        """
        Calculate center point of mask region.
        
        Args:
            mask: Binary mask
            
        Returns:
            Center coordinates (x, y)
        """
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Use largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return (cx, cy)
        
        # Fallback: use mask centroid
        y_coords, x_coords = np.where(mask > 0)
        if len(x_coords) > 0 and len(y_coords) > 0:
            cx = int(np.mean(x_coords))
            cy = int(np.mean(y_coords))
            return (cx, cy)
        
        # Last resort: image center
        return (mask.shape[1] // 2, mask.shape[0] // 2)
    
    def _alpha_blend(self, source_image: np.ndarray, target_image: np.ndarray,
                    mask: np.ndarray) -> np.ndarray:
        """
        Simple alpha blending with feathered edges.
        
        Args:
            source_image: Source image
            target_image: Target image
            mask: Blending mask
            
        Returns:
            Alpha blended image
        """
        # Create feathered mask
        feathered_mask = self._create_feathered_mask(mask)
        
        # Convert to 3-channel mask
        if len(feathered_mask.shape) == 2:
            feathered_mask = np.stack([feathered_mask] * 3, axis=2)
        
        # Normalize mask to [0, 1]
        feathered_mask = feathered_mask.astype(np.float32) / 255.0
        
        # Blend images
        blended = (source_image * feathered_mask + 
                  target_image * (1 - feathered_mask)).astype(np.uint8)
        
        return blended
    
    def _poisson_blend(self, source_image: np.ndarray, target_image: np.ndarray,
                      mask: np.ndarray, center: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        Poisson blending using OpenCV's seamlessClone.
        
        Args:
            source_image: Source image
            target_image: Target image
            mask: Blending mask
            center: Center point for blending
            
        Returns:
            Poisson blended image or None if failed
        """
        try:
            # Ensure mask is 8-bit single channel
            if len(mask.shape) == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            
            mask = mask.astype(np.uint8)
            
            # Apply Poisson blending
            blended = cv2.seamlessClone(
                source_image, target_image, mask, center, cv2.NORMAL_CLONE
            )
            
            return blended
            
        except Exception as e:
            logger.warning(f"Poisson blending failed, falling back to alpha blend: {e}")
            return self._alpha_blend(source_image, target_image, mask)
    
    def _multiband_blend(self, source_image: np.ndarray, target_image: np.ndarray,
                        mask: np.ndarray) -> np.ndarray:
        """
        Multi-band blending using Laplacian pyramids.
        
        Args:
            source_image: Source image
            target_image: Target image
            mask: Blending mask
            
        Returns:
            Multi-band blended image
        """
        try:
            # Convert images to float
            src_float = source_image.astype(np.float32) / 255.0
            tgt_float = target_image.astype(np.float32) / 255.0
            mask_float = mask.astype(np.float32) / 255.0
            
            # Create Gaussian pyramid for mask
            levels = 6
            mask_pyramid = self._create_gaussian_pyramid(mask_float, levels)
            
            # Create Laplacian pyramids for images
            src_pyramid = self._create_laplacian_pyramid(src_float, levels)
            tgt_pyramid = self._create_laplacian_pyramid(tgt_float, levels)
            
            # Blend pyramids
            blended_pyramid = []
            for i in range(levels):
                # Ensure mask has same dimensions as pyramid level
                if len(mask_pyramid[i].shape) == 2:
                    mask_level = np.stack([mask_pyramid[i]] * 3, axis=2)
                else:
                    mask_level = mask_pyramid[i]
                
                # Resize mask if needed
                if mask_level.shape[:2] != src_pyramid[i].shape[:2]:
                    mask_level = cv2.resize(mask_level, 
                                          (src_pyramid[i].shape[1], src_pyramid[i].shape[0]))
                
                # Blend pyramid level
                blended_level = (src_pyramid[i] * mask_level + 
                               tgt_pyramid[i] * (1 - mask_level))
                blended_pyramid.append(blended_level)
            
            # Reconstruct image from pyramid
            blended = self._reconstruct_from_laplacian_pyramid(blended_pyramid)
            
            # Convert back to uint8
            blended = np.clip(blended * 255, 0, 255).astype(np.uint8)
            
            return blended
            
        except Exception as e:
            logger.warning(f"Multi-band blending failed, falling back to alpha blend: {e}")
            return self._alpha_blend(source_image, target_image, mask)
    
    def _gradient_blend(self, source_image: np.ndarray, target_image: np.ndarray,
                       mask: np.ndarray) -> np.ndarray:
        """
        Gradient domain blending.
        
        Args:
            source_image: Source image
            target_image: Target image
            mask: Blending mask
            
        Returns:
            Gradient blended image
        """
        try:
            result = target_image.copy().astype(np.float32)
            
            # Process each color channel
            for c in range(source_image.shape[2]):
                # Calculate gradients
                src_grad_x = cv2.Sobel(source_image[:, :, c], cv2.CV_32F, 1, 0, ksize=3)
                src_grad_y = cv2.Sobel(source_image[:, :, c], cv2.CV_32F, 0, 1, ksize=3)
                
                tgt_grad_x = cv2.Sobel(target_image[:, :, c], cv2.CV_32F, 1, 0, ksize=3)
                tgt_grad_y = cv2.Sobel(target_image[:, :, c], cv2.CV_32F, 0, 1, ksize=3)
                
                # Blend gradients based on mask
                mask_norm = mask.astype(np.float32) / 255.0
                blended_grad_x = src_grad_x * mask_norm + tgt_grad_x * (1 - mask_norm)
                blended_grad_y = src_grad_y * mask_norm + tgt_grad_y * (1 - mask_norm)
                
                # Reconstruct from gradients (simplified approach)
                # In a full implementation, this would use Poisson solver
                reconstruction = cv2.integral(blended_grad_x) + cv2.integral(blended_grad_y)
                
                # Normalize and apply
                if reconstruction.size > 0:
                    reconstruction = reconstruction[1:, 1:]  # Remove integral padding
                    reconstruction = cv2.resize(reconstruction, (result.shape[1], result.shape[0]))
                    
                    # Blend with original
                    result[:, :, c] = (reconstruction * mask_norm + 
                                     result[:, :, c] * (1 - mask_norm))
            
            return np.clip(result, 0, 255).astype(np.uint8)
            
        except Exception as e:
            logger.warning(f"Gradient blending failed, falling back to alpha blend: {e}")
            return self._alpha_blend(source_image, target_image, mask)
    
    def _seamless_blend(self, source_image: np.ndarray, target_image: np.ndarray,
                       mask: np.ndarray, center: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        Advanced seamless blending with color correction.
        
        Args:
            source_image: Source image
            target_image: Target image
            mask: Blending mask
            center: Center point for blending
            
        Returns:
            Seamlessly blended image
        """
        try:
            # First apply color correction
            color_corrected = self._color_match(source_image, target_image, mask)
            
            # Then apply Poisson blending
            blended = self._poisson_blend(color_corrected, target_image, mask, center)
            
            if blended is None:
                # Fallback to multi-band blending
                blended = self._multiband_blend(color_corrected, target_image, mask)
            
            return blended
            
        except Exception as e:
            logger.warning(f"Seamless blending failed, falling back to alpha blend: {e}")
            return self._alpha_blend(source_image, target_image, mask)
    
    def _create_feathered_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Create feathered (soft-edge) mask.
        
        Args:
            mask: Binary mask
            
        Returns:
            Feathered mask
        """
        if self.feathering <= 0:
            return mask
        
        # Apply Gaussian blur for feathering
        kernel_size = max(3, self.feathering * 2 + 1)
        feathered = cv2.GaussianBlur(mask.astype(np.float32), 
                                   (kernel_size, kernel_size), 
                                   self.feathering / 3.0)
        
        return (feathered * 255).astype(np.uint8)
    
    def _create_gaussian_pyramid(self, image: np.ndarray, levels: int) -> list:
        """
        Create Gaussian pyramid.
        
        Args:
            image: Input image
            levels: Number of pyramid levels
            
        Returns:
            List of pyramid levels
        """
        pyramid = [image.copy()]
        
        for i in range(levels - 1):
            current = pyramid[-1]
            down = cv2.pyrDown(current)
            pyramid.append(down)
        
        return pyramid
    
    def _create_laplacian_pyramid(self, image: np.ndarray, levels: int) -> list:
        """
        Create Laplacian pyramid.
        
        Args:
            image: Input image
            levels: Number of pyramid levels
            
        Returns:
            List of Laplacian pyramid levels
        """
        gaussian_pyramid = self._create_gaussian_pyramid(image, levels)
        laplacian_pyramid = []
        
        for i in range(levels - 1):
            current = gaussian_pyramid[i]
            next_level = gaussian_pyramid[i + 1]
            
            # Upsample next level
            upsampled = cv2.pyrUp(next_level)
            
            # Ensure same size
            if upsampled.shape[:2] != current.shape[:2]:
                upsampled = cv2.resize(upsampled, (current.shape[1], current.shape[0]))
            
            # Calculate Laplacian
            laplacian = current - upsampled
            laplacian_pyramid.append(laplacian)
        
        # Add the last Gaussian level
        laplacian_pyramid.append(gaussian_pyramid[-1])
        
        return laplacian_pyramid
    
    def _reconstruct_from_laplacian_pyramid(self, pyramid: list) -> np.ndarray:
        """
        Reconstruct image from Laplacian pyramid.
        
        Args:
            pyramid: Laplacian pyramid levels
            
        Returns:
            Reconstructed image
        """
        current = pyramid[-1]  # Start with smallest level
        
        for i in range(len(pyramid) - 2, -1, -1):
            # Upsample current level
            upsampled = cv2.pyrUp(current)
            
            # Ensure same size as target level
            target = pyramid[i]
            if upsampled.shape[:2] != target.shape[:2]:
                upsampled = cv2.resize(upsampled, (target.shape[1], target.shape[0]))
            
            # Add Laplacian level
            current = upsampled + target
        
        return current
    
    def _color_match(self, source_image: np.ndarray, target_image: np.ndarray,
                    mask: np.ndarray) -> np.ndarray:
        """
        Match colors between source and target in masked region.
        
        Args:
            source_image: Source image
            target_image: Target image
            mask: Region mask
            
        Returns:
            Color-matched source image
        """
        try:
            result = source_image.copy().astype(np.float32)
            mask_bool = mask > 127
            
            # Process each color channel
            for c in range(source_image.shape[2]):
                src_channel = source_image[:, :, c].astype(np.float32)
                tgt_channel = target_image[:, :, c].astype(np.float32)
                
                # Calculate statistics in masked region
                src_masked = src_channel[mask_bool]
                tgt_masked = tgt_channel[mask_bool]
                
                if len(src_masked) > 0 and len(tgt_masked) > 0:
                    src_mean = np.mean(src_masked)
                    src_std = np.std(src_masked)
                    tgt_mean = np.mean(tgt_masked)
                    tgt_std = np.std(tgt_masked)
                    
                    # Apply color transfer
                    if src_std > 0:
                        matched = (src_channel - src_mean) * (tgt_std / src_std) + tgt_mean
                        result[:, :, c] = matched
            
            return np.clip(result, 0, 255).astype(np.uint8)
            
        except Exception as e:
            logger.warning(f"Color matching failed: {e}")
            return source_image
    
    def create_face_mask(self, landmarks: np.ndarray, image_shape: Tuple[int, int],
                        region: str = "full") -> np.ndarray:
        """
        Create face mask from landmarks.
        
        Args:
            landmarks: Facial landmarks
            image_shape: Image dimensions (height, width)
            region: Face region ("full", "inner", "lower")
            
        Returns:
            Face mask
        """
        try:
            mask = np.zeros(image_shape[:2], dtype=np.uint8)
            
            if landmarks.shape[0] < 68:
                logger.warning("Insufficient landmarks for mask creation")
                return mask
            
            # Define different face regions
            if region == "full":
                # Full face including jaw
                contour_indices = list(range(0, 17)) + list(range(17, 27)) + [27, 28, 29, 30]
            elif region == "inner":
                # Inner face excluding jaw
                contour_indices = list(range(17, 27)) + list(range(27, 36)) + list(range(36, 48)) + list(range(48, 60))
            elif region == "lower":
                # Lower face including mouth and chin
                contour_indices = list(range(4, 13)) + list(range(48, 68))
            else:
                contour_indices = list(range(0, 17))
            
            # Create contour
            contour_points = landmarks[contour_indices].astype(np.int32)
            
            # Fill mask
            cv2.fillPoly(mask, [contour_points], 255)
            
            return mask
            
        except Exception as e:
            logger.error(f"Face mask creation failed: {e}")
            return np.zeros(image_shape[:2], dtype=np.uint8)