"""
Delaunay Triangulation for Face Swapping

Implements Delaunay triangulation for precise face warping and morphing.
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional
from scipy.spatial import Delaunay

from ..face_detection.landmarks import FaceLandmarks

logger = logging.getLogger(__name__)


class FaceTriangulation:
    """
    Handles Delaunay triangulation for face swapping.
    
    Creates a triangular mesh from facial landmarks for precise warping.
    """
    
    def __init__(self, include_boundary: bool = True):
        """
        Initialize triangulation handler.
        
        Args:
            include_boundary: Whether to include image boundary points
        """
        self.include_boundary = include_boundary
        self.triangles = None
        self.points = None
    
    def create_triangulation(self, landmarks: FaceLandmarks, 
                           image_shape: Tuple[int, int]) -> Delaunay:
        """
        Create Delaunay triangulation from facial landmarks.
        
        Args:
            landmarks: Facial landmarks
            image_shape: Image dimensions (height, width)
            
        Returns:
            Delaunay triangulation object
        """
        points = landmarks.points.copy()
        
        if self.include_boundary:
            # Add image boundary points for complete coverage
            h, w = image_shape[:2]
            boundary_points = np.array([
                [0, 0], [w//2, 0], [w-1, 0],           # Top
                [w-1, h//2], [w-1, h-1],               # Right
                [w//2, h-1], [0, h-1],                 # Bottom
                [0, h//2]                               # Left
            ], dtype=np.float32)
            
            points = np.vstack([points, boundary_points])
        
        # Create triangulation
        self.points = points
        self.triangles = Delaunay(points)
        
        return self.triangles
    
    def get_triangles(self) -> List[Tuple[int, int, int]]:
        """
        Get list of triangle indices.
        
        Returns:
            List of triangles as tuples of vertex indices
        """
        if self.triangles is None:
            return []
        
        return [(tri[0], tri[1], tri[2]) for tri in self.triangles.simplices]
    
    def get_triangle_points(self, triangle_idx: int) -> np.ndarray:
        """
        Get coordinates of a specific triangle.
        
        Args:
            triangle_idx: Triangle index
            
        Returns:
            Array of shape (3, 2) with triangle vertices
        """
        if self.triangles is None or triangle_idx >= len(self.triangles.simplices):
            return np.array([])
        
        indices = self.triangles.simplices[triangle_idx]
        return self.points[indices]
    
    def find_corresponding_triangles(self, source_triangulation: 'FaceTriangulation',
                                   target_triangulation: 'FaceTriangulation') -> List[Tuple[int, int]]:
        """
        Find corresponding triangles between source and target triangulations.
        
        Args:
            source_triangulation: Source face triangulation
            target_triangulation: Target face triangulation
            
        Returns:
            List of (source_idx, target_idx) pairs
        """
        # For landmarks-based triangulation, triangles should correspond directly
        # if both use the same landmark ordering and point count
        correspondences = []
        
        # Only match triangles if point sets are identical in structure
        if (source_triangulation.points.shape == target_triangulation.points.shape):
            min_triangles = min(len(source_triangulation.triangles.simplices),
                               len(target_triangulation.triangles.simplices))
            
            for i in range(min_triangles):
                correspondences.append((i, i))
        
        return correspondences
    
    def visualize_triangulation(self, image: np.ndarray, 
                              color: Tuple[int, int, int] = (0, 255, 0),
                              thickness: int = 1) -> np.ndarray:
        """
        Visualize triangulation on image.
        
        Args:
            image: Input image
            color: Line color (BGR)
            thickness: Line thickness
            
        Returns:
            Image with drawn triangulation
        """
        result = image.copy()
        
        if self.triangles is None:
            return result
        
        # Draw all triangles
        for triangle in self.triangles.simplices:
            pts = self.points[triangle].astype(np.int32)
            
            # Draw triangle edges
            cv2.line(result, tuple(pts[0]), tuple(pts[1]), color, thickness)
            cv2.line(result, tuple(pts[1]), tuple(pts[2]), color, thickness)
            cv2.line(result, tuple(pts[2]), tuple(pts[0]), color, thickness)
        
        return result


class TriangularWarper:
    """
    Performs triangular warping for face morphing.
    """
    
    @staticmethod
    def warp_triangle(source_img: np.ndarray, target_img: np.ndarray,
                     source_tri: np.ndarray, target_tri: np.ndarray) -> None:
        """
        Warp a single triangle from source to target image.
        
        Args:
            source_img: Source image
            target_img: Target image (modified in-place)
            source_tri: Source triangle vertices (3x2)
            target_tri: Target triangle vertices (3x2)
        """
        try:
            # Validate triangle data
            if source_tri.shape != (3, 2) or target_tri.shape != (3, 2):
                return
            
            # Check for degenerate triangles
            source_area = cv2.contourArea(source_tri.astype(np.float32))
            target_area = cv2.contourArea(target_tri.astype(np.float32))
            
            if source_area < 1.0 or target_area < 1.0:
                return  # Skip degenerate triangles
            
            # Get bounding rectangles
            source_rect = cv2.boundingRect(source_tri.astype(np.float32))
            target_rect = cv2.boundingRect(target_tri.astype(np.float32))
            
            # Validate rectangles
            if source_rect[2] <= 0 or source_rect[3] <= 0 or target_rect[2] <= 0 or target_rect[3] <= 0:
                return
            
            # Offset triangles to rectangle coordinates
            source_tri_rect = source_tri - np.array([source_rect[0], source_rect[1]])
            target_tri_rect = target_tri - np.array([target_rect[0], target_rect[1]])
            
            # Validate bounds
            if (source_rect[1] + source_rect[3] > source_img.shape[0] or 
                source_rect[0] + source_rect[2] > source_img.shape[1] or
                target_rect[1] + target_rect[3] > target_img.shape[0] or
                target_rect[0] + target_rect[2] > target_img.shape[1]):
                return
            
            # Get affine transform
            transform_matrix = cv2.getAffineTransform(
                source_tri_rect.astype(np.float32),
                target_tri_rect.astype(np.float32)
            )
            
            # Extract and warp source region
            source_crop = source_img[source_rect[1]:source_rect[1] + source_rect[3],
                                    source_rect[0]:source_rect[0] + source_rect[2]]
            
            if source_crop.size == 0:
                return
            
            target_crop_size = (target_rect[2], target_rect[3])
            warped = cv2.warpAffine(source_crop, transform_matrix, target_crop_size, 
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            
            # Create triangle mask
            mask = np.zeros((target_rect[3], target_rect[2]), dtype=np.uint8)
            cv2.fillPoly(mask, [target_tri_rect.astype(np.int32)], 255)
            
            # Apply to target image
            target_region = target_img[target_rect[1]:target_rect[1] + target_rect[3],
                                      target_rect[0]:target_rect[0] + target_rect[2]]
            
            if warped.shape[:2] == target_region.shape[:2] == mask.shape:
                # Apply mask
                mask_bool = mask > 0
                for c in range(target_img.shape[2]):
                    target_region[:, :, c] = np.where(
                        mask_bool,
                        warped[:, :, c],
                        target_region[:, :, c]
                    )
                    
        except Exception as e:
            # Silently skip problematic triangles
            pass
    
    @staticmethod
    def warp_face(source_img: np.ndarray, target_img: np.ndarray,
                 source_landmarks: FaceLandmarks, target_landmarks: FaceLandmarks) -> np.ndarray:
        """
        Warp entire face using triangular warping.
        
        Args:
            source_img: Source image
            target_img: Target image
            source_landmarks: Source facial landmarks
            target_landmarks: Target facial landmarks
            
        Returns:
            Warped result image
        """
        result = target_img.copy()
        
        # Create triangulations
        source_triangulation = FaceTriangulation()
        target_triangulation = FaceTriangulation()
        
        source_triangulation.create_triangulation(source_landmarks, source_img.shape)
        target_triangulation.create_triangulation(target_landmarks, target_img.shape)
        
        # Find corresponding triangles
        correspondences = source_triangulation.find_corresponding_triangles(
            source_triangulation, target_triangulation
        )
        
        # Warp each triangle
        for source_idx, target_idx in correspondences:
            source_tri = source_triangulation.get_triangle_points(source_idx)
            target_tri = target_triangulation.get_triangle_points(target_idx)
            
            if source_tri.size > 0 and target_tri.size > 0:
                TriangularWarper.warp_triangle(
                    source_img, result, source_tri, target_tri
                )
        
        return result


def create_face_mesh(landmarks: FaceLandmarks, image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Create a face mesh suitable for morphing.
    
    Args:
        landmarks: Facial landmarks
        image_shape: Image dimensions
        
    Returns:
        Array of mesh points
    """
    # Start with landmark points
    mesh_points = landmarks.points.copy()
    
    # Add intermediate points for smoother morphing
    # Add points between eyes
    left_eye_center = np.mean(landmarks.get_region('left_eye'), axis=0)
    right_eye_center = np.mean(landmarks.get_region('right_eye'), axis=0)
    between_eyes = (left_eye_center + right_eye_center) / 2
    
    # Add forehead points
    eyebrow_center = np.mean(np.vstack([
        landmarks.get_region('left_eyebrow'),
        landmarks.get_region('right_eyebrow')
    ]), axis=0)
    
    forehead_point = eyebrow_center.copy()
    forehead_point[1] -= 30  # Move up
    
    # Add cheek points
    nose_bottom = np.array(landmarks.get_point(33))
    jaw_left = np.array(landmarks.get_point(3))
    jaw_right = np.array(landmarks.get_point(13))
    
    left_cheek = (nose_bottom + jaw_left) / 2
    right_cheek = (nose_bottom + jaw_right) / 2
    
    # Combine all points
    additional_points = np.array([
        between_eyes,
        forehead_point,
        left_cheek,
        right_cheek
    ])
    
    mesh_points = np.vstack([mesh_points, additional_points])
    
    # Add boundary points
    h, w = image_shape[:2]
    boundary_points = np.array([
        [0, 0], [w//4, 0], [w//2, 0], [3*w//4, 0], [w-1, 0],
        [w-1, h//4], [w-1, h//2], [w-1, 3*h//4], [w-1, h-1],
        [3*w//4, h-1], [w//2, h-1], [w//4, h-1], [0, h-1],
        [0, 3*h//4], [0, h//2], [0, h//4]
    ], dtype=np.float32)
    
    mesh_points = np.vstack([mesh_points, boundary_points])
    
    return mesh_points