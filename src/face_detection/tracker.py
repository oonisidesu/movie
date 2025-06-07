"""
Face Tracking Module - FaceTracker Class

This module provides face tracking capabilities across video frames.
It maintains consistent face IDs throughout a video sequence and handles
face appearance/disappearance scenarios.

Author: Face Swapping Tool
License: Personal/Educational Use Only
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict, deque
import time

from .detector import FaceDetector, FaceDetection, DetectionBackend


@dataclass
class TrackedFace:
    """Container for a tracked face across frames."""
    
    face_id: int
    detection: FaceDetection
    frame_number: int
    last_seen: int
    track_length: int = 1
    velocity: Tuple[float, float] = (0.0, 0.0)
    confidence_history: List[float] = None
    
    def __post_init__(self):
        if self.confidence_history is None:
            self.confidence_history = [self.detection.confidence]
    
    @property
    def avg_confidence(self) -> float:
        """Average confidence over tracking history."""
        return sum(self.confidence_history) / len(self.confidence_history)
    
    @property
    def stability_score(self) -> float:
        """Calculate stability based on confidence variance and track length."""
        if len(self.confidence_history) < 2:
            return self.detection.confidence
        
        # Lower variance = higher stability
        variance = np.var(self.confidence_history)
        stability = 1.0 / (1.0 + variance)
        
        # Longer tracks are more stable
        length_factor = min(1.0, self.track_length / 30.0)
        
        return stability * length_factor


class FaceTracker:
    """
    Face tracking class for maintaining consistent face IDs across video frames.
    
    This class uses a combination of spatial proximity, appearance similarity,
    and temporal consistency to track faces throughout a video sequence.
    """
    
    def __init__(self, 
                 detector: Optional[FaceDetector] = None,
                 max_distance: float = 50.0,
                 max_disappeared: int = 10,
                 min_track_length: int = 3,
                 iou_threshold: float = 0.3):
        """
        Initialize the face tracker.
        
        Args:
            detector: FaceDetector instance (creates default if None)
            max_distance: Maximum distance for face association
            max_disappeared: Maximum frames a face can be missing before removal
            min_track_length: Minimum frames to consider a track valid
            iou_threshold: Minimum IoU for face association
        """
        self.detector = detector or FaceDetector(DetectionBackend.OPENCV_DNN)
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared
        self.min_track_length = min_track_length
        self.iou_threshold = iou_threshold
        
        # Tracking state
        self.tracked_faces: Dict[int, TrackedFace] = {}
        self.next_face_id = 0
        self.frame_number = 0
        
        # History for trajectory prediction
        self.position_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=5)
        )
        
        self.logger = logging.getLogger(__name__)
        
        # Performance metrics
        self.processing_times = deque(maxlen=100)
    
    def update(self, frame: np.ndarray) -> List[TrackedFace]:
        """
        Update tracker with new frame and return tracked faces.
        
        Args:
            frame: Input video frame
            
        Returns:
            List of currently tracked faces
        """
        start_time = time.time()
        
        # Detect faces in current frame
        detections = self.detector.detect_faces(frame)
        
        # Update tracking
        self._update_tracking(detections)
        
        # Increment frame counter
        self.frame_number += 1
        
        # Record processing time
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        # Return currently active tracks
        return self.get_active_tracks()
    
    def _update_tracking(self, detections: List[FaceDetection]) -> None:
        """Update tracking state with new detections."""
        
        # If no current tracks, initialize with detections
        if not self.tracked_faces:
            for detection in detections:
                self._create_new_track(detection)
            return
        
        # Match detections to existing tracks
        matches, unmatched_detections, unmatched_tracks = self._associate_detections(
            detections
        )
        
        # Update matched tracks
        for track_id, detection in matches:
            self._update_track(track_id, detection)
        
        # Create new tracks for unmatched detections
        for detection in unmatched_detections:
            self._create_new_track(detection)
        
        # Handle disappeared tracks
        for track_id in unmatched_tracks:
            self._handle_disappeared_track(track_id)
    
    def _associate_detections(self, detections: List[FaceDetection]) -> Tuple[
        List[Tuple[int, FaceDetection]], 
        List[FaceDetection], 
        List[int]
    ]:
        """
        Associate detections with existing tracks.
        
        Returns:
            (matches, unmatched_detections, unmatched_tracks)
        """
        if not detections:
            return [], [], list(self.tracked_faces.keys())
        
        if not self.tracked_faces:
            return [], detections, []
        
        # Calculate association costs
        track_ids = list(self.tracked_faces.keys())
        cost_matrix = np.zeros((len(track_ids), len(detections)))
        
        for i, track_id in enumerate(track_ids):
            tracked_face = self.tracked_faces[track_id]
            predicted_pos = self._predict_position(track_id)
            
            for j, detection in enumerate(detections):
                cost = self._calculate_association_cost(
                    tracked_face, detection, predicted_pos
                )
                cost_matrix[i, j] = cost
        
        # Perform assignment using simple greedy matching
        # In production, you might want to use Hungarian algorithm
        matches = []
        unmatched_detections = list(detections)
        unmatched_tracks = list(track_ids)
        
        # Greedy assignment: find minimum cost associations
        while cost_matrix.size > 0 and np.min(cost_matrix) < float('inf'):
            min_idx = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
            track_idx, det_idx = min_idx
            
            track_id = track_ids[track_idx]
            detection = detections[det_idx]
            
            # Check if assignment is valid
            if cost_matrix[track_idx, det_idx] < self.max_distance:
                matches.append((track_id, detection))
                unmatched_tracks.remove(track_id)
                unmatched_detections.remove(detection)
            
            # Remove this track and detection from consideration
            cost_matrix = np.delete(cost_matrix, track_idx, axis=0)
            cost_matrix = np.delete(cost_matrix, det_idx, axis=1)
            track_ids.pop(track_idx)
            detections.pop(det_idx)
        
        return matches, unmatched_detections, unmatched_tracks
    
    def _calculate_association_cost(self, 
                                  tracked_face: TrackedFace,
                                  detection: FaceDetection,
                                  predicted_pos: Optional[Tuple[int, int]] = None) -> float:
        """
        Calculate cost for associating a detection with a tracked face.
        
        Args:
            tracked_face: Existing tracked face
            detection: New detection
            predicted_pos: Predicted position for the tracked face
            
        Returns:
            Association cost (lower is better)
        """
        # Use predicted position if available, otherwise last known position
        if predicted_pos:
            ref_center = predicted_pos
        else:
            ref_center = tracked_face.detection.center
        
        det_center = detection.center
        
        # Euclidean distance
        distance = np.sqrt(
            (ref_center[0] - det_center[0]) ** 2 + 
            (ref_center[1] - det_center[1]) ** 2
        )
        
        # IoU similarity (converted to distance)
        iou = self._calculate_iou(tracked_face.detection, detection)
        iou_distance = 1.0 - iou
        
        # Size similarity
        size_ratio = min(
            detection.area / max(tracked_face.detection.area, 1),
            tracked_face.detection.area / max(detection.area, 1)
        )
        size_distance = 1.0 - size_ratio
        
        # Combine costs with weights
        total_cost = (
            0.5 * distance +
            0.3 * iou_distance * 100 +  # Scale IoU to similar range
            0.2 * size_distance * 100
        )
        
        # Penalize if above thresholds
        if distance > self.max_distance or iou < self.iou_threshold:
            total_cost = float('inf')
        
        return total_cost
    
    def _calculate_iou(self, det1: FaceDetection, det2: FaceDetection) -> float:
        """Calculate Intersection over Union (IoU) between two detections."""
        x1_inter = max(det1.x, det2.x)
        y1_inter = max(det1.y, det2.y)
        x2_inter = min(det1.x + det1.width, det2.x + det2.width)
        y2_inter = min(det1.y + det1.height, det2.y + det2.height)
        
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
        
        intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        union = det1.area + det2.area - intersection
        
        return intersection / max(union, 1)
    
    def _predict_position(self, track_id: int) -> Optional[Tuple[int, int]]:
        """Predict next position based on velocity."""
        if track_id not in self.tracked_faces:
            return None
        
        tracked_face = self.tracked_faces[track_id]
        current_center = tracked_face.detection.center
        
        # Simple linear prediction based on velocity
        predicted_x = int(current_center[0] + tracked_face.velocity[0])
        predicted_y = int(current_center[1] + tracked_face.velocity[1])
        
        return (predicted_x, predicted_y)
    
    def _update_track(self, track_id: int, detection: FaceDetection) -> None:
        """Update existing track with new detection."""
        tracked_face = self.tracked_faces[track_id]
        
        # Calculate velocity
        old_center = tracked_face.detection.center
        new_center = detection.center
        velocity = (
            new_center[0] - old_center[0],
            new_center[1] - old_center[1]
        )
        
        # Update position history
        self.position_history[track_id].append(new_center)
        
        # Update tracked face
        tracked_face.detection = detection
        tracked_face.last_seen = self.frame_number
        tracked_face.track_length += 1
        tracked_face.velocity = velocity
        tracked_face.confidence_history.append(detection.confidence)
        
        # Limit confidence history size
        if len(tracked_face.confidence_history) > 50:
            tracked_face.confidence_history.pop(0)
    
    def _create_new_track(self, detection: FaceDetection) -> None:
        """Create new track for unmatched detection."""
        track_id = self.next_face_id
        self.next_face_id += 1
        
        tracked_face = TrackedFace(
            face_id=track_id,
            detection=detection,
            frame_number=self.frame_number,
            last_seen=self.frame_number
        )
        
        self.tracked_faces[track_id] = tracked_face
        self.position_history[track_id].append(detection.center)
        
        self.logger.debug(f"Created new track {track_id}")
    
    def _handle_disappeared_track(self, track_id: int) -> None:
        """Handle track that wasn't matched in current frame."""
        tracked_face = self.tracked_faces[track_id]
        frames_since_seen = self.frame_number - tracked_face.last_seen
        
        if frames_since_seen >= self.max_disappeared:
            # Remove track if disappeared too long
            del self.tracked_faces[track_id]
            if track_id in self.position_history:
                del self.position_history[track_id]
            
            self.logger.debug(f"Removed track {track_id} after {frames_since_seen} frames")
    
    def get_active_tracks(self) -> List[TrackedFace]:
        """Get list of currently active tracks."""
        return [
            face for face in self.tracked_faces.values()
            if face.track_length >= self.min_track_length
        ]
    
    def get_stable_tracks(self, min_stability: float = 0.5) -> List[TrackedFace]:
        """Get tracks with high stability scores."""
        active_tracks = self.get_active_tracks()
        return [
            face for face in active_tracks 
            if face.stability_score >= min_stability
        ]
    
    def get_track_by_id(self, track_id: int) -> Optional[TrackedFace]:
        """Get specific track by ID."""
        return self.tracked_faces.get(track_id)
    
    def reset(self) -> None:
        """Reset tracker state."""
        self.tracked_faces.clear()
        self.position_history.clear()
        self.next_face_id = 0
        self.frame_number = 0
        self.processing_times.clear()
        
        self.logger.info("Tracker reset")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get tracking statistics."""
        active_tracks = self.get_active_tracks()
        
        stats = {
            'total_tracks': len(self.tracked_faces),
            'active_tracks': len(active_tracks),
            'frame_number': self.frame_number,
            'next_face_id': self.next_face_id,
            'avg_processing_time': np.mean(self.processing_times) if self.processing_times else 0.0,
            'max_processing_time': max(self.processing_times) if self.processing_times else 0.0,
        }
        
        if active_tracks:
            stats.update({
                'avg_track_length': np.mean([f.track_length for f in active_tracks]),
                'avg_confidence': np.mean([f.avg_confidence for f in active_tracks]),
                'avg_stability': np.mean([f.stability_score for f in active_tracks]),
            })
        
        return stats
    
    def visualize_tracks(self, frame: np.ndarray, 
                        show_ids: bool = True,
                        show_trails: bool = False) -> np.ndarray:
        """
        Visualize tracks on frame.
        
        Args:
            frame: Input frame to draw on
            show_ids: Whether to show track IDs
            show_trails: Whether to show position trails
            
        Returns:
            Frame with visualizations
        """
        vis_frame = frame.copy()
        
        for track_id, tracked_face in self.tracked_faces.items():
            detection = tracked_face.detection
            
            # Choose color based on track stability
            if tracked_face.stability_score > 0.7:
                color = (0, 255, 0)  # Green for stable
            elif tracked_face.stability_score > 0.4:
                color = (0, 255, 255)  # Yellow for moderate
            else:
                color = (0, 0, 255)  # Red for unstable
            
            # Draw bounding box
            cv2.rectangle(
                vis_frame,
                (detection.x, detection.y),
                (detection.x + detection.width, detection.y + detection.height),
                color, 2
            )
            
            # Draw track ID
            if show_ids:
                label = f"ID: {track_id}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(
                    vis_frame,
                    (detection.x, detection.y - label_size[1] - 10),
                    (detection.x + label_size[0], detection.y),
                    color, -1
                )
                cv2.putText(
                    vis_frame, label,
                    (detection.x, detection.y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
                )
            
            # Draw position trail
            if show_trails and track_id in self.position_history:
                positions = list(self.position_history[track_id])
                for i in range(1, len(positions)):
                    cv2.line(vis_frame, positions[i-1], positions[i], color, 2)
        
        return vis_frame