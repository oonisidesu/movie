import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import logging

from .detector import FaceDetector

logger = logging.getLogger(__name__)


class FaceTracker:
    def __init__(self, 
                 detector: Optional[FaceDetector] = None,
                 max_missing_frames: int = 10,
                 iou_threshold: float = 0.5):
        self.detector = detector or FaceDetector()
        self.max_missing_frames = max_missing_frames
        self.iou_threshold = iou_threshold
        
        self.tracks = {}  # Track ID -> Track info
        self.next_track_id = 0
        self.frame_count = 0
        
    def process_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        Process a single frame and track faces.
        
        Args:
            frame: Input frame
            
        Returns:
            List of tracked faces with consistent IDs
        """
        # Detect faces in current frame
        detections = self.detector.detect_faces(frame)
        
        # Match detections to existing tracks
        matched_tracks = self._match_detections_to_tracks(detections)
        
        # Update tracks
        self._update_tracks(detections, matched_tracks)
        
        # Increment frame counter
        self.frame_count += 1
        
        # Return active tracks
        return self._get_active_tracks()
    
    def _match_detections_to_tracks(self, detections: List[Dict]) -> Dict[int, int]:
        """
        Match current detections to existing tracks using IoU.
        
        Returns:
            Dictionary mapping detection index to track ID
        """
        if not self.tracks or not detections:
            return {}
        
        matched = {}
        used_tracks = set()
        
        # Calculate IoU matrix
        for det_idx, detection in enumerate(detections):
            best_iou = 0
            best_track_id = None
            
            for track_id, track in self.tracks.items():
                if track_id in used_tracks:
                    continue
                    
                iou = self._calculate_iou(detection['bbox'], track['bbox'])
                
                if iou > self.iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id
            
            if best_track_id is not None:
                matched[det_idx] = best_track_id
                used_tracks.add(best_track_id)
        
        return matched
    
    def _calculate_iou(self, bbox1: Tuple[int, int, int, int], 
                      bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union between two bounding boxes."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union
        bbox1_area = w1 * h1
        bbox2_area = w2 * h2
        union_area = bbox1_area + bbox2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0
    
    def _update_tracks(self, detections: List[Dict], matched: Dict[int, int]):
        """Update existing tracks and create new ones."""
        # Update matched tracks
        for det_idx, track_id in matched.items():
            detection = detections[det_idx]
            self.tracks[track_id].update({
                'bbox': detection['bbox'],
                'confidence': detection['confidence'],
                'landmarks': detection.get('landmarks'),
                'last_seen': self.frame_count,
                'missing_frames': 0,
                'history': self.tracks[track_id]['history'] + [detection['bbox']]
            })
        
        # Create new tracks for unmatched detections
        for det_idx, detection in enumerate(detections):
            if det_idx not in matched:
                self.tracks[self.next_track_id] = {
                    'id': self.next_track_id,
                    'bbox': detection['bbox'],
                    'confidence': detection['confidence'],
                    'landmarks': detection.get('landmarks'),
                    'first_seen': self.frame_count,
                    'last_seen': self.frame_count,
                    'missing_frames': 0,
                    'history': [detection['bbox']]
                }
                self.next_track_id += 1
        
        # Update missing frames for unmatched tracks
        matched_track_ids = set(matched.values())
        for track_id in list(self.tracks.keys()):
            if track_id not in matched_track_ids:
                self.tracks[track_id]['missing_frames'] += 1
                
                # Remove tracks that have been missing for too long
                if self.tracks[track_id]['missing_frames'] > self.max_missing_frames:
                    del self.tracks[track_id]
    
    def _get_active_tracks(self) -> List[Dict]:
        """Get list of currently active tracks."""
        active_tracks = []
        
        for track_id, track in self.tracks.items():
            if track['missing_frames'] == 0:
                active_tracks.append({
                    'track_id': track_id,
                    'bbox': track['bbox'],
                    'confidence': track['confidence'],
                    'landmarks': track.get('landmarks'),
                    'age': self.frame_count - track['first_seen'],
                    'history_length': len(track['history'])
                })
        
        return active_tracks
    
    def get_track_history(self, track_id: int) -> Optional[List[Tuple[int, int, int, int]]]:
        """Get movement history of a specific track."""
        if track_id in self.tracks:
            return self.tracks[track_id]['history']
        return None
    
    def reset(self):
        """Reset tracker state."""
        self.tracks = {}
        self.next_track_id = 0
        self.frame_count = 0


class MultiPersonTracker:
    def __init__(self, detector: Optional[FaceDetector] = None):
        self.detector = detector or FaceDetector()
        self.trackers = {}  # Person ID -> FaceTracker
        self.person_assignments = {}  # Track ID -> Person ID
        
    def assign_person_to_track(self, track_id: int, person_id: str):
        """Manually assign a person ID to a track."""
        self.person_assignments[track_id] = person_id
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, List[Dict]]:
        """
        Process frame and return faces grouped by person.
        
        Returns:
            Dictionary mapping person ID to list of face detections
        """
        # Get all face tracks
        if 'main' not in self.trackers:
            self.trackers['main'] = FaceTracker(self.detector)
        
        tracks = self.trackers['main'].process_frame(frame)
        
        # Group by person
        result = defaultdict(list)
        for track in tracks:
            track_id = track['track_id']
            person_id = self.person_assignments.get(track_id, f"unknown_{track_id}")
            result[person_id].append(track)
        
        return dict(result)