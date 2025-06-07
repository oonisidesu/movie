import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class FaceDetector:
    def __init__(self, 
                 detection_confidence: float = 0.5,
                 model_selection: int = 0):
        self.detection_confidence = detection_confidence
        self.model_selection = model_selection
        
        # Initialize face detection models
        self._init_opencv_detector()
        self._init_mediapipe_detector()
        
    def _init_opencv_detector(self):
        try:
            # Try to load Haar Cascade classifier
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            # Try to load DNN-based detector
            prototxt_path = 'models/deploy.prototxt'
            model_path = 'models/res10_300x300_ssd_iter_140000.caffemodel'
            try:
                self.dnn_net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
                self.use_dnn = True
            except:
                logger.warning("DNN model not found, falling back to Haar Cascade")
                self.use_dnn = False
        except Exception as e:
            logger.error(f"Failed to initialize OpenCV detector: {e}")
            raise
    
    def _init_mediapipe_detector(self):
        try:
            import mediapipe as mp
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_face_mesh = mp.solutions.face_mesh
            self.mp_drawing = mp.solutions.drawing_utils
            self.use_mediapipe = True
        except ImportError:
            logger.warning("MediaPipe not installed, some features will be limited")
            self.use_mediapipe = False
    
    def detect_faces(self, image: np.ndarray, method: str = 'auto') -> List[Dict]:
        """
        Detect faces in an image using specified method.
        
        Args:
            image: Input image as numpy array
            method: Detection method ('opencv', 'mediapipe', 'auto')
            
        Returns:
            List of dictionaries containing face information
        """
        if method == 'auto':
            if self.use_mediapipe:
                faces = self._detect_faces_mediapipe(image)
                if len(faces) == 0 and self.use_dnn:
                    faces = self._detect_faces_dnn(image)
            else:
                faces = self._detect_faces_opencv(image)
        elif method == 'opencv':
            faces = self._detect_faces_opencv(image)
        elif method == 'mediapipe' and self.use_mediapipe:
            faces = self._detect_faces_mediapipe(image)
        else:
            faces = self._detect_faces_opencv(image)
            
        return faces
    
    def _detect_faces_opencv(self, image: np.ndarray) -> List[Dict]:
        """Detect faces using OpenCV (Haar Cascade or DNN)"""
        if self.use_dnn:
            return self._detect_faces_dnn(image)
        else:
            return self._detect_faces_haar(image)
    
    def _detect_faces_haar(self, image: np.ndarray) -> List[Dict]:
        """Detect faces using Haar Cascade"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        results = []
        for i, (x, y, w, h) in enumerate(faces):
            results.append({
                'id': i,
                'bbox': (x, y, w, h),
                'confidence': 1.0,  # Haar doesn't provide confidence
                'landmarks': None
            })
        return results
    
    def _detect_faces_dnn(self, image: np.ndarray) -> List[Dict]:
        """Detect faces using DNN (Deep Neural Network)"""
        h, w = image.shape[:2]
        
        # Prepare input blob
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 
            1.0, 
            (300, 300), 
            (104.0, 177.0, 123.0)
        )
        
        self.dnn_net.setInput(blob)
        detections = self.dnn_net.forward()
        
        results = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > self.detection_confidence:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x, y, x2, y2 = box.astype("int")
                
                results.append({
                    'id': i,
                    'bbox': (x, y, x2-x, y2-y),
                    'confidence': float(confidence),
                    'landmarks': None
                })
        
        return results
    
    def _detect_faces_mediapipe(self, image: np.ndarray) -> List[Dict]:
        """Detect faces using MediaPipe"""
        if not self.use_mediapipe:
            return []
            
        results = []
        with self.mp_face_detection.FaceDetection(
            model_selection=self.model_selection,
            min_detection_confidence=self.detection_confidence
        ) as face_detection:
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_results = face_detection.process(image_rgb)
            
            if mp_results.detections:
                h, w = image.shape[:2]
                
                for i, detection in enumerate(mp_results.detections):
                    bbox = detection.location_data.relative_bounding_box
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    
                    # Extract key points
                    landmarks = []
                    if detection.location_data.relative_keypoints:
                        for keypoint in detection.location_data.relative_keypoints:
                            landmarks.append((
                                int(keypoint.x * w),
                                int(keypoint.y * h)
                            ))
                    
                    results.append({
                        'id': i,
                        'bbox': (x, y, width, height),
                        'confidence': detection.score[0],
                        'landmarks': landmarks if landmarks else None
                    })
        
        return results
    
    def draw_faces(self, image: np.ndarray, faces: List[Dict], 
                   draw_landmarks: bool = True) -> np.ndarray:
        """
        Draw detected faces on image.
        
        Args:
            image: Input image
            faces: List of detected faces
            draw_landmarks: Whether to draw facial landmarks
            
        Returns:
            Image with drawn faces
        """
        output = image.copy()
        
        for face in faces:
            x, y, w, h = face['bbox']
            confidence = face.get('confidence', 1.0)
            
            # Draw bounding box
            color = (0, 255, 0) if confidence > 0.8 else (0, 255, 255)
            cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
            
            # Draw confidence score
            label = f"Face {face['id']}: {confidence:.2f}"
            cv2.putText(output, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw landmarks if available
            if draw_landmarks and face.get('landmarks'):
                for point in face['landmarks']:
                    cv2.circle(output, point, 2, (0, 0, 255), -1)
        
        return output
    
    def get_face_region(self, image: np.ndarray, face: Dict, 
                       padding: float = 0.2) -> np.ndarray:
        """
        Extract face region from image with padding.
        
        Args:
            image: Input image
            face: Face detection result
            padding: Padding ratio around face
            
        Returns:
            Cropped face region
        """
        x, y, w, h = face['bbox']
        
        # Add padding
        pad_w = int(w * padding)
        pad_h = int(h * padding)
        
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(image.shape[1], x + w + pad_w)
        y2 = min(image.shape[0], y + h + pad_h)
        
        return image[y1:y2, x1:x2]