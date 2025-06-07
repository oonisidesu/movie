import unittest
import numpy as np
import cv2
import time
from unittest.mock import Mock, patch
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from video.frame_processor import (
    FrameProcessor,
    FaceSwappingFrameProcessor,
    FrameProcessingStats,
    resize_frame,
    normalize_frame,
    denoise_frame,
    enhance_contrast
)


class TestFrameProcessingStats(unittest.TestCase):
    
    def test_init(self):
        """Test FrameProcessingStats initialization."""
        stats = FrameProcessingStats()
        
        self.assertEqual(stats.total_frames, 0)
        self.assertEqual(stats.processed_frames, 0)
        self.assertEqual(stats.failed_frames, 0)
        self.assertEqual(stats.total_processing_time, 0.0)
        self.assertEqual(stats.average_processing_time, 0.0)
        self.assertEqual(stats.fps, 0.0)


class TestFrameProcessor(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = FrameProcessor(enable_stats=True)
        
        # Create test frames
        self.test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        self.empty_frame = np.array([])
        
    def test_init(self):
        """Test FrameProcessor initialization."""
        processor = FrameProcessor(enable_stats=False)
        
        self.assertFalse(processor.enable_stats)
        self.assertIsInstance(processor.stats, FrameProcessingStats)
        self.assertEqual(len(processor.processors), 0)
        self.assertEqual(len(processor.preprocessing_functions), 0)
        self.assertEqual(len(processor.postprocessing_functions), 0)
        self.assertFalse(processor.resize_frames)
        self.assertIsNone(processor.target_size)
        self.assertEqual(processor.color_space, 'BGR')
    
    def test_add_preprocessor(self):
        """Test adding preprocessor function."""
        def test_preprocess(frame):
            return frame
        
        self.processor.add_preprocessor(test_preprocess)
        
        self.assertEqual(len(self.processor.preprocessing_functions), 1)
        self.assertEqual(self.processor.preprocessing_functions[0], test_preprocess)
    
    def test_add_processor(self):
        """Test adding main processor function."""
        def test_process(frame):
            return frame
        
        self.processor.add_processor(test_process)
        
        self.assertEqual(len(self.processor.processors), 1)
        self.assertEqual(self.processor.processors[0], test_process)
    
    def test_add_postprocessor(self):
        """Test adding postprocessor function."""
        def test_postprocess(frame):
            return frame
        
        self.processor.add_postprocessor(test_postprocess)
        
        self.assertEqual(len(self.processor.postprocessing_functions), 1)
        self.assertEqual(self.processor.postprocessing_functions[0], test_postprocess)
    
    def test_set_resize_target(self):
        """Test setting resize target."""
        self.processor.set_resize_target(320, 240)
        
        self.assertTrue(self.processor.resize_frames)
        self.assertEqual(self.processor.target_size, (320, 240))
    
    def test_disable_resize(self):
        """Test disabling resize."""
        self.processor.set_resize_target(320, 240)
        self.processor.disable_resize()
        
        self.assertFalse(self.processor.resize_frames)
        self.assertIsNone(self.processor.target_size)
    
    def test_process_frame_simple(self):
        """Test simple frame processing."""
        processed = self.processor.process_frame(self.test_frame)
        
        # Should return the same frame if no processors added
        np.testing.assert_array_equal(processed, self.test_frame)
    
    def test_process_frame_with_processors(self):
        """Test frame processing with processors."""
        def add_border(frame):
            bordered = frame.copy()
            bordered[0:5, :] = [255, 0, 0]  # Red border
            return bordered
        
        def increase_brightness(frame):
            bright = frame.copy()
            bright = cv2.add(bright, np.ones(bright.shape, dtype=np.uint8) * 10)
            return bright
        
        self.processor.add_preprocessor(add_border)
        self.processor.add_processor(increase_brightness)
        
        processed = self.processor.process_frame(self.test_frame)
        
        # Frame should be different from original
        self.assertFalse(np.array_equal(processed, self.test_frame))
        
        # Should have red border from preprocessor
        np.testing.assert_array_equal(processed[0, :], [255, 10, 10])  # Red + brightness
    
    def test_process_frame_with_resize(self):
        """Test frame processing with resizing."""
        self.processor.set_resize_target(320, 240)
        
        processed = self.processor.process_frame(self.test_frame)
        
        self.assertEqual(processed.shape, (240, 320, 3))
    
    def test_process_frame_empty(self):
        """Test processing empty frame."""
        processed = self.processor.process_frame(None)
        
        self.assertIsNone(processed)
    
    def test_process_frame_error_handling(self):
        """Test error handling in frame processing."""
        def failing_processor(frame):
            raise ValueError("Test error")
        
        self.processor.add_processor(failing_processor)
        
        # Should return original frame on error
        processed = self.processor.process_frame(self.test_frame)
        np.testing.assert_array_equal(processed, self.test_frame)
    
    def test_process_frame_batch(self):
        """Test batch frame processing."""
        frames = [self.test_frame.copy() for _ in range(5)]
        
        def add_frame_number(frame):
            # This is a mock - normally would modify frame based on position
            return frame
        
        self.processor.add_processor(add_frame_number)
        
        processed_frames = self.processor.process_frame_batch(frames)
        
        self.assertEqual(len(processed_frames), 5)
        for processed in processed_frames:
            self.assertEqual(processed.shape, self.test_frame.shape)
    
    def test_stats_update(self):
        """Test statistics updating."""
        # Process some frames
        for _ in range(5):
            self.processor.process_frame(self.test_frame)
        
        stats = self.processor.get_stats()
        
        self.assertEqual(stats.total_frames, 5)
        self.assertEqual(stats.processed_frames, 5)
        self.assertEqual(stats.failed_frames, 0)
        self.assertGreater(stats.total_processing_time, 0)
        self.assertGreater(stats.average_processing_time, 0)
        self.assertGreater(stats.fps, 0)
    
    def test_reset_stats(self):
        """Test statistics reset."""
        # Process a frame first
        self.processor.process_frame(self.test_frame)
        
        # Reset stats
        self.processor.reset_stats()
        
        stats = self.processor.get_stats()
        self.assertEqual(stats.total_frames, 0)
        self.assertEqual(stats.processed_frames, 0)
        self.assertEqual(stats.failed_frames, 0)
        self.assertEqual(stats.total_processing_time, 0.0)
    
    def test_clear_pipeline(self):
        """Test pipeline clearing."""
        # Add some processors
        self.processor.add_preprocessor(lambda x: x)
        self.processor.add_processor(lambda x: x)
        self.processor.add_postprocessor(lambda x: x)
        
        self.processor.clear_pipeline()
        
        self.assertEqual(len(self.processor.processors), 0)
        self.assertEqual(len(self.processor.preprocessing_functions), 0)
        self.assertEqual(len(self.processor.postprocessing_functions), 0)
    
    def test_stats_disabled(self):
        """Test processing with stats disabled."""
        processor = FrameProcessor(enable_stats=False)
        
        processor.process_frame(self.test_frame)
        
        stats = processor.get_stats()
        # Stats should still be initialized but not updated
        self.assertEqual(stats.total_frames, 0)


class TestFaceSwappingFrameProcessor(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = FaceSwappingFrameProcessor(enable_stats=True)
        self.test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    def test_init(self):
        """Test FaceSwappingFrameProcessor initialization."""
        self.assertIsNone(self.processor.face_detector)
        self.assertIsNone(self.processor.face_tracker)
        self.assertIsNone(self.processor.quality_assessor)
        self.assertEqual(self.processor.quality_threshold, 0.6)
        self.assertTrue(self.processor.enable_face_tracking)
        self.assertTrue(self.processor.enable_quality_filtering)
    
    def test_setup_face_processing(self):
        """Test face processing setup."""
        mock_detector = Mock()
        mock_tracker = Mock()
        mock_assessor = Mock()
        
        self.processor.setup_face_processing(
            face_detector=mock_detector,
            face_tracker=mock_tracker,
            quality_assessor=mock_assessor
        )
        
        self.assertEqual(self.processor.face_detector, mock_detector)
        self.assertEqual(self.processor.face_tracker, mock_tracker)
        self.assertEqual(self.processor.quality_assessor, mock_assessor)
    
    def test_add_face_swap_processor(self):
        """Test adding face swap processor."""
        mock_detector = Mock()
        mock_tracker = Mock()
        mock_assessor = Mock()
        
        # Setup mock detector to return faces
        mock_faces = [{'bbox': (10, 10, 100, 100)}]
        mock_detector.detect_faces.return_value = mock_faces
        mock_detector.get_face_region.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Setup mock tracker
        mock_tracked_faces = [{'bbox': (10, 10, 100, 100), 'track_id': 0}]
        mock_tracker.process_frame.return_value = mock_tracked_faces
        
        # Setup mock quality assessor
        mock_quality = {'overall_score': 0.8}
        mock_assessor.assess_quality.return_value = mock_quality
        
        # Setup face processing components
        self.processor.setup_face_processing(
            face_detector=mock_detector,
            face_tracker=mock_tracker,
            quality_assessor=mock_assessor
        )
        
        # Add face swap processor
        self.processor.add_face_swap_processor("source_face.jpg")
        
        # Process a frame
        processed = self.processor.process_frame(self.test_frame)
        
        # Should have called the detector
        mock_detector.detect_faces.assert_called_once()
        
        # Should return processed frame
        self.assertEqual(processed.shape, self.test_frame.shape)
    
    def test_face_swap_without_components(self):
        """Test face swap processing without face detection components."""
        self.processor.add_face_swap_processor("source_face.jpg")
        
        # Should work but not detect faces
        processed = self.processor.process_frame(self.test_frame)
        
        # Should return the frame unchanged
        np.testing.assert_array_equal(processed, self.test_frame)


class TestUtilityFunctions(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    def test_resize_frame(self):
        """Test frame resizing utility."""
        resize_func = resize_frame(320, 240)
        
        resized = resize_func(self.test_frame)
        
        self.assertEqual(resized.shape, (240, 320, 3))
        self.assertEqual(resize_func.__name__, "resize_320x240")
    
    def test_normalize_frame(self):
        """Test frame normalization utility."""
        normalize_func = normalize_frame()
        
        # Create frame with non-standard range
        test_frame = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
        normalized = normalize_func(test_frame)
        
        # Should be normalized to 0-255 range
        self.assertEqual(normalized.min(), 0)
        self.assertEqual(normalized.max(), 255)
        self.assertEqual(normalize_func.__name__, "normalize_frame")
    
    def test_denoise_frame(self):
        """Test frame denoising utility."""
        denoise_func = denoise_frame(strength=5)
        
        # Create a small noisy frame for testing
        small_frame = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        denoised = denoise_func(small_frame)
        
        self.assertEqual(denoised.shape, small_frame.shape)
        self.assertEqual(denoise_func.__name__, "denoise_strength_5")
    
    def test_enhance_contrast(self):
        """Test contrast enhancement utility."""
        enhance_func = enhance_contrast(alpha=1.5, beta=20)
        
        enhanced = enhance_func(self.test_frame)
        
        self.assertEqual(enhanced.shape, self.test_frame.shape)
        self.assertEqual(enhance_func.__name__, "enhance_contrast_a1.5_b20")


class TestFrameProcessorIntegration(unittest.TestCase):
    """Integration tests for frame processor."""
    
    def test_complex_processing_pipeline(self):
        """Test complex processing pipeline with multiple stages."""
        processor = FrameProcessor(enable_stats=True)
        
        # Create test frame
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Add multiple processing stages
        processor.add_preprocessor(resize_frame(320, 240))
        processor.add_processor(normalize_frame())
        processor.add_processor(enhance_contrast(alpha=1.2, beta=10))
        processor.add_postprocessor(lambda x: cv2.GaussianBlur(x, (3, 3), 0))
        
        # Process frame
        processed = processor.process_frame(test_frame)
        
        # Check final result
        self.assertEqual(processed.shape, (240, 320, 3))
        
        # Check stats
        stats = processor.get_stats()
        self.assertEqual(stats.processed_frames, 1)
        self.assertGreater(stats.total_processing_time, 0)


if __name__ == '__main__':
    unittest.main()