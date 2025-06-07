import numpy as np
from typing import Callable, Optional, Any, List
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class FrameProcessor:
    """Pipeline for processing video frames."""
    
    def __init__(self):
        self.processors: List[Callable[[np.ndarray], np.ndarray]] = []
        
    def add_processor(self, processor: Callable[[np.ndarray], np.ndarray]):
        """Add a frame processor to the pipeline.
        
        Args:
            processor: Function that takes a frame and returns a processed frame
        """
        self.processors.append(processor)
        
    def remove_processor(self, processor: Callable[[np.ndarray], np.ndarray]):
        """Remove a processor from the pipeline.
        
        Args:
            processor: Function to remove
        """
        if processor in self.processors:
            self.processors.remove(processor)
            
    def clear_processors(self):
        """Clear all processors from the pipeline."""
        self.processors.clear()
        
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame through the pipeline.
        
        Args:
            frame: Input frame
            
        Returns:
            np.ndarray: Processed frame
        """
        processed_frame = frame.copy()
        
        for processor in self.processors:
            try:
                processed_frame = processor(processed_frame)
            except Exception as e:
                logger.error(f"Error in frame processor {processor.__name__}: {e}")
                return frame
                
        return processed_frame
        
    def process_frames(self, frames: List[np.ndarray], 
                      show_progress: bool = True) -> List[np.ndarray]:
        """Process multiple frames through the pipeline.
        
        Args:
            frames: List of input frames
            show_progress: Show progress bar
            
        Returns:
            List[np.ndarray]: List of processed frames
        """
        processed_frames = []
        
        iterator = tqdm(frames, desc="Processing frames") if show_progress else frames
        
        for frame in iterator:
            processed_frame = self.process_frame(frame)
            processed_frames.append(processed_frame)
            
        return processed_frames