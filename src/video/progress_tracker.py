"""
Progress Tracker - Progress tracking utilities for video processing

Provides progress tracking capabilities for video processing operations
with support for different output formats and real-time updates.
"""

import time
import logging
from typing import Callable, Optional, Dict, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class ProgressInfo:
    """Progress information container."""
    current_frame: int
    total_frames: int
    percentage: float
    elapsed_time: float
    estimated_total_time: float
    estimated_remaining_time: float
    fps: float
    current_operation: str = "Processing"


class ProgressTracker(ABC):
    """Abstract base class for progress tracking."""
    
    @abstractmethod
    def update(self, current: int, total: int, operation: str = "Processing") -> None:
        """Update progress."""
        pass
    
    @abstractmethod
    def finish(self) -> None:
        """Mark progress as finished."""
        pass


class ConsoleProgressTracker(ProgressTracker):
    """Console-based progress tracker with text output."""
    
    def __init__(self, show_bar: bool = True, bar_length: int = 50,
                 update_interval: float = 1.0):
        """
        Initialize console progress tracker.
        
        Args:
            show_bar: Whether to show progress bar
            bar_length: Length of progress bar in characters
            update_interval: Minimum interval between updates (seconds)
        """
        self.show_bar = show_bar
        self.bar_length = bar_length
        self.update_interval = update_interval
        
        self.start_time = time.time()
        self.last_update_time = 0
        self.last_percentage = -1
        
    def update(self, current: int, total: int, operation: str = "Processing") -> None:
        """Update console progress display."""
        current_time = time.time()
        
        # Check if enough time has passed for update
        if current_time - self.last_update_time < self.update_interval:
            return
        
        if total <= 0:
            return
        
        # Calculate progress
        percentage = min((current / total) * 100, 100)
        
        # Skip if percentage hasn't changed significantly
        if abs(percentage - self.last_percentage) < 0.1:
            return
        
        elapsed_time = current_time - self.start_time
        
        # Calculate speed and time estimates
        if current > 0 and elapsed_time > 0:
            fps = current / elapsed_time
            estimated_total_time = total / fps
            estimated_remaining_time = max(0, estimated_total_time - elapsed_time)
        else:
            fps = 0
            estimated_total_time = 0
            estimated_remaining_time = 0
        
        # Create progress display
        if self.show_bar:
            filled_length = int(self.bar_length * current // total)
            bar = '█' * filled_length + '-' * (self.bar_length - filled_length)
            
            progress_text = (
                f"\r{operation}: |{bar}| "
                f"{current}/{total} ({percentage:.1f}%) "
                f"[{self._format_time(elapsed_time)} < "
                f"{self._format_time(estimated_remaining_time)}, "
                f"{fps:.1f} fps]"
            )
        else:
            progress_text = (
                f"\r{operation}: {current}/{total} ({percentage:.1f}%) "
                f"[{self._format_time(elapsed_time)} < "
                f"{self._format_time(estimated_remaining_time)}, "
                f"{fps:.1f} fps]"
            )
        
        print(progress_text, end='', flush=True)
        
        self.last_update_time = current_time
        self.last_percentage = percentage
    
    def finish(self) -> None:
        """Finish progress tracking."""
        elapsed_time = time.time() - self.start_time
        print(f"\nCompleted in {self._format_time(elapsed_time)}")
    
    def _format_time(self, seconds: float) -> str:
        """Format time in human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            seconds = seconds % 60
            return f"{minutes:02d}:{seconds:04.1f}"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            seconds = seconds % 60
            return f"{hours:02d}:{minutes:02d}:{seconds:04.1f}"


class CallbackProgressTracker(ProgressTracker):
    """Progress tracker that calls user-provided callbacks."""
    
    def __init__(self, callback: Callable[[ProgressInfo], None],
                 update_interval: float = 0.5):
        """
        Initialize callback progress tracker.
        
        Args:
            callback: Function to call with progress updates
            update_interval: Minimum interval between updates (seconds)
        """
        self.callback = callback
        self.update_interval = update_interval
        
        self.start_time = time.time()
        self.last_update_time = 0
    
    def update(self, current: int, total: int, operation: str = "Processing") -> None:
        """Update progress via callback."""
        current_time = time.time()
        
        # Check if enough time has passed for update
        if current_time - self.last_update_time < self.update_interval:
            return
        
        if total <= 0:
            return
        
        elapsed_time = current_time - self.start_time
        percentage = min((current / total) * 100, 100)
        
        # Calculate estimates
        if current > 0 and elapsed_time > 0:
            fps = current / elapsed_time
            estimated_total_time = total / fps
            estimated_remaining_time = max(0, estimated_total_time - elapsed_time)
        else:
            fps = 0
            estimated_total_time = 0
            estimated_remaining_time = 0
        
        # Create progress info
        progress_info = ProgressInfo(
            current_frame=current,
            total_frames=total,
            percentage=percentage,
            elapsed_time=elapsed_time,
            estimated_total_time=estimated_total_time,
            estimated_remaining_time=estimated_remaining_time,
            fps=fps,
            current_operation=operation
        )
        
        # Call user callback
        try:
            self.callback(progress_info)
        except Exception as e:
            logger.error(f"Progress callback failed: {e}")
        
        self.last_update_time = current_time
    
    def finish(self) -> None:
        """Finish progress tracking."""
        # Call callback one final time with 100% completion
        elapsed_time = time.time() - self.start_time
        
        final_progress = ProgressInfo(
            current_frame=0,  # Will be set by final update
            total_frames=0,   # Will be set by final update
            percentage=100.0,
            elapsed_time=elapsed_time,
            estimated_total_time=elapsed_time,
            estimated_remaining_time=0.0,
            fps=0.0,
            current_operation="Completed"
        )
        
        try:
            self.callback(final_progress)
        except Exception as e:
            logger.error(f"Final progress callback failed: {e}")


class SilentProgressTracker(ProgressTracker):
    """Silent progress tracker that logs but doesn't display progress."""
    
    def __init__(self, log_interval: float = 10.0):
        """
        Initialize silent progress tracker.
        
        Args:
            log_interval: Interval for logging progress (seconds)
        """
        self.log_interval = log_interval
        self.start_time = time.time()
        self.last_log_time = 0
    
    def update(self, current: int, total: int, operation: str = "Processing") -> None:
        """Log progress at intervals."""
        current_time = time.time()
        
        # Check if enough time has passed for logging
        if current_time - self.last_log_time < self.log_interval:
            return
        
        if total <= 0:
            return
        
        percentage = min((current / total) * 100, 100)
        elapsed_time = current_time - self.start_time
        
        logger.info(f"{operation}: {current}/{total} ({percentage:.1f}%) "
                   f"- Elapsed: {elapsed_time:.1f}s")
        
        self.last_log_time = current_time
    
    def finish(self) -> None:
        """Log completion."""
        elapsed_time = time.time() - self.start_time
        logger.info(f"Processing completed in {elapsed_time:.1f}s")


class MultiProgressTracker(ProgressTracker):
    """Progress tracker that manages multiple sub-trackers."""
    
    def __init__(self, trackers: list[ProgressTracker]):
        """
        Initialize multi-progress tracker.
        
        Args:
            trackers: List of progress trackers to manage
        """
        self.trackers = trackers
    
    def update(self, current: int, total: int, operation: str = "Processing") -> None:
        """Update all managed trackers."""
        for tracker in self.trackers:
            try:
                tracker.update(current, total, operation)
            except Exception as e:
                logger.error(f"Progress tracker update failed: {e}")
    
    def finish(self) -> None:
        """Finish all managed trackers."""
        for tracker in self.trackers:
            try:
                tracker.finish()
            except Exception as e:
                logger.error(f"Progress tracker finish failed: {e}")


def create_progress_callback(total_frames: int, 
                           tracker_type: str = "console",
                           **kwargs) -> Callable[[int, int], None]:
    """
    Create a progress callback for video processing.
    
    Args:
        total_frames: Total number of frames to process
        tracker_type: Type of tracker ('console', 'silent', 'callback')
        **kwargs: Additional arguments for tracker
        
    Returns:
        Progress callback function
    """
    # Create appropriate tracker
    if tracker_type == "console":
        tracker = ConsoleProgressTracker(**kwargs)
    elif tracker_type == "silent":
        tracker = SilentProgressTracker(**kwargs)
    elif tracker_type == "callback":
        if "callback" not in kwargs:
            raise ValueError("Callback tracker requires 'callback' parameter")
        tracker = CallbackProgressTracker(**kwargs)
    else:
        raise ValueError(f"Unknown tracker type: {tracker_type}")
    
    def progress_callback(current_frame: int, total_frames_param: int = total_frames):
        """Progress callback function."""
        tracker.update(current_frame, total_frames_param)
        
        # Auto-finish when complete
        if current_frame >= total_frames_param:
            tracker.finish()
    
    return progress_callback


class ProgressManager:
    """Manager for progress tracking across multiple operations."""
    
    def __init__(self):
        """Initialize progress manager."""
        self.active_trackers: Dict[str, ProgressTracker] = {}
        self.operation_stats: Dict[str, Dict[str, Any]] = {}
    
    def start_operation(self, operation_id: str, 
                       tracker: ProgressTracker,
                       total_items: int = 0) -> None:
        """
        Start tracking a new operation.
        
        Args:
            operation_id: Unique identifier for the operation
            tracker: Progress tracker to use
            total_items: Total number of items to process
        """
        self.active_trackers[operation_id] = tracker
        self.operation_stats[operation_id] = {
            'start_time': time.time(),
            'total_items': total_items,
            'current_items': 0
        }
        
        logger.info(f"Started tracking operation: {operation_id}")
    
    def update_operation(self, operation_id: str, 
                        current: int, total: Optional[int] = None,
                        operation_name: str = "Processing") -> None:
        """
        Update progress for an operation.
        
        Args:
            operation_id: Operation identifier
            current: Current progress
            total: Total items (uses stored total if None)
            operation_name: Name to display for operation
        """
        if operation_id not in self.active_trackers:
            logger.warning(f"Unknown operation: {operation_id}")
            return
        
        tracker = self.active_trackers[operation_id]
        stats = self.operation_stats[operation_id]
        
        total_items = total or stats['total_items']
        stats['current_items'] = current
        
        tracker.update(current, total_items, operation_name)
    
    def finish_operation(self, operation_id: str) -> Dict[str, Any]:
        """
        Finish tracking an operation.
        
        Args:
            operation_id: Operation identifier
            
        Returns:
            Final statistics for the operation
        """
        if operation_id not in self.active_trackers:
            logger.warning(f"Unknown operation: {operation_id}")
            return {}
        
        tracker = self.active_trackers[operation_id]
        stats = self.operation_stats[operation_id]
        
        # Finish tracking
        tracker.finish()
        
        # Calculate final stats
        end_time = time.time()
        duration = end_time - stats['start_time']
        
        final_stats = {
            'duration': duration,
            'total_items': stats['total_items'],
            'items_processed': stats['current_items'],
            'average_rate': stats['current_items'] / duration if duration > 0 else 0
        }
        
        # Clean up
        del self.active_trackers[operation_id]
        del self.operation_stats[operation_id]
        
        logger.info(f"Finished operation: {operation_id} "
                   f"({final_stats['items_processed']}/{final_stats['total_items']} "
                   f"in {duration:.1f}s)")
        
        return final_stats
    
    def get_active_operations(self) -> list[str]:
        """Get list of active operation IDs."""
        return list(self.active_trackers.keys())
    
    def cancel_operation(self, operation_id: str) -> None:
        """Cancel tracking for an operation."""
        if operation_id in self.active_trackers:
            del self.active_trackers[operation_id]
            del self.operation_stats[operation_id]
            logger.info(f"Cancelled operation: {operation_id}")


# Global progress manager instance
progress_manager = ProgressManager()