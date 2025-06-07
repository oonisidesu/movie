"""
Progress indicator utilities for CLI operations.
"""

import sys
import time
from typing import Optional


class ProgressBar:
    """Simple progress bar for terminal output."""
    
    def __init__(self, total: int, width: int = 50, desc: str = "Processing"):
        self.total = total
        self.width = width
        self.desc = desc
        self.current = 0
        self.start_time = time.time()
        self.last_update = 0
    
    def update(self, current: Optional[int] = None):
        """Update progress bar."""
        if current is not None:
            self.current = current
        else:
            self.current += 1
        
        # Throttle updates to avoid too frequent redraws
        now = time.time()
        if now - self.last_update < 0.1 and self.current < self.total:
            return
        self.last_update = now
        
        # Calculate progress
        percent = min(100, (self.current / self.total) * 100)
        filled = int(self.width * percent / 100)
        bar = '█' * filled + '▒' * (self.width - filled)
        
        # Calculate time estimates
        elapsed = now - self.start_time
        if self.current > 0:
            rate = self.current / elapsed
            eta = (self.total - self.current) / rate if rate > 0 else 0
            eta_str = self._format_time(eta)
        else:
            eta_str = "?:??"
        
        # Format output
        elapsed_str = self._format_time(elapsed)
        
        # Write progress line
        line = f"\r{self.desc}: |{bar}| {self.current}/{self.total} "
        line += f"({percent:.1f}%) {elapsed_str}<{eta_str}"
        
        sys.stdout.write(line)
        sys.stdout.flush()
        
        # Print newline when complete
        if self.current >= self.total:
            sys.stdout.write('\n')
            sys.stdout.flush()
    
    def _format_time(self, seconds: float) -> str:
        """Format time duration as MM:SS."""
        if seconds < 0:
            return "?:??"
        
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes}:{seconds:02d}"
    
    def finish(self):
        """Mark progress as complete."""
        self.update(self.total)


class Spinner:
    """Simple spinner for indeterminate progress."""
    
    def __init__(self, desc: str = "Working"):
        self.desc = desc
        self.chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        self.index = 0
        self.running = False
    
    def start(self):
        """Start spinner."""
        self.running = True
        self._update()
    
    def stop(self):
        """Stop spinner and clear line."""
        self.running = False
        sys.stdout.write('\r' + ' ' * (len(self.desc) + 10) + '\r')
        sys.stdout.flush()
    
    def _update(self):
        """Update spinner display."""
        if not self.running:
            return
        
        char = self.chars[self.index % len(self.chars)]
        line = f"\r{char} {self.desc}..."
        
        sys.stdout.write(line)
        sys.stdout.flush()
        
        self.index += 1
        
        # Schedule next update
        import threading
        timer = threading.Timer(0.1, self._update)
        timer.daemon = True
        timer.start()


def create_progress_callback(total_frames: int, desc: str = "Processing frames"):
    """Create a progress callback function for video processing."""
    progress = ProgressBar(total_frames, desc=desc)
    
    def callback(current_frame: int, total: int):
        progress.update(current_frame)
    
    return callback