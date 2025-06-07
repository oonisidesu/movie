"""
Logging Configuration for Video Face Swapping Tool

Provides centralized logging setup and configuration for the entire application.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

try:
    import colorlog
    HAS_COLORLOG = True
except ImportError:
    colorlog = None
    HAS_COLORLOG = False


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    enable_colors: bool = True,
    format_string: Optional[str] = None
) -> None:
    """
    Setup logging configuration for the application.
    
    Args:
        level: Logging level (e.g., logging.INFO)
        log_file: Optional path to log file
        enable_colors: Whether to enable colored console output
        format_string: Custom format string for log messages
    """
    # Default format string
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Color format for console
    color_format = (
        '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    if enable_colors and HAS_COLORLOG:
        # Colored console output
        console_formatter = colorlog.ColoredFormatter(
            color_format,
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
    else:
        # Plain console output
        console_formatter = logging.Formatter(
            format_string,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use rotating file handler to prevent large log files
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(level)
        
        file_formatter = logging.Formatter(
            format_string,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Set logging levels for specific modules
    configure_module_logging()


def configure_module_logging() -> None:
    """Configure logging levels for specific modules."""
    # Reduce noise from external libraries
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    # OpenCV can be verbose
    logging.getLogger('cv2').setLevel(logging.WARNING)
    
    # MediaPipe can be verbose
    logging.getLogger('mediapipe').setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class ProgressLogHandler(logging.Handler):
    """
    Custom log handler that works with progress tracking.
    
    Ensures log messages don't interfere with progress bars or counters.
    """
    
    def __init__(self, progress_tracker=None):
        super().__init__()
        self.progress_tracker = progress_tracker
    
    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a log record.
        
        Args:
            record: Log record to emit
        """
        try:
            msg = self.format(record)
            
            # If we have a progress tracker, handle output carefully
            if self.progress_tracker and hasattr(self.progress_tracker, 'log_message'):
                self.progress_tracker.log_message(msg)
            else:
                # Fallback to standard output
                print(msg, file=sys.stderr)
                
        except Exception:
            self.handleError(record)


def setup_cli_logging(
    verbose: bool = False,
    quiet: bool = False,
    log_file: Optional[str] = None,
    progress_tracker=None
) -> None:
    """
    Setup logging specifically for CLI usage.
    
    Args:
        verbose: Enable verbose (DEBUG) logging
        quiet: Enable quiet mode (ERROR only)
        log_file: Optional log file path
        progress_tracker: Optional progress tracker instance
    """
    if quiet:
        level = logging.ERROR
        enable_colors = False
    elif verbose:
        level = logging.DEBUG
        enable_colors = True
    else:
        level = logging.INFO
        enable_colors = True
    
    # Setup basic logging
    setup_logging(
        level=level,
        log_file=log_file,
        enable_colors=enable_colors
    )
    
    # Add progress-aware handler if needed
    if progress_tracker:
        root_logger = logging.getLogger()
        
        # Remove console handler
        handlers_to_remove = [
            h for h in root_logger.handlers 
            if isinstance(h, logging.StreamHandler) and h.stream == sys.stdout
        ]
        
        for handler in handlers_to_remove:
            root_logger.removeHandler(handler)
        
        # Add progress-aware handler
        progress_handler = ProgressLogHandler(progress_tracker)
        progress_handler.setLevel(level)
        
        if enable_colors and HAS_COLORLOG:
            formatter = colorlog.ColoredFormatter(
                '%(log_color)s%(levelname)s: %(message)s',
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
                }
            )
        else:
            formatter = logging.Formatter('%(levelname)s: %(message)s')
        
        progress_handler.setFormatter(formatter)
        root_logger.addHandler(progress_handler)


def log_system_info() -> None:
    """Log system information for debugging purposes."""
    logger = get_logger(__name__)
    
    logger.info("=== System Information ===")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Platform: {sys.platform}")
    
    # Log memory info if available
    try:
        import psutil
        memory = psutil.virtual_memory()
        logger.info(f"Total memory: {memory.total / (1024**3):.1f} GB")
        logger.info(f"Available memory: {memory.available / (1024**3):.1f} GB")
    except ImportError:
        logger.debug("psutil not available for memory info")
    
    # Log GPU info if available
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"CUDA available: {torch.cuda.device_count()} devices")
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                memory = torch.cuda.get_device_properties(i).total_memory
                logger.info(f"  GPU {i}: {name} ({memory / (1024**3):.1f} GB)")
        else:
            logger.info("CUDA not available")
    except ImportError:
        logger.debug("PyTorch not available for GPU info")
    
    logger.info("=" * 30)


class LoggingContext:
    """
    Context manager for temporary logging configuration.
    
    Useful for changing logging behavior for specific operations.
    """
    
    def __init__(self, level: int, logger_name: Optional[str] = None):
        """
        Initialize logging context.
        
        Args:
            level: Temporary logging level
            logger_name: Specific logger name (None for root logger)
        """
        self.level = level
        self.logger_name = logger_name
        self.original_level = None
        self.logger = None
    
    def __enter__(self):
        if self.logger_name:
            self.logger = logging.getLogger(self.logger_name)
        else:
            self.logger = logging.getLogger()
        
        self.original_level = self.logger.level
        self.logger.setLevel(self.level)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.logger:
            self.logger.setLevel(self.original_level)


# Convenience functions for common logging operations
def debug_mode(func):
    """Decorator to enable debug logging for a function."""
    def wrapper(*args, **kwargs):
        with LoggingContext(logging.DEBUG):
            return func(*args, **kwargs)
    return wrapper


def quiet_mode(func):
    """Decorator to enable quiet logging for a function."""
    def wrapper(*args, **kwargs):
        with LoggingContext(logging.ERROR):
            return func(*args, **kwargs)
    return wrapper