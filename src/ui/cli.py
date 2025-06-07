"""
CLI Interface for Video Face Swapping Tool

Provides command-line interface for the video face-swapping tool with
comprehensive argument parsing, validation, and user feedback.
"""

import argparse
import sys
import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from ..face_detection import FaceDetector, DetectionBackend
from ..video import VideoProcessor, ProgressTracker, ConsoleProgressTracker
from ..utils.logging_config import setup_cli_logging
from .config import CLIConfig, load_config


class CLIApp:
    """
    Main CLI application class for video face swapping.
    
    Handles argument parsing, validation, and orchestrates the
    face swapping pipeline execution.
    """
    
    def __init__(self):
        """Initialize CLI application."""
        self.config = CLIConfig()
        self.logger = None
        
        # Processing components
        self.face_detector = None
        self.video_processor = None
        self.progress_tracker = None
    
    def setup_logging(self, verbose: bool = False, quiet: bool = False) -> None:
        """
        Setup logging configuration.
        
        Args:
            verbose: Enable verbose logging
            quiet: Enable quiet mode (errors only)
        """
        if quiet:
            level = logging.ERROR
        elif verbose:
            level = logging.DEBUG
        else:
            level = logging.INFO
        
        setup_cli_logging(verbose=(level==logging.DEBUG), quiet=(level==logging.ERROR))
        self.logger = logging.getLogger(__name__)
    
    def create_argument_parser(self) -> argparse.ArgumentParser:
        """
        Create and configure argument parser.
        
        Returns:
            Configured ArgumentParser instance
        """
        parser = argparse.ArgumentParser(
            prog='face-swap',
            description='Video Face Swapping Tool - Replace faces in videos',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog='''
Examples:
  %(prog)s -i input.mp4 -f face.jpg -o output.mp4
  %(prog)s -i video.mp4 -f target.jpg -o result.mp4 --quality high
  %(prog)s -i input.mp4 -f face.jpg -o output.mp4 --verbose
            '''
        )
        
        # Required arguments (made optional for info commands)
        required = parser.add_argument_group('required arguments')
        required.add_argument(
            '-i', '--input',
            type=str,
            metavar='VIDEO',
            help='Input video file path'
        )
        required.add_argument(
            '-f', '--face',
            type=str,
            metavar='IMAGE',
            help='Target face image file path'
        )
        required.add_argument(
            '-o', '--output',
            type=str,
            metavar='VIDEO',
            help='Output video file path'
        )
        
        # Optional arguments
        parser.add_argument(
            '--quality',
            choices=['low', 'medium', 'high'],
            default='medium',
            help='Output video quality (default: medium)'
        )
        parser.add_argument(
            '--fps',
            type=float,
            metavar='FPS',
            help='Output video frame rate (default: same as input)'
        )
        parser.add_argument(
            '--resolution',
            type=str,
            metavar='WIDTHxHEIGHT',
            help='Output video resolution (e.g., 1920x1080)'
        )
        parser.add_argument(
            '--face-detector',
            choices=['opencv', 'mediapipe'],
            default='opencv',
            help='Face detection backend (default: opencv)'
        )
        parser.add_argument(
            '--confidence',
            type=float,
            default=0.5,
            metavar='THRESHOLD',
            help='Face detection confidence threshold (default: 0.5)'
        )
        parser.add_argument(
            '--temp-dir',
            type=str,
            metavar='DIR',
            help='Temporary files directory'
        )
        parser.add_argument(
            '--config',
            type=str,
            metavar='FILE',
            help='Configuration file path'
        )
        
        # Processing options
        processing = parser.add_argument_group('processing options')
        processing.add_argument(
            '--no-audio',
            action='store_true',
            help='Disable audio preservation'
        )
        processing.add_argument(
            '--start-time',
            type=float,
            metavar='SECONDS',
            help='Start processing from specified time'
        )
        processing.add_argument(
            '--duration',
            type=float,
            metavar='SECONDS',
            help='Process only specified duration'
        )
        processing.add_argument(
            '--chunk-size',
            type=int,
            default=30,
            metavar='FRAMES',
            help='Number of frames to process at once (default: 30)'
        )
        
        # Output options
        output = parser.add_argument_group('output options')
        output.add_argument(
            '--overwrite',
            action='store_true',
            help='Overwrite output file if it exists'
        )
        output.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Enable verbose output'
        )
        output.add_argument(
            '--quiet', '-q',
            action='store_true',
            help='Suppress all output except errors'
        )
        output.add_argument(
            '--progress',
            choices=['console', 'silent'],
            default='console',
            help='Progress display mode (default: console)'
        )
        
        # Information options
        info = parser.add_argument_group('information')
        info.add_argument(
            '--version',
            action='version',
            version='%(prog)s 1.0.0'
        )
        info.add_argument(
            '--list-formats',
            action='store_true',
            help='List supported video formats and exit'
        )
        info.add_argument(
            '--check-deps',
            action='store_true',
            help='Check system dependencies and exit'
        )
        
        return parser
    
    def parse_resolution(self, resolution_str: str) -> tuple:
        """
        Parse resolution string into width and height.
        
        Args:
            resolution_str: Resolution string (e.g., "1920x1080")
            
        Returns:
            Tuple of (width, height)
            
        Raises:
            ValueError: If resolution format is invalid
        """
        try:
            parts = resolution_str.lower().split('x')
            if len(parts) != 2:
                raise ValueError("Resolution must be in format WIDTHxHEIGHT")
            
            width = int(parts[0])
            height = int(parts[1])
            
            if width <= 0 or height <= 0:
                raise ValueError("Width and height must be positive")
            
            return width, height
            
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid resolution format: {resolution_str}") from e
    
    def validate_arguments(self, args: argparse.Namespace) -> None:
        """
        Validate command line arguments.
        
        Args:
            args: Parsed command line arguments
            
        Raises:
            ValueError: If validation fails
        """
        # Check required arguments for processing commands
        if not args.input:
            raise ValueError("Input video file is required")
        if not args.face:
            raise ValueError("Target face image file is required")
        if not args.output:
            raise ValueError("Output video file is required")
        
        # Check input file exists
        if not os.path.exists(args.input):
            raise ValueError(f"Input video file not found: {args.input}")
        
        # Check face image exists
        if not os.path.exists(args.face):
            raise ValueError(f"Face image file not found: {args.face}")
        
        # Check output directory exists or can be created
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
            except OSError as e:
                raise ValueError(f"Cannot create output directory: {output_dir}") from e
        
        # Check if output file exists and overwrite flag
        if os.path.exists(args.output) and not args.overwrite:
            raise ValueError(
                f"Output file already exists: {args.output}\n"
                f"Use --overwrite to replace it"
            )
        
        # Validate resolution format
        if args.resolution:
            try:
                self.parse_resolution(args.resolution)
            except ValueError as e:
                raise ValueError(f"Invalid resolution: {e}") from e
        
        # Validate confidence threshold
        if not 0.0 <= args.confidence <= 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
        
        # Validate FPS
        if args.fps is not None and args.fps <= 0:
            raise ValueError("FPS must be positive")
        
        # Validate time parameters
        if args.start_time is not None and args.start_time < 0:
            raise ValueError("Start time must be non-negative")
        
        if args.duration is not None and args.duration <= 0:
            raise ValueError("Duration must be positive")
        
        # Check conflicting options
        if args.verbose and args.quiet:
            raise ValueError("Cannot use --verbose and --quiet together")
    
    def load_configuration(self, config_path: Optional[str] = None) -> None:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
        """
        if config_path:
            self.config = load_config(config_path)
            self.logger.info(f"Loaded configuration from: {config_path}")
    
    def initialize_components(self, args: argparse.Namespace) -> None:
        """
        Initialize processing components based on arguments.
        
        Args:
            args: Parsed command line arguments
        """
        # Initialize face detector
        backend_map = {
            'opencv': DetectionBackend.OPENCV_HAAR,
            'mediapipe': DetectionBackend.MEDIAPIPE
        }
        backend = backend_map[args.face_detector]
        
        self.face_detector = FaceDetector(
            backend=backend,
            confidence_threshold=args.confidence
        )
        
        # Initialize video processor
        self.video_processor = VideoProcessor(
            input_path=args.input,
            output_path=args.output,
            temp_dir=args.temp_dir
        )
        
        # Initialize progress tracker
        if args.progress == 'console' and not args.quiet:
            self.progress_tracker = ConsoleProgressTracker()
        else:
            from ..video import SilentProgressTracker
            self.progress_tracker = SilentProgressTracker()
        
        self.logger.info("Components initialized successfully")
    
    def display_system_info(self) -> None:
        """Display system and dependency information."""
        print("Video Face Swapping Tool - System Information")
        print("=" * 50)
        print(f"Python version: {sys.version}")
        print(f"Platform: {sys.platform}")
        
        # Check dependencies
        dependencies = {
            'opencv-python': 'cv2',
            'numpy': 'numpy',
            'mediapipe': 'mediapipe'
        }
        
        print("\nDependency Status:")
        for package, module in dependencies.items():
            try:
                __import__(module)
                print(f"✓ {package}: Available")
            except ImportError:
                print(f"✗ {package}: Not available")
        
        # Check FFmpeg
        try:
            import subprocess
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✓ FFmpeg: Available")
            else:
                print("✗ FFmpeg: Not available")
        except FileNotFoundError:
            print("✗ FFmpeg: Not available")
    
    def list_supported_formats(self) -> None:
        """List supported video formats."""
        print("Supported Video Formats")
        print("=" * 30)
        print("Input formats:")
        print("  - MP4 (.mp4)")
        print("  - AVI (.avi)")
        print("  - MOV (.mov)")
        print("  - MKV (.mkv)")
        print("  - WMV (.wmv)")
        print("  - FLV (.flv)")
        print("  - WebM (.webm)")
        print()
        print("Output formats:")
        print("  - MP4 (.mp4) - Recommended")
        print("  - AVI (.avi)")
        print()
        print("Supported image formats for face input:")
        print("  - JPEG (.jpg, .jpeg)")
        print("  - PNG (.png)")
        print("  - BMP (.bmp)")
    
    def run_processing_pipeline(self, args: argparse.Namespace) -> bool:
        """
        Run the main processing pipeline.
        
        Args:
            args: Parsed command line arguments
            
        Returns:
            True if processing successful
        """
        try:
            self.logger.info("Starting video processing pipeline")
            
            # Load and validate video
            video_info = self.video_processor.load_video(args.input)
            self.logger.info(f"Loaded video: {video_info}")
            
            # Setup output configuration
            output_fps = args.fps
            output_resolution = None
            if args.resolution:
                output_resolution = self.parse_resolution(args.resolution)
            
            self.video_processor.setup_output(
                args.output,
                fps=output_fps,
                resolution=output_resolution,
                codec='mp4v'
            )
            
            # Extract audio if enabled
            if not args.no_audio:
                self.video_processor.extract_audio_track()
            
            # Define frame processing function
            def process_frame(frame_num: int, frame) -> any:
                """Process individual frame with face swapping."""
                # TODO: Implement actual face swapping
                # For now, just return original frame
                self.logger.debug(f"Processing frame {frame_num}")
                return frame
            
            # Define progress callback
            def progress_callback(current: int, total: int) -> None:
                """Handle progress updates."""
                if self.progress_tracker:
                    progress_info = {
                        'current': current,
                        'total': total,
                        'percentage': (current / total) * 100
                    }
                    self.progress_tracker.update(progress_info)
            
            # Process video
            success = self.video_processor.process_frames(
                frame_callback=process_frame,
                progress_callback=progress_callback
            )
            
            if success:
                # Finalize output
                success = self.video_processor.finalize_output()
                
                if success:
                    self.logger.info(f"Processing completed successfully: {args.output}")
                    return True
                else:
                    self.logger.error("Failed to finalize output")
                    return False
            else:
                self.logger.error("Video processing failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Processing pipeline failed: {e}")
            return False
        finally:
            # Cleanup
            if self.video_processor:
                self.video_processor.cleanup()
    
    def run(self, argv: Optional[list] = None) -> int:
        """
        Main entry point for CLI application.
        
        Args:
            argv: Command line arguments (default: sys.argv)
            
        Returns:
            Exit code (0 for success, non-zero for error)
        """
        try:
            # Parse arguments
            parser = self.create_argument_parser()
            args = parser.parse_args(argv)
            
            # Setup logging
            self.setup_logging(args.verbose, args.quiet)
            
            # Handle information commands first (don't require validation)
            if args.check_deps:
                self.display_system_info()
                return 0
            
            if args.list_formats:
                self.list_supported_formats()
                return 0
            
            # Load configuration
            self.load_configuration(args.config)
            
            # Validate arguments for processing commands
            self.validate_arguments(args)
            
            # Initialize components
            self.initialize_components(args)
            
            # Run processing pipeline
            success = self.run_processing_pipeline(args)
            
            return 0 if success else 1
            
        except ValueError as e:
            try:
                if not args.quiet:
                    print(f"Error: {e}", file=sys.stderr)
            except:
                print(f"Error: {e}", file=sys.stderr)
            return 1
        except KeyboardInterrupt:
            try:
                if not args.quiet:
                    print("\nOperation cancelled by user", file=sys.stderr)
            except:
                print("\nOperation cancelled by user", file=sys.stderr)
            return 130
        except Exception as e:
            try:
                if not args.quiet:
                    print(f"Unexpected error: {e}", file=sys.stderr)
            except:
                print(f"Unexpected error: {e}", file=sys.stderr)
            return 1


def main() -> int:
    """Main entry point for command line interface."""
    app = CLIApp()
    return app.run()


if __name__ == '__main__':
    sys.exit(main())