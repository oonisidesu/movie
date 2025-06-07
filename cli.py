#!/usr/bin/env python3
"""
Video Face Swapper CLI Interface

A command-line interface for swapping faces in videos.
For personal/educational use only.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional, List
import logging

__version__ = "0.1.0"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Video Face Swapper - Replace faces in videos",
        epilog="For personal/educational use only. Please ensure you have appropriate permissions to use source images and videos.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Version
    parser.add_argument(
        '--version', '-V',
        action='version',
        version=f'%(prog)s {__version__}'
    )
    
    # Input/Output
    parser.add_argument(
        'input_video',
        type=str,
        help='Path to input video file (MP4, AVI, MOV)'
    )
    
    parser.add_argument(
        'source_face',
        type=str,
        help='Path to source face image (face to extract)'
    )
    
    parser.add_argument(
        'target_face',
        type=str,
        help='Path to target face image (face to replace with)'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output video path (default: input_swapped.mp4)'
    )
    
    # Processing options
    parser.add_argument(
        '-q', '--quality',
        type=str,
        choices=['low', 'medium', 'high', 'best'],
        default='medium',
        help='Processing quality (default: medium)'
    )
    
    parser.add_argument(
        '--fps',
        type=int,
        default=None,
        help='Output video FPS (default: same as input)'
    )
    
    parser.add_argument(
        '--face-index',
        type=int,
        default=0,
        help='Index of face to swap in multi-face videos (default: 0)'
    )
    
    # Advanced options
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Enable batch processing mode'
    )
    
    parser.add_argument(
        '--no-audio',
        action='store_true',
        help='Remove audio from output video'
    )
    
    parser.add_argument(
        '--preview',
        action='store_true',
        help='Preview face detection before processing'
    )
    
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Use GPU acceleration if available'
    )
    
    # Logging
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    return parser.parse_args()


def validate_file_path(filepath: str, file_type: str) -> Path:
    """Validate that a file path exists and is readable."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"{file_type} file not found: {filepath}")
    if not path.is_file():
        raise ValueError(f"{file_type} path is not a file: {filepath}")
    return path


def validate_video_format(filepath: Path) -> bool:
    """Check if video format is supported."""
    supported_formats = {'.mp4', '.avi', '.mov'}
    return filepath.suffix.lower() in supported_formats


def get_output_path(input_path: Path, output_arg: Optional[str]) -> Path:
    """Generate output path based on input and arguments."""
    if output_arg:
        return Path(output_arg)
    
    # Default: add _swapped before extension
    stem = input_path.stem
    suffix = input_path.suffix
    return input_path.parent / f"{stem}_swapped{suffix}"


def show_consent_warning():
    """Display usage warning and get user consent."""
    warning = """
╔════════════════════════════════════════════════════════════════╗
║                         WARNING                                ║
║                                                                ║
║  This tool is for PERSONAL/EDUCATIONAL USE ONLY.              ║
║                                                                ║
║  By using this software, you acknowledge that:                 ║
║  - You have appropriate permissions for all media used         ║
║  - You will not use this for malicious purposes               ║
║  - You understand the ethical implications                     ║
║  - All processing is done locally on your machine             ║
║                                                                ║
║  Continue? (yes/no):                                          ║
╚════════════════════════════════════════════════════════════════╝
"""
    print(warning)
    response = input().strip().lower()
    return response in ('yes', 'y')


def main():
    """Main CLI entry point."""
    args = parse_arguments()
    
    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARNING)
    
    try:
        # Validate input files
        input_video = validate_file_path(args.input_video, "Input video")
        source_face = validate_file_path(args.source_face, "Source face")
        target_face = validate_file_path(args.target_face, "Target face")
        
        # Check video format
        if not validate_video_format(input_video):
            logger.error(f"Unsupported video format: {input_video.suffix}")
            logger.error("Supported formats: MP4, AVI, MOV")
            return 1
        
        # Get output path
        output_path = get_output_path(input_video, args.output)
        
        # Show consent warning on first use
        consent_file = Path.home() / '.faceswap_consent'
        if not consent_file.exists():
            if not show_consent_warning():
                logger.info("User declined. Exiting.")
                return 0
            consent_file.touch()
        
        # Log processing information
        logger.info(f"Input video: {input_video}")
        logger.info(f"Source face: {source_face}")
        logger.info(f"Target face: {target_face}")
        logger.info(f"Output path: {output_path}")
        logger.info(f"Quality: {args.quality}")
        logger.info(f"GPU: {'Enabled' if args.gpu else 'Disabled'}")
        
        # TODO: Import and call actual processing functions
        # from face_swap import process_video
        # process_video(
        #     input_video=str(input_video),
        #     source_face=str(source_face),
        #     target_face=str(target_face),
        #     output_path=str(output_path),
        #     quality=args.quality,
        #     face_index=args.face_index,
        #     use_gpu=args.gpu,
        #     preview=args.preview,
        #     keep_audio=not args.no_audio,
        #     fps=args.fps
        # )
        
        logger.info("Processing complete! (Note: Core processing not yet implemented)")
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return 1
    except KeyboardInterrupt:
        logger.info("\nProcessing cancelled by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=args.debug)
        return 1


if __name__ == "__main__":
    sys.exit(main())