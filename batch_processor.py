"""
Batch processing utilities for the face swapping CLI.

Handles processing multiple videos or configurations in batch mode.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
import time
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class BatchJob:
    """Represents a single batch processing job."""
    input_video: str
    source_face: str
    target_face: str
    output_path: str
    quality: str = 'medium'
    face_index: int = 0
    fps: Optional[int] = None
    no_audio: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BatchJob':
        """Create from dictionary."""
        return cls(**data)


class BatchProcessor:
    """Handles batch processing operations."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path) if config_path else None
        self.jobs: List[BatchJob] = []
        self.results: List[Dict[str, Any]] = []
    
    def load_config(self, config_path: str):
        """Load batch configuration from JSON file."""
        config_file = Path(config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(f"Batch config file not found: {config_path}")
        
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Validate config structure
            if 'jobs' not in config_data:
                raise ValueError("Config file must contain 'jobs' array")
            
            # Load jobs
            self.jobs = []
            for i, job_data in enumerate(config_data['jobs']):
                try:
                    job = BatchJob.from_dict(job_data)
                    self.jobs.append(job)
                except Exception as e:
                    logger.warning(f"Skipping invalid job {i}: {e}")
            
            logger.info(f"Loaded {len(self.jobs)} jobs from {config_path}")
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading config: {e}")
    
    def add_job(self, job: BatchJob):
        """Add a job to the batch."""
        self.jobs.append(job)
    
    def validate_jobs(self) -> List[str]:
        """Validate all jobs and return list of errors."""
        errors = []
        
        for i, job in enumerate(self.jobs):
            job_errors = []
            
            # Check input files exist
            if not Path(job.input_video).exists():
                job_errors.append(f"Input video not found: {job.input_video}")
            
            if not Path(job.source_face).exists():
                job_errors.append(f"Source face not found: {job.source_face}")
            
            if not Path(job.target_face).exists():
                job_errors.append(f"Target face not found: {job.target_face}")
            
            # Check output directory exists
            output_dir = Path(job.output_path).parent
            if not output_dir.exists():
                job_errors.append(f"Output directory not found: {output_dir}")
            
            # Add job-specific errors
            if job_errors:
                errors.append(f"Job {i}: " + "; ".join(job_errors))
        
        return errors
    
    def process_batch(self, dry_run: bool = False) -> bool:
        """
        Process all jobs in the batch.
        
        Args:
            dry_run: If True, validate jobs but don't process
            
        Returns:
            True if all jobs completed successfully
        """
        if not self.jobs:
            logger.warning("No jobs to process")
            return True
        
        # Validate jobs
        errors = self.validate_jobs()
        if errors:
            logger.error("Batch validation failed:")
            for error in errors:
                logger.error(f"  {error}")
            return False
        
        if dry_run:
            logger.info(f"Dry run: {len(self.jobs)} jobs validated successfully")
            return True
        
        # Process jobs
        successful = 0
        failed = 0
        
        start_time = time.time()
        
        for i, job in enumerate(self.jobs):
            logger.info(f"Processing job {i+1}/{len(self.jobs)}: {Path(job.input_video).name}")
            
            try:
                result = self._process_single_job(job, i)
                self.results.append(result)
                
                if result['success']:
                    successful += 1
                    logger.info(f"Job {i+1} completed successfully")
                else:
                    failed += 1
                    logger.error(f"Job {i+1} failed: {result.get('error', 'Unknown error')}")
                
            except KeyboardInterrupt:
                logger.info("Batch processing cancelled by user")
                break
            except Exception as e:
                failed += 1
                error_result = {
                    'job_index': i,
                    'input_video': job.input_video,
                    'success': False,
                    'error': str(e),
                    'processing_time': 0
                }
                self.results.append(error_result)
                logger.error(f"Job {i+1} failed with error: {e}")
        
        # Summary
        total_time = time.time() - start_time
        logger.info(f"Batch processing complete:")
        logger.info(f"  Total jobs: {len(self.jobs)}")
        logger.info(f"  Successful: {successful}")
        logger.info(f"  Failed: {failed}")
        logger.info(f"  Total time: {total_time:.1f}s")
        
        return failed == 0
    
    def _process_single_job(self, job: BatchJob, job_index: int) -> Dict[str, Any]:
        """Process a single job and return result."""
        start_time = time.time()
        
        try:
            # Import processing modules
            from video_processor import VideoProcessor
            from progress import create_progress_callback
            
            # Setup paths
            input_video = Path(job.input_video)
            output_path = Path(job.output_path)
            
            # Handle audio extraction
            audio_path = None
            if not job.no_audio:
                audio_path = output_path.with_suffix('.aac')
                VideoProcessor.extract_audio(str(input_video), str(audio_path))
            
            # Process video
            with VideoProcessor(str(input_video), str(output_path), job.quality) as processor:
                processor.setup_writer(job.fps)
                
                video_info = processor.get_video_info()
                progress_callback = create_progress_callback(
                    video_info['total_frames'],
                    f"Job {job_index+1} ({job.quality})"
                )
                
                def frame_processor(frame_num: int, frame):
                    """Process individual frame."""
                    # TODO: Implement actual face swapping
                    return frame
                
                processor.process_with_callback(frame_processor, progress_callback)
            
            # Merge audio
            if audio_path and audio_path.exists():
                temp_video = output_path.with_suffix('.temp.mp4')
                output_path.rename(temp_video)
                
                if VideoProcessor.merge_audio(str(temp_video), str(audio_path), str(output_path)):
                    temp_video.unlink()
                    audio_path.unlink()
                else:
                    temp_video.rename(output_path)
            
            processing_time = time.time() - start_time
            
            return {
                'job_index': job_index,
                'input_video': job.input_video,
                'output_path': job.output_path,
                'success': True,
                'processing_time': processing_time
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            return {
                'job_index': job_index,
                'input_video': job.input_video,
                'success': False,
                'error': str(e),
                'processing_time': processing_time
            }
    
    def save_results(self, output_path: str):
        """Save batch processing results to JSON file."""
        results_data = {
            'total_jobs': len(self.jobs),
            'successful_jobs': sum(1 for r in self.results if r['success']),
            'failed_jobs': sum(1 for r in self.results if not r['success']),
            'total_processing_time': sum(r['processing_time'] for r in self.results),
            'jobs': self.results
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Results saved to: {output_path}")
    
    @staticmethod
    def create_sample_config(output_path: str):
        """Create a sample batch configuration file."""
        sample_config = {
            "jobs": [
                {
                    "input_video": "video1.mp4",
                    "source_face": "face1.jpg",
                    "target_face": "target1.jpg",
                    "output_path": "output1.mp4",
                    "quality": "medium",
                    "face_index": 0,
                    "fps": None,
                    "no_audio": False
                },
                {
                    "input_video": "video2.mp4",
                    "source_face": "face2.jpg",
                    "target_face": "target2.jpg",
                    "output_path": "output2.mp4",
                    "quality": "high",
                    "face_index": 0,
                    "fps": 30,
                    "no_audio": True
                }
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(sample_config, f, indent=2)
        
        logger.info(f"Sample config created at: {output_path}")


def auto_generate_batch_jobs(video_dir: str, source_face: str, target_face: str, 
                           output_dir: str, **kwargs) -> List[BatchJob]:
    """
    Auto-generate batch jobs for all videos in a directory.
    
    Args:
        video_dir: Directory containing input videos
        source_face: Path to source face image
        target_face: Path to target face image
        output_dir: Directory for output videos
        **kwargs: Additional job parameters
        
    Returns:
        List of generated batch jobs
    """
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    
    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")
    
    # Find all video files
    video_extensions = {'.mp4', '.avi', '.mov'}
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(video_dir.glob(f"*{ext}"))
        video_files.extend(video_dir.glob(f"*{ext.upper()}"))
    
    if not video_files:
        logger.warning(f"No video files found in {video_dir}")
        return []
    
    # Generate jobs
    jobs = []
    for video_file in video_files:
        output_name = f"{video_file.stem}_swapped{video_file.suffix}"
        output_path = output_dir / output_name
        
        job = BatchJob(
            input_video=str(video_file),
            source_face=source_face,
            target_face=target_face,
            output_path=str(output_path),
            **kwargs
        )
        jobs.append(job)
    
    logger.info(f"Generated {len(jobs)} batch jobs")
    return jobs