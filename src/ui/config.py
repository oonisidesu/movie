"""
Configuration Management for CLI

Handles configuration file loading, validation, and default settings
for the video face swapping CLI application.
"""

import json
from pathlib import Path

try:
    import yaml
    HAS_YAML = True
except ImportError:
    yaml = None
    HAS_YAML = False
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class CLIConfig:
    """
    CLI configuration settings.
    
    Contains default settings and user-configurable options
    for the video face swapping application.
    """
    
    # Processing settings
    default_quality: str = "medium"
    default_face_detector: str = "opencv"
    default_confidence: float = 0.5
    default_chunk_size: int = 30
    
    # Output settings
    preserve_audio: bool = True
    overwrite_output: bool = False
    default_progress_mode: str = "console"
    
    # Performance settings
    max_memory_gb: float = 8.0
    enable_gpu: bool = True
    num_workers: int = 1
    
    # Directories
    temp_dir: Optional[str] = None
    cache_dir: Optional[str] = None
    
    # Logging settings
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    # Advanced settings
    face_quality_threshold: float = 0.6
    enable_face_tracking: bool = True
    tracking_smoothing: float = 0.7
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CLIConfig':
        """Create config from dictionary."""
        # Filter out unknown keys
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        
        return cls(**filtered_data)
    
    def validate(self) -> None:
        """
        Validate configuration settings.
        
        Raises:
            ValueError: If any setting is invalid
        """
        # Validate quality setting
        valid_qualities = ["low", "medium", "high"]
        if self.default_quality not in valid_qualities:
            raise ValueError(f"Invalid quality: {self.default_quality}")
        
        # Validate face detector
        valid_detectors = ["opencv", "mediapipe"]
        if self.default_face_detector not in valid_detectors:
            raise ValueError(f"Invalid face detector: {self.default_face_detector}")
        
        # Validate confidence threshold
        if not 0.0 <= self.default_confidence <= 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
        
        # Validate memory limit
        if self.max_memory_gb <= 0:
            raise ValueError("Max memory must be positive")
        
        # Validate chunk size
        if self.chunk_size <= 0:
            raise ValueError("Chunk size must be positive")
        
        # Validate face quality threshold
        if not 0.0 <= self.face_quality_threshold <= 1.0:
            raise ValueError("Face quality threshold must be between 0.0 and 1.0")
        
        # Validate tracking smoothing
        if not 0.0 <= self.tracking_smoothing <= 1.0:
            raise ValueError("Tracking smoothing must be between 0.0 and 1.0")
        
        # Validate log level
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level not in valid_levels:
            raise ValueError(f"Invalid log level: {self.log_level}")
        
        # Validate progress mode
        valid_modes = ["console", "silent"]
        if self.default_progress_mode not in valid_modes:
            raise ValueError(f"Invalid progress mode: {self.default_progress_mode}")


def load_config(config_path: str) -> CLIConfig:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to configuration file (.json or .yaml/.yml)
        
    Returns:
        CLIConfig instance
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file is invalid
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            if config_file.suffix.lower() in ['.yaml', '.yml']:
                if not HAS_YAML:
                    raise ValueError("PyYAML is required for YAML configuration files")
                data = yaml.safe_load(f)
            elif config_file.suffix.lower() == '.json':
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_file.suffix}")
        
        if not isinstance(data, dict):
            raise ValueError("Configuration file must contain a dictionary")
        
        config = CLIConfig.from_dict(data)
        config.validate()
        
        logger.info(f"Loaded configuration from: {config_path}")
        return config
        
    except (yaml.YAMLError, json.JSONDecodeError) as e:
        raise ValueError(f"Invalid configuration file format: {e}") from e
    except Exception as e:
        raise ValueError(f"Failed to load configuration: {e}") from e


def save_config(config: CLIConfig, config_path: str, format: str = "yaml") -> None:
    """
    Save configuration to file.
    
    Args:
        config: CLIConfig instance to save
        config_path: Path to save configuration file
        format: File format ("yaml" or "json")
        
    Raises:
        ValueError: If format is unsupported
    """
    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    data = config.to_dict()
    
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            if format.lower() == "yaml":
                if not HAS_YAML:
                    raise ValueError("PyYAML is required for YAML format")
                yaml.dump(data, f, default_flow_style=False, indent=2)
            elif format.lower() == "json":
                json.dump(data, f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved configuration to: {config_path}")
        
    except Exception as e:
        raise ValueError(f"Failed to save configuration: {e}") from e


def get_default_config_path() -> Path:
    """
    Get default configuration file path.
    
    Returns:
        Path to default config file
    """
    # Try different locations in order of preference
    locations = [
        Path.cwd() / "face_swap_config.yaml",
        Path.home() / ".config" / "face_swap" / "config.yaml",
        Path.home() / ".face_swap_config.yaml"
    ]
    
    for path in locations:
        if path.exists():
            return path
    
    # Return first location as default
    return locations[0]


def create_sample_config(output_path: str) -> None:
    """
    Create a sample configuration file.
    
    Args:
        output_path: Path to save sample config
    """
    config = CLIConfig()
    
    # Add comments for the sample config
    sample_data = {
        "# Processing settings": None,
        "default_quality": config.default_quality,
        "default_face_detector": config.default_face_detector,
        "default_confidence": config.default_confidence,
        "default_chunk_size": config.default_chunk_size,
        
        "# Output settings": None,
        "preserve_audio": config.preserve_audio,
        "overwrite_output": config.overwrite_output,
        "default_progress_mode": config.default_progress_mode,
        
        "# Performance settings": None,
        "max_memory_gb": config.max_memory_gb,
        "enable_gpu": config.enable_gpu,
        "num_workers": config.num_workers,
        
        "# Directories (null for auto-detection)": None,
        "temp_dir": config.temp_dir,
        "cache_dir": config.cache_dir,
        
        "# Logging settings": None,
        "log_level": config.log_level,
        "log_file": config.log_file,
        
        "# Advanced settings": None,
        "face_quality_threshold": config.face_quality_threshold,
        "enable_face_tracking": config.enable_face_tracking,
        "tracking_smoothing": config.tracking_smoothing
    }
    
    # Filter out comment keys for actual saving
    clean_data = {k: v for k, v in sample_data.items() if not k.startswith("#")}
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Video Face Swapping Tool - Configuration File\n")
        f.write("# This file contains default settings for the CLI application\n\n")
        
        if HAS_YAML:
            yaml.dump(clean_data, f, default_flow_style=False, indent=2)
        else:
            json.dump(clean_data, f, indent=2)
    
    logger.info(f"Created sample configuration: {output_path}")


# Configuration presets for different use cases
PRESETS = {
    "fast": CLIConfig(
        default_quality="low",
        default_confidence=0.3,
        default_chunk_size=50,
        max_memory_gb=4.0,
        face_quality_threshold=0.4,
        enable_face_tracking=False
    ),
    
    "balanced": CLIConfig(
        default_quality="medium",
        default_confidence=0.5,
        default_chunk_size=30,
        max_memory_gb=8.0,
        face_quality_threshold=0.6,
        enable_face_tracking=True
    ),
    
    "quality": CLIConfig(
        default_quality="high",
        default_confidence=0.7,
        default_chunk_size=10,
        max_memory_gb=16.0,
        face_quality_threshold=0.8,
        enable_face_tracking=True,
        tracking_smoothing=0.9
    )
}


def get_preset_config(preset_name: str) -> CLIConfig:
    """
    Get configuration preset by name.
    
    Args:
        preset_name: Name of the preset ("fast", "balanced", "quality")
        
    Returns:
        CLIConfig instance for the preset
        
    Raises:
        ValueError: If preset name is invalid
    """
    if preset_name not in PRESETS:
        available = ", ".join(PRESETS.keys())
        raise ValueError(f"Unknown preset: {preset_name}. Available: {available}")
    
    return PRESETS[preset_name]