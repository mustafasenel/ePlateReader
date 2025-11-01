"""Configuration management."""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Application configuration."""
    
    # Model settings
    device: str = 'cpu'
    detection_model: str = 'yolov8n.pt'
    ocr_engine: str = 'paddleocr'
    model_dir: Path = Path('models')
    
    # Processing settings
    detection_threshold: float = 0.25
    ocr_confidence_threshold: float = 0.5
    max_image_size: int = 1920
    min_plate_width: int = 50
    min_plate_height: int = 20
    aspect_ratio_min: float = 2.0
    aspect_ratio_max: float = 6.0
    
    # Logging
    log_level: str = 'INFO'
    log_file: Optional[str] = None
    
    def __post_init__(self):
        """Post initialization."""
        # Check CUDA availability
        if self.device == 'cuda':
            try:
                import torch
                if not torch.cuda.is_available():
                    self.device = 'cpu'
            except ImportError:
                self.device = 'cpu'
        
        # Create model directory
        self.model_dir = Path(self.model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)


_global_config = None


def get_config() -> Config:
    """Get global configuration."""
    global _global_config
    if _global_config is None:
        _global_config = Config()
    return _global_config


def set_config(config: Config):
    """Set global configuration."""
    global _global_config
    _global_config = config