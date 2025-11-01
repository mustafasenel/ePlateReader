"""ePlateReader - Professional Turkish License Plate Recognition."""

__version__ = "1.0.0"
__author__ = "Senel"

from .core.pipeline import PlateReaderPipeline, PipelineResult
from .utils.logger import setup_logger, get_logger
from .utils.config import Config, get_config

__all__ = [
    '__version__',
    'PlateReaderPipeline',
    'PipelineResult',
    'setup_logger',
    'get_logger',
    'Config',
    'get_config',
]