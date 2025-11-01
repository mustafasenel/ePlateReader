"""Core modules for plate recognition."""

from .pipeline import PlateReaderPipeline, PipelineResult
from .detector import PlateDetector
from .preprocessor import PlatePreprocessor

__all__ = [
    'PlateReaderPipeline',
    'PipelineResult',
    'PlateDetector',
    'PlatePreprocessor',
]