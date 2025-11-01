"""Configuration for LLM service."""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Service settings."""
    
    # Model settings
    model_name: str = "Qwen/Qwen3-VL-2B-Instruct"
    device: str = "auto"  # auto, cuda, mps, cpu
    max_new_tokens: int = 50
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_title: str = "LLM Vision Service"
    api_version: str = "1.0.0"
    
    # CORS settings
    allow_origins: list = ["*"]
    allow_credentials: bool = True
    allow_methods: list = ["*"]
    allow_headers: list = ["*"]
    
    # Logging
    log_level: str = "INFO"
    
    class Config:
        env_prefix = "LLM_SERVICE_"
        case_sensitive = False


settings = Settings()
