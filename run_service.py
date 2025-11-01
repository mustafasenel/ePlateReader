#!/usr/bin/env python3
"""Run the LLM Vision Service."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import uvicorn
from eplatereader.service.config import settings


def main():
    """Run the FastAPI service."""
    print(f"""
╔══════════════════════════════════════════════════════════╗
║          LLM Vision Service                              ║
║          Qwen3-VL License Plate Recognition              ║
╚══════════════════════════════════════════════════════════╝

Starting service on {settings.api_host}:{settings.api_port}
Model: {settings.model_name}
Device: {settings.device}

API Documentation: http://localhost:{settings.api_port}/docs
Health Check: http://localhost:{settings.api_port}/health

Endpoints:
  - POST /api/v1/recognize/plate  (License Plate Recognition)
  - POST /api/v1/query             (General LLM Query)

Press CTRL+C to stop the service
""")
    
    uvicorn.run(
        "eplatereader.service.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
        log_level=settings.log_level.lower()
    )


if __name__ == "__main__":
    main()
