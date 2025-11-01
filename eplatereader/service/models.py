"""Pydantic models for API requests and responses."""

from pydantic import BaseModel, Field
from typing import Optional


class PlateRecognitionRequest(BaseModel):
    """Request model for plate recognition."""
    
    image_base64: str = Field(..., description="Base64 encoded image")
    
    class Config:
        json_schema_extra = {
            "example": {
                "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
            }
        }


class PlateRecognitionResponse(BaseModel):
    """Response model for plate recognition."""
    
    success: bool = Field(..., description="Whether recognition was successful")
    plate_text: Optional[str] = Field(None, description="Recognized plate number")
    confidence: Optional[float] = Field(None, description="Recognition confidence (0-1)")
    error: Optional[str] = Field(None, description="Error message if failed")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "plate_text": "34ABC123",
                "confidence": 0.95,
                "error": None,
                "processing_time": 2.5
            }
        }


class GeneralQueryRequest(BaseModel):
    """Request model for general LLM queries."""
    
    image_base64: str = Field(..., description="Base64 encoded image")
    prompt: str = Field(..., description="Question or instruction for the LLM")
    max_tokens: Optional[int] = Field(50, description="Maximum tokens to generate")
    
    class Config:
        json_schema_extra = {
            "example": {
                "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                "prompt": "What do you see in this image?",
                "max_tokens": 100
            }
        }


class GeneralQueryResponse(BaseModel):
    """Response model for general LLM queries."""
    
    success: bool = Field(..., description="Whether query was successful")
    response: Optional[str] = Field(None, description="LLM response")
    error: Optional[str] = Field(None, description="Error message if failed")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "response": "I see a license plate with the number 34ABC123",
                "error": None,
                "processing_time": 2.5
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    device: str = Field(..., description="Device being used")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "device": "mps"
            }
        }
