"""FastAPI application for LLM Vision Service."""

import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import settings
from .models import (
    PlateRecognitionRequest,
    PlateRecognitionResponse,
    GeneralQueryRequest,
    GeneralQueryResponse,
    HealthResponse
)
from .llm_service import llm_service


# Create FastAPI app
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description="LLM-based Vision Service for License Plate Recognition and General Queries",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allow_origins,
    allow_credentials=settings.allow_credentials,
    allow_methods=settings.allow_methods,
    allow_headers=settings.allow_headers,
)


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    print("Starting LLM Vision Service...")
    print(f"Loading model: {settings.model_name}")
    llm_service.load_model()
    print("Service ready!")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "service": settings.api_title,
        "version": settings.api_version,
        "status": "running",
        "endpoints": {
            "health": "/health",
            "plate_recognition": "/api/v1/recognize/plate",
            "general_query": "/api/v1/query",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if llm_service.is_model_loaded() else "model_not_loaded",
        model_loaded=llm_service.is_model_loaded(),
        device=llm_service.device
    )


@app.post("/api/v1/recognize/plate", response_model=PlateRecognitionResponse, tags=["Plate Recognition"])
async def recognize_plate(request: PlateRecognitionRequest):
    """Recognize Turkish license plate from image.
    
    This endpoint accepts a base64 encoded image and returns the recognized plate number.
    
    Args:
        request: PlateRecognitionRequest with base64 encoded image
        
    Returns:
        PlateRecognitionResponse with plate text and confidence
    """
    start_time = time.time()
    
    try:
        if not llm_service.is_model_loaded():
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Recognize plate
        plate_text, confidence = llm_service.recognize_plate(request.image_base64)
        
        processing_time = time.time() - start_time
        
        if plate_text:
            return PlateRecognitionResponse(
                success=True,
                plate_text=plate_text,
                confidence=confidence,
                error=None,
                processing_time=processing_time
            )
        else:
            return PlateRecognitionResponse(
                success=False,
                plate_text=None,
                confidence=0.0,
                error="Failed to recognize plate",
                processing_time=processing_time
            )
    
    except Exception as e:
        processing_time = time.time() - start_time
        return PlateRecognitionResponse(
            success=False,
            plate_text=None,
            confidence=0.0,
            error=str(e),
            processing_time=processing_time
        )


@app.post("/api/v1/query", response_model=GeneralQueryResponse, tags=["General Query"])
async def general_query(request: GeneralQueryRequest):
    """General purpose LLM query with image.
    
    This endpoint accepts a base64 encoded image and a text prompt,
    and returns the LLM's response.
    
    Args:
        request: GeneralQueryRequest with image and prompt
        
    Returns:
        GeneralQueryResponse with LLM response
    """
    start_time = time.time()
    
    try:
        if not llm_service.is_model_loaded():
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Query LLM
        response = llm_service.query(
            request.image_base64,
            request.prompt,
            request.max_tokens
        )
        
        processing_time = time.time() - start_time
        
        if response:
            return GeneralQueryResponse(
                success=True,
                response=response,
                error=None,
                processing_time=processing_time
            )
        else:
            return GeneralQueryResponse(
                success=False,
                response=None,
                error="Failed to generate response",
                processing_time=processing_time
            )
    
    except Exception as e:
        processing_time = time.time() - start_time
        return GeneralQueryResponse(
            success=False,
            response=None,
            error=str(e),
            processing_time=processing_time
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": str(exc),
            "detail": "Internal server error"
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower()
    )
