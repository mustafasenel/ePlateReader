"""LLM-based license plate recognizer using Qwen3-VL."""

import base64
import requests
from typing import List, Tuple, Optional
import cv2
import numpy as np


class LLMPlateRecognizer:
    """License plate recognizer using LLM service via HTTP API."""
    
    def __init__(self, debug: bool = False, service_url: str = "http://localhost:8000"):
        """Initialize the LLM recognizer.
        
        Args:
            debug: Enable debug output
            service_url: URL of the LLM service
        """
        self.debug = debug
        self.service_url = service_url
        
        # Check if service is available
        self._check_service()
        
        if self.debug:
            print(f"LLM Recognizer initialized (Service: {self.service_url})")
    
    def _check_service(self):
        """Check if LLM service is running."""
        try:
            response = requests.get(f"{self.service_url}/health", timeout=2)
            if response.status_code == 200:
                health = response.json()
                if self.debug:
                    print(f"✓ LLM Service is healthy (device: {health.get('device', 'unknown')})")
                return True
        except Exception as e:
            print(f"⚠️  Warning: LLM Service not available at {self.service_url}")
            print(f"   Please start the service with: python run_service.py")
            print(f"   Error: {e}")
            raise ConnectionError(f"LLM Service not available. Please start it first.")
        
        return False
    
    def recognize(self, plate_img: np.ndarray) -> Tuple[Optional[str], float]:
        """Recognize text from plate image using Qwen3-VL via HTTP API.
        
        Args:
            plate_img: Preprocessed plate image (BGR format)
            
        Returns:
            Tuple of (recognized_text, confidence)
        """
        try:
            if self.debug:
                print("LLM Recognizer: Converting image to base64...")
            
            # Ensure image is not empty
            if plate_img is None or plate_img.size == 0:
                raise ValueError("Input plate_img is empty or None.")
            
            # Encode image to JPEG format in memory
            success, buffer = cv2.imencode('.jpg', plate_img)
            if not success:
                raise ValueError("Failed to encode image to JPEG format")
            
            # Convert to base64
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            if self.debug:
                print("LLM Recognizer: Sending request to LLM service...")
            
            # Send HTTP request to service
            response = requests.post(
                f"{self.service_url}/api/v1/recognize/plate",
                json={"image_base64": image_base64},
                timeout=30  # 30 second timeout
            )
            
            if response.status_code != 200:
                raise Exception(f"Service returned status code {response.status_code}")
            
            result = response.json()
            
            if not result.get("success"):
                error = result.get("error", "Unknown error")
                raise Exception(f"Recognition failed: {error}")
            
            plate_text = result.get("plate_text")
            confidence = result.get("confidence", 0.0)
            
            if self.debug:
                print(f"LLM Recognizer: Result: '{plate_text}' (Confidence: {confidence:.2%})")

            return plate_text, confidence
            
        except Exception as e:
            if self.debug:
                print(f"LLM Recognizer: Recognition error: {e}")
            return None, 0.0
    
        
    def recognize_ensemble(self, variants: List[Tuple[str, np.ndarray]]) -> Tuple[Optional[str], float, Optional[str]]:
        """Recognize text from multiple variants of the plate image.
        
        Args:
            variants: List of (variant_name, image) tuples
            
        Returns:
            Tuple of (recognized_text, confidence, variant_name)
        """
        if not variants:
            return None, 0.0, None
            
        # For LLM-based recognition, running on multiple variants can be
        # computationally expensive. We'll typically use the "best" or first
        # variant from the preprocessor.
        # If true ensemble with LLM is desired, a more complex voting logic
        # would be implemented here, potentially involving multiple LLM calls.
        
        variant_name, variant_img = variants[0] # Use the first variant for LLM
        if self.debug:
            print(f"LLM Recognizer: Ensemble (using first variant '{variant_name}' for LLM recognition)...")
        
        text, confidence = self.recognize(variant_img)
        
        if self.debug:
            print(f"LLM Recognizer: Result for '{variant_name}': '{text}' ({confidence:.2%})")
            
        return text, confidence, variant_name