"""LLM Service wrapper for FastAPI."""

import base64
import tempfile
import os
import time
from typing import Tuple, Optional
import torch
import numpy as np
import cv2
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from .config import settings


class LLMService:
    """LLM Service for vision-language tasks."""
    
    def __init__(self):
        """Initialize the LLM service."""
        self.model = None
        self.processor = None
        self.device = self._get_device()
        self.model_name = settings.model_name
        
    def _get_device(self) -> str:
        """Determine the best available device."""
        if settings.device != "auto":
            return settings.device
            
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    
    def load_model(self):
        """Load the Qwen3-VL model and processor."""
        if self.model is None:
            print(f"Loading {self.model_name} on {self.device}...")
            
            try:
                self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                    self.model_name,
                    device_map="auto",
                    torch_dtype="auto"
                )
                if str(self.model.device) != self.device:
                    self.model.to(self.device)

                self.processor = AutoProcessor.from_pretrained(self.model_name)
                print("Model and Processor loaded successfully.")
            except Exception as e:
                print(f"ERROR: Failed to load model: {e}")
                raise
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None and self.processor is not None
    
    def recognize_plate(self, image_base64: str) -> Tuple[Optional[str], float]:
        """Recognize license plate from base64 image.
        
        Args:
            image_base64: Base64 encoded image string
            
        Returns:
            Tuple of (plate_text, confidence)
        """
        if not self.is_model_loaded():
            self.load_model()
        
        temp_file_path = None
        try:
            # Decode base64 to image
            image_data = base64.b64decode(image_base64)
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                temp_file_path = temp_file.name
                temp_file.write(image_data)
            
            # Prepare prompt for Turkish license plates
            prompt = """This is a Turkish license plate image. Carefully read the license plate number.
            The format is typically: 2 digits, then 1-3 uppercase letters, then 2-4 digits.
            
            Example: 34ABC123 or 06XYZ9876
            
            Strict rules for the output:
            1. ONLY the license plate number. No other text, punctuation, or descriptions.
            2. All characters must be alphanumeric (A-Z, 0-9).
            3. All letters must be uppercase.
            4. No spaces between characters.
            
            If you are unable to detect a plate number or are unsure, return an empty string."""
            
            # Prepare messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"data:image/jpeg;base64,{image_base64}"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Process and generate
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=settings.max_new_tokens,
                )
            
            # Decode response
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0].strip()
            
            # Clean output
            plate_text = ''.join(c.upper() for c in output_text if c.isalnum())
            confidence = 0.95 if plate_text else 0.0
            
            return plate_text, confidence
            
        except Exception as e:
            print(f"Error in recognize_plate: {e}")
            return None, 0.0
        finally:
            # Clean up temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
    
    def query(self, image_base64: str, prompt: str, max_tokens: int = 50) -> Optional[str]:
        """General purpose LLM query with image.
        
        Args:
            image_base64: Base64 encoded image string
            prompt: User's question or instruction
            max_tokens: Maximum tokens to generate
            
        Returns:
            LLM response text
        """
        if not self.is_model_loaded():
            self.load_model()
        
        temp_file_path = None
        try:
            # Decode base64 to image
            image_data = base64.b64decode(image_base64)
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                temp_file_path = temp_file.name
                temp_file.write(image_data)
            
            # Prepare messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"data:image/jpeg;base64,{image_base64}"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Process and generate
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                )
            
            # Decode response
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0].strip()
            
            return output_text
            
        except Exception as e:
            print(f"Error in query: {e}")
            return None
        finally:
            # Clean up temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)


# Global service instance
llm_service = LLMService()
