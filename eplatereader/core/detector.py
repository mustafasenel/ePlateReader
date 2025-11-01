"""Plate detection module using YOLO and contour-based methods."""

import cv2
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install ultralytics")
    raise


class PlateDetector:
    """License plate detector using YOLO with contour fallback."""
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu'):
        """Initialize detector.
        
        Args:
            model_path: Path to custom YOLO model (default: best.pt)
            device: Device to run on ('cpu' or 'cuda')
        """
        self.device = device
        
        # Load YOLO model
        if model_path and Path(model_path).exists():
            self.yolo = YOLO(model_path)
        else:
            self.yolo = YOLO('best.pt')
    
    def detect(self, image: np.ndarray, use_fallback: bool = True) -> List[Dict]:
        """Detect license plates in image.
        
        Args:
            image: Input image (BGR)
            use_fallback: Use contour-based fallback if YOLO fails
            
        Returns:
            List of detections with bbox, confidence, method
        """
        # Try YOLO first
        detections = self._detect_yolo(image)
        
        # Fallback to contour-based
        if not detections and use_fallback:
            print("  ⚠️  YOLO failed, trying contour-based detection...")
            detections = self._detect_contour(image)
        
        return detections
    
    def _detect_yolo(self, image: np.ndarray) -> List[Dict]:
        """YOLO-based detection."""
        results = self.yolo.predict(
            image,
            conf=0.1,  # Low threshold for better recall
            verbose=False
        )
        
        detections = []
        
        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            print(f"  YOLO found {len(boxes)} object(s)")
            
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                
                w, h = x2 - x1, y2 - y1
                aspect_ratio = w / h if h > 0 else 0
                
                print(f"  - Object: size={w}x{h}, AR={aspect_ratio:.2f}, conf={conf:.2f}")
                
                # Filter by aspect ratio and size
                # Support both horizontal (2.0-6.0) and square/vertical plates (1.5-2.5)
                is_horizontal = 2.5 <= aspect_ratio <= 6.0 and w > 80 and h > 20
                is_square = 1.5 <= aspect_ratio <= 2.8 and w > 50 and h > 40  # Two-line plates
                
                if is_horizontal or is_square:
                    plate_type = 'horizontal' if is_horizontal else 'square'
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'method': 'yolo',
                        'type': plate_type
                    })
                    print(f"    ✓ Valid {plate_type} plate candidate")
        
        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        return detections
    
    def _detect_contour(self, image: np.ndarray) -> List[Dict]:
        """Contour-based fallback detection."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 200)
            
            # Find contours
            contours, _ = cv2.findContours(
                edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            
            detections = []
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # Filter by aspect ratio and size
                is_horizontal = 2.5 <= aspect_ratio <= 6.0 and w > 80 and h > 20
                is_square = 1.5 <= aspect_ratio <= 2.8 and w > 50 and h > 40
                
                if is_horizontal or is_square:
                    plate_type = 'horizontal' if is_horizontal else 'square'
                    detections.append({
                        'bbox': [x, y, x + w, y + h],
                        'confidence': 0.5,
                        'method': 'contour',
                        'type': plate_type
                    })
            
            # Sort by area (larger is better)
            detections.sort(
                key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]),
                reverse=True
            )
            
            if detections:
                print(f"  Contour detection found {len(detections)} candidate(s)")
            
        except Exception as e:
            print(f"  Contour detection error: {e}")
            return []
        
        return detections
