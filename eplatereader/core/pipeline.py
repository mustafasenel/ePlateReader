"""Main pipeline for license plate recognition."""

import time
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass

from .detector import PlateDetector
from .preprocessor import PlatePreprocessor
from .llm_recognizer import LLMPlateRecognizer
from ..utils.validation import validate_plate


@dataclass
class PipelineResult:
    """Pipeline result."""
    success: bool
    plate_text: Optional[str] = None
    confidence: Optional[float] = None
    is_valid: Optional[bool] = None
    processing_time: Optional[float] = None
    detection_method: Optional[str] = None
    error: Optional[str] = None


class PlateReaderPipeline:
    """Complete license plate reading pipeline (orchestrator)."""
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None, 
                 debug: bool = False, use_llm: bool = False):
        """Initialize pipeline.
        
        Args:
            model_path: Path to custom YOLO model
            device: Device to run on ('cpu' or 'cuda')
            debug: Enable debug output
            use_llm: Whether to use LLM-based recognition (Qwen3-VL) instead of PaddleOCR
        """
        self.debug = debug
        self.device = device or self._check_cuda()
        self.use_llm = use_llm
        
        # Initialize components
        print("Loading YOLO model...")
        self.detector = PlateDetector(model_path=model_path, device=self.device)
        self.recognizer = LLMPlateRecognizer(debug=debug)
        self.preprocessor = PlatePreprocessor(debug=debug)
        
        print(f"✅ Pipeline ready on {self.device}")
    
    def _check_cuda(self) -> str:
        """Check CUDA availability."""
        try:
            import torch
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        except:
            return 'cpu'
    
    def process_image(self, image_path: Union[str, Path]) -> PipelineResult:
        """Process image and extract plate text.
        
        Args:
            image_path: Path to input image
            
        Returns:
            PipelineResult with plate text and metadata
        """
        start_time = time.time()
        
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                return PipelineResult(
                    success=False,
                    error="Failed to load image",
                    processing_time=time.time() - start_time
                )
            
            # Step 1: Detect plates
            print("🔍 Detecting plates with YOLO...")
            detections = self.detector.detect(image, use_fallback=True)
            
            if not detections:
                return PipelineResult(
                    success=False,
                    error="No plates detected",
                    processing_time=time.time() - start_time
                )
            
            print(f"✓ Found {len(detections)} candidate(s)")
            
            # Get best detection
            best_det = detections[0]
            x1, y1, x2, y2 = best_det['bbox']
            detection_method = best_det['method']
            
            # Extract plate region
            plate_img = image[y1:y2, x1:x2]
            
            # Setup debug directory
            debug_dir = Path('debug_output') if self.debug else None
            if debug_dir:
                debug_dir.mkdir(exist_ok=True)
                cv2.imwrite(str(debug_dir / '1_cropped_plate.jpg'), plate_img)
                print(f"  💾 Saved: debug_output/1_cropped_plate.jpg")
            
            # Step 2: Preprocess (deskew only for LLM)
            print("\n  ⚙️  Preprocessing plate image...")
            preprocessed_variants = self.preprocessor.preprocess_multi(
                plate_img, 
                debug_dir, 
                just_deskew=True  # LLM only needs deskewed image
            )
            
            if debug_dir:
                print(f"  💾 Saved {len(preprocessed_variants)} preprocessing variants")
            
            # Step 3: OCR with ensemble voting
            print("\n  🔍 Running ensemble OCR on variants...")
            plate_text, confidence, best_variant = self.recognizer.recognize_ensemble(preprocessed_variants)
            
            if best_variant:
                print(f"\n  🏆 Ensemble winner: '{plate_text}' from {best_variant} ({confidence:.2%})")
            
            if not plate_text:
                return PipelineResult(
                    success=False,
                    error="OCR failed",
                    processing_time=time.time() - start_time,
                    detection_method=detection_method
                )
            
            # Step 4: Validate
            validation = validate_plate(plate_text)
            
            return PipelineResult(
                success=True,
                plate_text=plate_text,
                confidence=confidence,
                is_valid=validation.is_valid,
                processing_time=time.time() - start_time,
                detection_method=detection_method
            )
            
        except Exception as e:
            return PipelineResult(
                success=False,
                error=str(e),
                processing_time=time.time() - start_time
            )
    
    def _detect_plates_yolo(self, image: np.ndarray) -> list:
        """Detect plates using YOLO."""
        # Lower threshold for better detection
        results = self.yolo.predict(image, conf=0.1, verbose=False)
        
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            
            print(f"  YOLO found {len(boxes)} object(s)")
            
            for box, conf in zip(boxes, confidences):
                x1, y1, x2, y2 = map(int, box)
                w, h = x2 - x1, y2 - y1
                aspect_ratio = w / h if h > 0 else 0
                
                print(f"  - Object: size={w}x{h}, AR={aspect_ratio:.2f}, conf={conf:.2f}")
                
                # Filter by aspect ratio (Turkish plates ~3:1 to 5:1)
                if 2.5 <= aspect_ratio <= 5.5 and w >= 80 and h >= 20:
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': float(conf),
                        'aspect_ratio': aspect_ratio,
                        'method': 'yolo'
                    })
                    print(f"    ✓ Valid plate candidate")
        
        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        return detections
    
    def _detect_plates_contour(self, image: np.ndarray) -> list:
        """Fallback: Detect plates using contour detection."""
        detections = []
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply bilateral filter to reduce noise
            bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
            
            # Edge detection
            edged = cv2.Canny(bilateral, 30, 200)
            
            # Find contours
            contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
            
            print(f"  Contour method found {len(contours)} contour(s)")
            
            for contour in contours:
                # Approximate contour
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                area = w * h
                
                # Filter by aspect ratio and size
                if (2.5 <= aspect_ratio <= 5.5 and 
                    w >= 80 and h >= 20 and 
                    area >= 2000 and
                    len(approx) >= 4):
                    
                    detections.append({
                        'bbox': (x, y, x+w, y+h),
                        'confidence': 0.5,  # Default confidence for contour method
                        'aspect_ratio': aspect_ratio,
                        'method': 'contour'
                    })
                    print(f"  - Contour: size={w}x{h}, AR={aspect_ratio:.2f}")
                    print(f"    ✓ Valid plate candidate")
            
            # Sort by area (larger is better)
            detections.sort(key=lambda x: (x['bbox'][2]-x['bbox'][0]) * (x['bbox'][3]-x['bbox'][1]), reverse=True)
            
        except Exception as e:
            print(f"  Contour detection error: {e}")
        
        return detections
    
    def _preprocess_plate_multi(self, plate_img: np.ndarray, debug_dir: Path) -> list:
        """Generate multiple preprocessing variants and return all."""
        variants = []
        
        # Variant 1: Original (baseline)
        variants.append(('original', plate_img.copy()))
        cv2.imwrite(str(debug_dir / '2a_original.jpg'), plate_img)
        
        # Variant 2: Simple grayscale + CLAHE
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        variants.append(('clahe_only', enhanced_bgr))
        cv2.imwrite(str(debug_dir / '2b_clahe.jpg'), enhanced_bgr)
        
        # Variant 3: Illumination correction (soft)
        blur = cv2.GaussianBlur(gray, (31, 31), 0)
        normalized = cv2.divide(gray, blur, scale=255)
        clahe2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        corrected = clahe2.apply(normalized)
        corrected_bgr = cv2.cvtColor(corrected, cv2.COLOR_GRAY2BGR)
        variants.append(('illumination_corrected', corrected_bgr))
        cv2.imwrite(str(debug_dir / '2c_illumination.jpg'), corrected_bgr)
        
        # Variant 4: Bilateral filter + soft threshold
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        # Use Otsu's method (automatic threshold)
        _, otsu = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Invert if needed
        if cv2.mean(otsu)[0] < 127:
            otsu = cv2.bitwise_not(otsu)
        otsu_bgr = cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR)
        variants.append(('otsu_threshold', otsu_bgr))
        cv2.imwrite(str(debug_dir / '2d_otsu.jpg'), otsu_bgr)
        
        # Variant 5: Advanced (original method but softer)
        advanced = self._preprocess_plate_advanced(plate_img)
        variants.append(('advanced', advanced))
        cv2.imwrite(str(debug_dir / '2e_advanced.jpg'), advanced)
        
        return variants
    
    def _preprocess_plate_advanced(self, plate_img: np.ndarray) -> np.ndarray:
        """Advanced preprocessing for optimal OCR with glare removal."""
        # Step 1: Resize if too small
        h, w = plate_img.shape[:2]
        if h < 50 or w < 150:
            scale = max(50 / h, 150 / w)
            new_w, new_h = int(w * scale), int(h * scale)
            plate_img = cv2.resize(plate_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Step 2: Convert to grayscale
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        
        # Step 3: Morphological gradient to enhance edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph_grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        
        # Step 4: Remove glare using illumination correction
        # Estimate background illumination
        blur = cv2.GaussianBlur(gray, (51, 51), 0)
        # Normalize by dividing by background
        normalized = cv2.divide(gray, blur, scale=255)
        
        # Step 5: CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(normalized)
        
        # Step 6: Bilateral filter (denoise while preserving edges)
        bilateral = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # Step 7: Adaptive thresholding
        adaptive = cv2.adaptiveThreshold(
            bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Step 8: Morphological operations to clean up
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel_clean)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_clean)
        
        # Step 9: Invert if needed (text should be dark on light background)
        if cv2.mean(cleaned)[0] < 127:
            cleaned = cv2.bitwise_not(cleaned)
        
        # Step 10: Convert back to BGR for PaddleOCR
        result = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
        
        # Step 11: Sharpen
        kernel_sharp = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(result, -1, kernel_sharp)
        
        return sharpened
    
    def _recognize_plate(self, plate_img: np.ndarray) -> tuple:
        """Recognize text using PaddleOCR."""
        try:
            print("  Running OCR...")
            results = self.ocr.ocr(plate_img)
            
            # DEBUG: Print full raw results
            print("\n" + "="*60)
            print("DEBUG: PaddleOCR RAW OUTPUT")
            print("="*60)
            if results:
                for idx, res in enumerate(results):
                    print(f"\nResult {idx}:")
                    if isinstance(res, dict):
                        print(f"  Type: dict")
                        print(f"  Keys: {res.keys()}")
                        if 'rec_texts' in res:
                            print(f"  rec_texts: {res['rec_texts']}")
                        if 'rec_scores' in res:
                            print(f"  rec_scores: {res['rec_scores']}")
                        if 'dt_polys' in res:
                            print(f"  dt_polys count: {len(res.get('dt_polys', []))}")
                        if 'rec_polys' in res:
                            print(f"  rec_polys count: {len(res.get('rec_polys', []))}")
                    elif isinstance(res, list):
                        print(f"  Type: list, length: {len(res)}")
                        for i, item in enumerate(res[:3]):  # Show first 3 items
                            print(f"    Item {i}: {item}")
                    else:
                        print(f"  Type: {type(res)}")
                        print(f"  Value: {res}")
            print("="*60 + "\n")
            
            if results and len(results) > 0:
                result = results[0]
                
                # New PaddleOCR format: check for 'rec_texts' field
                if isinstance(result, dict) and 'rec_texts' in result:
                    rec_texts = result['rec_texts']
                    rec_scores = result.get('rec_scores', [])
                    rec_polys = result.get('rec_polys', [])
                    
                    print(f"  ✓ OCR detected {len(rec_texts)} text block(s)")
                    
                    if rec_texts:
                        # Collect all text blocks with their positions
                        text_blocks = []
                        
                        for i, text in enumerate(rec_texts):
                            conf = rec_scores[i] if i < len(rec_scores) else 0.9
                            
                            # Get x position for sorting (left to right)
                            x_pos = 0
                            if i < len(rec_polys) and rec_polys[i] is not None:
                                try:
                                    x_pos = int(rec_polys[i][0][0])  # Top-left x coordinate
                                except:
                                    x_pos = i * 1000  # Fallback: use index
                            else:
                                x_pos = i * 1000
                            
                            text_blocks.append({
                                'text': str(text),
                                'conf': float(conf),
                                'x_pos': x_pos
                            })
                            print(f"    Block {i+1}: '{text}' at x={x_pos} (conf: {conf:.2%})")
                        
                        # Sort by x position (left to right)
                        text_blocks.sort(key=lambda x: x['x_pos'])
                        
                        # Combine all texts in correct order
                        texts = [block['text'] for block in text_blocks]
                        confidences = [block['conf'] for block in text_blocks]
                        
                        # Join with space first, then clean
                        full_text = ' '.join(texts)
                        print(f"  → Combined: '{full_text}'")
                        
                        # Clean text (removes spaces and non-alphanumeric)
                        full_text = self._clean_text(full_text)
                        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
                        
                        print(f"  ✓ Final text: '{full_text}' (avg conf: {avg_conf:.2%})")
                        return full_text, avg_conf
                
                # Fallback: old format
                elif isinstance(result, list):
                    print("  Using old OCR format...")
                    texts = []
                    confidences = []
                    
                    for line in result:
                        if line and len(line) >= 2:
                            try:
                                if isinstance(line[1], (list, tuple)) and len(line[1]) >= 2:
                                    text = str(line[1][0])
                                    conf = float(line[1][1])
                                    texts.append(text)
                                    confidences.append(conf)
                                    print(f"    Text: '{text}' (conf: {conf:.2f})")
                            except Exception as e:
                                print(f"    Error parsing line: {e}")
                                continue
                    
                    if texts:
                        full_text = ''.join(texts)
                        full_text = self._clean_text(full_text)
                        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
                        print(f"  Final text: '{full_text}' (avg conf: {avg_conf:.2f})")
                        return full_text, avg_conf
            
            print("  ⚠️  No text detected")
            return None, 0.0
            
        except Exception as e:
            print(f"❌ OCR error: {e}")
            import traceback
            traceback.print_exc()
            return None, 0.0
    
    def _clean_text(self, text: str) -> str:
        """Clean OCR text."""
        # Remove whitespace and convert to uppercase
        text = ''.join(text.split()).upper()
        # Remove non-alphanumeric
        text = ''.join(c for c in text if c.isalnum())
        return text
    
    def _ensemble_vote(self, ocr_results: list) -> tuple:
        """Ensemble voting: combine OCR confidence + validation score.
        
        Scoring:
        - OCR confidence: 0-100%
        - Validation score: +20% if valid format, -10% if invalid
        - Text frequency: +10% for most common result
        """
        from collections import Counter
        from ..utils.validation import validate_plate
        
        # Count text frequencies
        text_counts = Counter([r['text'] for r in ocr_results])
        most_common_text = text_counts.most_common(1)[0][0] if text_counts else None
        
        scored_results = []
        
        for result in ocr_results:
            text = result['text']
            ocr_conf = result['confidence']
            variant = result['variant']
            
            # Start with OCR confidence
            score = ocr_conf
            
            # Validate plate format
            validation = validate_plate(text)
            if validation.is_valid:
                score += 0.20  # Bonus for valid format
                # Extra bonus for high validation confidence
                score += validation.confidence * 0.10
            else:
                score -= 0.10  # Penalty for invalid format
            
            # Bonus for most common result (consensus)
            if text == most_common_text and text_counts[text] > 1:
                score += 0.10
            
            # Variant-specific bonuses (based on empirical performance)
            variant_weights = {
                'clahe_only': 0.05,  # CLAHE often works best
                'illumination_corrected': 0.03,
                'original': 0.02,
                'otsu_threshold': 0.0,
                'advanced': -0.02  # Can be too aggressive
            }
            score += variant_weights.get(variant, 0.0)
            
            # Cap score at 1.0
            score = min(score, 1.0)
            
            scored_results.append({
                'text': text,
                'score': score,
                'variant': variant,
                'ocr_conf': ocr_conf,
                'is_valid': validation.is_valid
            })
        
        # Sort by score
        scored_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Return best result
        best = scored_results[0]
        return best['text'], best['score'], best['variant']
    
    def _validate_plate(self, plate_text: str) -> bool:
        """Basic validation for Turkish plates."""
        import re
        
        # Turkish plate patterns
        patterns = [
            r'^\d{2}[A-Z]{1,3}\d{2,4}$',  # 34ABC123, 34AB1234, etc.
        ]
        
        for pattern in patterns:
            if re.match(pattern, plate_text):
                # Check city code (01-81)
                try:
                    city_code = int(plate_text[:2])
                    if 1 <= city_code <= 81:
                        return True
                except:
                    pass
        
        return False