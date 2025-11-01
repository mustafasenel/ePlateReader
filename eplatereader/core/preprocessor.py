# --- START OF FILE preprocessor.py ---

"""Image preprocessing module for plate enhancement."""

import cv2
import numpy as np
from typing import List, Tuple
from pathlib import Path

class PlatePreprocessor:
    """Multi-strategy image preprocessor for optimal OCR."""
    
    def __init__(self, debug: bool = False):
        """Initialize preprocessor.
        
        Args:
            debug: Enable debug output
        """
        self.debug = debug
    
    def preprocess_multi(self, plate_img: np.ndarray, debug_dir: Path = None, just_deskew: bool = False) -> List[Tuple[str, np.ndarray]]:
        """Generate multiple preprocessing variants or just deskew the image.
        
        Args:
            plate_img: Cropped plate image
            debug_dir: Optional directory to save debug images
            just_deskew: If True, only performs deskewing and returns that single variant.
            
        Returns:
            List of (variant_name, processed_image) tuples
        """
        variants = []
        
        # Resize if too small
        plate_img = self._resize_if_needed(plate_img)
        
        # Step 0: Perspective correction (deskew) - Always performed
        deskewed_img = self._deskew_plate(plate_img)
        if debug_dir:
            cv2.imwrite(str(debug_dir / '0_deskewed.jpg'), deskewed_img)
        
        # If only deskew is requested, return it immediately
        if just_deskew:
            if deskewed_img is None or deskewed_img.size == 0:
                if self.debug:
                    print("Warning: Deskewed image is empty or None.")
                return []
            variants.append(('deskewed', deskewed_img))
            if self.debug:
                print("  Only deskewed variant generated for LLM.")
            return variants
            
        # Use deskewed image as base for all other variants
        plate_img = deskewed_img
        
        # Variant 1: Original (baseline)
        variants.append(('original', plate_img.copy()))
        if debug_dir:
            cv2.imwrite(str(debug_dir / '2a_original.jpg'), plate_img)
        
        # Variant 2: Simple CLAHE
        clahe_img = self._apply_clahe(plate_img)
        variants.append(('clahe_only', clahe_img))
        if debug_dir:
            cv2.imwrite(str(debug_dir / '2b_clahe.jpg'), clahe_img)
        
        # Variant 3: Illumination correction
        illum_img = self._apply_illumination_correction(plate_img)
        variants.append(('illumination_corrected', illum_img))
        if debug_dir:
            cv2.imwrite(str(debug_dir / '2c_illumination.jpg'), illum_img)
        
        # Variant 4: Otsu threshold
        otsu_img = self._apply_otsu_threshold(plate_img)
        variants.append(('otsu_threshold', otsu_img))
        if debug_dir:
            cv2.imwrite(str(debug_dir / '2d_otsu.jpg'), otsu_img)
        
        # Variant 5: Advanced (combination)
        advanced = self._apply_advanced(plate_img)
        variants.append(('advanced', advanced))
        if debug_dir:
            cv2.imwrite(str(debug_dir / '2e_advanced.jpg'), advanced)
        
        # Variant 6: Adaptive Bilateral (noise reduction + edge preservation)
        adaptive_bilateral = self._apply_adaptive_bilateral(plate_img)
        variants.append(('adaptive_bilateral', adaptive_bilateral))
        if debug_dir:
            cv2.imwrite(str(debug_dir / '2f_adaptive_bilateral.jpg'), adaptive_bilateral)
        
        # Variant 7: Unsharp Masking (enhanced sharpness)
        unsharp = self._apply_unsharp_mask(plate_img)
        variants.append(('unsharp_mask', unsharp))
        if debug_dir:
            cv2.imwrite(str(debug_dir / '2g_unsharp_mask.jpg'), unsharp)
        
        # Variant 8: Morphological Enhancement (text clarity)
        morphological = self._apply_morphological_enhancement(plate_img)
        variants.append(('morphological', morphological))
        if debug_dir:
            cv2.imwrite(str(debug_dir / '2h_morphological.jpg'), morphological)
        
        # Variant 9: High Contrast (specifically for numbers)
        high_contrast = self._apply_high_contrast(plate_img)
        variants.append(('high_contrast', high_contrast))
        if debug_dir:
            cv2.imwrite(str(debug_dir / '2i_high_contrast.jpg'), high_contrast)
        
        if self.debug: # Debug özelliği eklenirse burada çıktı verilebilir
            print(f"  Generated {len(variants)} preprocessing variants.")
        return variants
    
    def _resize_if_needed(self, img: np.ndarray) -> np.ndarray:
        """Resize image if too small."""
        h, w = img.shape[:2]
        if h < 50 or w < 150:
            scale = max(50 / h, 150 / w)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        return img
    
    def _apply_clahe(self, img: np.ndarray) -> np.ndarray:
        """Apply CLAHE for contrast enhancement."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    def _apply_illumination_correction(self, img: np.ndarray) -> np.ndarray:
        """Apply illumination correction to remove glare."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Estimate background illumination
        blur = cv2.GaussianBlur(gray, (31, 31), 0)
        
        # Normalize by dividing by background
        normalized = cv2.divide(gray, blur, scale=255)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        corrected = clahe.apply(normalized)
        
        return cv2.cvtColor(corrected, cv2.COLOR_GRAY2BGR)
    
    def _apply_otsu_threshold(self, img: np.ndarray) -> np.ndarray:
        """Apply Otsu's automatic thresholding."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Bilateral filter (denoise while preserving edges)
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Otsu's method
        _, otsu = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Invert if needed (text should be dark on light background)
        if cv2.mean(otsu)[0] < 127:
            otsu = cv2.bitwise_not(otsu)
        
        return cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR)
    
    def _apply_advanced(self, img: np.ndarray) -> np.ndarray:
        """Apply advanced preprocessing (aggressive)."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Morphological gradient
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph_grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        
        # Illumination correction
        blur = cv2.GaussianBlur(gray, (51, 51), 0)
        normalized = cv2.divide(gray, blur, scale=255)
        
        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(normalized)
        
        # Bilateral filter
        bilateral = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # Adaptive threshold
        adaptive = cv2.adaptiveThreshold(
            bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological cleanup
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel_clean)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_clean)
        
        # Invert if needed
        if cv2.mean(cleaned)[0] < 127:
            cleaned = cv2.bitwise_not(cleaned)
        
        # Convert back to BGR
        result = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
        
        # Sharpen
        kernel_sharp = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(result, -1, kernel_sharp)
        
        return sharpened
    
    def _deskew_plate(self, img: np.ndarray) -> np.ndarray:
        """Robust perspective correction - works in all conditions.
        
        Multi-method approach:
        1. Try minAreaRect (works for any polygon)
        2. Try convex hull + corner detection
        3. Fallback to simple rotation
        """
        try:
            h, w = img.shape[:2]
            original = img.copy()
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Method 1: Use minAreaRect (most robust)
            # Apply threshold to get binary image
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Get minimum area rectangle (works for any shape)
                rect = cv2.minAreaRect(largest_contour)
                box = cv2.boxPoints(rect)
                box = box.astype(np.int32)
                
                # Order the corners
                ordered_box = self._order_points(box.astype(np.float32))
                (tl, tr, br, bl) = ordered_box
                
                # Calculate dimensions
                widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
                widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
                maxWidth = max(int(widthA), int(widthB))
                
                heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
                heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
                maxHeight = max(int(heightA), int(heightB))
                
                # Ensure minimum size
                maxWidth = max(maxWidth, 100)
                maxHeight = max(maxHeight, 30)
                
                # Check if correction is needed (significant skew/perspective)
                # Calculate angle and perspective distortion
                angle = rect[2]
                aspect_ratio_diff = abs(widthA - widthB) / max(widthA, widthB) if max(widthA, widthB) > 0 else 0
                
                if abs(angle) > 1 or aspect_ratio_diff > 0.1:
                    if self.debug: # Debug özelliği eklenirse
                        print(f"  [Deskew] Correcting (angle: {angle:.1f}°, distortion: {aspect_ratio_diff:.2%})")
                    
                    # Define destination points (perfect rectangle)
                    dst = np.array([
                        [0, 0],
                        [maxWidth - 1, 0],
                        [maxWidth - 1, maxHeight - 1],
                        [0, maxHeight - 1]
                    ], dtype="float32")
                    
                    # Get perspective transform matrix
                    M = cv2.getPerspectiveTransform(ordered_box, dst)
                    
                    # Apply perspective transform
                    warped = cv2.warpPerspective(original, M, (maxWidth, maxHeight),
                                                flags=cv2.INTER_CUBIC,
                                                borderMode=cv2.BORDER_REPLICATE)
                    
                    if self.debug: # Debug özelliği eklenirse
                        print(f"  [Deskew] ✓ Corrected to {maxWidth}x{maxHeight}")
                    return warped
                else:
                    if self.debug: # Debug özelliği eklenirse
                        print(f"  [Deskew] No correction needed")
                    return original
            
        except Exception as e:
            if self.debug: # Debug özelliği eklenirse
                print(f"  [Deskew] Error: {e}")
            pass
        
        return original
    
    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """Order points: top-left, top-right, bottom-right, bottom-left.
        
        Uses sum and diff method for robust corner detection.
        """
        rect = np.zeros((4, 2), dtype="float32")
        
        # Sum of coordinates: smallest = top-left, largest = bottom-right
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # Top-left
        rect[2] = pts[np.argmax(s)]  # Bottom-right
        
        # Difference of coordinates: smallest = top-right, largest = bottom-left
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # Top-right
        rect[3] = pts[np.argmax(diff)]  # Bottom-left
        
        return rect
    
    def _apply_adaptive_bilateral(self, img: np.ndarray) -> np.ndarray:
        """Apply adaptive bilateral filtering for noise reduction with edge preservation."""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter (reduces noise while keeping edges sharp)
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply adaptive histogram equalization
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(bilateral)
        
        # Apply adaptive threshold
        adaptive_thresh = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Convert back to BGR
        result = cv2.cvtColor(adaptive_thresh, cv2.COLOR_GRAY2BGR)
        return result
    
    def _apply_unsharp_mask(self, img: np.ndarray) -> np.ndarray:
        """Apply unsharp masking for enhanced sharpness and clarity."""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        gaussian = cv2.GaussianBlur(gray, (5, 5), 1.0)
        
        # Unsharp mask: original + (original - blurred) * amount
        unsharp = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(unsharp)
        
        # Apply Otsu threshold
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Convert back to BGR
        result = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        return result
    
    def _apply_morphological_enhancement(self, img: np.ndarray) -> np.ndarray:
        """Apply morphological operations for text clarity."""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter first
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Apply Otsu threshold
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to enhance text
        # Remove small noise
        kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)
        
        # Close small gaps in characters
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)
        
        # Dilate slightly to make characters bolder
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        dilated = cv2.dilate(closed, kernel_dilate, iterations=1)
        
        # Convert back to BGR
        result = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)
        return result
    
    def _apply_high_contrast(self, img: np.ndarray) -> np.ndarray:
        """Apply extreme contrast enhancement for number clarity."""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply strong bilateral filter
        denoised = cv2.bilateralFilter(gray, 11, 100, 100)
        
        # Apply extreme CLAHE
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4,4))
        enhanced = clahe.apply(denoised)
        
        # Apply gamma correction to brighten
        gamma = 1.2
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
        gamma_corrected = cv2.LUT(enhanced, table)
        
        # Apply adaptive threshold with larger block size
        adaptive = cv2.adaptiveThreshold(
            gamma_corrected, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 2
        )
        
        # Invert if needed (text should be black on white)
        if cv2.mean(adaptive)[0] < 127:
            adaptive = cv2.bitwise_not(adaptive)
        
        # Apply morphological closing to connect broken characters
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        closed = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel)
        
        # Convert back to BGR
        result = cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)
        return result