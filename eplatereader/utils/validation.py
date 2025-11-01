"""Turkish license plate validation and correction."""

import re
from typing import Optional, Dict
from dataclasses import dataclass

try:
    from Levenshtein import distance as levenshtein_distance
except ImportError:
    # Fallback implementation
    def levenshtein_distance(s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]


# Turkish city codes (01-81)
VALID_CITY_CODES = set(range(1, 82))

# Forbidden letters in Turkish plates
FORBIDDEN_LETTERS = {'I', 'O', 'Q', 'W'}
VALID_LETTERS = set('ABCDEFGHJKLMNPRSTUVYZ')

# OCR confusion pairs
CONFUSION_PAIRS = {
    'O': '0', 'I': '1', 'S': '5', 'B': '8', 'Z': '2',
    '0': 'O', '1': 'I', '5': 'S', '8': 'B', '2': 'Z',
    'D': '0', 'G': '6', 'Q': 'O'
}


@dataclass
class PlateValidationResult:
    """Plate validation result."""
    is_valid: bool
    original_text: str
    corrected_text: Optional[str] = None
    confidence: float = 0.0
    format_type: Optional[str] = None
    errors: list = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class TurkishPlateValidator:
    """Validator for Turkish license plates."""
    
    # Regex patterns for Turkish plates
    PATTERNS = {
        'new_3letter': re.compile(r'^(\d{2})\s*([A-Z]{3})\s*(\d{2,3})$'),  # 34 ABC 123
        'new_2letter': re.compile(r'^(\d{2})\s*([A-Z]{2})\s*(\d{3,4})$'),  # 34 AB 1234
        'old_1letter': re.compile(r'^(\d{2})\s*([A-Z]{1})\s*(\d{4})$'),     # 34 A 1234
        'old_2letter': re.compile(r'^(\d{2})\s*([A-Z]{2})\s*(\d{4})$'),    # 34 AB 1234
    }
    
    def validate(self, plate_text: str) -> PlateValidationResult:
        """Validate and correct Turkish license plate."""
        # Clean input
        cleaned = self._clean_text(plate_text)
        
        # Try to match patterns
        match_result = self._match_pattern(cleaned)
        
        if match_result['matched']:
            validation = self._validate_components(
                match_result['city_code'],
                match_result['letters'],
                match_result['numbers']
            )
            
            if validation['valid']:
                corrected = self._format_plate(
                    match_result['city_code'],
                    match_result['letters'],
                    match_result['numbers']
                )
                
                return PlateValidationResult(
                    is_valid=True,
                    original_text=plate_text,
                    corrected_text=corrected,
                    confidence=validation['confidence'],
                    format_type=match_result['format_type']
                )
        
        # Try correction
        corrected_result = self._attempt_correction(cleaned)
        
        if corrected_result['success']:
            return PlateValidationResult(
                is_valid=True,
                original_text=plate_text,
                corrected_text=corrected_result['corrected'],
                confidence=corrected_result['confidence'],
                format_type=corrected_result['format_type'],
                errors=['Corrected from OCR errors']
            )
        
        return PlateValidationResult(
            is_valid=False,
            original_text=plate_text,
            errors=['No valid pattern matched']
        )
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize plate text."""
        text = ' '.join(text.split())
        text = text.upper()
        text = re.sub(r'[^A-Z0-9\s]', '', text)
        return text.strip()
    
    def _match_pattern(self, text: str) -> Dict:
        """Try to match text against known patterns."""
        for format_name, pattern in self.PATTERNS.items():
            match = pattern.match(text)
            if match:
                groups = match.groups()
                format_type = 'new' if 'new' in format_name else 'old'
                
                return {
                    'matched': True,
                    'format_type': format_type,
                    'city_code': groups[0],
                    'letters': groups[1],
                    'numbers': groups[2]
                }
        
        return {'matched': False}
    
    def _validate_components(self, city_code: str, letters: str, numbers: str) -> Dict:
        """Validate individual components."""
        errors = []
        confidence = 1.0
        
        # Validate city code
        try:
            city_num = int(city_code)
            if city_num not in VALID_CITY_CODES:
                errors.append(f'Invalid city code: {city_code}')
                confidence *= 0.5
        except ValueError:
            return {'valid': False, 'errors': errors, 'confidence': 0.0}
        
        # Validate letters
        for letter in letters:
            if letter in FORBIDDEN_LETTERS:
                errors.append(f'Forbidden letter: {letter}')
                confidence *= 0.7
            elif letter not in VALID_LETTERS:
                errors.append(f'Invalid letter: {letter}')
                confidence *= 0.5
        
        # Validate numbers
        if not numbers.isdigit():
            return {'valid': False, 'errors': errors, 'confidence': 0.0}
        
        is_valid = len(errors) == 0 or confidence > 0.6
        
        return {
            'valid': is_valid,
            'errors': errors,
            'confidence': confidence
        }
    
    def _format_plate(self, city_code: str, letters: str, numbers: str) -> str:
        """Format plate text."""
        return f"{city_code}{letters}{numbers}"
    
    def _attempt_correction(self, text: str) -> Dict:
        """Attempt to correct OCR errors."""
        variations = self._generate_variations(text)
        
        for variation in variations:
            match_result = self._match_pattern(variation)
            if match_result['matched']:
                validation = self._validate_components(
                    match_result['city_code'],
                    match_result['letters'],
                    match_result['numbers']
                )
                
                if validation['valid']:
                    corrected = self._format_plate(
                        match_result['city_code'],
                        match_result['letters'],
                        match_result['numbers']
                    )
                    
                    edit_dist = levenshtein_distance(text, variation)
                    confidence = max(0.5, 1.0 - (edit_dist * 0.1))
                    
                    return {
                        'success': True,
                        'corrected': corrected,
                        'confidence': confidence * validation['confidence'],
                        'format_type': match_result['format_type']
                    }
        
        return {'success': False}
    
    def _generate_variations(self, text: str) -> list:
        """Generate possible variations by correcting OCR errors."""
        variations = [text]
        
        # Try all confusion pairs
        for old_char, new_char in CONFUSION_PAIRS.items():
            if old_char in text:
                variations.append(text.replace(old_char, new_char))
        
        # Try positional corrections (O/0 in different positions)
        parts = text.split()
        if len(parts) >= 2:
            # City code should be digits
            city_part = parts[0].replace('O', '0').replace('I', '1')
            
            # Letters part
            if len(parts) >= 2:
                letter_part = parts[1].replace('0', 'O').replace('1', 'I')
                
                # Numbers part
                if len(parts) >= 3:
                    number_part = parts[2].replace('O', '0').replace('I', '1')
                    variations.append(f"{city_part} {letter_part} {number_part}")
                else:
                    variations.append(f"{city_part} {letter_part}")
        
        return list(set(variations))


def validate_plate(plate_text: str) -> PlateValidationResult:
    """Convenience function to validate a plate."""
    validator = TurkishPlateValidator()
    return validator.validate(plate_text)