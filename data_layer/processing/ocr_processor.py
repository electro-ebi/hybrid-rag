"""
Advanced OCR processor for extracting text from images, charts, and graphs.
Supports multiple image formats and preprocessing for better accuracy.
"""
from __future__ import annotations

import base64
import io
import os
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter

from logger import get_logger

logger = get_logger("ocr_processor")

class OCRProcessor:
    """Enterprise-grade OCR with image preprocessing."""
    
    def __init__(self):
        # Configure Tesseract for better results
        self.tesseract_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
        self.tesseract_config_dense = r'--oem 3 --psm 4 -c preserve_interword_spaces=1'
        
    def extract_text_from_image(
        self, 
        image_path: str | None = None, 
        image_base64: str | None = None,
        preprocess: bool = True,
        dense_text: bool = False
    ) -> Dict[str, Any]:
        """
        Extract text from image with optional preprocessing.
        
        Args:
            image_path: Path to image file
            image_base64: Base64 encoded image string
            preprocess: Apply image preprocessing for better OCR
            dense_text: Use dense text configuration for documents
            
        Returns:
            Dictionary with extracted text, confidence, and metadata
        """
        try:
            # Load image
            image = self._load_image(image_path, image_base64)
            if image is None:
                return {"text": "", "confidence": 0.0, "error": "Failed to load image"}
            
            original_size = image.size
            
            # Preprocess if requested
            if preprocess:
                image = self._preprocess_image(image)
            
            # Choose Tesseract config
            config = self.tesseract_config_dense if dense_text else self.tesseract_config
            
            # Extract text with confidence
            text = pytesseract.image_to_string(image, config=config)
            
            # Get detailed data including confidence
            data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
            
            # Calculate average confidence
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            # Extract words with positions for structured data
            words_with_positions = []
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 0 and data['text'][i].strip():
                    words_with_positions.append({
                        'text': data['text'][i],
                        'confidence': int(data['conf'][i]),
                        'bbox': {
                            'x': data['left'][i],
                            'y': data['top'][i],
                            'width': data['width'][i],
                            'height': data['height'][i]
                        }
                    })
            
            result = {
                "text": text.strip(),
                "confidence": avg_confidence,
                "original_size": original_size,
                "processed_size": image.size,
                "word_count": len(words_with_positions),
                "words_with_positions": words_with_positions,
                "preprocessing_applied": preprocess
            }
            
            logger.info(
                "OCR extracted %d words with %.1f%% confidence",
                len(words_with_positions), avg_confidence
            )
            
            return result
            
        except Exception as e:
            logger.error("OCR processing failed: %s", e)
            return {"text": "", "confidence": 0.0, "error": str(e)}
    
    def extract_structured_data(self, image_path: str | None = None, image_base64: str | None = None) -> Dict[str, Any]:
        """
        Extract structured data from charts, tables, and graphs.
        Enhanced processing for data visualization content.
        """
        try:
            # Load and preprocess for data extraction
            image = self._load_image(image_path, image_base64)
            if image is None:
                return {"structured_data": [], "error": "Failed to load image"}
            
            # Apply specialized preprocessing for charts/graphs
            processed_image = self._preprocess_for_data_extraction(image)
            
            # Extract text with detailed layout info
            data = pytesseract.image_to_data(
                processed_image, 
                config=r'--oem 3 --psm 4 -c preserve_interword_spaces=1',
                output_type=pytesseract.Output.DICT
            )
            
            # Group words into potential data elements
            structured_data = self._group_data_elements(data)
            
            # Detect if this looks like a chart/table
            content_type = self._detect_content_type(structured_data)
            
            result = {
                "content_type": content_type,
                "structured_data": structured_data,
                "text_elements": len(structured_data)
            }
            
            logger.info(
                "Structured data extraction: %s with %d elements",
                content_type, len(structured_data)
            )
            
            return result
            
        except Exception as e:
            logger.error("Structured data extraction failed: %s", e)
            return {"structured_data": [], "error": str(e)}
    
    def _load_image(self, image_path: str | None, image_base64: str | None) -> Image.Image | None:
        """Load image from path or base64 string."""
        try:
            if image_base64:
                # Decode base64
                image_data = base64.b64decode(image_base64)
                return Image.open(io.BytesIO(image_data))
            elif image_path and os.path.exists(image_path):
                return Image.open(image_path)
            else:
                logger.error("No valid image source provided")
                return None
        except Exception as e:
            logger.error("Failed to load image: %s", e)
            return None
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Apply preprocessing for better OCR accuracy."""
        try:
            # Convert to grayscale
            if image.mode != 'L':
                image = image.convert('L')
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.5)
            
            # Apply slight blur to reduce noise
            image = image.filter(ImageFilter.MedianFilter(size=3))
            
            # Resize if too small (minimum 300 DPI equivalent)
            width, height = image.size
            if width < 1000 or height < 1000:
                scale = max(1000/width, 1000/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            return image
            
        except Exception as e:
            logger.warning("Image preprocessing failed, using original: %s", e)
            return image
    
    def _preprocess_for_data_extraction(self, image: Image.Image) -> Image.Image:
        """Specialized preprocessing for charts and graphs."""
        try:
            # Convert to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Convert to grayscale
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive threshold for better text detection in charts
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # Remove small noise
            kernel = np.ones((2, 2), np.uint8)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # Convert back to PIL
            return Image.fromarray(cleaned)
            
        except Exception as e:
            logger.warning("Data extraction preprocessing failed, using original: %s", e)
            return image
    
    def _group_data_elements(self, data: Dict[str, List]) -> List[Dict[str, Any]]:
        """Group OCR words into logical data elements."""
        elements = []
        
        for i in range(len(data['text'])):
            if int(data['conf'][i]) > 30 and data['text'][i].strip():  # Confidence threshold
                text = data['text'][i].strip()
                
                # Skip very short text (likely noise)
                if len(text) < 2:
                    continue
                
                element = {
                    'text': text,
                    'confidence': int(data['conf'][i]),
                    'position': {
                        'x': data['left'][i],
                        'y': data['top'][i],
                        'width': data['width'][i],
                        'height': data['height'][i]
                    }
                }
                
                # Try to identify data types
                element['type'] = self._classify_text_element(text)
                
                elements.append(element)
        
        return elements
    
    def _classify_text_element(self, text: str) -> str:
        """Classify text element type (number, label, etc.)."""
        # Check if it's a number
        try:
            float(text.replace('%', '').replace(',', ''))
            return 'number'
        except ValueError:
            pass
        
        # Check if it's a percentage
        if '%' in text:
            return 'percentage'
        
        # Check if it's a date
        if any(char in text for char in '/-') and len(text) > 5:
            return 'date'
        
        # Check if it's all caps (likely label)
        if text.isupper() and len(text) > 1:
            return 'label'
        
        # Default to text
        return 'text'
    
    def _detect_content_type(self, elements: List[Dict[str, Any]]) -> str:
        """Detect if content is chart, table, or general text."""
        if not elements:
            return 'unknown'
        
        # Count different element types
        types = [elem['type'] for elem in elements]
        number_count = types.count('number')
        total_count = len(types)
        
        # If many numbers, likely chart/table
        if number_count / total_count > 0.3:
            return 'chart_data'
        
        # If structured layout, likely table
        y_positions = [elem['position']['y'] for elem in elements]
        unique_rows = len(set([y // 50 for y in y_positions]))  # Group by rows
        
        if unique_rows > 2 and number_count > 0:
            return 'table'
        
        return 'general_text'

# Global instance
_ocr_instance = None

def get_ocr_processor() -> OCRProcessor:
    """Get or create OCR processor instance."""
    global _ocr_instance
    if _ocr_instance is None:
        _ocr_instance = OCRProcessor()
    return _ocr_instance
