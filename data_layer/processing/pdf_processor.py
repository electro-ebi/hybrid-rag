"""
PDF processor implementation.
"""

import PyPDF2
from typing import Dict, Any, List
from pathlib import Path

from logger import get_logger

logger = get_logger("pdf_processor")

class PDFProcessor:
    """PDF processor for text extraction."""
    
    def __init__(self):
        self.name = "PDFProcessor"
    
    def extract_text(self, file_path: str) -> str:
        """Extract text from PDF file."""
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                return text
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {e}")
            return ""
    
    def process(self, file_path: str) -> Dict[str, Any]:
        """Process PDF file and return extracted data."""
        text = self.extract_text(file_path)
        
        return {
            'text': text,
            'file_path': file_path,
            'processor': self.name,
            'success': len(text.strip()) > 0
        }
