"""
Text processing utilities.
"""

import re
from typing import List, Dict, Any

from ..core.base import BaseProcessor
from ..core.types import ProcessingResult


class TextProcessor(BaseProcessor):
    """Text processor for document content."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize text processor."""
        self.config = config or {}
    
    def process(self, content: Any, **kwargs) -> ProcessingResult:
        """Process text content."""
        try:
            if isinstance(content, str):
                processed_text = self._clean_text(content)
            else:
                processed_text = str(content)
            
            return ProcessingResult(
                success=True,
                content={"text": processed_text},
                metadata={"processor": "text_processor"},
                processing_time=0.0,
                errors=[]
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                content={},
                metadata={},
                processing_time=0.0,
                errors=[str(e)]
            )
    
    def supports_file_type(self, file_type: str) -> bool:
        """Check if processor supports file type."""
        return file_type.lower() in ['txt', 'md', 'text']
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-]', '', text)
        # Strip leading/trailing whitespace
        text = text.strip()
        return text
