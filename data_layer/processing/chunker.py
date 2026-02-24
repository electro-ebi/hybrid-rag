"""
Document chunking implementation.
"""

from typing import List, Dict, Any
import re

from ..core.base import BaseProcessor
from ..core.types import ProcessingResult, Chunk


class DocumentChunker(BaseProcessor):
    """Document chunker for splitting content into manageable pieces."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize chunker."""
        self.config = config or {}
        self.chunk_size = self.config.get('chunk_size', 512)
        self.chunk_overlap = self.config.get('chunk_overlap', 50)
    
    def process(self, content: Any, **kwargs) -> ProcessingResult:
        """Process content into chunks."""
        try:
            if isinstance(content, str):
                chunks = self._chunk_text(content)
            elif isinstance(content, dict) and 'text' in content:
                chunks = self._chunk_text(content['text'])
            else:
                chunks = self._chunk_text(str(content))
            
            return ProcessingResult(
                success=True,
                content={"chunks": chunks},
                metadata={
                    "processor": "document_chunker",
                    "chunk_count": len(chunks),
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap
                },
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
        return True  # Chunker supports all file types
    
    def _chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """Split text into chunks."""
        # Clean text first
        text = re.sub(r'\s+', ' ', text).strip()
        
        if len(text) <= self.chunk_size:
            return [{"content": text, "chunk_id": "0"}]
        
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size
            
            if end >= len(text):
                # Last chunk
                chunk_text = text[start:].strip()
                if chunk_text:
                    chunks.append({
                        "content": chunk_text,
                        "chunk_id": str(chunk_id)
                    })
                break
            
            # Try to break at word boundary
            chunk_text = text[start:end]
            last_space = chunk_text.rfind(' ')
            
            if last_space > 0 and (end - start - last_space) < 50:
                # Adjust to word boundary
                chunk_text = chunk_text[:last_space]
                end = start + last_space
            
            chunk_text = chunk_text.strip()
            if chunk_text:
                chunks.append({
                    "content": chunk_text,
                    "chunk_id": str(chunk_id)
                })
            
            # Move start position with overlap
            start = max(start + 1, end - self.chunk_overlap)
            chunk_id += 1
        
        return chunks
