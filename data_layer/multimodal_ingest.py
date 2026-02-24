"""
Multi-modal ingestion pipeline for text, images, charts, and complex documents.
Handles PDFs with images, OCR, vision analysis, and data extraction.
"""
from __future__ import annotations

import os
from typing import Dict, Any, List, Optional

from logger import get_logger
from .loaders.loader_factory import load_file

logger = get_logger("multimodal_ingest")

# Simplified multimodal ingestor without complex legacy dependencies
class MultiModalIngestor:
    """Multi-modal document ingestion pipeline."""
    
    def __init__(self):
        self.name = "MultiModalIngestor"
    
    def ingest_document(self, file_path: str) -> Dict[str, Any]:
        """Ingest a document and extract content."""
        try:
            # Use the loader factory to process the file
            result = load_file(file_path)
            
            return {
                'file_path': file_path,
                'content': result.get('content', ''),  # Fixed: was 'text', now 'content'
                'metadata': result.get('metadata', {}),
                'success': True,
                'processor': self.name
            }
            
        except Exception as e:
            logger.error(f"Error ingesting {file_path}: {e}")
            return {
                'file_path': file_path,
                'content': '',
                'metadata': {},
                'success': False,
                'error': str(e),
                'processor': self.name
            }
    
    def batch_ingest(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Ingest multiple documents."""
        results = []
        
        for file_path in file_paths:
            result = self.ingest_document(file_path)
            results.append(result)
        
        return results
    
    def add_chunks(self, chunks: List[Dict[str, Any]]) -> bool:
        """Add chunks to storage (simplified version)."""
        try:
            from data_layer.storage.sparse_store import add_sparse_chunks
            from data_layer.storage.vector_store import add_chunks
            
            # Add to sparse store
            sparse_chunks = []
            for chunk in chunks:
                sparse_chunk = {
                    'uid': chunk.get('chunk_uid', f"chunk_{len(sparse_chunks)}"),
                    'title': chunk.get('title', 'Unknown'),
                    'content': chunk.get('content', ''),
                    'metadata': chunk.get('metadata', '')
                }
                sparse_chunks.append(sparse_chunk)
            
            add_sparse_chunks(sparse_chunks, 'data/sparse_index')
            
            # Add to vector store
            add_chunks(chunks)
            
            logger.info(f"Added {len(chunks)} chunks to storage")
            return True
            
        except Exception as e:
            logger.error(f"Error adding chunks: {e}")
            return False
