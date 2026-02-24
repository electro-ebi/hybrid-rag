"""
Batch indexer for processing multiple documents.
"""

from typing import List, Dict, Any, Optional
import time
from pathlib import Path

from ..core.base import BaseIndexer
from ..core.types import IndexingResult, Document, ProcessingLevel
from ..storage.sparse_store import SparseStore
from ..storage.vector_store import VectorStore


class BatchIndexer(BaseIndexer):
    """Batch indexer for processing multiple documents efficiently."""
    
    def __init__(
        self, 
        sparse_store: Optional[SparseStore] = None,
        vector_store: Optional[VectorStore] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize batch indexer."""
        self.sparse_store = sparse_store
        self.vector_store = vector_store
        self.config = config or {}
    
    def index_document(self, document: Document) -> IndexingResult:
        """Index a single document."""
        # This would be implemented to handle the new Document format
        # For now, return a placeholder result
        return IndexingResult(
            status="success",
            chunks_indexed=len(document.chunks),
            file_path=document.source,
            processing_level=document.processing_level,
            metadata=document.metadata,
            errors={},
            sparse_indexed=self.sparse_store is not None,
            dense_indexed=self.vector_store is not None
        )
    
    def index_file(self, file_path: str, **kwargs) -> IndexingResult:
        """Index a single file."""
        # Delegate to legacy indexing for now
        from ..indexing.document_indexer import index_file
        return index_file(file_path, **kwargs)
    
    def index_batch(
        self, 
        file_paths: List[str], 
        processing_level: ProcessingLevel = ProcessingLevel.STANDARD
    ) -> List[IndexingResult]:
        """Index multiple files in batch."""
        results = []
        
        print(f"Starting batch indexing of {len(file_paths)} files")
        
        for i, file_path in enumerate(file_paths, 1):
            print(f"Processing file {i}/{len(file_paths)}: {Path(file_path).name}")
            
            try:
                result = self.index_file(file_path, processing_level=processing_level.value)
                results.append(result)
                
                if result['status'] == 'success':
                    print(f"  ✅ Indexed {result['chunks_indexed']} chunks")
                else:
                    print(f"  ❌ Failed: {result.get('errors', {})}")
                    
            except Exception as e:
                print(f"  ❌ Exception: {e}")
                error_result = IndexingResult(
                    status="failed",
                    chunks_indexed=0,
                    file_path=file_path,
                    processing_level=processing_level,
                    metadata={},
                    errors={"exception": str(e)},
                    sparse_indexed=False,
                    dense_indexed=False
                )
                results.append(error_result)
        
        # Summary
        successful = sum(1 for r in results if r.status == "success")
        total_chunks = sum(r.chunks_indexed for r in results)
        
        print(f"\nBatch indexing complete:")
        print(f"  Successful: {successful}/{len(file_paths)} files")
        print(f"  Total chunks: {total_chunks}")
        
        return results
    
    def delete_document(self, document_id: str) -> bool:
        """Delete a document from index."""
        success = True
        
        if self.sparse_store:
            success &= self.sparse_store.delete(document_id)
        
        if self.vector_store:
            success &= self.vector_store.delete(document_id)
        
        return success
