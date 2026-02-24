"""
Sparse retriever implementation.
"""

from typing import List, Dict, Any

from ..core.base import BaseRetriever
from ..core.types import RetrievalResult, RetrievalMode
from ..storage.sparse_store import SparseStore


class SparseRetriever(BaseRetriever):
    """Sparse retriever using keyword search."""
    
    def __init__(self, sparse_store: SparseStore, config: Dict[str, Any] = None):
        """Initialize sparse retriever."""
        self.sparse_store = sparse_store
        self.config = config or {}
    
    def retrieve(self, query: str, mode: RetrievalMode, top_k: int = 5, **kwargs) -> RetrievalResult:
        """Retrieve chunks using sparse search."""
        if mode != RetrievalMode.SPARSE:
            raise ValueError("SparseRetriever only supports SPARSE mode")
        
        # Search sparse store
        results = self.sparse_store.search(query, top_k)
        
        return RetrievalResult(
            chunks=[r['content'] for r in results],
            chunk_ids=[r['chunk_uid'] for r in results],
            mode=mode,
            query=query,
            scores=[r['score'] for r in results],
            metadata={},
            latency_ms=0.0
        )
    
    def is_available(self) -> bool:
        """Check if sparse retriever is available."""
        return self.sparse_store is not None
