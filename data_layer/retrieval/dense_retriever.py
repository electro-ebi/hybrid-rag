"""
Dense retriever implementation.
"""

from typing import List, Dict, Any

from ..core.base import BaseRetriever
from ..core.types import RetrievalResult, RetrievalMode
from ..storage.vector_store import VectorStore


class DenseRetriever(BaseRetriever):
    """Dense retriever using vector embeddings."""
    
    def __init__(self, vector_store: VectorStore, config: Dict[str, Any] = None):
        """Initialize dense retriever."""
        self.vector_store = vector_store
        self.config = config or {}
    
    def retrieve(self, query: str, mode: RetrievalMode, top_k: int = 5, **kwargs) -> RetrievalResult:
        """Retrieve chunks using dense search."""
        if mode != RetrievalMode.DENSE:
            raise ValueError("DenseRetriever only supports DENSE mode")
        
        # Query vector store
        results = self.vector_store.query(query, top_k)
        
        return RetrievalResult(
            chunks=results.get('documents', []),
            chunk_ids=results.get('ids', []),
            mode=mode,
            query=query,
            scores=results.get('distances', []),
            metadata=results.get('metadatas', {}),
            latency_ms=0.0
        )
    
    def is_available(self) -> bool:
        """Check if dense retriever is available."""
        return self.vector_store is not None
