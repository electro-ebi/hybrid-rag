"""
Hybrid retriever implementation combining dense and sparse retrieval.
"""

import time
from typing import List, Dict, Any, Optional

from ..core.base import BaseRetriever
from ..core.types import RetrievalResult, RetrievalMode
from ..storage.sparse_store import SparseStore
from ..storage.vector_store import VectorStore
from .query_processor import QueryProcessor
from .fusion import RRFFusion
from ..embeddings import generate_embedding


class HybridRetriever(BaseRetriever):
    """Hybrid retriever combining dense and sparse search."""
    
    def __init__(
        self, 
        sparse_store: SparseStore,
        vector_store: VectorStore,
        query_processor: QueryProcessor,
        config: Dict[str, Any] = None
    ):
        """Initialize hybrid retriever."""
        self.sparse_store = sparse_store
        self.vector_store = vector_store
        self.query_processor = query_processor
        self.config = config or {}
        self.fusion = RRFFusion(self.config.get('rrf_k', 60))
    
    def retrieve(
        self, 
        query: str, 
        mode: RetrievalMode,
        top_k: int = 5,
        **kwargs
    ) -> RetrievalResult:
        """Retrieve chunks using specified mode."""
        start_time = time.perf_counter()
        
        if mode == RetrievalMode.DENSE:
            return self._retrieve_dense(query, top_k, start_time)
        elif mode == RetrievalMode.SPARSE:
            return self._retrieve_sparse(query, top_k, start_time)
        elif mode == RetrievalMode.HYBRID:
            return self._retrieve_hybrid(query, top_k, start_time)
        elif mode == RetrievalMode.HYBRID_RERANK:
            return self._retrieve_hybrid_rerank(query, top_k, start_time)
        else:
            raise ValueError(f"Unsupported retrieval mode: {mode}")
    
    def _retrieve_dense(self, query: str, top_k: int, start_time: float) -> RetrievalResult:
        """Retrieve using dense search only."""
        try:
            # Generate embedding
            query_embedding = generate_embedding(query)
            results = self.vector_store.query(query, top_k)
            
            chunks = results.get('documents', [])
            chunk_ids = results.get('ids', [])
            scores = results.get('distances', [])
            
            latency = (time.perf_counter() - start_time) * 1000
            
            return RetrievalResult(
                chunks=chunks,
                chunk_ids=chunk_ids,
                mode=RetrievalMode.DENSE,
                query=query,
                scores=scores,
                metadata=results.get('metadatas', {}),
                latency_ms=latency
            )
        except Exception as e:
            print(f"Dense retrieval failed: {e}")
            return RetrievalResult(
                chunks=[], chunk_ids=[], mode=RetrievalMode.DENSE,
                query=query, scores=[], metadata={}, latency_ms=0
            )
    
    def _retrieve_sparse(self, query: str, top_k: int, start_time: float) -> RetrievalResult:
        """Retrieve using sparse search only."""
        try:
            # Preprocess query for sparse retrieval
            processed_query = self.query_processor.preprocess_for_sparse(query)
            
            results = self.sparse_store.search(processed_query, top_k)
            
            chunks = [r['content'] for r in results]
            chunk_ids = [r['chunk_uid'] for r in results]
            scores = [r['score'] for r in results]
            
            latency = (time.perf_counter() - start_time) * 1000
            
            return RetrievalResult(
                chunks=chunks,
                chunk_ids=chunk_ids,
                mode=RetrievalMode.SPARSE,
                query=query,
                scores=scores,
                metadata={},
                latency_ms=latency
            )
        except Exception as e:
            print(f"Sparse retrieval failed: {e}")
            return RetrievalResult(
                chunks=[], chunk_ids=[], mode=RetrievalMode.SPARSE,
                query=query, scores=[], metadata={}, latency_ms=0
            )
    
    def _retrieve_hybrid(self, query: str, top_k: int, start_time: float) -> RetrievalResult:
        """Retrieve using hybrid search."""
        try:
            # Get dense results
            dense_results = self.vector_store.query(query, top_k * 2)
            dense_tuples = list(zip(
                dense_results.get('ids', []),
                dense_results.get('documents', [])
            ))
            
            # Get sparse results
            processed_query = self.query_processor.preprocess_for_sparse(query)
            sparse_results = self.sparse_store.search(processed_query, top_k * 2)
            sparse_tuples = list(zip(
                [r['chunk_uid'] for r in sparse_results],
                [r['content'] for r in sparse_results]
            ))
            
            # Fuse results
            final_ids, final_chunks = self.fusion.fuse(
                dense_tuples, sparse_tuples, top_k
            )
            
            latency = (time.perf_counter() - start_time) * 1000
            
            return RetrievalResult(
                chunks=final_chunks,
                chunk_ids=final_ids,
                mode=RetrievalMode.HYBRID,
                query=query,
                scores=[],  # RRF doesn't provide traditional scores
                metadata={},
                latency_ms=latency
            )
        except Exception as e:
            print(f"Hybrid retrieval failed: {e}")
            return RetrievalResult(
                chunks=[], chunk_ids=[], mode=RetrievalMode.HYBRID,
                query=query, scores=[], metadata={}, latency_ms=0
            )
    
    def _retrieve_hybrid_rerank(self, query: str, top_k: int, start_time: float) -> RetrievalResult:
        """Retrieve using hybrid search with reranking."""
        try:
            # Get hybrid results first (more candidates)
            hybrid_result = self._retrieve_hybrid(query, top_k * 2, start_time)
            
            # Apply reranking if available
            if 'reranker' in self.config:
                from ..cross_encoder_reranker import get_reranker
                reranker = get_reranker()
                reranked_chunks = reranker.rerank(query, hybrid_result.chunks)
                final_chunks = reranked_chunks[:top_k]
                final_ids = hybrid_result.chunk_ids[:top_k]
            else:
                final_chunks = hybrid_result.chunks[:top_k]
                final_ids = hybrid_result.chunk_ids[:top_k]
            
            latency = (time.perf_counter() - start_time) * 1000
            
            return RetrievalResult(
                chunks=final_chunks,
                chunk_ids=final_ids,
                mode=RetrievalMode.HYBRID_RERANK,
                query=query,
                scores=[],
                metadata={},
                latency_ms=latency
            )
        except Exception as e:
            print(f"Hybrid rerank retrieval failed: {e}")
            return RetrievalResult(
                chunks=[], chunk_ids=[], mode=RetrievalMode.HYBRID_RERANK,
                query=query, scores=[], metadata={}, latency_ms=0
            )
    
    def is_available(self) -> bool:
        """Check if hybrid retriever is available."""
        return (self.sparse_store is not None and 
                self.vector_store is not None and 
                self.query_processor is not None)
